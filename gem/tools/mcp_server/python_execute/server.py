#!/usr/bin/env python3
"""
Python Execute MCP Server

An MCP server that provides Python code execution capabilities in an isolated environment.
Based on mcpbench_dev/utils/aux_tools/python_interpretor.py
"""

import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, Optional

# Suppress FastMCP banner and reduce log level (must be before import)
os.environ["FASTMCP_SHOW_CLI_BANNER"] = "false"
os.environ["FASTMCP_LOG_LEVEL"] = "ERROR"

# Suppress logging unless verbose mode is enabled
if os.environ.get('LOCA_QUIET', '').lower() in ('1', 'true', 'yes'):
    logging.basicConfig(level=logging.ERROR, force=True)
    logging.getLogger().setLevel(logging.ERROR)
    for _logger_name in ["mcp", "fastmcp", "mcp.server", "mcp.client", "uvicorn", "uvicorn.error", "uvicorn.access"]:
        logging.getLogger(_logger_name).setLevel(logging.ERROR)
    # Suppress rich console output
    try:
        from rich.console import Console
        Console._force_terminal = False
    except ImportError:
        pass

# Add parent directory to path for imports
gem_root = Path(__file__).parent.parent.parent.parent.parent
if str(gem_root) not in sys.path:
    sys.path.insert(0, str(gem_root))

from fastmcp import FastMCP

# Load bwrap_confine directly from its file path instead of via the gem.tools
# package: importing the package runs gem/tools/__init__.py, which pulls in
# mcp_tool and replaced this server's sys.stdout with a filter object, crashing
# the MCP stdio transport at startup (the server then silently ran without
# being discoverable).
import importlib.util as _importlib_util

_bwrap_path = Path(__file__).parent.parent / "bwrap_confine.py"
_bwrap_spec = _importlib_util.spec_from_file_location("bwrap_confine", str(_bwrap_path))
_bwrap_module = _importlib_util.module_from_spec(_bwrap_spec)
_bwrap_spec.loader.exec_module(_bwrap_module)
bwrap_usable = _bwrap_module.bwrap_usable
build_bwrap_argv = _bwrap_module.build_bwrap_argv

# Create FastMCP server
app = FastMCP("Python Execute Server")

# Default workspace (can be overridden by environment variable)
DEFAULT_WORKSPACE = "."

# Appended when output contains a bwrap EROFS denial, to steer the model back
# into the writable workspace instead of retrying the same out-of-bounds path.
_ROFS_HINT = (
    "A 'Read-only file system' error means the write target is OUTSIDE the agent "
    "workspace. Writes are confined to the workspace: {workspace} (and its "
    "subdirectories). Use a path inside it -- relative paths already resolve there."
)


def get_workspace() -> str:
    """Get the workspace directory from environment or use default."""
    return os.environ.get("PYTHON_EXECUTE_WORKSPACE", DEFAULT_WORKSPACE)


def _build_write_guard_preamble(workspace_path: str) -> str:
    """Return a single line of code that confines file writes to ``workspace_path``.

    Prepended to the model's code before it runs in the ``uv run`` subprocess.
    Installs a ``builtins.open`` wrapper that denies write-mode opens whose
    realpath is outside the workspace, and chdir's into the workspace so
    relative paths resolve there. This matches the Python-level guard the
    programmatic_tool_calling StatefulSandbox already applies (same
    PermissionError semantics); like that guard it only covers ``open()``, not
    ``os.open``/``subprocess``/``shutil``.

    The guard body is emitted as one physical line (an ``exec`` of a source
    string) so it shifts user-code line numbers in tracebacks by exactly one.
    """
    guard_src = (
        "import os as _os, builtins as _bi\n"
        f"_ws_real = _os.path.realpath({workspace_path!r})\n"
        "_orig_open = _bi.open\n"
        "_write_modes = frozenset('wax+')\n"
        "def _guarded_open(file, mode='r', *args, **kwargs):\n"
        "    if _write_modes & set(str(mode)):\n"
        "        _abs = _os.path.realpath(str(file))\n"
        "        if not (_abs.startswith(_ws_real + _os.sep) or _abs == _ws_real):\n"
        "            raise PermissionError(f'Write denied: {file!r} is outside the agent workspace {_ws_real!r}.')\n"
        "    return _orig_open(file, mode, *args, **kwargs)\n"
        "_bi.open = _guarded_open\n"
        "try:\n"
        "    _os.chdir(_ws_real)\n"
        "except OSError:\n"
        "    pass\n"
    )
    return f"exec(compile({guard_src!r}, '<loca_write_guard>', 'exec'))\n"


@app.tool()
def python_execute(
    code: Annotated[str, "Python code to execute (can be directly pasted into a .py file)"],
    filename: Annotated[Optional[str], "Filename for the Python file (including .py extension). If not provided, a random UUID will be used."] = None,
    timeout: Annotated[Optional[int], "Maximum execution time in seconds. Cannot exceed 120 seconds. If a value greater than 120 is provided, it will be automatically limited to 120 seconds. Default is 30 seconds."] = 30
) -> str:
    """Execute Python code directly under the agent workspace, and returns stdout, stderr, return code, and execution time in a structured format. This runs an isolated script and CANNOT access MCP tools (there is no `tools` object); to call env tools, use code_execution instead."""
    try:
        # Use provided filename or generate random UUID
        if filename is None:
            filename = f"{uuid.uuid4()}.py"

        # Ensure timeout does not exceed 120 seconds
        if timeout is None:
            timeout = 30
        if timeout > 120:
            timeout = 120

        # Ensure filename ends with .py
        if not filename.endswith(".py"):
            filename += ".py"

        # Get working directory
        agent_workspace = get_workspace()
        agent_workspace = os.path.abspath(agent_workspace)

        # Create .python_tmp directory
        tmp_dir = os.path.join(agent_workspace, '.python_tmp')
        os.makedirs(tmp_dir, exist_ok=True)

        # Create Python file. A write-guard preamble is prepended so file
        # writes from the model's code are confined to the workspace (parity
        # with the programmatic_tool_calling sandbox guard).
        file_path = os.path.join(tmp_dir, filename)
        guarded_code = _build_write_guard_preamble(agent_workspace) + code
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(guarded_code)

        # Record start time
        start_time = time.time()

        # Execute Python file. When bwrap is usable it is the real
        # write-confinement layer: the uv-run subprocess (and anything it
        # spawns, and any os/pathlib/subprocess writes in the model's code) can
        # only write inside the workspace. bwrap --chdir's into the workspace and
        # sets HOME there, so uv's cache stays writable. Without bwrap we fall
        # back to the open()-guard preamble prepended above.
        rel_script = f"./.python_tmp/{filename}"
        uv_argv = ["uv", "run", "--directory", agent_workspace, rel_script]
        use_bwrap = bwrap_usable()
        if use_bwrap:
            argv = build_bwrap_argv(agent_workspace) + uv_argv
            run_kwargs = dict(shell=False)
        else:
            argv = f"uv run --directory {agent_workspace} {rel_script}"
            run_kwargs = dict(shell=True)
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=timeout,
                **run_kwargs,
            )
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return f"=== EXECUTION TIMEOUT ===\nExecution timed out after {timeout} seconds\nExecution time: {execution_time:.3f} seconds"

        # Calculate execution time
        execution_time = time.time() - start_time

        # Build output
        output_parts = []

        # Add stdout
        if result.stdout:
            output_parts.append("=== STDOUT ===")
            output_parts.append(result.stdout.rstrip())

        # Add stderr
        if result.stderr:
            output_parts.append("=== STDERR ===")
            output_parts.append(result.stderr.rstrip())

        # Add execution info
        output_parts.append("=== EXECUTION INFO ===")
        output_parts.append(f"Return code: {result.returncode}")
        output_parts.append(f"Execution time: {execution_time:.3f} seconds")
        output_parts.append(f"Timeout limit: {timeout} seconds")

        # If no output at all
        if not result.stdout and not result.stderr:
            output_parts.insert(0, "No console output produced.")

        # bwrap denies writes outside the workspace with a bare EROFS ("Read-only
        # file system"), which reads like a disk problem rather than a policy.
        # Append a hint so the model retries into the workspace instead of
        # treating it as a transient error.
        if "Read-only file system" in (result.stderr or ""):
            output_parts.append("=== HINT ===")
            output_parts.append(_ROFS_HINT.format(workspace=agent_workspace))

        return "\n".join(output_parts)
        
    except Exception as e:
        return f"Error executing Python code: {str(e)}"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Python Execute MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport type (default: stdio)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (for HTTP transport)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8084,
        help="Port to bind to (for HTTP transport)"
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="Agent workspace directory (default: current directory)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set workspace environment variable
    os.environ["PYTHON_EXECUTE_WORKSPACE"] = os.path.abspath(args.workspace)
    
    # Run the server
    if args.transport == "stdio":
        app.run(transport="stdio", show_banner=False)
    else:
        app.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            show_banner=False
        )

