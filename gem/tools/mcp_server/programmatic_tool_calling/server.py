#!/usr/bin/env python3
"""
Programmatic Tool Calling MCP Server

Registers the ``code_execution`` tool so MCP discovery exposes it to the model.
In normal LOCA-Bench runs the tool body here never executes:
``ProgrammaticToolCallingTool`` (helper.py) intercepts code_execution calls and
runs the code in a persistent IPython kernel with a working ``tools`` bridge.
The body below is a standalone fallback (running this server directly without
the wrapper) that executes pure Python only -- ``tools[...]`` raises there.
"""

import os
import sys
import csv
import time
import uuid
import json
import traceback
from pathlib import Path
from typing import Annotated
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Suppress FastMCP banner and reduce log level (must be before import)
os.environ["FASTMCP_SHOW_CLI_BANNER"] = "false"
os.environ["FASTMCP_LOG_LEVEL"] = "ERROR"

# Suppress logging output
import logging
logging.basicConfig(level=logging.ERROR, force=True)
logging.getLogger().setLevel(logging.ERROR)
for _logger_name in ["mcp", "fastmcp", "mcp.server", "mcp.client", "httpx", "asyncio", "uvicorn", "uvicorn.error", "uvicorn.access"]:
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

# Add parent directory to path for imports
gem_root = Path(__file__).parent.parent.parent.parent.parent
if str(gem_root) not in sys.path:
    sys.path.insert(0, str(gem_root))

from fastmcp import FastMCP

# Create FastMCP server
app = FastMCP("Programmatic Tool Calling Server")

# Default workspace (can be overridden by environment variable)
DEFAULT_WORKSPACE = "."

# Model-facing description for the code_execution tool. This is what the agent
# sees, so it must match the tool's actual behavior. Execution is handled by
# ProgrammaticToolCallingTool (see helper.py) in a persistent IPython kernel
# (StatefulSandbox, vendored from verl for training-side alignment), NOT by
# this server's code_execution body: tools are called via the `tools["name"](...)`
# mapping and return native Python values, variables/imports persist across calls,
# and the result to the model is the kernel output (stdout + last-expression echo,
# stderr under a [stderr]: header, IPython traceback on error). The text below is
# byte-identical to verl's PTC_TOOL_DESCRIPTION_RICH.
PTC_TOOL_DESCRIPTION_RICH = (
    'Run Python that calls the tools listed above as `tools["tool_name"](*args, **kwargs)`. State (variables, imports) persists across calls. Use print() to see output.\n'
    "USE WHEN: loops, conditionals, error handling, or chaining multiple tool calls with intermediate processing.\n\n"
    "Notes:\n"
    "- Code runs in the workspace directory and file writes are restricted to it, don't write to `/tmp`; always use absolute paths for file writes; os, json, csv, sys are pre-imported.\n"
    "- Tools return native Python values; the type and structure vary by tool (e.g. dict, list, or str). Always print and inspect the first result before processing many items; do not assume a result is a list and loop over it.\n"
    "- Very large printed output is truncated; print summaries rather than large raw data.\n"
    "- Tools may raise an exception; wrap calls in try/except to handle failures.\n"
    "Usage examples:\n\n"
    "Batch + conditional workflow:\n"
    "```python\n"
    "print(type(tools[\"get_info\"](id='A001')))  # inspect return type first, then loop\n"
    "results = []\n"
    "for item in ['A001', 'A002', 'A003']:\n"
    "    info = tools[\"get_info\"](id=item)\n"
    "    if info.get('status') == 'active': # info is a dict, confirmed above\n"
    "        results.append(tools[\"get_details\"](id=item))\n"
    "    else:\n"
    "        print(f'Skipping {item}')\n"
    "    print(f'Processed {item}')\n"
    "print('Collected', len(results))\n"
    "```\n\n"
    "Error handling:\n"
    "```python\n"
    "ok, failed = [], []\n"
    "for item_id in ['A001', 'A002', 'A003']:\n"
    "    try:\n"
    "        r = tools[\"get_info\"](id=item_id)\n"
    "        ok.append(r)\n"
    "    except Exception as e:\n"
    "        failed.append((item_id, str(e)))\n"
    "    print(f'{len(ok)} ok, {len(failed)} failed')\n"
    "print('Failed:', failed[:3] if failed else 'none')\n"
    "```"
)

def get_workspace() -> str:
    """Get the workspace directory from environment or use default."""
    return os.environ.get("PROGRAMMATIC_TOOL_CALLING_WORKSPACE", DEFAULT_WORKSPACE)


class _UnavailableTools:
    """``tools`` stub for the standalone fallback body.

    Tool calls only work through the kernel bridge that
    ``ProgrammaticToolCallingTool`` sets up in the parent process; when this
    server runs standalone there is nothing to dispatch to, so any access
    fails loudly instead of returning placeholder garbage.
    """

    @staticmethod
    def _unavailable(name: str):
        raise RuntimeError(
            f"tools[{name!r}] is unavailable: this server is running standalone "
            "without the ProgrammaticToolCallingTool wrapper, so environment "
            "tools cannot be dispatched. Only pure Python runs here."
        )

    def __getitem__(self, name: str):
        self._unavailable(name)

    def __getattr__(self, name: str):
        self._unavailable(name)


@app.tool(description=PTC_TOOL_DESCRIPTION_RICH)
def code_execution(
    code: Annotated[str, 'Python code. Use tools["func_name"](*args, **kwargs) to call env tools.'],
) -> str:
    """Execute Python code that can call other tools via the ``tools`` mapping.

    The model-facing description is ``PTC_TOOL_DESCRIPTION_RICH`` (passed to
    ``@app.tool``). This docstring is for maintainers only.

    The signature exposes only ``code`` so the FastMCP-generated inputSchema
    matches the verl training-side programmatic_tool_call tool exactly (the
    parent-process wrapper additionally hard-overrides the parameters block in
    get_tool_function, see helper._CODE_EXECUTION_PARAMETERS).

    In normal LOCA-Bench runs this body never executes:
    ``ProgrammaticToolCallingTool`` intercepts code_execution and runs the code
    in a persistent IPython kernel (see helper.py). The body is kept as a
    standalone fallback (e.g. running this server directly); it executes pure
    Python only -- ``tools[...]`` raises, since there is no parent process to
    dispatch tool calls to.

    Args:
        code: Python code to execute.

    Returns:
        JSON string with keys: success, execution_time_seconds, timeout_limit_seconds,
        stdout, stderr, return_value, error, file_path.
    """
    try:
        timeout = 30
        filename = f"programmatic_{uuid.uuid4().hex[:8]}.py"

        # Ensure filename ends with .py
        if not filename.endswith(".py"):
            filename += ".py"

        # Get workspace
        agent_workspace = get_workspace()
        agent_workspace = os.path.abspath(agent_workspace)

        # Create .python_tmp directory
        tmp_dir = os.path.join(agent_workspace, '.python_tmp')
        os.makedirs(tmp_dir, exist_ok=True)

        # Save the code to a file for debugging
        file_path = os.path.join(tmp_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)

        # Prepare execution environment. os/json/csv/sys are pre-imported to match
        # the tool description so the model can use them without an import line.
        # A guarded open() confines write-mode file access to the workspace,
        # matching the StatefulSandbox guard used on the normal execution path.
        # This body runs in-process, so instead of chdir'ing the whole server we
        # resolve relative paths against the workspace explicitly (the real
        # kernel chdir's into it, so relative paths must behave the same here).
        _orig_open = open
        _write_modes = frozenset('wax+')
        _ws_real = os.path.realpath(agent_workspace)

        def _guarded_open(file, mode='r', *args, **kwargs):
            _target = str(file)
            if not os.path.isabs(_target):
                _target = os.path.join(_ws_real, _target)
            _abs = os.path.realpath(_target)
            if _write_modes & set(str(mode)):
                if not (_abs.startswith(_ws_real + os.sep) or _abs == _ws_real):
                    raise PermissionError(
                        f"Write denied: {file!r} is outside the agent workspace {_ws_real!r}."
                    )
            return _orig_open(_abs, mode, *args, **kwargs)

        exec_globals = {
            "__name__": "__main__",
            "__file__": file_path,
            "tools": _UnavailableTools(),  # Standalone mode: tool calls raise
            "WORKSPACE": agent_workspace,  # Provide workspace path for file operations
            "os": os,
            "sys": sys,
            "json": json,
            "csv": csv,
            "open": _guarded_open,
        }

        # Capture stdout and stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        # Track execution time
        start_time = time.time()

        # Execute the code
        execution_error = None
        return_value = None

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Compile and execute the code
                compiled_code = compile(code, file_path, 'exec')
                exec(compiled_code, exec_globals)

                # Check if there's a return value (if code defined a main function or similar)
                if 'result' in exec_globals:
                    return_value = exec_globals['result']

        except Exception as e:
            execution_error = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }

        # Calculate execution time
        execution_time = time.time() - start_time

        # Get captured output
        stdout_content = stdout_capture.getvalue()
        stderr_content = stderr_capture.getvalue()

        # Build structured result
        result = {
            "success": execution_error is None,
            "execution_time_seconds": round(execution_time, 3),
            "timeout_limit_seconds": timeout,
            "stdout": stdout_content if stdout_content else None,
            "stderr": stderr_content if stderr_content else None,
            "return_value": str(return_value) if return_value is not None else None,
            "error": execution_error,
            "file_path": file_path,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        # Top-level error (e.g., file I/O error)
        return json.dumps({
            "success": False,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
        }, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Programmatic Tool Calling MCP Server")
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
        default=8085,
        help="Port to bind to (for HTTP transport, default: 8085)"
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
    os.environ["PROGRAMMATIC_TOOL_CALLING_WORKSPACE"] = os.path.abspath(args.workspace)

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
