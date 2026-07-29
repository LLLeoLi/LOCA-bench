#!/usr/bin/env python3
"""
Programmatic Tool Calling MCP Server

An MCP server that provides Python code execution with embedded tool calling capabilities.
When code execution encounters a tool call, it pauses, executes the tool via the tool executor,
and continues with the tool result injected back into the code execution context.

Based on python_execute MCP server but with programmatic tool calling support.
"""

import os
import sys
import csv
import time
import uuid
import json
import traceback
from pathlib import Path
from typing import Annotated, Optional, List, Dict, Any, Callable
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

# Global tool executor - will be set by the tool that creates this server
_tool_executor: Optional[Callable[[str, Dict[str, Any]], tuple]] = None


def set_tool_executor(executor: Callable[[str, Dict[str, Any]], tuple]):
    """Set the tool executor function.

    The executor should be a callable that takes (tool_name, tool_args) and returns:
    (tool_parsed, tool_execute_error, observation, returned_tool_name, returned_tool_call_id)
    """
    global _tool_executor
    _tool_executor = executor


def get_workspace() -> str:
    """Get the workspace directory from environment or use default."""
    return os.environ.get("PROGRAMMATIC_TOOL_CALLING_WORKSPACE", DEFAULT_WORKSPACE)


class ToolCallInterceptor:
    """
    A class that intercepts function calls and records them for later execution.

    Since MCP servers run in separate processes, actual tool execution happens
    in the parent process. This class just records tool calls and returns
    placeholder values that will be replaced with actual results later.
    """

    def __init__(self, tool_results_cache: Optional[Dict[str, str]] = None):
        """
        Args:
            tool_results_cache: Pre-computed results from previous tool executions
                               Format: {tool_call_id: observation}
        """
        self.tool_calls_made = []
        self.tool_results = []
        self.tool_results_cache = tool_results_cache or {}

    def __getattr__(self, tool_name: str):
        """Intercept attribute access as a tool call: ``tools.tool_name(...)``."""
        return self._make_tool_function(tool_name)

    def __getitem__(self, tool_name: str):
        """Intercept subscript access as a tool call: ``tools["tool_name"](...)``."""
        return self._make_tool_function(tool_name)

    def _make_tool_function(self, tool_name: str):
        """Build the callable that records/returns the result for one tool name."""
        def tool_function(*args, **kwargs):
            """Record the tool call and return its cached native result if available."""
            if args:
                # MCP tools take named parameters; positional args cannot be mapped.
                raise TypeError(
                    f"tools[{tool_name!r}] must be called with keyword arguments, "
                    f"e.g. tools[{tool_name!r}](id='A001')"
                )

            # Generate a deterministic cache key based on tool name and args
            # This ensures the same call gets the same result across passes
            import json
            import hashlib
            args_str = json.dumps(kwargs, sort_keys=True)
            cache_key = f"{tool_name}:{args_str}"
            hash_suffix = hashlib.md5(cache_key.encode()).hexdigest()[:8]
            tool_call_id = f"call_{hash_suffix}"

            # Record the tool call
            self.tool_calls_made.append({
                "tool_name": tool_name,
                "args": kwargs,
                "tool_call_id": tool_call_id,
            })

            # Check if we have a cached result for this call
            if tool_call_id in self.tool_results_cache:
                raw = self.tool_results_cache[tool_call_id]
                # Record the (non-pending) result before decoding so a decode that
                # raises (tool errored) still leaves this call marked as resolved.
                self.tool_results.append({
                    "tool_call_id": tool_call_id,
                    "observation": raw,
                    "has_error": False,
                })
                return self._decode_cached(raw)

            # First pass - return a placeholder.
            # This will trigger re-execution after tools are actually run.
            observation = f"__TOOL_CALL_PENDING_{tool_call_id}__"
            self.tool_results.append({
                "tool_call_id": tool_call_id,
                "observation": observation,
                "has_error": False,
            })
            return observation

        return tool_function

    @staticmethod
    def _decode_cached(raw: str):
        """Turn a cached tool result into the native value user code receives.

        The parent process caches results in the wire envelope
        ``{"__ptc__": true, "ok": bool, "value": <raw tool output text>}``.
        - ``ok=False`` -> raise ``RuntimeError(value)`` so code can ``try/except``.
        - ``ok=True``  -> return the payload as a native Python value: ``json.loads``
          the text when it is JSON (dict/list/number/...), otherwise the raw string.
        Values that are not in the envelope form are returned as-is (best-effort
        JSON-decoded), keeping backward compatibility.
        """
        import json
        try:
            env = json.loads(raw)
        except (ValueError, TypeError):
            env = None

        if isinstance(env, dict) and env.get("__ptc__") is True:
            value = env.get("value", "")
            if not env.get("ok", True):
                raise RuntimeError(value if isinstance(value, str) else str(value))
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except (ValueError, TypeError):
                    return value
            return value

        # Fallback: treat the raw string as the value itself.
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return raw


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
    standalone fallback (e.g. running this server directly) and still returns
    the full structured JSON described below.

    Args:
        code: Python code to execute; tool calls go through the injected ``tools`` mapping.

    Returns:
        JSON string with keys: success, execution_time_seconds, timeout_limit_seconds,
        stdout, stderr, return_value, error, plus the internal fields tool_calls,
        tool_results, needs_tool_execution, file_path.
    """
    tool_results_cache: Optional[Dict[str, str]] = None

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

        # Create tool interceptor with cache
        interceptor = ToolCallInterceptor(tool_results_cache)

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
            "tools": interceptor,  # Inject the tool interceptor
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

        # Check if there are pending tool calls
        needs_tool_execution = any(
            "__TOOL_CALL_PENDING_" in str(tr.get("observation", ""))
            for tr in interceptor.tool_results
        )

        # Build structured result
        result = {
            "success": execution_error is None,
            "execution_time_seconds": round(execution_time, 3),
            "timeout_limit_seconds": timeout,
            "stdout": stdout_content if stdout_content else None,
            "stderr": stderr_content if stderr_content else None,
            "tool_calls": interceptor.tool_calls_made,
            "tool_results": interceptor.tool_results,
            "needs_tool_execution": needs_tool_execution,
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

    # Note: Tool executor must be set via set_tool_executor() before use
    print("Warning: Tool executor not configured. Use set_tool_executor() to configure.", file=sys.stderr)

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
