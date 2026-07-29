"""Programmatic Tool Calling (PTC)

Exposes a ``code_execution`` tool that runs model-written Python in a
persistent IPython kernel (StatefulSandbox); ``tools["name"](...)`` calls
inside the code dispatch to the environment's real MCP tools over a
unix-socket bridge. See README.md and helper.py for details.
"""

from .helper import (
    create_programmatic_tool_calling_tool_http,
    create_programmatic_tool_calling_tool_stdio,
    get_programmatic_tool_calling_stdio_config,
    ProgrammaticToolCallingTool,
)

__all__ = [
    "create_programmatic_tool_calling_tool_stdio",
    "create_programmatic_tool_calling_tool_http",
    "get_programmatic_tool_calling_stdio_config",
    "ProgrammaticToolCallingTool",
]
