# Programmatic Tool Calling (PTC)

Exposes a `code_execution` tool that lets the model write Python which calls
the environment's other MCP tools as `tools["tool_name"](...)` — loops,
conditionals, and chaining many tool calls with intermediate processing in a
single model turn.

## Architecture

`server.py` is a FastMCP server that only exists so MCP discovery presents the
`code_execution` tool (name, description, single `code` parameter — matching
the verl training-side `programmatic_tool_call` tool). Its body never runs in
normal LOCA-Bench evaluation.

Actual execution is handled in the parent process by
`ProgrammaticToolCallingTool` (`helper.py`), which intercepts `code_execution`
calls and runs the code in a persistent IPython kernel:

- **`stateful_kernel.py`** — `StatefulSandbox`, vendored from verl's
  `landlock_sandbox.py` (ipykernel subprocess; RLIMIT_AS 2 GiB; output =
  stdout + last-expression echo + `[stderr]:` section; 50k-char truncation;
  timeout with interrupt-to-idle). **Sync manually when verl changes.**
- **`helper.py`** — one kernel per tool instance (= per task), lazily started
  on the first `code_execution` call. Variables and imports persist across
  calls and across message turns. `tools[...]` inside the kernel RPCs over a
  unix socket back to the parent (`_ToolBridge`), which dispatches to the real
  MCP tools and returns native Python values (dict/list/str; tool errors raise
  `RuntimeError`, catchable with try/except). If the kernel dies unrecoverably
  (hard-killed timeout, crash, OOM), it is torn down and a fresh kernel starts
  on the next call — state is lost and the observation says so.

Timeout: 60s per execute (`LOCA_PTC_TIMEOUT` env or `ptc_timeout` kwarg;
not model-controllable).

## Usage

Enable in a task's MCP config alongside the task's other servers:

```yaml
programmatic_tool_calling:
  type: programmatic_tool_calling
  enabled: true
```

`run_react.py` detects it and constructs `ProgrammaticToolCallingTool` with the
merged server config; direct (non-code_execution) tool calls pass through
unchanged.

Programmatic construction:

```python
from gem.tools.mcp_server.programmatic_tool_calling import (
    create_programmatic_tool_calling_tool_stdio,
)

prog_tool = create_programmatic_tool_calling_tool_stdio(
    workspace_path="/path/to/workspace",
    tools=[other_tool_a, other_tool_b],  # or prog_tool.set_tools([...]) later
)
```

## Notes

- Tool names are presented MCPMark-style (server prefix stripped,
  collision-safe); dispatch translates back to the real prefixed names.
- Code runs chdir'ed into the workspace; `open()` in write mode is confined to
  the workspace.
- Running `server.py` standalone executes pure Python only — `tools[...]`
  raises, since there is no parent process to dispatch tool calls to.
