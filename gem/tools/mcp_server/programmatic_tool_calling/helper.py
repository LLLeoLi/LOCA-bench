"""Helper functions to create Programmatic Tool Calling MCP tool instances.

This module provides a specialized MCP tool that executes Python code with embedded
tool calling capabilities. Unlike regular python_execute, this allows code to invoke
tools as functions via the ``tools`` mapping.

Execution model (aligned with verl training-side PTC):
``code_execution`` runs in a per-instance :class:`StatefulSandbox` -- an IPython
kernel subprocess vendored from verl's landlock_sandbox -- so the observation the
model sees (stdout, last-expression echo, ``[stderr]:`` section, ``(no output)``,
IPython tracebacks, truncation format, timeout messages) is byte-identical to
training. Tool calls made inside the code (``tools["name"](...)``) are RPC'd back
to this process over a unix socket and dispatched to the real MCP tools.
"""

import ast
import copy
import json
import os
import re
import shutil
import socket
import tempfile
import textwrap
import threading
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from gem.tools.mcp_tool import MCPTool
from gem.tools.mcp_server.programmatic_tool_calling.stateful_kernel import StatefulSandbox

# Per-execute() timeout of the PTC kernel (seconds). 120s matches the ceiling
# python_execute allows, so I/O-heavy code and bulk tool-call loops have room
# to finish before the kernel is killed. Model code cannot override it;
# configure via the ptc_timeout kwarg or LOCA_PTC_TIMEOUT.
_DEFAULT_PTC_TIMEOUT = float(os.getenv("LOCA_PTC_TIMEOUT", "120.0"))

# Wall-clock budget for the one-shot setup code (tools proxy injection).
# Matches verl's _SANDBOX_SETUP_TIMEOUT.
_SANDBOX_SETUP_TIMEOUT = float(os.getenv("LOCA_SANDBOX_SETUP_TIMEOUT", "60.0"))

# Exact parameters schema of the verl training-side programmatic_tool_call tool
# (tasksync_agent_loop.get_tool_schemas). The FastMCP-generated schema for
# code_execution is replaced with this in get_tool_function so the model sees an
# identical parameter block (single required `code`, same description).
_CODE_EXECUTION_PARAMETERS = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": (
                "Python code. Use tools[\"func_name\"](*args, **kwargs) to call env tools."
            ),
        }
    },
    "required": ["code"],
}

# Opening line of the code_execution description in PTC-only mode, copied
# verbatim from verl task-sync eval.py PTC_TOOL_DESCRIPTION_ONLY. The full
# PTC-only description is this line + the RICH description body (everything
# after the first line of the server-provided description), mirroring verl's
# construction (opening + PTC_TOOL_DESCRIPTION_RICH.split("\n", 1)[1]) so the
# two stay byte-identical as long as the RICH text is.
_PTC_ONLY_DESCRIPTION_OPENING = (
    'Run Python that calls the tools listed above as `tools["tool_name"](*args, **kwargs)`. '
    "**This is the ONLY way to invoke env tools** — they cannot be called as standalone tool "
    "calls, so every env tool must go through this sandbox. "
    "State (variables, imports) persists across calls. Use print() to see output.\n"
)

# Kernel-side setup: the `tools` proxy injected into the PTC kernel. Mirrors
# verl's TOOLS_PROXY_SOURCE -- same class name, same unknown-tool message
# ("Unknown tool 'x'. Available tools: a, b"), raised at subscript/attribute
# access time -- except dispatch goes through a unix-socket RPC to the parent
# process (where the MCP clients live) instead of an in-kernel Task_Env.
# %% escapes are for the %-formatting pass that injects sock_path/tool_names.
_PTC_SETUP_TEMPLATE = textwrap.dedent(r'''
import json as _ptc_json
import socket as _ptc_socket

_PTC_SOCK_PATH = %(sock_path)r
_PTC_TOOL_NAMES = %(tool_names)r

def _ptc_dispatch(_name, _args, _kwargs):
    _req = _ptc_json.dumps({"name": _name, "args": list(_args), "kwargs": _kwargs})
    _s = _ptc_socket.socket(_ptc_socket.AF_UNIX, _ptc_socket.SOCK_STREAM)
    try:
        _s.connect(_PTC_SOCK_PATH)
        _s.sendall(_req.encode("utf-8") + b"\n")
        _buf = b""
        while not _buf.endswith(b"\n"):
            _chunk = _s.recv(65536)
            if not _chunk:
                break
            _buf += _chunk
    finally:
        _s.close()
    _resp = _ptc_json.loads(_buf.decode("utf-8"))
    if not _resp.get("ok", False):
        _val = _resp.get("value", "")
        if _resp.get("error_type") == "TypeError":
            raise TypeError(_val)
        raise RuntimeError(_val)
    _val = _resp.get("value", "")
    if isinstance(_val, str):
        try:
            return _ptc_json.loads(_val)
        except (ValueError, TypeError):
            return _val
    return _val

class _ToolsProxy:
    """Proxy that allows `tools["function_name"](*args, **kwargs)` inside the programmatic tool call sandbox.
    Also supports `tools.function_name(...)` for backward compatibility."""
    @staticmethod
    def _unknown(name):
        return "Unknown tool '%%s'. Available tools: %%s" %% (name, ", ".join(sorted(_PTC_TOOL_NAMES)))
    def __getitem__(self, name):
        if name not in _PTC_TOOL_NAMES:
            raise KeyError(self._unknown(name))
        def _call(*args, **kwargs):
            return _ptc_dispatch(name, args, kwargs)
        return _call
    def __getattr__(self, name):
        if name not in _PTC_TOOL_NAMES:
            raise AttributeError(self._unknown(name))
        def _call(*args, **kwargs):
            return _ptc_dispatch(name, args, kwargs)
        return _call
tools = _ToolsProxy()
''').strip()


class _ToolBridge:
    """Unix-socket RPC endpoint the kernel-side ``tools`` proxy dials.

    One connection per tool call: the kernel sends one JSON line
    ``{"name": ..., "args": [...], "kwargs": {...}}`` and receives one JSON
    line ``{"ok": bool, "value": ..., "error_type": ...}`` back. Dispatch runs
    on a daemon thread so it never blocks the caller's thread; the kernel is
    blocked on the socket for the duration of the tool call, which therefore
    counts against the execute() timeout -- same accounting as training, where
    env tools run inside the kernel.
    """

    def __init__(self, dispatch, sock_path: str):
        self._dispatch = dispatch
        self._sock_path = sock_path
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(sock_path)
        self._server.listen(16)
        self._closed = False
        self._thread = threading.Thread(
            target=self._serve, daemon=True, name="ptc-tool-bridge"
        )
        self._thread.start()

    def _serve(self):
        while not self._closed:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        try:
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
            if not buf.strip():
                return
            try:
                req = json.loads(buf.decode("utf-8"))
                resp = self._dispatch(
                    req.get("name", ""), req.get("args") or [], req.get("kwargs") or {}
                )
            except Exception as e:  # noqa: BLE001 - must always answer the kernel
                resp = {
                    "ok": False,
                    "error_type": "RuntimeError",
                    "value": f"{type(e).__name__}: {e}",
                }
            try:
                conn.sendall(json.dumps(resp, ensure_ascii=False).encode("utf-8") + b"\n")
            except OSError:
                pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self):
        self._closed = True
        try:
            self._server.close()
        except OSError:
            pass
        try:
            os.unlink(self._sock_path)
        except OSError:
            pass


class ProgrammaticToolCallingTool(MCPTool):
    """
    Extended MCPTool that wraps the programmatic_tool_calling server and
    provides tool execution capabilities to code running inside the kernel.

    This tool maintains a reference to other tools in the environment and
    executes them when Python code makes tool calls during execution.

    Execution model: ``code_execution`` is intercepted here and run in a
    per-instance IPython kernel (:class:`StatefulSandbox`, vendored from verl)
    rather than round-tripped to the MCP subprocess. Variables/imports persist
    across calls; ``tools["x"](...)`` RPCs back to this process and returns a
    native Python value (raising on tool error); the observation returned to
    the model is the kernel output in the exact training-side format.

    If the kernel dies unrecoverably (hard-killed after a timeout that could
    not be interrupted, or the process crashed), it is torn down and a fresh
    kernel is started on the next ``code_execution`` call. Kernel state
    (variables, imports) is lost across such a restart; the observation for
    the failing call tells the model so.
    """

    def __init__(
        self,
        config: dict,
        tools: Optional[List[Any]] = None,
        validate_on_init: bool = False,
        ptc_timeout: Optional[float] = None,
        ptc_only: bool = False,
        **kwargs
    ):
        """Initialize the Programmatic Tool Calling tool.

        Args:
            config: MCP server configuration
            tools: Optional list of other tools available for calling.
                   Can be set later using set_tools().
            validate_on_init: Whether to validate connection on initialization
            ptc_timeout: Per-execute() timeout (seconds) of the PTC kernel.
                         Defaults to LOCA_PTC_TIMEOUT or 120.0.
            ptc_only: If True, task tools remain visible in the model-facing
                      schema but direct calls to them are rejected with a
                      redirect error; only code_execution and claim_done can be
                      called directly. claim_done is also removed from the
                      in-kernel ``tools`` proxy so task completion must be
                      claimed with a direct call (a claim inside code would not
                      terminate the episode).
            **kwargs: Additional arguments to pass to MCPTool constructor
        """
        super().__init__(config, validate_on_init=validate_on_init, **kwargs)
        self._other_tools = tools or []
        self._ptc_only = ptc_only

        # MCPMark-alignment state (built lazily from discovered tools).
        # LOCA-Bench presents FastMCP multi-server tool names/descriptions
        # ("woocommerce_woo_products_list", "[woocommerce] ...") while MCPMark
        # presents the raw single-server names/descriptions ("woo_products_list").
        # We expose the MCPMark-style names to the model and translate them back
        # to the real prefixed names for dispatch. Names that would collide across
        # servers keep their prefix (collision-safe).
        self._name_maps_built = False
        self._real_to_display: Dict[str, str] = {}
        self._display_to_real_map: Dict[str, str] = {}

        # PTC kernel state, created lazily on the first code_execution call.
        self._ptc_timeout = ptc_timeout if ptc_timeout is not None else _DEFAULT_PTC_TIMEOUT
        self._sandbox: Optional[StatefulSandbox] = None
        self._sandbox_failed: Optional[str] = None
        self._bridge: Optional[_ToolBridge] = None
        self._bridge_dir: Optional[str] = None
        self._tool_registry: Dict[str, List[str]] = {}
        self._sandbox_lock = threading.Lock()
        self._exec_lock = threading.Lock()
        self._workspace_cache: Optional[str] = None

    def set_tools(self, tools: List[Any]):
        """Set the list of other tools that can be called.

        This should be called after the tool is added to ToolEnvWrapper,
        so it can reference the other tools in self.tools.

        Args:
            tools: List of tool instances that this tool can call
        """
        self._other_tools = [t for t in tools if t is not self]

    # ------------------------------------------------------------------
    # MCPMark alignment helpers
    # ------------------------------------------------------------------
    def _ensure_name_maps(self) -> None:
        """Build display<->real tool-name maps once, collision-safe.

        display name = MCPMark-style raw name (server prefix stripped).
        If two servers would produce the same display name, both keep their
        prefixed (real) name so dispatch stays unambiguous.
        """
        if self._name_maps_built:
            return

        available = self.get_available_tools()

        # First pass: compute candidate stripped name per real name.
        stripped: Dict[str, str] = {}
        display_counts: Dict[str, int] = {}
        for tool in available:
            real = tool.get("name", "")
            server = (tool.get("server_info", {}) or {}).get("detected_server")
            display = real
            if server and real.startswith(f"{server}_"):
                display = real[len(server) + 1:]
            stripped[real] = display
            display_counts[display] = display_counts.get(display, 0) + 1

        # Second pass: keep prefix for colliding display names.
        real_to_display: Dict[str, str] = {}
        display_to_real: Dict[str, str] = {}
        collisions: List[str] = []
        for real, display in stripped.items():
            if display_counts.get(display, 0) > 1:
                real_to_display[real] = real
                display_to_real[real] = real
                collisions.append(display)
            else:
                real_to_display[real] = display
                display_to_real[display] = real

        if collisions:
            try:
                from gem.tools.mcp_tool import logger as _logger
                _logger.warning(
                    "ProgrammaticToolCalling: kept server prefix for colliding "
                    "tool names across servers: %s", sorted(set(collisions))
                )
            except Exception:
                pass

        self._real_to_display = real_to_display
        self._display_to_real_map = display_to_real
        self._name_maps_built = True

    def _display_to_real(self, name: str) -> str:
        """Translate a model-visible (MCPMark-style) name to the real name."""
        self._ensure_name_maps()
        return self._display_to_real_map.get(name, name)

    @staticmethod
    def _strip_server_desc_prefix(description: str, servers: List[str]) -> str:
        """Remove a leading ``[server] `` prefix added for multi-server configs."""
        if not description:
            return description
        match = re.match(r"^\[([^\]]+)\]\s+", description)
        if match and match.group(1) in servers:
            return description[match.end():]
        return description

    @staticmethod
    def _is_code_execution_tool(real_name: str) -> bool:
        """True for the programmatic_tool_calling server's code_execution tool.

        With multiple servers FastMCP prefixes the name
        ("programmatic_tool_calling_code_execution"); with the PTC server as
        the only server the name is bare ("code_execution") -- without the
        bare-name match interception silently fails there and the stateless
        server-body fallback runs instead of the kernel.
        """
        if real_name == "code_execution":
            return True
        return "programmatic_tool_calling" in real_name and "code_execution" in real_name

    @staticmethod
    def _is_claim_done_tool(real_name: str) -> bool:
        """True for the claim_done server's tool.

        Bare name with a single server ("claim_done"), FastMCP-prefixed with
        multiple servers ("claim_done_claim_done").
        """
        return real_name == "claim_done" or real_name.endswith("_claim_done")

    # ------------------------------------------------------------------
    # PTC kernel (StatefulSandbox) + tool bridge
    # ------------------------------------------------------------------
    def _get_workspace(self) -> str:
        """Resolve the code_execution workspace directory (cached)."""
        if self._workspace_cache is not None:
            return self._workspace_cache
        ws = None
        try:
            servers = (self.normalized_config or {}).get("mcpServers", {})
            for cfg in servers.values():
                env = (cfg or {}).get("env", {}) or {}
                ws = env.get("PROGRAMMATIC_TOOL_CALLING_WORKSPACE") or cfg.get("cwd")
                if ws:
                    break
        except Exception:
            ws = None
        self._workspace_cache = ws or os.getcwd()
        return self._workspace_cache

    def _build_tool_registry(self) -> Dict[str, List[str]]:
        """Map every PTC-callable display name to its schema parameter order.

        The name list seeds the kernel-side proxy's unknown-tool check; the
        parameter order is used to map positional args to kwargs on dispatch
        (verl passes positional args straight to the underlying Python
        functions; MCP tools only take named parameters).
        """
        self._ensure_name_maps()
        registry: Dict[str, List[str]] = {}
        for tool in self.get_available_tools():
            real = tool.get("name", "")
            if not real or self._is_code_execution_tool(real):
                continue
            if self._ptc_only and self._is_claim_done_tool(real):
                continue
            display = self._real_to_display.get(real, real)
            props = (tool.get("parameters") or {}).get("properties") or {}
            registry[display] = list(props.keys())
        for other in self._other_tools:
            try:
                other_tools = other.get_available_tools()
            except Exception:
                continue
            for tool in other_tools:
                real = tool.get("name", "")
                if not real or real in registry or self._is_code_execution_tool(real):
                    continue
                if self._ptc_only and self._is_claim_done_tool(real):
                    continue
                props = (tool.get("parameters") or {}).get("properties") or {}
                registry[real] = list(props.keys())
        return registry

    def _bridge_dispatch(
        self, display_name: str, args: List[Any], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle one kernel-side ``tools[name](*args, **kwargs)`` call."""
        registry = self._tool_registry
        if display_name not in registry:
            # The kernel-side proxy already guards on the name list; this is
            # defensive against a registry/kernel mismatch.
            return {
                "ok": False,
                "error_type": "RuntimeError",
                "value": "Unknown tool '%s'. Available tools: %s"
                % (display_name, ", ".join(sorted(registry))),
            }
        if args:
            param_order = registry[display_name]
            if len(args) > len(param_order):
                return {
                    "ok": False,
                    "error_type": "TypeError",
                    "value": (
                        f"tools[{display_name!r}] takes at most {len(param_order)} "
                        f"positional arguments ({len(args)} given)"
                    ),
                }
            merged = dict(kwargs)
            for i, val in enumerate(args):
                pname = param_order[i]
                if pname in merged:
                    return {
                        "ok": False,
                        "error_type": "TypeError",
                        "value": (
                            f"tools[{display_name!r}] got multiple values for "
                            f"argument '{pname}'"
                        ),
                    }
                merged[pname] = val
            kwargs = merged

        real_name = self._display_to_real(display_name)
        try:
            parsed, has_error, observation = self._dispatch_real_tool(real_name, kwargs)
        except Exception as e:  # noqa: BLE001 - errors surface in the kernel
            return {
                "ok": False,
                "error_type": "RuntimeError",
                "value": f"{type(e).__name__}: {e}",
            }
        if not parsed:
            return {
                "ok": False,
                "error_type": "RuntimeError",
                "value": f"Tool '{display_name}' not found",
            }
        if has_error:
            return {"ok": False, "error_type": "RuntimeError", "value": observation}
        return {"ok": True, "value": self._observation_to_wire(observation)}

    # Matches the MCP result-object repr wrappers LOCA-Bench tools render, e.g.
    # ``Root(content='...')`` (filesystem) or ``read_data_from_excelOutput(result='...')``
    # (excel). Captures the single kwarg value so it can be literal_eval'd back
    # to the inner payload string.
    _WRAPPER_RE = re.compile(r"^\s*[\w.]+\((?:content|result|text|data)=(.*)\)\s*$", re.S)
    # Matches a trailing JSON array/object after a human-readable preamble, e.g.
    # bigquery's ``Query executed successfully\n...\nResults:\n[ {...} ]``.
    _TRAILING_JSON_RE = re.compile(r"(\[.*\]|\{.*\})\s*$", re.S)

    @classmethod
    def _observation_to_wire(cls, observation: str) -> str:
        """Normalize a tool observation so the kernel decodes a native value.

        The kernel-side ``tools[...]()`` proxy ``json.loads`` the wire value, so
        model code gets a native dict/list (matching verl training-side PTC).
        LOCA-Bench tool observations arrive in several non-JSON shapes; this
        recovers the structured payload from each, falling back to the raw text
        when nothing parses:

        - MCP result-object repr wrapper: ``Root(content='...')``,
          ``read_data_from_excelOutput(result='{...}')`` -- unwrap the kwarg and
          re-normalize the inner payload.
        - Human-readable preamble + trailing JSON body (bigquery ``run_query``):
          ``Query executed successfully\\n...\\nResults:\\n[ {...} ]`` -- extract
          the trailing array/object.
        - Python ``repr`` of a structure (``{'id': 'A001'}``) -- re-encode as JSON.
        - Already-JSON text, or plain prose -- returned unchanged.
        """
        payload = observation

        # Unwrap MCP result-object repr(s). Loop in case of nesting, but bound it.
        for _ in range(3):
            m = cls._WRAPPER_RE.match(payload)
            if not m:
                break
            try:
                inner = ast.literal_eval(m.group(1))
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                break
            if not isinstance(inner, str):
                # Wrapper held a real structure already -- encode it directly.
                try:
                    return json.dumps(inner, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    return payload
            payload = inner

        # Already valid JSON text -- kernel decodes it natively.
        try:
            json.loads(payload)
            return payload
        except (ValueError, TypeError):
            pass

        # Human-readable preamble + trailing JSON body (e.g. bigquery run_query).
        m = cls._TRAILING_JSON_RE.search(payload)
        if m:
            body = m.group(1)
            try:
                json.loads(body)
                return body
            except (ValueError, TypeError):
                pass

        # Python repr of a structure -- re-encode as JSON.
        try:
            obj = ast.literal_eval(payload)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return observation  # plain string value; keep the original text
        try:
            return json.dumps(obj, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return observation

    def _dispatch_real_tool(
        self, real_name: str, kwargs: Dict[str, Any]
    ) -> Tuple[bool, bool, str]:
        """Run one real (FastMCP-prefixed) tool. Returns (parsed, has_error, observation)."""
        tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
        if self._other_tools:
            for other_tool in self._other_tools:
                parsed, err, obs, _, _ = other_tool.execute_tool(
                    real_name, kwargs, tool_call_id
                )
                if parsed:
                    return parsed, err, obs
        parsed, err, obs, _, _ = super(ProgrammaticToolCallingTool, self).execute_tool(
            real_name, kwargs, tool_call_id
        )
        return parsed, err, obs

    def _ensure_sandbox(self) -> Tuple[Optional[StatefulSandbox], Optional[str]]:
        """Start the PTC kernel + tool bridge on first use.

        Returns (sandbox, None) on success or (None, error_message) on failure.
        A failed bring-up is cached so every subsequent call fails fast with
        the same message instead of re-paying the startup timeout.
        """
        with self._sandbox_lock:
            if self._sandbox is not None:
                return self._sandbox, None
            if self._sandbox_failed is not None:
                return None, self._sandbox_failed

            # The bridge socket must exist before the kernel starts: the kernel
            # runs under bwrap, which mounts a private tmpfs over /tmp, so its
            # directory has to be bind-mounted in explicitly (extra_bind_paths)
            # or `tools[...]` calls fail with FileNotFoundError on connect().
            try:
                self._bridge_dir = tempfile.mkdtemp(prefix="ptc_bridge_")
                sock_path = os.path.join(self._bridge_dir, "tools.sock")
                self._bridge = _ToolBridge(self._bridge_dispatch, sock_path)
            except Exception as e:
                self._teardown_bridge_locked()
                self._sandbox_failed = f"[Error] PTC tool bridge failed to start: {e}"
                return None, self._sandbox_failed

            try:
                sandbox = StatefulSandbox(
                    workspace_path=self._get_workspace(),
                    timeout=self._ptc_timeout,
                    extra_bind_paths=[self._bridge_dir],
                )
                sandbox.start()
            except Exception as e:
                self._teardown_bridge_locked()
                self._sandbox_failed = f"[Error] PTC sandbox failed to start: {e}"
                return None, self._sandbox_failed

            try:
                self._tool_registry = self._build_tool_registry()
                setup_code = _PTC_SETUP_TEMPLATE % {
                    "sock_path": sock_path,
                    "tool_names": sorted(self._tool_registry),
                }
                result, success = sandbox.execute(
                    setup_code, timeout=_SANDBOX_SETUP_TIMEOUT
                )
                if not success:
                    raise RuntimeError(result)
            except Exception as e:
                sandbox.cleanup()
                self._teardown_bridge_locked()
                self._sandbox_failed = f"[Error] PTC sandbox setup failed: {e}"
                return None, self._sandbox_failed

            self._sandbox = sandbox
            return sandbox, None

    def _teardown_bridge_locked(self):
        """Close the bridge socket and remove its dir. Caller holds _sandbox_lock."""
        bridge, self._bridge = self._bridge, None
        bridge_dir, self._bridge_dir = self._bridge_dir, None
        if bridge is not None:
            try:
                bridge.close()
            except Exception:
                pass
        if bridge_dir is not None:
            shutil.rmtree(bridge_dir, ignore_errors=True)

    def _teardown_sandbox(self):
        """Stop the PTC kernel and tool bridge. Idempotent."""
        with self._sandbox_lock:
            sandbox, self._sandbox = self._sandbox, None
            self._teardown_bridge_locked()
        if sandbox is not None:
            try:
                sandbox.cleanup()
            except Exception:
                pass

    def close(self):
        """Clean up the PTC kernel/bridge, then the MCP client resources."""
        self._teardown_sandbox()
        super().close()

    # ------------------------------------------------------------------
    # Model-facing schema / dispatch
    # ------------------------------------------------------------------
    def get_tool_function(self) -> List[Dict[str, Any]]:
        """Expose MCPMark-style tool names/descriptions to the model.

        Strips the FastMCP ``{server}_`` name prefix and the ``[server] ``
        description prefix so the schema matches what MCPMark presents for the
        same servers. The programmatic ``code_execution`` tool is exposed under
        its raw name with the exact verl training-side parameters block
        (single required ``code``).
        """
        funcs = super().get_tool_function()
        self._ensure_name_maps()
        servers = self._get_server_names()
        for func in funcs:
            fn = func.get("function", {})
            real = fn.get("name", "")
            fn["name"] = self._real_to_display.get(real, real)
            fn["description"] = self._strip_server_desc_prefix(
                fn.get("description", ""), servers
            )
            if self._is_code_execution_tool(real):
                fn["parameters"] = copy.deepcopy(_CODE_EXECUTION_PARAMETERS)
                if self._ptc_only:
                    # PTC-only mode: swap the description's first line for the
                    # verl PTC_TOOL_DESCRIPTION_ONLY opening; the body (RICH
                    # minus its first line) is reused from the server-provided
                    # text, matching verl's construction exactly.
                    desc = fn.get("description", "")
                    if "\n" in desc:
                        fn["description"] = (
                            _PTC_ONLY_DESCRIPTION_OPENING + desc.split("\n", 1)[1]
                        )
        return funcs

    def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        tool_call_id: str
    ) -> Tuple[bool, bool, str, str, str]:
        """Dispatch a tool call.

        - ``code_execution`` runs in the per-instance PTC kernel (see class
          docstring): stateful across calls, tool calls inside it dispatch back
          here over the bridge, and the observation is the kernel output in the
          training-side format. It never round-trips to the MCP subprocess.
        - Any other (direct) tool call passes straight through to the real tool,
          identical to the original LOCA-Bench behavior.

        Tool dispatch uses two modes:
        - Multi-tool mode: self._other_tools (set via set_tools()).
        - Single-tool mode: super() (same instance with multiple servers).

        Args:
            tool_name: The name of the tool to execute
            parameters: The parameters of the tool to execute
            tool_call_id: Unique ID for this tool call

        Returns:
            Tuple of (tool_parsed, tool_execute_error, observation,
                     returned_tool_name, returned_tool_call_id)
        """
        # Translate the model-visible (MCPMark-style, de-prefixed) tool name back
        # to the real FastMCP prefixed name used for dispatch.
        tool_name = self._display_to_real(tool_name)

        if self._is_code_execution_tool(tool_name):
            code = parameters.get("code", "")
            sandbox, error = self._ensure_sandbox()
            if sandbox is None:
                return (True, True, error, tool_name, tool_call_id)
            # Serialize executes on this instance; the kernel handles one
            # request at a time and interleaved calls would mix up outputs.
            with self._exec_lock:
                observation, success = sandbox.execute(code)
            # An unrecoverable kill (timeout that could not be interrupted, or
            # a kernel crash such as the RLIMIT_AS OOM kill) leaves the kernel
            # dead. Tear it down so the next code_execution starts a fresh
            # kernel instead of failing for the rest of the task. Kernel state
            # is lost; rewrite the (now stale) kernel message and tell the
            # model explicitly.
            if not success and getattr(sandbox, "_dead", False):
                self._teardown_sandbox()
                observation = observation.replace(
                    "Sandbox killed; subsequent code will not run.",
                    "Sandbox killed.",
                ).replace(
                    "(killed after a previous timeout / crash). Subsequent code will not run.",
                    "(killed after a previous timeout / crash).",
                )
                observation += (
                    "\n\n[note] The Python sandbox was killed and a fresh one will be "
                    "started automatically on your next code_execution call. All "
                    "variables, imports and function definitions are lost -- "
                    "re-create any state you still need."
                )
            # bwrap denies out-of-workspace writes with a bare EROFS ("Read-only
            # file system"), which reads like a disk fault rather than a policy.
            # Append a hint so the model retries into the workspace instead of
            # treating it as transient. (open()-guard denials already say
            # "outside workspace" and need no hint.)
            if "Read-only file system" in observation:
                observation += (
                    "\n\n[hint] A 'Read-only file system' error means the write "
                    "target is OUTSIDE the agent workspace. Writes are confined to "
                    f"the workspace: {self._get_workspace()} (and its "
                    "subdirectories). Use a path inside it -- relative paths and "
                    "the pre-defined `workspace` variable already resolve there."
                )
            return (True, not success, observation, tool_name, tool_call_id)

        else:
            # PTC-only mode: task tools stay visible in the model-facing schema,
            # but a direct call to anything except claim_done is answered with a
            # redirect error instead of the tool result. Unknown names fall
            # through so the wrapper's "tool not found" path is unchanged.
            if (
                self._ptc_only
                and tool_name in self._real_to_display
                and not self._is_claim_done_tool(tool_name)
            ):
                display = self._real_to_display.get(tool_name, tool_name)
                return (
                    True,
                    True,
                    (
                        f"[Error] Direct call to tool '{display}' is not allowed in "
                        "PTC-only mode. Invoke it from inside code_execution instead, "
                        f'e.g. tools["{display}"](...). Only code_execution and '
                        "claim_done can be called directly."
                    ),
                    tool_name,
                    tool_call_id,
                )

            # This is a direct (non-code_execution) tool call. Keep it identical to
            # the original LOCA-Bench behavior: pass straight through to the real
            # tool with no result re-wrapping.
            # Try multi-tool mode first (self._other_tools)
            if self._other_tools:
                for other_tool in self._other_tools:
                    result = other_tool.execute_tool(tool_name, parameters, tool_call_id)
                    if result[0]:  # tool_parsed
                        return result

            # Try single-tool mode (super() - same instance with multiple servers)
            return super().execute_tool(tool_name, parameters, tool_call_id)


def get_programmatic_tool_calling_stdio_config(
    workspace_path: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    server_name: str = "programmatic_tool_calling"
) -> dict:
    """Get Programmatic Tool Calling stdio server configuration (without creating MCPTool).

    Returns a config dict that can be merged with other server configs
    before creating a ProgrammaticToolCallingTool instance.

    Args:
        workspace_path: Path to the agent workspace directory (default: current directory)
        workspace_dir: Alias for workspace_path (for backward compatibility)
        server_name: Name for this server in the config (default: "programmatic_tool_calling")

    Returns:
        Dict with single server config: {server_name: {...}}

    Examples:
        # Get individual configs
        prog_config = get_programmatic_tool_calling_stdio_config(workspace_path="/path/to/workspace")
        claim_done_config = get_claim_done_stdio_config()

        # Merge configs
        merged_config = {
            "mcpServers": {
                **prog_config,
                **claim_done_config
            }
        }

        # Create combined tool with executor
        tool = ProgrammaticToolCallingTool(merged_config, tool_executor=my_executor, validate_on_init=False)
    """
    # Get path to programmatic_tool_calling server script
    server_script = Path(__file__).parent / "server.py"

    if not server_script.exists():
        raise FileNotFoundError(
            f"Programmatic Tool Calling server script not found at: {server_script}"
        )

    # Support both workspace_path and workspace_dir for backward compatibility
    workspace = workspace_dir if workspace_dir is not None else workspace_path
    if workspace is None:
        workspace = "."

    # Convert workspace path to absolute path
    abs_workspace = str(Path(workspace).absolute())

    # Build command arguments - pass workspace via command line argument
    args = [
        str(server_script),
        "--workspace", abs_workspace
    ]

    # Return single server config (without mcpServers wrapper)
    # Set cwd so that native Python file operations (like open()) use the workspace directory
    return {
        server_name: {
            "command": "python",
            "args": args,
            "cwd": abs_workspace,
            "env": {
                "PROGRAMMATIC_TOOL_CALLING_WORKSPACE": abs_workspace,
                "LOCA_QUIET": "1",
            }
        }
    }


def create_programmatic_tool_calling_tool_stdio(
    workspace_path: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    tools: Optional[List[Any]] = None,
    validate_on_init: bool = False,
    **kwargs
) -> ProgrammaticToolCallingTool:
    """Create a Programmatic Tool Calling MCP tool using stdio transport.

    Provides a 'programmatic_tool_calling' tool that executes Python code with
    embedded tool calling capabilities in a persistent IPython kernel; tool
    calls made by the code dispatch to the other tools in the environment.

    This starts the Programmatic Tool Calling server as a subprocess and connects via stdio.

    Args:
        workspace_path: Path to the agent workspace directory (default: current directory)
        workspace_dir: Alias for workspace_path (for backward compatibility)
        tools: Optional list of other tools that can be called.
               Use tool.set_tools(tools) after creation to update.
        validate_on_init: Whether to validate the connection on initialization.
                         Set to False for faster startup.
        **kwargs: Additional arguments to pass to ProgrammaticToolCallingTool constructor

    Returns:
        ProgrammaticToolCallingTool configured for Programmatic Tool Calling server via stdio

    Example:
        >>> from gem.tools.mcp_server.programmatic_tool_calling import create_programmatic_tool_calling_tool_stdio
        >>> from gem.tools.mcp_server.filesystem import create_filesystem_tool
        >>>
        >>> # Create other tools first
        >>> filesystem_tool = create_filesystem_tool(agent_workspace="/path/to/workspace")
        >>>
        >>> # Create programmatic tool and set tools
        >>> prog_tool = create_programmatic_tool_calling_tool_stdio(
        >>>     workspace_path="/path/to/workspace"
        >>> )
        >>> prog_tool.set_tools([filesystem_tool])  # Set after creation
        >>>
        >>> # Or pass tools during creation
        >>> prog_tool = create_programmatic_tool_calling_tool_stdio(
        >>>     workspace_path="/path/to/workspace",
        >>>     tools=[filesystem_tool]
        >>> )
        >>>
        >>> # Use in environment
        >>> env = ToolEnvWrapper(env, tools=[filesystem_tool, prog_tool])
        >>>
        >>> # After adding to ToolEnvWrapper, update tool references
        >>> prog_tool.set_tools(env.tools)  # Now it can call all tools in env
        >>>
        >>> # Example code that uses tools programmatically:
        >>> code = '''
        >>> # Get file list
        >>> files = tools.list_files(path=".")
        >>> print(f"Found files: {files}")
        >>>
        >>> # Read first file
        >>> content = tools.read_file(path=files[0])
        >>> print(f"Content: {content}")
        >>>
        >>> result = "Success"
        >>> '''
    """
    # Get server config
    server_config = get_programmatic_tool_calling_stdio_config(
        workspace_path=workspace_path,
        workspace_dir=workspace_dir
    )

    # Wrap in mcpServers
    config = {"mcpServers": server_config}

    return ProgrammaticToolCallingTool(
        config,
        tools=tools,
        validate_on_init=validate_on_init,
        **kwargs
    )


def create_programmatic_tool_calling_tool_http(
    host: str = "127.0.0.1",
    port: int = 8085,
    tools: Optional[List[Any]] = None,
    validate_on_init: bool = True,
    **kwargs
) -> ProgrammaticToolCallingTool:
    """Create a Programmatic Tool Calling MCP tool using HTTP transport.

    Note: You need to start the Programmatic Tool Calling server separately:
        python -m gem.tools.mcp_server.programmatic_tool_calling.server --transport streamable-http --port 8085 --workspace /path/to/workspace

    Args:
        host: Server host (default: 127.0.0.1)
        port: Server port (default: 8085)
        tools: Optional list of other tools. Can be set later using tool.set_tools().
        validate_on_init: Whether to validate the connection on initialization
        **kwargs: Additional arguments to pass to ProgrammaticToolCallingTool constructor

    Returns:
        ProgrammaticToolCallingTool configured for Programmatic Tool Calling server via HTTP

    Example:
        >>> from gem.tools.mcp_server.programmatic_tool_calling import create_programmatic_tool_calling_tool_http
        >>> tool = create_programmatic_tool_calling_tool_http(port=8085, tools=[filesystem_tool])
        >>> # Use in environment
        >>> env = ToolEnvWrapper(env, tools=[tool])
    """
    url = f"http://{host}:{port}"

    # Create config for HTTP transport
    config = {"mcpServers": {
        "programmatic_tool_calling": {
            "url": url
        }
    }}

    return ProgrammaticToolCallingTool(
        config,
        tools=tools,
        validate_on_init=validate_on_init,
        **kwargs
    )
