"""Stateful Python sandbox backed by an IPython kernel.

Vendored from verl (verl/experimental/agent_loop/landlock_sandbox.py) so the
LOCA-bench eval-side programmatic tool calling matches the training-side
implementation byte-for-byte in every model-observable way:

  - execute() output composition: stdout + execute_result echo of the last
    expression, stderr appended under a ``[stderr]:`` header, ``(no output)``
    for empty output;
  - IPython-formatted tracebacks (ANSI stripped) on error;
  - middle-truncation format and limits (50k chars, 70/30 head/tail, same
    marker text), plus the bounded in-flight stream buffer;
  - timeout semantics and messages (interrupt-to-idle preserving state, or
    kernel kill), 60s default per execute();
  - kernel init namespace: os/json/csv/sys pre-imported, ``workspace``
    variable, chdir to the workspace, and the Python-level open() guard that
    denies write-mode opens outside the workspace (same PermissionError text
    as training, which also runs the guard-only init_code branch).

Training-side machinery that has no model-observable effect is stripped:
bwrap/Landlock confinement, cgroup caps, rollout admission slots, env_dir
tracking. RLIMIT_AS (2GiB default) is kept as the memory backstop.
"""

import atexit
import collections
import concurrent.futures
import ctypes
import logging
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

from gem.tools.mcp_server.bwrap_confine import bwrap_usable, build_bwrap_argv

logger = logging.getLogger(__file__)

# Default per-kernel memory cap (bytes), enforced via RLIMIT_AS in the kernel
# child. Same default as training (VERL_SANDBOX_MEM_LIMIT_BYTES, 2GiB).
_DEFAULT_MEM_LIMIT_BYTES = int(
    os.getenv("LOCA_SANDBOX_MEM_LIMIT_BYTES", str(2 * 1024**3))
)

# Cap on the output string a single execute() call may return (chars). Same
# default as training (VERL_SANDBOX_MAX_OUTPUT_CHARS). 0 disables truncation.
_DEFAULT_MAX_OUTPUT_CHARS = int(os.getenv("LOCA_SANDBOX_MAX_OUTPUT_CHARS", "50000"))

# prctl(PR_SET_PDEATHSIG, sig) -- kernel sends `sig` to this process when its
# parent dies. Linux-only; key=1 per <sys/prctl.h>.
_PR_SET_PDEATHSIG = 1

_KERNEL_START_LOCK = threading.Lock()

# Track spawned kernel PIDs (pid -> /proc starttime) so stragglers can be
# force-killed at process exit without touching recycled PIDs.
_SPAWNED_PIDS: dict = {}
_PID_LOCK = threading.Lock()


def _proc_starttime(pid: int) -> Optional[int]:
    """Kernel starttime (clock ticks since boot) of `pid`, or None if gone.
    (pid, starttime) uniquely identifies a process across pid reuse."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
        return int(data[data.rfind(")") + 2:].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def _find_child_pid(ppid: int) -> Optional[int]:
    """Return the single child PID of `ppid`, or None. Used to locate the kernel
    when it runs under bwrap (bwrap is the immediate child / process-group
    leader; the kernel is bwrap's child). Tries the cheap children file, then
    falls back to scanning /proc."""
    try:
        with open(f"/proc/{ppid}/task/{ppid}/children") as f:
            kids = f.read().split()
        if kids:
            return int(kids[0])
    except (OSError, ValueError):
        pass
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    for e in entries:
        if not e.isdigit():
            continue
        try:
            with open(f"/proc/{e}/stat") as f:
                data = f.read()
            after = data[data.rfind(")") + 2:].split()
            if int(after[1]) == ppid:
                return int(e)
        except (OSError, IndexError, ValueError):
            continue
    return None


def _force_kill_pid(pid: int, expected_starttime: Optional[int] = None):
    """Best-effort kill a process group, then the PID itself. Skipped when the
    live process's starttime no longer matches (PID was recycled)."""
    if expected_starttime is not None:
        current = _proc_starttime(pid)
        if current is None:
            return
        if current != expected_starttime:
            return
    for sig in (signal.SIGKILL,):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def kill_all_kernels():
    """Force-kill every kernel process we ever spawned."""
    with _PID_LOCK:
        pids = dict(_SPAWNED_PIDS)
        _SPAWNED_PIDS.clear()
    for pid, starttime in pids.items():
        _force_kill_pid(pid, expected_starttime=starttime)
    for pid in pids:
        try:
            os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, OSError):
            pass


atexit.register(kill_all_kernels)


def _set_pdeathsig(sig: int) -> None:
    """Best-effort: ask the kernel to deliver `sig` when our parent dies.
    Called from preexec_fn (already in the forked child)."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(
            ctypes.c_int(_PR_SET_PDEATHSIG),
            ctypes.c_ulong(sig),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
        )
    except Exception:
        pass


# Dedicated, process-lifetime thread used ONLY to fork (Popen) sandbox kernels.
# PR_SET_PDEATHSIG fires when the *thread* that called fork() exits -- a
# transient pool/worker thread would SIGKILL live kernels when recycled.
_FORK_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="sandbox-fork"
)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


class _BoundedStreamBuffer:
    """Accumulates kernel stream text with a bounded memory footprint.

    Keeps a fixed head plus a rolling tail (so the final traceback stays
    visible) and drops the middle DURING accumulation.

    ``head_cap``/``tail_cap`` <= 0 disables the bound.
    """

    def __init__(self, head_cap: int, tail_cap: int):
        self._head: list = []
        self._head_len = 0
        self._tail: collections.deque = collections.deque()
        self._tail_len = 0
        self._dropped = 0
        self._head_cap = head_cap
        self._tail_cap = tail_cap

    def append(self, text: str) -> None:
        if not text:
            return
        if self._head_cap <= 0 or self._tail_cap <= 0:
            self._head.append(text)
            return
        if self._head_len < self._head_cap:
            take = min(len(text), self._head_cap - self._head_len)
            self._head.append(text[:take])
            self._head_len += take
            text = text[take:]
            if not text:
                return
        if len(text) > self._tail_cap:
            self._dropped += self._tail_len + (len(text) - self._tail_cap)
            self._tail.clear()
            self._tail_len = 0
            self._tail.append(text[-self._tail_cap:])
            self._tail_len = self._tail_cap
            return
        self._tail.append(text)
        self._tail_len += len(text)
        while self._tail_len - len(self._tail[0]) >= self._tail_cap:
            self._dropped += len(self._tail[0])
            self._tail_len -= len(self._tail.popleft())

    def value(self) -> str:
        head = "".join(self._head)
        tail = "".join(self._tail)
        if self._dropped:
            return f"{head}\n...[{self._dropped} chars dropped during execution]...\n{tail}"
        return head + tail


def truncate_output(text: str, limit: int) -> str:
    """Middle-truncate `text` to ~`limit` chars, keeping head and tail.

    The head usually carries the model's own progress prints and the tail the
    final summary/traceback, so both are preserved. `limit <= 0` disables.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    omitted = len(text) - head - tail
    return (
        f"{text[:head]}\n...[output truncated: {omitted} chars omitted; "
        f"print concise summaries instead of large raw data]...\n{text[-tail:]}"
    )


class StatefulSandbox:
    """A stateful Python sandbox backed by an IPython kernel.

    State (variables, imports, defined functions, etc.) persists across
    execute() calls. Use as a context manager for automatic cleanup.

    Example:
        with StatefulSandbox(workspace_path="/tmp/ws") as sb:
            sb.execute("x = 42")
            print(sb.execute("print(x)"))  # prints 42
    """

    def __init__(
        self,
        workspace_path: str,
        timeout: float = 60.0,
        mem_limit_bytes: Optional[int] = None,
        interrupt_grace_seconds: float = 2.0,
        max_output_chars: Optional[int] = None,
        extra_bind_paths: Sequence[str] = (),
    ):
        self.workspace_path = workspace_path
        # Host paths that must stay visible AND writable inside the bwrap
        # sandbox. Divergence from verl's sandbox: the eval-side `tools` proxy
        # reaches the MCP clients over a unix socket in the parent process, and
        # bwrap mounts a private tmpfs over /tmp, so the socket's directory has
        # to be bound explicitly or the kernel cannot see it at all.
        self.extra_bind_paths = list(extra_bind_paths)
        self.max_output_chars = (
            max_output_chars if max_output_chars is not None else _DEFAULT_MAX_OUTPUT_CHARS
        )
        self.timeout = timeout
        self.mem_limit_bytes = (
            mem_limit_bytes if mem_limit_bytes is not None else _DEFAULT_MEM_LIMIT_BYTES
        )
        self.interrupt_grace_seconds = interrupt_grace_seconds
        self._tmpdir: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._kernel_pid: Optional[int] = None
        self._client = None
        self._conn_file: Optional[Path] = None
        self._cleaned_up = False
        self._dead = False
        self._use_bwrap = False

    def start(self):
        """Start the IPython kernel and initialize workspace state."""
        self._cleaned_up = False
        try:
            self._start_impl()
        except Exception:
            self._cleaned_up = False
            self.cleanup()
            raise

    def _start_impl(self):
        try:
            from jupyter_client import BlockingKernelClient
        except ImportError:
            raise RuntimeError(
                "jupyter_client is required for StatefulSandbox. "
                "Install it with: pip install jupyter_client ipykernel"
            )

        self._tmpdir = tempfile.mkdtemp(prefix="sandbox_")
        self._conn_file = Path(self._tmpdir) / "kernel.json"

        os.makedirs(self.workspace_path, exist_ok=True)
        mem_limit = self.mem_limit_bytes

        def _preexec():
            # Die with the parent so orphan ipykernels cannot outlive the runner.
            _set_pdeathsig(signal.SIGKILL)
            if mem_limit and mem_limit > 0:
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
                except (ValueError, OSError):
                    pass

        kernel_cmd = [sys.executable, "-m", "ipykernel_launcher", "-f", str(self._conn_file)]

        # bwrap is the primary write-confinement layer (matches training). When
        # usable, the kernel -- and everything it spawns (subprocess) or writes
        # via os.open / pathlib / pandas -- can only write inside the workspace,
        # closing the holes the Python open() guard alone cannot. The kernel
        # connection-file dir (self._tmpdir) must stay writable so ipykernel can
        # create it. Falls back to the open() guard when bwrap is unavailable.
        self._use_bwrap = bwrap_usable()
        if self._use_bwrap:
            kernel_cmd = (
                build_bwrap_argv(
                    self.workspace_path,
                    write_paths_extra=[self._tmpdir] + self.extra_bind_paths,
                )
                + kernel_cmd
            )

        # Capture the kernel's stderr to a file so an early exit reports the
        # real cause instead of an opaque exit code.
        stderr_path = os.path.join(self._tmpdir, "kernel_stderr.log")
        stderr_f = open(stderr_path, "wb")

        def _stderr_tail(limit: int = 4000) -> str:
            try:
                with open(stderr_path, "r", errors="replace") as f:
                    txt = f.read().strip()
            except OSError:
                return ""
            return txt[-limit:] if txt else ""

        # Fork on the dedicated lifetime thread (see _FORK_EXECUTOR) so the
        # kernel's PR_SET_PDEATHSIG is bound to a thread that lives as long as
        # the runner process, not a recyclable worker thread.
        try:
            self._proc = _FORK_EXECUTOR.submit(
                subprocess.Popen,
                kernel_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr_f,
                cwd=self.workspace_path,
                start_new_session=True,
                preexec_fn=_preexec,
            ).result()
        finally:
            stderr_f.close()  # parent's copy; the child keeps its own dup

        with _PID_LOCK:
            _SPAWNED_PIDS[self._proc.pid] = _proc_starttime(self._proc.pid)

        deadline = time.time() + 15.0
        while time.time() < deadline:
            if self._conn_file.exists():
                break
            if self._proc.poll() is not None:
                tail = _stderr_tail()
                mode = "bwrap" if self._use_bwrap else "guard-only"
                raise RuntimeError(
                    f"IPython kernel exited early with code {self._proc.returncode} "
                    f"(confinement={mode}). stderr:\n{tail or '<empty>'}"
                )
            time.sleep(0.05)
        else:
            raise RuntimeError(
                f"Timeout waiting for IPython kernel to start. stderr:\n"
                f"{_stderr_tail() or '<empty>'}"
            )

        client = BlockingKernelClient(connection_file=str(self._conn_file))
        client.load_connection_file()

        acquired = _KERNEL_START_LOCK.acquire(timeout=30)
        if not acquired:
            raise RuntimeError("Timed out waiting for kernel start lock")
        try:
            # Only channel/connection bring-up needs serializing (jupyter_client
            # connection-file load + ZMQ port setup is not concurrency-safe).
            client.start_channels()
        finally:
            _KERNEL_START_LOCK.release()
        client.wait_for_ready(timeout=10.0)

        self._client = client
        # PID to target for SIGINT interrupts. Normally self._proc.pid; under
        # bwrap the kernel is bwrap's child, so SIGINT must hit the child, not
        # the process group -- the group leader is bwrap, which dies on SIGINT
        # and takes the kernel with it (--die-with-parent), turning a state-
        # preserving interrupt into a hard kill. Falls back to self._proc.pid
        # (killpg path) if the child can't be resolved.
        self._kernel_pid = self._proc.pid
        if self._use_bwrap:
            child = _find_child_pid(self._proc.pid)
            if child is not None:
                self._kernel_pid = child

        # Same init_code as the training sandbox: pre-imports, `workspace`
        # variable, chdir into the workspace, and a Python-level open() guard
        # denying write-mode opens outside the workspace. When bwrap is active
        # it is the real write-confinement layer (covering os/subprocess/pathlib
        # too); this guard is then defense-in-depth. Without bwrap it is the ONLY
        # layer, and only covers built-in open().
        _guard_note = (
            "bwrap OS-level write-confinement also active"
            if self._use_bwrap
            else "bwrap unavailable; Python-level guard only"
        )
        init_code = (
            f"import os, json, csv, sys\n"
            f"workspace = {repr(self.workspace_path)}\n"
            f"os.chdir({repr(self.workspace_path)})\n"
            f"_open_orig = open\n"
            f"_WRITE_MODES = frozenset('wax+')\n"
            f"def open(file, mode='r', *args, **kwargs):\n"
            f"    if _WRITE_MODES & set(str(mode)):\n"
            f"        _abs = os.path.realpath(str(file))\n"
            f"        _ws  = os.path.realpath({repr(self.workspace_path)})\n"
            f"        if not (_abs.startswith(_ws + os.sep) or _abs == _ws):\n"
            f"            raise PermissionError(\n"
            f"                f'Write denied: {{file!r}} is outside workspace. '\n"
            f"                f'({_guard_note})'\n"
            f"            )\n"
            f"    return _open_orig(file, mode, *args, **kwargs)\n"
        )

        self._run(init_code)

        if self._cleaned_up:
            # cleanup() raced this start; reap the kernel we just started.
            self._kill_kernel_now()
            try:
                client.stop_channels()
                if client.context is not None:
                    client.context.destroy(linger=0)
            except Exception:
                pass
            raise RuntimeError("Sandbox cleanup() ran concurrently with start(); kernel reaped")

    def _try_interrupt_to_idle(self, msg_id: str, grace_seconds: float) -> bool:
        """Send SIGINT to the kernel and wait up to grace_seconds for the
        in-flight execution (msg_id) to reach an idle status. Returns True if
        the kernel recovered, False otherwise. State (variables/imports) is
        preserved on a successful interrupt."""
        if self._proc is None or self._proc.poll() is not None:
            return False
        try:
            if self._kernel_pid is not None and self._kernel_pid != self._proc.pid:
                # Under bwrap: SIGINT only the kernel, NOT the process group --
                # the group leader is bwrap, which would die on SIGINT and take
                # the kernel with it (--die-with-parent), turning a state-
                # preserving interrupt into a hard kill.
                os.kill(self._kernel_pid, signal.SIGINT)
            else:
                os.killpg(self._proc.pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError, OSError):
            return False
        deadline = time.time() + grace_seconds
        client = self._client
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            try:
                msg = client.get_iopub_msg(timeout=remaining)
            except Exception:
                return False
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            msg_type = msg.get("header", {}).get("msg_type", "")
            content = msg.get("content", {})
            if msg_type == "status" and content.get("execution_state") == "idle":
                return True

    def _kill_kernel_now(self):
        """Hard-kill the kernel process group and mark sandbox dead."""
        self._dead = True
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        if proc.poll() is not None:
            with _PID_LOCK:
                _SPAWNED_PIDS.pop(proc.pid, None)

    def _run(self, code: str, timeout: Optional[float] = None) -> tuple:
        """Execute code in the kernel and return (stdout, stderr, success)."""
        if self._dead:
            raise RuntimeError("Kernel was killed after a previous timeout")
        if self._proc is not None and self._proc.poll() is not None:
            self._dead = True
            with _PID_LOCK:
                _SPAWNED_PIDS.pop(self._proc.pid, None)
            raise RuntimeError(
                f"Kernel process has exited unexpectedly (code {self._proc.returncode})"
            )
        client = self._client
        msg_id = client.execute(code, silent=False, store_history=True)
        effective_timeout = self.timeout if timeout is None else timeout
        deadline = time.time() + effective_timeout

        # Bound the in-flight accumulation (not just the final result): a
        # print-storm can stream far more than max_output_chars before the
        # execution finishes, and truncate_output only runs at the end.
        if self.max_output_chars > 0:
            head_cap = max(2 * self.max_output_chars, 131072)
            tail_cap = max(self.max_output_chars, 65536)
        else:
            head_cap = tail_cap = 0  # truncation disabled -> unbounded
        out_parts = _BoundedStreamBuffer(head_cap, tail_cap)
        err_parts = _BoundedStreamBuffer(head_cap, tail_cap)
        success = True

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                # Try to interrupt cleanly so kernel state survives; otherwise
                # SIGKILL the kernel so subsequent execute() calls don't block
                # for another full timeout each.
                recovered = self._try_interrupt_to_idle(msg_id, self.interrupt_grace_seconds)
                if not recovered:
                    self._kill_kernel_now()
                raise TimeoutError("Kernel execution timed out")
            try:
                msg = client.get_iopub_msg(timeout=remaining)
            except Exception:
                recovered = self._try_interrupt_to_idle(msg_id, self.interrupt_grace_seconds)
                if not recovered:
                    self._kill_kernel_now()
                raise TimeoutError("Kernel execution timed out")

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("header", {}).get("msg_type", "")
            content = msg.get("content", {})

            if msg_type == "stream":
                target = out_parts if content.get("name") == "stdout" else err_parts
                target.append(content.get("text", ""))
            elif msg_type in ("execute_result", "display_data"):
                data = content.get("data", {})
                if "text/plain" in data:
                    out_parts.append(f"{data['text/plain']}\n")
            elif msg_type == "error":
                success = False
                tb = content.get("traceback") or []
                if tb:
                    err_parts.append("\n".join(tb) + "\n")
                else:
                    err_parts.append(
                        f"{content.get('ename', 'Error')}: {content.get('evalue', '')}\n"
                    )
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        return strip_ansi(out_parts.value()), strip_ansi(err_parts.value()), success

    def execute(self, code: str, timeout: Optional[float] = None) -> tuple:
        """Execute code and return (output, success).

        State from previous calls is preserved. Returns an error string on
        failure rather than raising. ``success`` is False whenever the kernel
        reported an exception, the sandbox was unavailable, or execution timed
        out -- i.e. any case where ``output`` carries an error message.

        ``timeout`` overrides the sandbox-level default for this single call.
        """
        if not isinstance(code, str):
            return (
                f"[Error] The 'code' argument must be a string, got "
                f"{type(code).__name__}: {code!r}.",
                False,
            )
        if self._client is None:
            return "[Error] Sandbox not started. Call start() first.", False
        if self._dead:
            return (
                "[Error] Sandbox is no longer available "
                "(killed after a previous timeout / crash). Subsequent code will not run."
            ), False
        effective_timeout = self.timeout if timeout is None else timeout
        try:
            output, error, success = self._run(code, timeout=effective_timeout)
            result = output
            if error.strip():
                result += ("\n[stderr]:\n" if result else "") + error.strip()
            result = truncate_output(result.strip(), self.max_output_chars)
            return (result if result else "(no output)"), success
        except TimeoutError:
            suffix = (
                " Sandbox killed; subsequent code will not run."
                if self._dead
                else " Sandbox interrupted; state preserved."
            )
            return (
                f"[Error] Code execution timed out ({effective_timeout}-second limit).{suffix}",
                False,
            )
        except Exception as e:
            return f"[Error] {e}", False

    def cleanup(self):
        """Stop the kernel and remove temporary files."""
        if self._cleaned_up:
            return
        self._cleaned_up = True
        self._dead = True

        proc = self._proc
        self._proc = None
        if proc is not None:
            pid = proc.pid
            if proc.poll() is None:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                    try:
                        proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
            else:
                try:
                    proc.wait(timeout=1)
                except Exception:
                    pass
            with _PID_LOCK:
                _SPAWNED_PIDS.pop(pid, None)

        client = self._client
        self._client = None
        if client is not None:
            try:
                client.stop_channels()
            except Exception:
                pass
            # Force-destroy the underlying ZMQ context to release ports immediately
            try:
                if hasattr(client, 'context') and client.context is not None:
                    client.context.destroy(linger=0)
            except Exception:
                pass

        tmpdir = self._tmpdir
        self._tmpdir = None
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
