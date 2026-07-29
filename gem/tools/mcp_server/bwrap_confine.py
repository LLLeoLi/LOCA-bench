"""Bubblewrap (bwrap) filesystem confinement for LOCA-Bench code-execution tools.

Ported from the training-side sandbox
(``verl/experimental/agent_loop/landlock_sandbox.py``) so the eval-side PTC
kernel and ``python_execute`` subprocess confine writes to the agent workspace
the same way training does. On the training side bwrap is the PRIMARY
filesystem-confinement layer; the Python-level ``open()`` guard is only
defense-in-depth. The eval-side vendoring originally kept only the ``open()``
guard, which any of ``os.open`` / ``pathlib.Path.write_text`` / pandas /
``subprocess`` bypasses -- letting model code write to ``/mnt``, ``/root``,
``/workspace``, etc. Wrapping the interpreter subprocess in bwrap closes that.

Confinement policy (differs from training in ONE way -- reads):
  - Reads: the whole root is bind-mounted READ-ONLY by default
    (``--ro-bind / /``). Training uses a default-DENY read allowlist to hide
    sibling rollouts' ``data.db`` / model weights, but the eval interpreter and
    its deps live under ``/root`` (miniconda, ``uv``) which is not on that
    allowlist, and read-isolation is not the goal here -- WRITE-confinement is.
    Override the read policy with ``LOCA_SANDBOX_BWRAP_READ_PATHS`` (colon-
    separated) to opt into default-deny reads.
  - Writes: confined to the workspace + a private (tmpfs) ``/tmp`` & ``/dev/shm``
    + any explicit ``write_paths_extra`` (e.g. the kernel connection-file dir).
    Everything else is read-only, so writes outside the workspace fail at the
    kernel level regardless of how the model's code issues them.
  - Network is intentionally NOT isolated (kernel<->client ZMQ on 127.0.0.1).

Set ``LOCA_SANDBOX_USE_BWRAP=0`` to disable and fall back to the ``open()``
guard alone.
"""

import logging
import os
import shutil
import subprocess
import threading

logger = logging.getLogger(__name__)

_BWRAP_BIN = shutil.which("bwrap")
_USE_BWRAP = os.getenv("LOCA_SANDBOX_USE_BWRAP", "1") != "0" and _BWRAP_BIN is not None

# Read policy. Default "/" = expose the whole root read-only (write-confinement
# only). Set to a colon-separated allowlist (e.g. "/usr:/lib:/lib64:/bin:/etc")
# to opt into default-DENY reads like training. Empty entries are dropped.
_BWRAP_READ_PATHS = [
    p
    for p in os.getenv("LOCA_SANDBOX_BWRAP_READ_PATHS", "/").split(":")
    if p
]
# When the read policy is the whole root we bind "/" read-only directly; a
# per-subdir allowlist instead needs a fresh /proc + /dev + tmpfs /tmp.
_READ_WHOLE_ROOT = _BWRAP_READ_PATHS == ["/"]

_BWRAP_USABLE = None
_BWRAP_PROBE_LOCK = threading.Lock()
_LOGGED_ONCE = set()
_LOG_ONCE_LOCK = threading.Lock()


def _log_once(key: str, level: int, msg: str) -> None:
    """Emit ``msg`` at most once per process (keyed by ``key``)."""
    with _LOG_ONCE_LOCK:
        if key in _LOGGED_ONCE:
            return
        _LOGGED_ONCE.add(key)
    logger.log(level, msg)


def bwrap_usable() -> bool:
    """True iff bwrap is enabled and can actually create a namespace here.

    Probed once with a trivial sandbox and cached process-wide. Many
    locked-down containers (no CAP_SYS_ADMIN / unprivileged-userns disabled)
    have the ``bwrap`` binary but reject namespace creation; those fall back to
    the ``open()`` guard.
    """
    global _BWRAP_USABLE
    if not _USE_BWRAP:
        return False
    if _BWRAP_USABLE is not None:
        return _BWRAP_USABLE
    with _BWRAP_PROBE_LOCK:
        if _BWRAP_USABLE is not None:
            return _BWRAP_USABLE
        try:
            r = subprocess.run(
                [_BWRAP_BIN, "--unshare-user-try", "--ro-bind", "/", "/", "--", "true"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=10,
            )
            ok = r.returncode == 0
            if not ok:
                _log_once(
                    "bwrap_unusable",
                    logging.WARNING,
                    "[sandbox] bwrap is present but cannot create namespaces here "
                    f"({r.stderr.decode(errors='replace').strip() or 'unknown error'}); "
                    "falling back to the Python open() write-guard ONLY. Writes via "
                    "os.open / pathlib / subprocess will NOT be confined to the "
                    "workspace. Enable userns / CAP_SYS_ADMIN, or set "
                    "LOCA_SANDBOX_USE_BWRAP=0 to silence this.",
                )
        except (OSError, subprocess.SubprocessError) as e:
            _log_once(
                "bwrap_probe_error",
                logging.WARNING,
                f"[sandbox] bwrap probe failed ({e}); falling back to the Python "
                "open() write-guard only (writes not confined via os/subprocess).",
            )
            ok = False
        _BWRAP_USABLE = ok
        return ok


def build_bwrap_argv(workspace, write_paths_extra=()):
    """Return the bwrap argv prefix that confines writes to ``workspace``.

    Prepend this to the real command (e.g. the ipykernel launcher or the
    ``uv run`` invocation). Callers should only prepend when
    :func:`bwrap_usable` is True.

    Reads: whole root read-only by default (see module docstring), or a
    default-deny allowlist when ``LOCA_SANDBOX_BWRAP_READ_PATHS`` is set.
    Writes: ``workspace`` + private ``/tmp`` & ``/dev/shm`` + ``write_paths_extra``.

    No ``--unshare-*`` beyond the user namespace, so the child keeps loopback
    (kernel ZMQ) and stays in the caller's process group (``killpg``).
    """
    # --unshare-user-try: when running as root WITHOUT CAP_SYS_ADMIN, bwrap
    # won't create a user namespace by default and the mount namespace then
    # fails; forcing a user ns first grants the caps needed. No-op elsewhere.
    argv = [_BWRAP_BIN, "--unshare-user-try"]

    if _READ_WHOLE_ROOT:
        # Whole root read-only; overlay writable bits on top.
        argv += ["--ro-bind", "/", "/"]
        argv += [
            "--dev", "/dev",          # minimal writable device nodes
            "--tmpfs", "/tmp",        # private writable /tmp
            "--tmpfs", "/dev/shm",    # private writable shm
        ]
    else:
        # Default-deny reads: only the allowlist is visible.
        for p in _BWRAP_READ_PATHS:
            argv += ["--ro-bind-try", p, p]
        argv += [
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            "--tmpfs", "/dev/shm",
        ]

    # Workspace bound writable AFTER the read layer so it overlays read-only
    # even when it sits inside a read-only subtree.
    os.makedirs(workspace, exist_ok=True)
    argv += ["--bind", workspace, workspace]
    for p in write_paths_extra:
        if p:
            argv += ["--bind", p, p]
    argv += [
        "--setenv", "HOME", workspace,   # libs write ~/.cache into the workspace
        "--chdir", workspace,
        "--die-with-parent",
        "--",
    ]
    return argv


__all__ = ["bwrap_usable", "build_bwrap_argv"]
