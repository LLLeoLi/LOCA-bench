#!/usr/bin/env python3
"""Launch excel-mcp-server (stdio) with workspace confinement actually enforced.

Upstream excel-mcp-server only assigns EXCEL_FILES_PATH in SSE / streamable-http
mode. In stdio mode the global stays ``None`` and ``get_excel_path()`` then
REQUIRES an absolute path and returns it as-is -- i.e. the model can create or
overwrite ``.xlsx`` files anywhere on the filesystem. This wrapper assigns the
module-level ``EXCEL_FILES_PATH`` from the environment before serving, which
activates the containment logic upstream already ships (relative paths only,
realpath must stay inside the base directory).

It also deals with upstream's import-time file log: ``excel_mcp.server``
configures a ``logging.FileHandler`` pointing NEXT TO site-packages
(``<venv>/lib/pythonX.Y/excel-mcp.log``) the moment it is imported -- a stray
write outside the workspace, and a hard crash under bwrap's read-only root. We
force ``delay=True`` during the import so no file is opened, then swap the
handler for one inside the workspace.
"""

import logging
import os
import sys


def main() -> None:
    base = os.environ.get("EXCEL_FILES_PATH") or "."
    base = os.path.realpath(base)
    os.makedirs(base, exist_ok=True)

    original_file_handler = logging.FileHandler

    class _DeferredFileHandler(original_file_handler):
        def __init__(self, filename, mode="a", encoding=None, delay=False, errors=None):
            super().__init__(filename, mode=mode, encoding=encoding, delay=True, errors=errors)

    logging.FileHandler = _DeferredFileHandler
    try:
        import excel_mcp.server as server
    finally:
        logging.FileHandler = original_file_handler

    # Never opened thanks to delay=True; replace with a workspace-local log.
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()
    root_logger.addHandler(
        logging.FileHandler(os.path.join(base, ".excel-mcp.log"), encoding="utf-8")
    )

    server.EXCEL_FILES_PATH = base

    # With EXCEL_FILES_PATH set, upstream rejects ALL absolute paths — but the
    # benchmark's models legitimately pass absolute paths inside the workspace
    # (that was the only accepted form before this wrapper). Accept those;
    # reject absolute paths that resolve outside the workspace.
    upstream_get_excel_path = server.get_excel_path

    def get_excel_path(filename: str):
        if filename and "\x00" not in filename and os.path.isabs(filename):
            real = os.path.realpath(filename)
            if real == base or real.startswith(base + os.sep):
                return real
            raise ValueError(
                f"Invalid filename: {filename} is outside the workspace {base}. "
                "Use a path inside the workspace (relative paths resolve there)."
            )
        return upstream_get_excel_path(filename)

    server.get_excel_path = get_excel_path
    server.run_stdio()


if __name__ == "__main__":
    sys.exit(main())
