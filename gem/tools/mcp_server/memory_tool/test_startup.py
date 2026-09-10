"""Regression coverage for storage removed between configuration and discovery."""

import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path

from fastmcp import Client

from gem.tools.mcp_server.config_loader import build_server_config
from gem.tools.mcp_tool import MCPTool


class MemoryStartupTest(unittest.TestCase):
    def test_discovery_and_storage_after_task_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task"
            workspace = task / "agent_workspace"
            base = workspace / "memory"
            base.mkdir(parents=True)
            servers = build_server_config("memory_tool", {"base_path": str(base)})
            # A second server exercises the runner's multi-server prefix check.
            servers.update(build_server_config("claim_done", {}))

            # Environments reset after server configuration has been built.
            shutil.rmtree(task)
            workspace.mkdir(parents=True)
            self.assertFalse(base.exists())

            config = {"mcpServers": servers}
            tool = MCPTool(config, validate_on_init=False)
            try:
                names = {entry["name"] for entry in tool.get_available_tools()}
                self.assertTrue({
                    "memory_tool_view", "memory_tool_create",
                    "memory_tool_str_replace", "memory_tool_insert",
                    "memory_tool_delete", "memory_tool_rename",
                }.issubset(names), names)
            finally:
                tool.close()

            async def check_storage():
                async with Client(config, timeout=20) as client:
                    created = await client.call_tool("memory_tool_create", {
                        "path": "/memories/check.txt", "file_text": "reset survived",
                    })
                    self.assertFalse(created.is_error)
                    viewed = await client.call_tool("memory_tool_view", {
                        "path": "/memories/check.txt",
                    })
                    self.assertFalse(viewed.is_error)
                    self.assertIn("reset survived", str(viewed))

            asyncio.run(check_storage())
            self.assertEqual((base / "memories/check.txt").read_text(), "reset survived")


if __name__ == "__main__":
    unittest.main()
