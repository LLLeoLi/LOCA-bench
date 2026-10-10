# Copyright 2025 LOCA-bench Contributors. All Rights Reserved.
#
# Licensed under the MIT License.

"""Strategy-specific MCP server injection for LOCA-bench CLI."""

import copy
import json
from pathlib import Path
from typing import Any, Dict

from loca.cli.utils.config_resolver import Strategy


# MCP server configurations for each strategy
STRATEGY_MCP_SERVERS: Dict[Strategy, Dict[str, Any]] = {
    Strategy.REACT: {},  # No additional servers
    Strategy.HYBRID: {
        "programmatic_tool_calling": {
            "enabled": True,
            "type": "programmatic_tool_calling",
            "params": {
                "workspace_path": "{agent_workspace}"
            }
        }
    },
    Strategy.PTC: {
        "programmatic_tool_calling": {
            "enabled": True,
            "type": "programmatic_tool_calling",
            # Marker read by the runner (not by setup_mcp_servers): task tool
            # schemas stay visible to the model, but direct calls to anything
            # other than code_execution / claim_done are rejected.
            "ptc_only": True,
            "params": {
                "workspace_path": "{agent_workspace}"
            }
        }
    },
    Strategy.MEMORY_TOOL: {
        "memory_tool": {
            "enabled": True,
            "type": "memory_tool",
            "params": {
                "base_path": "{agent_workspace}/memory",
                "server_name": "memory_tool"
            }
        }
    },
}


def inject_strategy_servers(
    config_path: Path,
    strategy: Strategy,
    output_path: Path,
    suppress_errors: bool = False,
) -> Path:
    """Inject strategy-specific MCP servers into a config file.

    Reads the config file, adds strategy-specific MCP servers to each
    configuration, and writes the result to output_path.

    Args:
        config_path: Path to the original config file.
        strategy: The context management strategy.
        output_path: Path to write the modified config.
        suppress_errors: Ablation marker read by the runner. When True, the
            injected programmatic_tool_calling server gets ``"suppress_errors":
            True`` so tool errors inside code_execution are returned as plain
            values instead of raising, and observations are never flagged
            is_error. Only meaningful for the hybrid / ptc strategies.

    Returns:
        Path to the modified config file.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    additional_servers = copy.deepcopy(STRATEGY_MCP_SERVERS.get(strategy, {}))
    if suppress_errors and "programmatic_tool_calling" in additional_servers:
        additional_servers["programmatic_tool_calling"]["suppress_errors"] = True

    if additional_servers:
        # Inject servers into each configuration
        for cfg in config.get("configurations", []):
            mcp_servers = cfg.get("mcp_servers", {})
            mcp_servers.update(additional_servers)
            cfg["mcp_servers"] = mcp_servers

    # Write modified config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    return output_path
