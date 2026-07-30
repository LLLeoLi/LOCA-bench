"""YAML-based MCP server configuration loader.

This module provides a generic loader for MCP server configurations,
replacing the individual helper.py files with declarative YAML configs.
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ServerConfigLoader:
    """Loads and processes MCP server configurations from YAML files."""

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the config loader.

        Args:
            base_dir: Base directory containing server subdirectories.
                     Defaults to the directory of this file.
        """
        self.base_dir = base_dir or Path(__file__).parent

    def load_config(self, server_type: str) -> Dict[str, Any]:
        """Load and validate YAML config for a server type.

        Args:
            server_type: Type of server (e.g., 'canvas', 'claim_done')

        Returns:
            Parsed YAML configuration dictionary

        Raises:
            FileNotFoundError: If config YAML doesn't exist
            ValueError: If YAML is invalid or missing required fields
        """
        # New location: config/{server_type}.yaml
        config_path = self.base_dir / "config" / f"{server_type}.yaml"

        # Fallback to old location for backward compatibility during transition
        if not config_path.exists():
            old_config_path = self.base_dir / server_type / "server_config.yaml"
            if old_config_path.exists():
                config_path = old_config_path
            else:
                raise FileNotFoundError(
                    f"No YAML config found for server type '{server_type}'. "
                    f"Tried: {config_path} and {old_config_path}"
                )

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}") from e

        # Validate required fields
        if "name" not in config:
            raise ValueError(f"Missing 'name' field in {config_path}")
        if "execution" not in config:
            raise ValueError(f"Missing 'execution' field in {config_path}")

        return config

    def build_stdio_config(
        self,
        server_type: str,
        params: Dict[str, Any],
        server_name: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Build stdio configuration for a server.

        Args:
            server_type: Type of server (e.g., 'canvas', 'claim_done')
            params: Runtime parameters from JSON config
            server_name: Override server name (defaults to config name)

        Returns:
            Configuration dict in format: {server_name: {"command": ..., "args": [...], "env": {...}}}

        Raises:
            FileNotFoundError: If config doesn't exist
            ValueError: If config is invalid or required params missing
        """
        config = self.load_config(server_type)

        # Resolve server name
        actual_server_name = server_name or config["name"]

        # Build command
        command, args = self._build_command(config, params)

        # Build environment variables
        env = self._build_env_vars(config, params)

        # Determine working directory
        cwd = self._determine_cwd(config, params)

        # OS-level write confinement (opt-in per server via workspace.confine_writes).
        # Needed for servers that execute model-controlled code/commands or accept
        # model-controlled output paths without validating them themselves (e.g.
        # cli-mcp-server lets `python -c`/`awk` program strings through its path
        # checks). Wrapping the server process in bwrap confines every write it or
        # its children make to the workspace, no matter how the write is issued.
        #
        # The confinement root doubles as the process cwd, so it must stay the
        # server's own cwd; the task workspace and agent workspace are bound
        # writable alongside it, since a server's legitimate targets can sit in
        # a sibling directory (pdf_tools' tempfile_dir, the local_db data dirs).
        if config.get("workspace", {}).get("confine_writes", False):
            task_ws = params.get("task_workspace")
            agent_ws = params.get("agent_workspace")
            confine_root = cwd or task_ws or agent_ws
            if confine_root:
                also_writable = [
                    str(Path(p).resolve())
                    for p in (task_ws, agent_ws)
                    if p
                ]
                command, args, env = self._apply_write_confinement(
                    command, args, env, str(Path(confine_root).resolve()), also_writable
                )
            else:
                warnings.warn(
                    f"confine_writes set for server '{actual_server_name}' but no "
                    "workspace root could be determined (no cwd/task_workspace/"
                    "agent_workspace); launching WITHOUT write confinement.",
                    RuntimeWarning,
                )

        # Build stdio config
        stdio_config = {
            "command": command,
            "args": args,
        }

        if env:
            stdio_config["env"] = env

        if cwd:
            stdio_config["cwd"] = cwd

        return {actual_server_name: stdio_config}

    _uv_cache_dir_cached: Optional[str] = None

    @classmethod
    def _uv_cache_dir(cls) -> Optional[str]:
        """Effective uv cache directory, as uv itself resolves it.

        uv's cache location comes from a chain of sources (UV_CACHE_DIR,
        XDG_CACHE_HOME, uv.toml, HOME); asking uv is the only robust way to
        find the directory the confined server would actually use.
        """
        if cls._uv_cache_dir_cached is None:
            import subprocess

            try:
                result = subprocess.run(
                    ["uv", "cache", "dir"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                cls._uv_cache_dir_cached = (
                    result.stdout.strip() if result.returncode == 0 else ""
                )
            except (OSError, subprocess.SubprocessError):
                cls._uv_cache_dir_cached = ""
        return cls._uv_cache_dir_cached or None

    @staticmethod
    def _load_bwrap_confine():
        """Import bwrap_confine without importing the gem.tools package.

        Importing ``gem.tools`` runs its ``__init__`` chain (mcp_tool), which is
        unsafe from MCP server subprocesses; loading by file path works in every
        context this loader runs in.
        """
        import importlib.util

        path = Path(__file__).parent / "bwrap_confine.py"
        spec = importlib.util.spec_from_file_location("bwrap_confine", str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _apply_write_confinement(
        self,
        command: str,
        args: List[str],
        env: Dict[str, str],
        cwd: str,
        also_writable: Optional[List[str]] = None,
    ) -> tuple[str, List[str], Dict[str, str]]:
        """Wrap the server launch in bwrap so it can only write inside ``cwd``.

        ``also_writable`` paths are bound writable too (the task/agent
        workspaces), while ``cwd`` stays the process working directory.

        The whole filesystem stays readable; writes outside these paths fail
        with EROFS at the kernel level for the server process AND everything it
        spawns. Falls back (with a warning) to the unconfined launch when bwrap
        cannot create namespaces on this host.
        """
        bwrap = self._load_bwrap_confine()
        if not bwrap.bwrap_usable():
            warnings.warn(
                "confine_writes requested but bwrap is unusable on this host; "
                "the server will run WITHOUT OS-level write confinement.",
                RuntimeWarning,
            )
            return command, args, env

        env = dict(env or {})
        write_paths_extra = []
        for path in also_writable or []:
            if path and path != cwd:
                os.makedirs(path, exist_ok=True)
                write_paths_extra.append(path)
        if command in ("uv", "uvx"):
            # uv needs write access to its cache even for a fully synced
            # `uv run` (lock files under the cache dir), and bwrap remaps HOME
            # into the workspace, which would otherwise send the cache there
            # (cold resolve + downloads per launch). Pin the effective cache
            # dir and bind it writable.
            cache_dir = self._uv_cache_dir()
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                write_paths_extra.append(cache_dir)
                env.setdefault("UV_CACHE_DIR", cache_dir)
            python_install_dir = os.environ.get("UV_PYTHON_INSTALL_DIR")
            if python_install_dir:
                env.setdefault("UV_PYTHON_INSTALL_DIR", python_install_dir)

        argv = bwrap.build_bwrap_argv(cwd, write_paths_extra=write_paths_extra)
        argv += [command] + args
        return argv[0], argv[1:], env

    def _build_command(
        self, config: Dict[str, Any], params: Dict[str, Any]
    ) -> tuple[str, List[str]]:
        """Build command and arguments from config.

        Args:
            config: Server configuration dict
            params: Runtime parameters

        Returns:
            Tuple of (command, args_list)

        Raises:
            ValueError: If command type is unknown or required fields missing
        """
        execution = config["execution"]
        command_type = execution.get("command_type")

        if command_type == "python":
            return self._build_python_command(config, params)
        elif command_type == "uv":
            return self._build_uv_command(config, params)
        elif command_type == "uvx":
            return self._build_uvx_command(config, params)
        elif command_type == "node":
            return self._build_node_command(config, params)
        elif command_type == "npx":
            return self._build_npx_command(config, params)
        elif command_type == "direct":
            return self._build_direct_command(config, params)
        else:
            raise ValueError(f"Unknown command_type: {command_type}")

    def _build_python_command(
        self, config: Dict[str, Any], params: Dict[str, Any]
    ) -> tuple[str, List[str]]:
        """Build Python command."""
        execution = config["execution"]
        script_path = self._resolve_script_path(execution.get("script", {}))

        args = [str(script_path)]
        args.extend(self._build_cli_args(config, params))

        return "python", args

    def _build_uv_command(
        self, config: Dict[str, Any], params: Dict[str, Any]
    ) -> tuple[str, List[str]]:
        """Build UV command."""
        execution = config["execution"]
        script_path = self._resolve_script_path(execution.get("script", {}))

        # Determine project root
        project_root = self._get_project_root(script_path)

        args = ["--directory", str(project_root), "run", "python", str(script_path)]
        args.extend(self._build_cli_args(config, params))

        return "uv", args

    def _build_uvx_command(
        self, config: Dict[str, Any], params: Dict[str, Any]
    ) -> tuple[str, List[str]]:
        """Build UVX command."""
        execution = config["execution"]
        package_name = execution.get("package_name")

        if not package_name:
            raise ValueError("uvx command_type requires 'package_name' in execution config")

        args = []
        # Extra requirement specs resolved into the ephemeral uvx environment,
        # used to pin transitive dependencies (e.g. "mcp<2" for cli-mcp-server)
        for requirement in execution.get("with_requirements", []):
            args.extend(["--with", requirement])
        args.append(package_name)
        args.extend(self._build_cli_args(config, params))

        return "uvx", args

    def _build_node_command(
        self, config: Dict[str, Any], params: Dict[str, Any]
    ) -> tuple[str, List[str]]:
        """Build Node command with optional fallback to NPX."""
        execution = config["execution"]

        # Try to resolve script path
        try:
            script_path = self._resolve_script_path(execution.get("script", {}))
            args = [str(script_path)]
            args.extend(self._build_cli_args(config, params))
            return "node", args
        except FileNotFoundError:
            # Fall back to NPX if defined
            fallback = execution.get("fallback_command")
            if fallback and fallback.get("command_type") == "npx":
                package_name = fallback.get("package_name")
                if package_name:
                    args = [package_name]
                    args.extend(self._build_cli_args(config, params))
                    return "npx", args
            # Re-raise if no fallback
            raise

    def _build_npx_command(
        self, config: Dict[str, Any], params: Dict[str, Any]
    ) -> tuple[str, List[str]]:
        """Build NPX command."""
        execution = config["execution"]
        package_name = execution.get("package_name")

        if not package_name:
            raise ValueError("npx command_type requires 'package_name' in execution config")

        args = [package_name]
        args.extend(self._build_cli_args(config, params))

        return "npx", args

    def _build_direct_command(
        self, config: Dict[str, Any], params: Dict[str, Any]
    ) -> tuple[str, List[str]]:
        """Build direct executable command."""
        execution = config["execution"]
        executable_name = execution.get("executable_name")

        if not executable_name:
            raise ValueError("direct command_type requires 'executable_name' in execution config")

        args = execution.get("additional_args", []).copy()
        args.extend(self._build_cli_args(config, params))

        return executable_name, args

    def _resolve_param_value(
        self, param_name: str, param_spec: Dict[str, Any], params: Dict[str, Any]
    ) -> Optional[Any]:
        """Resolve a parameter value: explicit value, then aliases, then default.

        Placeholders like {agent_workspace} are replaced from params; a value
        that resolves to an empty string is treated as unresolved.

        Raises:
            ValueError: If the parameter is required but not provided
        """
        param_value = params.get(param_name)
        if param_value is None and "aliases" in param_spec:
            for alias in param_spec["aliases"]:
                if params.get(alias) is not None:
                    param_value = params[alias]
                    break

        if param_value is None:
            if param_spec.get("required", False):
                raise ValueError(f"Required parameter '{param_name}' not provided")
            param_value = param_spec.get("default")

        if isinstance(param_value, str):
            param_value = self._replace_placeholders(param_value, params)
            if param_value == "":
                return None

        return param_value

    def _build_cli_args(self, config: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
        """Build CLI arguments from parameters.

        Args:
            config: Server configuration dict
            params: Runtime parameters

        Returns:
            List of CLI argument strings
        """
        args = []
        param_config = config.get("parameters", {})

        # Process parameters in config order
        for param_name, param_spec in param_config.items():
            # Handle parameter aliases
            if "alias_for" in param_spec:
                continue  # Skip aliases, they'll be resolved when processing the target param

            param_value = self._resolve_param_value(param_name, param_spec, params)

            # Skip if no value
            if param_value is None:
                continue

            # Add to args if cli_arg specified
            cli_arg = param_spec.get("cli_arg")
            if cli_arg is not None:
                if cli_arg == "":
                    # Positional argument (no flag)
                    args.append(str(param_value))
                else:
                    # Flag-based argument
                    args.extend([cli_arg, str(param_value)])

        return args

    def _build_env_vars(self, config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, str]:
        """Build environment variables from parameters.

        Args:
            config: Server configuration dict
            params: Runtime parameters

        Returns:
            Dictionary of environment variables
        """
        env = {}
        explicit_env_vars = set()
        param_config = config.get("parameters", {})

        for param_name, param_spec in param_config.items():
            # Handle parameter aliases
            if "alias_for" in param_spec:
                continue

            param_value = self._resolve_param_value(param_name, param_spec, params)

            # Skip if no value
            if param_value is None:
                continue

            # Add to env if env_var specified
            env_var = param_spec.get("env_var")
            if env_var:
                # When several parameters map to the same env var, a value that
                # falls back to a spec default must not clobber a value the
                # caller provided explicitly (e.g. terminal's ALLOWED_DIR).
                was_explicit = params.get(param_name) is not None or any(
                    params.get(alias) is not None
                    for alias in param_spec.get("aliases", [])
                )
                if env_var in explicit_env_vars and not was_explicit:
                    continue
                env[env_var] = str(param_value)
                if was_explicit:
                    explicit_env_vars.add(env_var)

        # Add terminal-specific env vars if present
        if config["name"] == "terminal":
            self._add_terminal_env_vars(env, params)

        return env

    def _add_terminal_env_vars(self, env: Dict[str, str], params: Dict[str, Any]) -> None:
        """Add terminal-specific environment variables.

        The terminal server uses many env vars that don't follow the standard pattern.
        """
        terminal_env_mapping = {
            "allowed_commands": "ALLOWED_COMMANDS",
            "allowed_flags": "ALLOWED_FLAGS",
            "max_command_length": "MAX_COMMAND_LENGTH",
            "command_timeout": "COMMAND_TIMEOUT",
            "allow_shell_operators": "ALLOW_SHELL_OPERATORS",
            "max_output_length": "MAX_OUTPUT_LENGTH",
            "max_stdout_length": "MAX_STDOUT_LENGTH",
            "max_stderr_length": "MAX_STDERR_LENGTH",
            "cli_proxy_enabled": "CLI_PROXY_ENABLED",
            "cli_proxy_url": "CLI_PROXY_URL",
        }

        for param_name, env_name in terminal_env_mapping.items():
            if param_name in params:
                value = params[param_name]
                if isinstance(value, list):
                    value = ",".join(str(v) for v in value)
                env[env_name] = str(value)

    def _determine_cwd(self, config: Dict[str, Any], params: Dict[str, Any]) -> Optional[str]:
        """Determine working directory for server.

        Args:
            config: Server configuration dict
            params: Runtime parameters

        Returns:
            Working directory path or None
        """
        workspace = config.get("workspace", {})

        if not workspace.get("set_cwd", False):
            return None

        # Determine which parameter contains the workspace path
        cwd_param = workspace.get("cwd_param")
        if cwd_param:
            # Resolve through the parameter spec so aliases and defaults are
            # honoured (task configs typically pass e.g. `workspace_dir` for a
            # `workspace_path` cwd_param).
            param_spec = config.get("parameters", {}).get(cwd_param, {})
            cwd_value = self._resolve_param_value(cwd_param, param_spec, params)
            if cwd_value:
                cwd_path = Path(cwd_value).resolve()

                # Create directory if it doesn't exist and mkdir_if_needed is True
                if workspace.get("mkdir_if_needed", True):
                    cwd_path.mkdir(parents=True, exist_ok=True)

                return str(cwd_path)

        return None

    def _resolve_script_path(self, script_config: Dict[str, Any]) -> Path:
        """Resolve script path with fallback logic.

        Args:
            script_config: Script configuration dict with 'primary' and optional 'fallbacks'

        Returns:
            Resolved Path object

        Raises:
            FileNotFoundError: If no valid script path found
        """
        if not script_config:
            raise ValueError("Script configuration is empty")

        primary = script_config.get("primary")
        if not primary:
            raise ValueError("Script configuration missing 'primary' field")

        fallbacks = script_config.get("fallbacks", [])

        # Try primary path (relative to gem root)
        gem_root = self._get_gem_root()
        primary_path = gem_root / primary

        if primary_path.exists():
            return primary_path

        # Try fallback paths (absolute)
        for fallback in fallbacks:
            fallback_path = Path(fallback)
            if fallback_path.exists():
                return fallback_path

        # Try relative to current directory as last resort
        if Path(primary).exists():
            return Path(primary).resolve()

        raise FileNotFoundError(
            f"Script not found. Tried:\n"
            f"  - {primary_path}\n"
            + "\n".join(f"  - {fb}" for fb in fallbacks)
        )

    def _get_project_root(self, script_path: Path) -> Path:
        """Get project root for uv commands.

        For mcp_convert servers, returns mcp_convert directory.
        Otherwise returns gem root.

        Args:
            script_path: Path to the server script

        Returns:
            Project root path
        """
        if "mcp_convert" in script_path.parts:
            # Find mcp_convert directory
            for i, part in enumerate(script_path.parts):
                if part == "mcp_convert":
                    return Path(*script_path.parts[: i + 1])

        return self._get_gem_root()

    def _get_gem_root(self) -> Path:
        """Get GEM project root directory.

        Returns:
            Path to gem root (4 levels up from this file)
        """
        return Path(__file__).parent.parent.parent.parent

    def _replace_placeholders(self, value: str, params: Dict[str, Any]) -> str:
        """Replace placeholders like {task_workspace} in values.

        Args:
            value: String value potentially containing placeholders
            params: Runtime parameters

        Returns:
            String with placeholders replaced
        """
        if not isinstance(value, str):
            return value

        # Replace common placeholders plus any {param_name} whose value is a
        # plain string in params (e.g. {workspace_path} used in YAML defaults)
        replacements = {
            "{task_workspace}": params.get("task_workspace", ""),
            "{agent_workspace}": params.get("agent_workspace", ""),
        }
        for key, param_value in params.items():
            if isinstance(param_value, str) and "{" not in param_value:
                replacements.setdefault(f"{{{key}}}", param_value)

        for placeholder, replacement in replacements.items():
            if placeholder in value:
                value = value.replace(placeholder, str(replacement))

        return value


# Global loader instance
_loader = ServerConfigLoader()


def build_server_config(
    server_type: str,
    params: Dict[str, Any],
    server_name: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build stdio configuration for a server (convenience function).

    Args:
        server_type: Type of server (e.g., 'canvas', 'claim_done')
        params: Runtime parameters from JSON config
        server_name: Override server name (defaults to config name)

    Returns:
        Configuration dict in format: {server_name: {"command": ..., "args": [...], "env": {...}}}

    Raises:
        FileNotFoundError: If config doesn't exist
        ValueError: If config is invalid or required params missing

    Example:
        >>> config = build_server_config("canvas", {"data_dir": "/path/to/data"})
        >>> print(config)
        {"canvas": {"command": "python", "args": [...], "env": {...}}}
    """
    return _loader.build_stdio_config(server_type, params, server_name)
