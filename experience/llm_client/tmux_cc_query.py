"""Tmux CC (Claude Coding) query interface.

This module provides a clean interface for querying tmux_cc (Claude in interactive mode),
completely decoupled from implementation details through TmuxCcConfig.
"""

import os
import subprocess
import sys
from typing import Optional
from experience.llm_client.agent_config import TmuxCcConfig
from experience.llm_client.agent_config_factory import AgentConfigFactory


def tmux_cc_query(
    prompt: str,
    cwd: Optional[str] = None,
    config: Optional[TmuxCcConfig] = None,
    permission_mode: str = "bypassPermissions",
    allowed_tools: str = "Read,Edit,Write",
    effort: str = "low",
) -> str:
    """Query tmux_cc (Claude) with the given prompt in non-interactive mode.
    
    This is a simple synchronous interface for single tmux_cc queries,
    parallel to raw_llm_query() and coding_agent_query().
    
    Args:
        prompt: The task prompt for tmux_cc.
        cwd: Working directory for the task.
        config: Optional TmuxCcConfig. If None, creates default config.
        permission_mode: Permission mode (default: "bypassPermissions").
        allowed_tools: Comma-separated list of allowed tools.
        effort: Effort level (low/medium/high).
    
    Returns:
        The tmux_cc output as a string.
    
    Raises:
        RuntimeError: If tmux_cc execution fails.
    """
    # Create config if not provided
    if config is None:
        config = AgentConfigFactory.create_tmux_cc_config()
    
    cli_path = config.get_cli_path()
    if cli_path is None:
        raise RuntimeError("Cannot find tmux_cc (Claude) binary. Please configure properly.")
    
    # Build command
    cmd = [
        cli_path,
        "-p", prompt,
        "--allowedTools", allowed_tools,
        "--permission-mode", permission_mode,
        "--effort", effort,
    ]
    
    # Add model if configured
    if config.model:
        cmd.extend(["--model", config.model])
    
    # Prepare environment
    env = os.environ.copy()
    
    # Configure for external API (Qianfan) if base_url is set
    if config.base_url:
        is_internal = "comate" in config.base_url or "baidu-int" in config.base_url
        
        if not is_internal and config.api_key:
            # External API: set env vars and disable settings
            env["ANTHROPIC_BASE_URL"] = config.base_url
            env["ANTHROPIC_API_KEY"] = config.api_key
            env.pop("ANTHROPIC_AUTH_TOKEN", None)
            env.pop("ANTHROPIC_CUSTOM_HEADERS", None)
            cmd.extend(["--setting-sources", "", "--settings", "{}"])
    
    # Execute tmux_cc
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            env=env,
        )
        
        if result.returncode != 0:
            error_msg = f"tmux_cc failed with code {result.returncode}\nstderr: {result.stderr}"
            print(error_msg, file=sys.stderr)
            raise RuntimeError(error_msg)
        
        return result.stdout
        
    except Exception as e:
        raise RuntimeError(f"Failed to execute tmux_cc: {e}") from e


if __name__ == "__main__":
    # Example usage
    result = tmux_cc_query(
        prompt="Hello, print 'Hello World'",
        cwd="/tmp"
    )
    print(result)
