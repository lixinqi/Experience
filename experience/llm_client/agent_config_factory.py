"""Agent configuration factory.

Factory for creating configurations for all LLM methods (raw_llm_api, coding_agent, tmux_cc).
This completely decouples the query/handler layer from implementation details.
"""

import os
from typing import Optional
from experience.llm_client.agent_config import (
    AgentConfig,
    RawLlmConfig,
    ClaudeAgentConfig,
    TmuxCcConfig,
)
from experience.llm_client.config import get_config
from experience.llm_client.agent_binary_provider import AgentBinaryProvider, get_default_provider

# Backward compatibility
from experience.llm_client.agent_binary_provider import ClaudeProvider


class AgentConfigFactory:
    """Factory for creating agent configurations.
    
    This is the only place that knows about Claude, settings.json, environment variables,
    and other implementation-specific details.
    """
    
    @staticmethod
    def create_raw_llm_config(
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        username: Optional[str] = None,
    ) -> RawLlmConfig:
        """Create configuration for raw_llm_api method.
        
        Args:
            base_url: API base URL (e.g., https://qianfan.baidubce.com/v2).
            api_key: API key.
            model: Model name.
            username: Username for custom headers.
        
        Returns:
            RawLlmConfig instance.
        """
        # Load from ~/.experience.json if not provided
        config = get_config("raw_llm_api")
        
        return RawLlmConfig(
            base_url=base_url or config.get("base_url") or os.environ.get("LLM_BASE_URL"),
            api_key=api_key or config.get("api_key") or os.environ.get("LLM_API_KEY"),
            model=model or config.get("model") or os.environ.get("LLM_MODEL"),
            username=username or config.get("username") or os.environ.get("LLM_USERNAME"),
        )
    
    @staticmethod
    def create_claude_agent_config(
        cwd: Optional[str] = None,
        allowed_tools: Optional[list[str]] = None,
        permission_mode: str = "acceptEdits",
        model: Optional[str] = None,
        agent_provider: Optional[AgentBinaryProvider] = None,
        claude_provider: Optional["ClaudeProvider"] = None,  # Backward compatibility
    ) -> ClaudeAgentConfig:
        """Create configuration for coding_agent method (agent programmatic mode).
        
        Args:
            cwd: Working directory for the agent.
            allowed_tools: List of allowed tools.
            permission_mode: Permission mode.
            model: Model name override.
            agent_provider: Optional agent binary provider (for custom installations).
            claude_provider: Deprecated. Use agent_provider instead. Kept for backward compatibility.
        
        Returns:
            ClaudeAgentConfig instance.
        """
        # Handle backward compatibility: claude_provider -> agent_provider
        if claude_provider is not None and agent_provider is None:
            agent_provider = claude_provider
        
        # Use provided provider or default
        if agent_provider is None:
            agent_provider = get_default_provider()
        
        # Get agent CLI path from provider
        cli_path = agent_provider.get_cli_path()
        
        # Check if using external API (Qianfan) or internal (Comate)
        config = get_config("coding_agent")
        if config.get("base_url"):
            # External API: disable default settings, use env vars
            settings_path = "{}"
            setting_sources = ""
        else:
            # Internal API: use agent settings
            settings_path = agent_provider.get_settings_path()
            setting_sources = None
        
        # Get model from parameter or environment
        if model is None:
            model = os.environ.get("ANTHROPIC_MODEL")
        
        # Set default tools if not specified
        if allowed_tools is None:
            allowed_tools = ["Read", "Edit"]
        
        return ClaudeAgentConfig(
            cli_path=cli_path,
            settings_path=settings_path,
            setting_sources=setting_sources,
            model=model,
            cwd=cwd,
            allowed_tools=allowed_tools,
            permission_mode=permission_mode,
        )
    
    @staticmethod
    def create_tmux_cc_config(
        model: Optional[str] = None,
        agent_provider: Optional[AgentBinaryProvider] = None,
        claude_provider: Optional["ClaudeProvider"] = None,  # Backward compatibility
    ) -> TmuxCcConfig:
        """Create configuration for tmux_cc method (agent interactive/batch mode).
        
        Args:
            model: Model name override.
            agent_provider: Optional agent binary provider (for custom installations).
            claude_provider: Deprecated. Use agent_provider instead. Kept for backward compatibility.
        
        Returns:
            TmuxCcConfig instance.
        """
        # Handle backward compatibility: claude_provider -> agent_provider
        if claude_provider is not None and agent_provider is None:
            agent_provider = claude_provider
        
        # Use provided provider or default
        if agent_provider is None:
            agent_provider = get_default_provider()
        
        # Get agent paths from provider
        cli_path = agent_provider.get_cli_path()
        settings_path = agent_provider.get_settings_path()
        
        # Get configuration from ~/.experience.json
        config = get_config("tmux_cc")
        base_url = config.get("base_url") or os.environ.get("ANTHROPIC_BASE_URL")
        api_key = config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        
        # Get model
        if model is None:
            model = config.get("model") or os.environ.get("ANTHROPIC_MODEL")
        
        # Get workspace root
        workspace_root = os.environ.get("TMUX_CC_WORKSPACE_ROOT") or os.path.expanduser("~/.tmux_cc_tmp")
        
        return TmuxCcConfig(
            cli_path=cli_path,
            settings_path=settings_path,
            model=model,
            base_url=base_url,
            api_key=api_key,
            workspace_root=workspace_root,
        )
    
    @staticmethod
    def create_config_by_method(
        method: str,
        **kwargs
    ) -> AgentConfig:
        """Create configuration by method name.
        
        Args:
            method: LLM method name ("raw_llm_api", "coding_agent", "tmux_cc").
            **kwargs: Method-specific parameters.
        
        Returns:
            AgentConfig instance for the specified method.
        
        Raises:
            ValueError: If method is not supported.
        """
        if method == "raw_llm_api":
            return AgentConfigFactory.create_raw_llm_config(**kwargs)
        elif method == "coding_agent":
            return AgentConfigFactory.create_claude_agent_config(**kwargs)
        elif method == "tmux_cc":
            return AgentConfigFactory.create_tmux_cc_config(**kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}. Must be one of: raw_llm_api, coding_agent, tmux_cc")


# Convenience functions for backward compatibility
def create_agent_config(
    cwd: Optional[str] = None,
    allowed_tools: Optional[list[str]] = None,
    permission_mode: str = "acceptEdits",
    model: Optional[str] = None,
    agent_provider: Optional[AgentBinaryProvider] = None,
    claude_provider: Optional["ClaudeProvider"] = None,  # Backward compatibility
) -> ClaudeAgentConfig:
    """Create coding_agent configuration with smart defaults.
    
    This is a convenience function that wraps the factory.
    For backward compatibility with coding_agent_query.
    
    Args:
        claude_provider: Deprecated. Use agent_provider instead.
    """
    return AgentConfigFactory.create_claude_agent_config(
        cwd=cwd,
        allowed_tools=allowed_tools,
        permission_mode=permission_mode,
        model=model,
        agent_provider=agent_provider,
        claude_provider=claude_provider,
    )
