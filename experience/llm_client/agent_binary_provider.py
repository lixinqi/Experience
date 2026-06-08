"""Agent binary provider abstraction.

This module provides a clean abstraction for locating agent CLI binaries and settings,
supporting different implementations (Claude Code, future OpenAI agents, etc.).
"""

import glob
import os
import shutil
from abc import ABC, abstractmethod
from typing import Optional


class AgentBinaryProvider(ABC):
    """Abstract provider for agent CLI binary and settings paths.
    
    This abstraction allows different agent implementations (Claude Code, OpenAI, etc.)
    to provide their binary locations without coupling to specific tool names.
    """

    @abstractmethod
    def get_cli_path(self) -> Optional[str]:
        """Get the path to agent CLI binary.
        
        Returns:
            Path to agent binary, or None if not found.
        """
        pass

    @abstractmethod
    def get_settings_path(self) -> Optional[str]:
        """Get the path to agent settings file.
        
        Returns:
            Path to settings file, or None if not found.
        """
        pass


class ClaudeCodeProvider(AgentBinaryProvider):
    """Provider for Claude Code binary from Comate extension.
    
    Search order:
    1. TMUX_CC_BIN environment variable
    2. ~/.comate/extensions/baidu.baidu-cc-*/resources/native-binary/bin/claude (glob)
    3. claude in PATH (fallback)
    """

    def get_cli_path(self) -> Optional[str]:
        # 1. Check environment variable
        if env_bin := os.environ.get("TMUX_CC_BIN"):
            return env_bin

        # 2. Glob pattern for comate extension (preferred)
        pattern = os.path.expanduser(
            "~/.comate/extensions/baidu.baidu-cc-*/resources/native-binary/bin/claude"
        )
        matches = sorted(glob.glob(pattern), reverse=True)
        if matches:
            return matches[0]

        # 3. Check PATH (fallback)
        if path_bin := shutil.which("claude"):
            return path_bin

        return None

    def get_settings_path(self) -> Optional[str]:
        pattern = os.path.expanduser(
            "~/.comate/extensions/baidu.baidu-cc-*/resources/settings.json"
        )
        matches = sorted(glob.glob(pattern), reverse=True)
        return matches[0] if matches else None


class CustomAgentProvider(AgentBinaryProvider):
    """Provider with explicit paths for custom agent installations."""

    def __init__(self, cli_path: str, settings_path: Optional[str] = None):
        self._cli_path = cli_path
        self._settings_path = settings_path

    def get_cli_path(self) -> Optional[str]:
        return self._cli_path

    def get_settings_path(self) -> Optional[str]:
        return self._settings_path


class EnvironmentAgentProvider(AgentBinaryProvider):
    """Provider that reads agent paths from environment variables only."""

    def get_cli_path(self) -> Optional[str]:
        return os.environ.get("TMUX_CC_BIN") or shutil.which("claude")

    def get_settings_path(self) -> Optional[str]:
        return os.environ.get("AGENT_SETTINGS_PATH")


# Singleton default provider
_default_provider: Optional[AgentBinaryProvider] = None


def get_default_provider() -> AgentBinaryProvider:
    """Get the default agent provider (lazy initialization).
    
    Returns:
        Default provider instance (ClaudeCodeProvider).
    """
    global _default_provider
    if _default_provider is None:
        _default_provider = ClaudeCodeProvider()
    return _default_provider


def set_default_provider(provider: AgentBinaryProvider) -> None:
    """Set the default agent provider.
    
    Args:
        provider: The provider instance to use as default.
    """
    global _default_provider
    _default_provider = provider


# Backward compatibility aliases
ClaudeProvider = AgentBinaryProvider
ComateClaudeProvider = ClaudeCodeProvider
CustomClaudeProvider = CustomAgentProvider
EnvironmentClaudeProvider = EnvironmentAgentProvider
