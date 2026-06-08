"""Coding agent query interface.

This module provides a clean interface for querying coding agents,
completely decoupled from implementation details (Claude, settings, etc.).
"""

import asyncio
import os
import sys
from typing import Dict, Optional
from claude_agent_sdk import query, ClaudeAgentOptions
from experience.llm_client.agent_config import AgentConfig
from experience.llm_client.agent_config_factory import create_agent_config


async def coding_agent_query(
    prompt: str,
    cwd: str | None = None,
    allowed_tools: list[str] = None,
    permission_mode: str = "acceptEdits",
    llm_env: Optional[Dict[str, str]] = None,
    model: str = None,
    agent_config: Optional[AgentConfig] = None,
):
    """Query coding agent with the given prompt.
    
    This function is completely decoupled from implementation details.
    It only knows about the abstract AgentConfig interface.
    
    Args:
        prompt: The task prompt for the agent.
        cwd: Working directory for the agent.
        allowed_tools: List of allowed tools (default: ["Read", "Edit"]).
        permission_mode: Permission mode (default: "acceptEdits").
        llm_env: Optional environment variables to set.
        model: Model name override.
        agent_config: Optional pre-configured AgentConfig. If None, creates default config.
    
    Yields:
        Messages from the agent.
    """
    # Backup environment variables if needed
    env_backup = {}
    if llm_env is not None:
        for key, val in llm_env.items():
            env_backup[key] = os.environ.get(key)
            os.environ[key] = val

    try:
        # Create config if not provided
        if agent_config is None:
            agent_config = create_agent_config(
                cwd=cwd,
                allowed_tools=allowed_tools,
                permission_mode=permission_mode,
                model=model,
            )
        
        # Convert abstract config to concrete options
        options_dict = agent_config.to_options()
        
        # Create ClaudeAgentOptions from dict
        options = ClaudeAgentOptions(**options_dict)
        
        # Query agent
        async for item in query(
            prompt=prompt,
            options=options,
        ):
            yield item
            
    finally:
        # Restore environment variables
        if llm_env is not None:
            for key in llm_env:
                if env_backup.get(key) is not None:
                    os.environ[key] = env_backup[key]
                else:
                    os.environ.pop(key, None)


def async_run(handlers: list):
    """Helper to run async handlers."""
    async def func():
        await asyncio.gather(*handlers)
    asyncio.run(func())


if __name__ == '__main__':
    async def foo():
        async for message in coding_agent_query(
            prompt="hello, save your greeting response to a file named './foo.txt'",
            cwd='/tmp/demo',
        ):
            print(message)
    
    import os
    os.makedirs('/tmp/demo', exist_ok=True)
    async_run([foo()])
