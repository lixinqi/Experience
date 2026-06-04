"""Raw LLM API query interface.

This module provides a clean interface for querying OpenAI-compatible LLM APIs,
decoupled from configuration details through AgentConfig.
"""

import asyncio
import contextlib
import json
import os
import re
import sys
from typing import Optional
from openai import AsyncOpenAI, BadRequestError, InternalServerError, RateLimitError
from experience.llm_client.agent_config import RawLlmConfig
from experience.llm_client.agent_config_factory import AgentConfigFactory

_RATE_LIMIT_RETRIES = int(os.environ.get("LLM_RATE_LIMIT_RETRIES", "5"))
_RATE_LIMIT_BASE_DELAY = float(os.environ.get("LLM_RATE_LIMIT_BASE_DELAY", "10.0"))
_SERVER_ERROR_RETRIES = int(os.environ.get("LLM_SERVER_ERROR_RETRIES", "6"))
_SERVER_ERROR_BASE_DELAY = float(os.environ.get("LLM_SERVER_ERROR_BASE_DELAY", "10.0"))
_BAD_REQUEST_RETRIES = int(os.environ.get("LLM_BAD_REQUEST_RETRIES", "3"))
_BAD_REQUEST_BASE_DELAY = float(os.environ.get("LLM_BAD_REQUEST_BASE_DELAY", "15.0"))

# Global concurrency control (0 = unlimited).
# Per-event-loop semaphore: recreated lazily whenever the running loop changes,
# so it works correctly across multiple asyncio.run() calls.
_max_concurrent: int = 0
_current_semaphore: Optional[asyncio.Semaphore] = None
_current_loop_id: int = -1


def set_max_concurrent(n: int) -> None:
    """Set max concurrent LLM API requests globally. 0 = unlimited."""
    global _max_concurrent, _current_semaphore, _current_loop_id
    _max_concurrent = n
    _current_semaphore = None
    _current_loop_id = -1


def _strip_think_tags(content: str) -> str:
    """Remove <think>...</think> blocks from LLM output.

    Some models (e.g., DeepSeek, MiniMax) output thinking process in <think> tags.
    """
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content = re.sub(r'^<think>.*', '', content, flags=re.DOTALL)
    return content.strip()


async def _call_with_retry(client: AsyncOpenAI, model: str, messages: list, extra_body=None) -> str:
    """Call the LLM API with automatic retry on rate-limit, server, and bad-request errors."""
    rate_limit_attempts = 0
    server_error_attempts = 0
    bad_request_attempts = 0
    while True:
        try:
            kwargs = dict(model=model, messages=messages, stream=False,
                          temperature=0, top_p=1)
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            response = await client.chat.completions.create(**kwargs)
            if isinstance(response, str):
                return response
            return response.choices[0].message.content
        except RateLimitError:
            rate_limit_attempts += 1
            if rate_limit_attempts >= _RATE_LIMIT_RETRIES:
                raise
            delay = _RATE_LIMIT_BASE_DELAY * (2 ** (rate_limit_attempts - 1))
            print(f"[raw_llm_query] 429 rate limit, retry {rate_limit_attempts}/{_RATE_LIMIT_RETRIES-1} in {delay:.0f}s", file=sys.stderr)
            await asyncio.sleep(delay)
        except InternalServerError as e:
            server_error_attempts += 1
            if server_error_attempts >= _SERVER_ERROR_RETRIES:
                raise
            delay = _SERVER_ERROR_BASE_DELAY * (2 ** (server_error_attempts - 1))
            print(f"[raw_llm_query] 500 server error, retry {server_error_attempts}/{_SERVER_ERROR_RETRIES-1} in {delay:.0f}s: {e}", file=sys.stderr)
            await asyncio.sleep(delay)
        except BadRequestError as e:
            # Some API providers return 400 for transient network errors (e.g., code 1210).
            bad_request_attempts += 1
            if bad_request_attempts >= _BAD_REQUEST_RETRIES:
                raise
            delay = _BAD_REQUEST_BASE_DELAY * (2 ** (bad_request_attempts - 1))
            print(f"[raw_llm_query] 400 bad request, retry {bad_request_attempts}/{_BAD_REQUEST_RETRIES-1} in {delay:.0f}s: {e}", file=sys.stderr)
            await asyncio.sleep(delay)


async def raw_llm_query(
    prompt: str,
    config: Optional[RawLlmConfig] = None,
    extra_body: Optional[dict] = None,
):
    """Query OpenAI-compatible LLM API with the given prompt.

    This function is completely decoupled from configuration details.

    Args:
        prompt: The prompt to send to the LLM.
        config: Optional RawLlmConfig. If None, creates default config from environment/~/.experience.json.
        extra_body: Optional dict passed as extra_body to the API call (e.g. {'thinking': {'type': 'disabled'}}).

    Returns:
        The LLM response text.
    """
    # Create config if not provided
    if config is None:
        config = AgentConfigFactory.create_raw_llm_config()

    # Lazily create a semaphore for the current event loop (reset on loop change).
    # asyncio is single-threaded; this assignment happens before any `await`, so no race.
    global _current_semaphore, _current_loop_id
    if _max_concurrent > 0:
        loop = asyncio.get_running_loop()
        if id(loop) != _current_loop_id:
            _current_semaphore = asyncio.Semaphore(_max_concurrent)
            _current_loop_id = id(loop)
    _sem_ctx = _current_semaphore if _current_semaphore is not None else contextlib.AsyncExitStack()

    # Get custom headers for authentication
    custom_headers = {}
    if config.username:
        custom_header_value = json.dumps({
            "agentId": f"raw_llm_api:user:{config.username}",
            "username": config.username,
            "repo": "",
            "source": "raw_llm_api",
        })
        custom_headers["comate_custom_header"] = custom_header_value
        print(f"[DEBUG] Using username: {config.username}", file=sys.stderr)
    
    # Create OpenAI client
    async with _sem_ctx:
        client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            default_headers=custom_headers,
        )
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Output raw content only, no thinking process."},
                {"role": "user", "content": prompt},
            ]
            content = await _call_with_retry(client, config.model, messages, extra_body=extra_body)
            return _strip_think_tags(content)
        finally:
            await client.close()


if __name__ == "__main__":
    ret = asyncio.run(raw_llm_query("Hello"))
    print(ret)
