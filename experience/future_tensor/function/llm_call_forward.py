"""
llm_call_forward :=
    FutureTensor
    <- $prompt_ft FutureTensor
    <- $llm_method str
    <- $llm_env dict | None
    # inline

Call an LLM with prompt text, return the response.
Decomposed from ft_expert_forward step 10 (LLM execution).
Uses raw_llm_query directly — no workspace/TaskHandler indirection.
"""

from typing import Dict, List, Optional, Tuple

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.symbolic_tensor.function.st_moe_forward import _read_file_content
from experience.symbolic_tensor.tensor_util.st_setitem import st_setitem


def llm_call_forward(
    prompt_ft: FutureTensor,
    llm_method: str = "raw_llm_api",
    llm_env: Optional[Dict[str, str]] = None,
) -> FutureTensor:
    """Forward: create a lazy FutureTensor whose ft_async_get calls the LLM.

    For each element:
      1. Pull prompt text from upstream FutureTensor.
      2. Call raw_llm_query with the prompt.
      3. Return the LLM response.

    Args:
        prompt_ft: FutureTensor containing prompt text per element.
        llm_method: LLM method name (for backward/dispatch key).
        llm_env: Optional environment variables for LLM config.

    Returns:
        FutureTensor whose elements are LLM response strings.
    """
    input_shape = prompt_ft.ft_capacity_shape
    input_schema = prompt_ft.ft_shape_schema
    relative_to = prompt_ft.ft_static_tensor.st_relative_to

    async def _async_get(
        coordinates: List[int], trajactory: str,
    ) -> Tuple[str, Status]:
        # If already computed, read from disk
        if output.ft_static_tensor.data[tuple(coordinates)].item() > 0:
            flat_idx = sum(
                c * s for c, s in zip(coordinates, output.ft_static_tensor.stride())
            )
            content = _read_file_content(output.ft_static_tensor, flat_idx)
            if content is not None:
                return (content, Status.confidence(1.0))

        if isinstance(trajactory, dict):
            actual_trajactory = trajactory.get("trajactory", "")
        else:
            actual_trajactory = trajactory

        # Ensure upstream prompt is materialized
        if (
            not prompt_ft.ft_forwarded
            and prompt_ft.ft_static_tensor.data[tuple(coordinates)].item() == 0
        ):
            prompt_text, prompt_status = await prompt_ft.ft_async_get(
                coordinates, actual_trajactory,
            )
            if prompt_text:
                st_setitem(
                    prompt_ft.ft_static_tensor, coordinates, prompt_text,
                    coefficient=Status.convert_status_to_float(prompt_status),
                )
        else:
            flat_idx = sum(
                c * s
                for c, s in zip(coordinates, prompt_ft.ft_static_tensor.stride())
            )
            prompt_text = _read_file_content(prompt_ft.ft_static_tensor, flat_idx)
            prompt_status = Status.confidence(1.0)

        if not prompt_text:
            return ("", Status.self_confidence_but_failed(0.5))

        if not prompt_status.is_confidence:
            return ("", prompt_status)

        # Call LLM in thread executor
        import asyncio
        loop = asyncio.get_event_loop()

        def _sync_llm_call() -> str:
            import asyncio as _aio
            from experience.llm_client.raw_llm_query import raw_llm_query
            from experience.llm_client.agent_config_factory import AgentConfigFactory

            config = AgentConfigFactory.create_raw_llm_config()
            if llm_env:
                if "base_url" in llm_env:
                    config = config._replace(base_url=llm_env["base_url"]) if hasattr(config, '_replace') else config
                if "api_key" in llm_env:
                    config = config._replace(api_key=llm_env["api_key"]) if hasattr(config, '_replace') else config
                if "model" in llm_env:
                    config = config._replace(model=llm_env["model"]) if hasattr(config, '_replace') else config

            return _aio.run(raw_llm_query(prompt_text, config))

        try:
            result_content = await loop.run_in_executor(None, _sync_llm_call)
        except Exception as e:
            import sys
            print(f"[llm_call_forward] LLM call failed: {e}", file=sys.stderr)
            return ("", Status.self_confidence_but_failed(0.5))

        if not result_content or not result_content.strip():
            return ("", Status.self_confidence_but_failed(0.5))

        result_content = result_content.strip()

        # Write-through
        st_setitem(
            output.ft_static_tensor, coordinates, result_content,
            coefficient=1.0,
        )

        return (result_content, Status.confidence(1.0))

    output = FutureTensor(
        relative_to,
        _async_get,
        list(input_schema),
    )

    return output
