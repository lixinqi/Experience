"""
switch_forward :=
    FutureTensor
    <- $condition FutureTensor
    <- $cases list[($symbol str, $summary str, $description str, $branch FutureTensor)]
    # inline

Pure lazy switch: control-flow evaluation is deferred to the returned
ft_async_get.  No condition is evaluated inside switch_forward itself.
"""

from typing import List, Tuple

from experience.future_tensor.future_tensor import FutureTensor


def switch_forward(
    condition: FutureTensor,
    cases: List[Tuple[str, str, str, FutureTensor]],
) -> FutureTensor:
    """Lazy forward pass of switch.

    Returns a **lazy** FutureTensor whose ``ft_async_get`` will, at pull time:
      1. Pull the condition to obtain its control symbol.
      2. Match the symbol against ``cases``.
      3. Delegate to the selected branch's ``ft_async_get``.

    Args:
        condition: FutureTensor whose symbolic content determines the branch.
        cases: List of (symbol, summary, description, branch) tuples.

    Returns:
        A lazy FutureTensor.
    """
    if not cases:
        raise ValueError("switch_forward: cases list is empty")

    symbols = [case[0] for case in cases]
    summaries = [case[1] for case in cases]
    descriptions = [case[2] for case in cases]
    branches = [case[3] for case in cases]

    # Validate uniform shape across all branches
    expected_shape = branches[0].ft_capacity_shape
    for i, branch in enumerate(branches):
        if branch.ft_capacity_shape != expected_shape:
            raise ValueError(
                f"switch_forward: branch {i} has shape {branch.ft_capacity_shape}, "
                f"expected {expected_shape}"
            )

    async def switch_async_get(coordinates: List[int], trajactory: str):
        # Pull the condition at coordinates (broadcast to condition's shape).
        cond_shape = condition.ft_capacity_shape or []
        if len(cond_shape) == len(coordinates):
            cond_coords = coordinates
        elif len(cond_shape) < len(coordinates):
            # Condition is lower-rank: use trailing coordinates
            cond_coords = coordinates[len(coordinates) - len(cond_shape):]
        else:
            # Condition higher-rank: pad with zeros
            cond_coords = [0] * (len(cond_shape) - len(coordinates)) + list(coordinates)
        condition_result = await condition.ft_async_get(
            cond_coords,
            {
                "trajactory": trajactory,
                "symbols": symbols,
                "summaries": summaries,
                "descriptions": descriptions,
            },
        )
        condition_symbol = condition_result[0].strip() if condition_result[0] else ""

        selected_branch = branches[0]
        for i, sym in enumerate(symbols):
            if condition_symbol == sym or condition_symbol.startswith(sym):
                selected_branch = branches[i]
                break

        return await selected_branch.ft_async_get(coordinates, trajactory)

    result = FutureTensor(
        branches[0].ft_initial_static_tensor.st_relative_to,
        switch_async_get,
        branches[0].ft_shape_schema,
    )
    result.ft_capacity_shape = list(expected_shape)
    # ft_forwarded stays False — purely lazy

    return result
