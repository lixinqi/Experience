"""
call_llm_until_success — Async retry loop: call LLM, validate output, retry
on failure with an improved prompt until the output passes validation.

    call_llm_until_success :=
      $data DataResult
      <- $llm_fn (Awaitable[str] <- $args LLmArgs)
      <- $validate_and_transform (($data DataResult | $err list[str]) <- str)
      <- $construct_retry_prompt (Awaitable[LLmArgs] <- $err list[str] <- $args LLmArgs)
"""

from __future__ import annotations
from typing import Awaitable, Callable, List, TypeVar, Union

LLmArgs = TypeVar("LLmArgs")
DataResult = TypeVar("DataResult")


class ValidationOk:
    """Wrapper for a successful validation result."""
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"ValidationOk({self.data!r})"


class ValidationErr:
    """Wrapper for validation errors that should trigger a retry."""
    def __init__(self, errors: List[str]):
        self.errors = errors

    def __repr__(self):
        return f"ValidationErr({self.errors!r})"


async def call_llm_until_success(
    llm_fn: Callable[[LLmArgs], Awaitable[str]],
    validate_and_transform: Callable[[str], Union[ValidationOk, ValidationErr]],
    construct_retry_prompt: Callable[[List[str], LLmArgs], Awaitable[LLmArgs]],
    initial_args: LLmArgs,
    *,
    max_retries: int = 8,
):
    """Call LLM, validate output, retry with improved prompt on failure.

    Args:
        llm_fn: Async — calls the LLM with the given args, returns raw string.
        validate_and_transform: Sync — takes raw LLM output, returns either
            ValidationOk(data) on success or ValidationErr(errors) on failure.
        construct_retry_prompt: Async — takes (errors, current_args), returns
            new args for the next LLM call.
        initial_args: Args for the first LLM call.
        max_retries: Maximum number of retries before raising RuntimeError.

    Returns:
        The DataResult from the first successful validation.

    Raises:
        RuntimeError: If max_retries is exceeded without success.
    """
    args = initial_args

    for attempt in range(max_retries + 1):
        raw = await llm_fn(args)
        result = validate_and_transform(raw)

        if isinstance(result, ValidationOk):
            return result.data

        # Validation failed — build retry prompt if we have attempts left.
        if attempt < max_retries:
            args = await construct_retry_prompt(result.errors, args)

    raise RuntimeError(
        f"call_llm_until_success: exceeded {max_retries} retries. "
        f"Last errors: {result.errors}"
    )


# ── tests ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import asyncio

    if "--test" in sys.argv or "-t" in sys.argv:
        passed = 0
        failed = 0

        def ok(condition, label):
            global passed, failed
            if condition:
                passed += 1
            else:
                print(f"  FAIL {label}")
                failed += 1

        async def _run():
            global passed, failed

            # ── success on first attempt ────────────────────────────

            calls = []
            async def llm_ok(args):
                calls.append(args)
                return "valid output"

            def validate_ok(raw):
                if raw == "valid output":
                    return ValidationOk(raw.upper())
                return ValidationErr(["bad"])

            async def retry_prompt(errors, args):
                return f"retry:{args}"

            result = await call_llm_until_success(
                llm_fn=llm_ok,
                validate_and_transform=validate_ok,
                construct_retry_prompt=retry_prompt,
                initial_args="prompt1",
            )
            ok(result == "VALID OUTPUT", "success on first attempt")
            ok(len(calls) == 1, "one LLM call")
            ok(calls[0] == "prompt1", "correct initial args")

            # ── success after retry ─────────────────────────────────

            attempts = []
            async def llm_retry(args):
                attempts.append(args)
                if len(attempts) >= 3:
                    return "finally correct"
                return f"bad attempt {len(attempts)}"

            def validate_retry(raw):
                if raw == "finally correct":
                    return ValidationOk("done")
                return ValidationErr([f"invalid: {raw}"])

            async def retry_prompt_2(errors, args):
                return f"{args} | fixed"

            result = await call_llm_until_success(
                llm_fn=llm_retry,
                validate_and_transform=validate_retry,
                construct_retry_prompt=retry_prompt_2,
                initial_args="start",
            )
            ok(result == "done", "success after retries")
            ok(len(attempts) == 3, "three LLM calls")

            # ── max retries exceeded ────────────────────────────────

            def always_fail(raw):
                return ValidationErr(["always wrong"])

            async def noop_retry(errors, args):
                return args

            try:
                await call_llm_until_success(
                    llm_fn=lambda a: _bad(),
                    validate_and_transform=always_fail,
                    construct_retry_prompt=noop_retry,
                    initial_args="x",
                    max_retries=2,
                )
                ok(False, "should have raised RuntimeError")
            except RuntimeError:
                ok(True, "max retries raises RuntimeError")

            # ── zero retries — tries once ───────────────────────────

            calls0 = []
            async def bad_once(args):
                calls0.append(args)
                return "x"

            try:
                await call_llm_until_success(
                    llm_fn=bad_once,
                    validate_and_transform=always_fail,
                    construct_retry_prompt=noop_retry,
                    initial_args="x",
                    max_retries=0,
                )
                ok(False, "zero retries should raise with always_fail")
            except RuntimeError:
                ok(len(calls0) == 1, "zero retries still calls once")

        async def _bad():
            return "bad"

        asyncio.run(_run())

        total = passed + failed
        print(f"\n  call_llm_until_success: {passed}/{total} passed"
              f"{' ✓' if failed == 0 else ''}")
        sys.exit(0 if failed == 0 else 1)

    else:
        print("Usage: python3 call_llm_until_success.py --test")
        sys.exit(1)
