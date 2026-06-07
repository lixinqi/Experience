"""
gen_keystrokes_until_success — Async: call LLM to generate a keystroke
program, parse + expand it, retry on parse failure with an improved prompt.

    gen_keystrokes_until_success :=
      $data list[str]
      <- $llm_fn (Awaitable[str] <- $args LLmArgs)
      <- $construct_retry_prompt (Awaitable[LLmArgs] <- $err list[str] <- $args LLmArgs)
"""

from __future__ import annotations
from typing import Awaitable, Callable, List, TypeVar

from experience.high_order_fn.call_llm_until_success import (
    call_llm_until_success,
    ValidationOk,
    ValidationErr,
)
from experience.keystroke_dsl.parse_and_expand import (
    parse_and_expand,
    ParseOk,
    ParseErr,
)

LLmArgs = TypeVar("LLmArgs")


async def gen_keystrokes_until_success(
    llm_fn: Callable[[LLmArgs], Awaitable[str]],
    construct_retry_prompt: Callable[[List[str], LLmArgs], Awaitable[LLmArgs]],
    initial_args: LLmArgs,
    *,
    max_retries: int = 8,
) -> List[str]:
    """Generate keystroke commands via LLM, parse, retry on failure.

    Calls llm_fn(args) to get a raw keystroke DSL program, parses and
    expands it.  If the program parses successfully the flat command list
    is returned.  On parse error the error list is fed to
    construct_retry_prompt and the LLM is called again.

    Args:
        llm_fn: Async — calls the LLM, returns a raw keystroke DSL program.
        construct_retry_prompt: Async — takes (errors, current_args), returns
            new args for the next LLM call.
        initial_args: Args for the first LLM call.
        max_retries: Maximum retries before raising RuntimeError.

    Returns:
        Flat list of "tmux-ctrl ..." / "tmux-text ..." command strings.

    Raises:
        RuntimeError: If max_retries is exceeded.
    """

    def _validate_and_transform(raw: str):
        result = parse_and_expand(raw)
        if isinstance(result, ParseOk):
            return ValidationOk(result.ok)
        return ValidationErr(result.err)

    return await call_llm_until_success(
        llm_fn=llm_fn,
        validate_and_transform=_validate_and_transform,
        construct_retry_prompt=construct_retry_prompt,
        initial_args=initial_args,
        max_retries=max_retries,
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

            # ── valid program on first attempt ──────────────────────

            async def llm_valid(args):
                return "tmux-text echo hello\ntmux-ctrl Enter"

            async def retry(errors, args):
                return f"retry:{args}"

            result = await gen_keystrokes_until_success(
                llm_fn=llm_valid,
                construct_retry_prompt=retry,
                initial_args="make a bash echo",
            )
            ok(result == ["tmux-text echo hello", "tmux-ctrl Enter"],
               "valid program first attempt")

            # ── invalid then valid ──────────────────────────────────

            calls = []
            async def llm_retry(args):
                calls.append(args)
                if len(calls) == 1:
                    return "tmux-bad Enter"
                return "tmux-text echo ok\ntmux-ctrl Enter"

            async def retry2(errors, args):
                return f"{args} | fix parse errors: {'; '.join(errors)}"

            result = await gen_keystrokes_until_success(
                llm_fn=llm_retry,
                construct_retry_prompt=retry2,
                initial_args="echo ok",
            )
            ok(result == ["tmux-text echo ok", "tmux-ctrl Enter"],
               "retry after parse error")
            ok(len(calls) == 2, "two LLM calls")

            # ── max retries exceeded ────────────────────────────────

            async def retry_noop(errors, args):
                return args

            try:
                await gen_keystrokes_until_success(
                    llm_fn=lambda a: _bad(),
                    construct_retry_prompt=retry_noop,
                    initial_args="x",
                    max_retries=1,
                )
                ok(False, "should have raised RuntimeError")
            except RuntimeError:
                ok(True, "max retries raises RuntimeError")

            # ── macro stored, not emitted ───────────────────────────

            result = await gen_keystrokes_until_success(
                llm_fn=lambda a: _macro_src(),
                construct_retry_prompt=retry,
                initial_args="x",
            )
            ok(result == ["tmux-text outer"], "macro stored not emitted")

        async def _bad():
            return "garbage not valid"

        async def _macro_src():
            return "tmux-macro m\nbegin\n    tmux-text inner\nend\ntmux-text outer"

        asyncio.run(_run())

        total = passed + failed
        print(f"\n  gen_keystrokes_until_success: {passed}/{total} passed"
              f"{' ✓' if failed == 0 else ''}")
        sys.exit(0 if failed == 0 else 1)

    else:
        print("Usage: python3 gen_keystrokes_until_success.py --test")
        sys.exit(1)
