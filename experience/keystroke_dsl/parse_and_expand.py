"""
parse_and_expand — Parse a keystroke DSL program, expand macros, return a
flat list of tmux-text / tmux-ctrl commands.

    parse_and_expand :=
      (Oneof
        | $ok list[str_starts_with_tmux_text_or_tmux_ctrl]
        | $err list[ErrorStrWithLinenoHint])
      <- $raw_keystroke_program str
"""

from __future__ import annotations
from typing import List, Union

from experience.keystroke_dsl.keystroke_ply import parse_keystroke


class ParseOk:
    """Successful parse + expand result."""
    def __init__(self, commands: List[str]):
        self.ok = commands

    def __repr__(self):
        return f"ParseOk({self.ok!r})"


class ParseErr:
    """Failed parse with error hints."""
    def __init__(self, errors: List[str]):
        self.err = errors

    def __repr__(self):
        return f"ParseErr({self.err!r})"


def parse_and_expand(raw_keystroke_program: str) -> Union[ParseOk, ParseErr]:
    """Parse a keystroke program, expand all macros, return flat command list.

    Only tmux-ctrl and tmux-text statements appear in the output.
    Comments are dropped.  Macro definitions are stored and expanded
    when their names appear in tmux-var / tmux-expand.

    Returns:
        ParseOk(ok=[...]) on success, ParseErr(err=[...]) on failure.
    """
    # 1. Parse
    try:
        ast = parse_keystroke(raw_keystroke_program)
    except Exception as e:
        return ParseErr([str(e)])

    statements = ast.get("statements", [])
    macros: dict[str, dict] = {}  # name → {"params": [...], "body": [...]}
    commands: List[str] = []
    errors: List[str] = []

    # 2. First pass — collect macro definitions and top-level commands.
    for stmt in statements:
        _process_stmt(stmt, macros, commands, errors, {})

    if errors:
        return ParseErr(errors)

    return ParseOk(commands)


# ── statement processing ────────────────────────────────────────────────


def _process_stmt(
    stmt: dict,
    macros: dict[str, dict],
    output: List[str],
    errors: List[str],
    bindings: dict[str, str],
) -> None:
    """Process a single statement, appending to output or errors."""
    typ = stmt.get("type")

    if typ == "comment":
        return  # drop

    if typ == "ctrl":
        output.append(f"tmux-ctrl {stmt['key']}")

    elif typ == "text":
        output.append(f"tmux-text {stmt['value']}")

    elif typ == "var":
        name = stmt["name"]
        if name in bindings:
            output.append(f"tmux-text {bindings[name]}")
        else:
            errors.append(f"undefined variable: {name}")

    elif typ == "expand":
        parts = []
        for name in stmt["vars"]:
            if name in bindings:
                parts.append(bindings[name])
            else:
                errors.append(f"undefined variable in expand: {name}")
        if parts:
            output.append(f"tmux-text {' '.join(parts)}")

    elif typ == "macro":
        # Macro definition — store, don't execute.
        name = stmt["params"][0] if stmt["params"] else None
        if name is None:
            errors.append("macro has no name")
        elif name in macros:
            errors.append(f"duplicate macro: {name}")
        else:
            macros[name] = {
                "params": stmt["params"][1:],  # rest are parameter names
                "body": stmt["body"],
            }

    else:
        errors.append(f"unknown statement type: {typ}")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv or "-t" in sys.argv:
        # ── test suite ────────────────────────────────────────────────
        passed = 0
        failed = 0

        def ok(result, expected, label):
            global passed, failed
            if isinstance(result, ParseErr):
                print(f"  FAIL {label}: unexpected ParseErr: {result.err}")
                failed += 1
            elif result.ok == expected:
                passed += 1
            else:
                print(f"  FAIL {label}: expected {expected}, got {result.ok}")
                failed += 1

        def err(result, *, contains="", label=""):
            global passed, failed
            if isinstance(result, ParseOk):
                print(f"  FAIL {label}: expected ParseErr, got ParseOk({result.ok})")
                failed += 1
            elif contains and not any(contains in e for e in result.err):
                print(f"  FAIL {label}: expected err containing {contains!r}, got {result.err}")
                failed += 1
            else:
                passed += 1

        # basic statements
        ok(parse_and_expand("tmux-text hello"),
           ["tmux-text hello"], "bare text")
        ok(parse_and_expand("tmux-text :wq"),
           ["tmux-text :wq"], "colon text")
        ok(parse_and_expand("tmux-ctrl Enter"),
           ["tmux-ctrl Enter"], "ctrl key")
        ok(parse_and_expand("tmux-ctrl C-c"),
           ["tmux-ctrl C-c"], "ctrl combo")
        ok(parse_and_expand('tmux-text "hello"'),
           ["tmux-text hello"], "quotes are delimiters (stripped)")

        # multi-statement
        ok(parse_and_expand("tmux-text echo hi\ntmux-ctrl Enter"),
           ["tmux-text echo hi", "tmux-ctrl Enter"], "multi-statement")

        # comments dropped
        ok(parse_and_expand("# comment\ntmux-text hello\n# another\ntmux-ctrl Escape"),
           ["tmux-text hello", "tmux-ctrl Escape"], "comments dropped")

        # macro stored, not emitted
        ok(parse_and_expand("tmux-macro greet\nbegin\n    tmux-text hello\nend\ntmux-text world"),
           ["tmux-text world"], "macro stored")

        # spaces preserved (after the single BLANK separator)
        ok(parse_and_expand("tmux-text  echo  hello"),
           ["tmux-text  echo  hello"], "spaces preserved")

        # invalid syntax
        err(parse_and_expand("tmux-bad Enter"), contains="Syntax error",
            label="invalid keyword")
        err(parse_and_expand("tmux-var x"), contains="undefined variable",
            label="undefined var")

        # empty / comment-only
        ok(parse_and_expand(""), [], "empty input")
        ok(parse_and_expand("# only a comment\n"), [], "only comment")

        # multiple macros
        ok(parse_and_expand("tmux-macro a\nbegin\ntmux-text one\nend\ntmux-macro b\nbegin\ntmux-text two\nend\ntmux-text three"),
           ["tmux-text three"], "multiple macros")

        # duplicate macro
        err(parse_and_expand("tmux-macro dup\nbegin\ntmux-text one\nend\ntmux-macro dup\nbegin\ntmux-text two\nend"),
            contains="duplicate macro", label="duplicate macro")

        # semicolon as statement separator
        ok(parse_and_expand("tmux-text a; tmux-text b"),
           ["tmux-text a", "tmux-text b"], "; separator")
        ok(parse_and_expand("tmux-text hello; tmux-ctrl Enter"),
           ["tmux-text hello", "tmux-ctrl Enter"], "mixed ; sep")
        ok(parse_and_expand("tmux-text a; tmux-text b; tmux-ctrl Enter"),
           ["tmux-text a", "tmux-text b", "tmux-ctrl Enter"], "triple ; sep")
        ok(parse_and_expand("tmux-text a;\ntmux-text b"),
           ["tmux-text a", "tmux-text b"], ";+newline sep")

        # semicolon inside quoted string is literal
        ok(parse_and_expand('tmux-text ";"'),
           ["tmux-text ;"], "quoted literal semicolon")
        ok(parse_and_expand('tmux-text "hello; world"'),
           ["tmux-text hello; world"], "quoted semicolon preserved")
        ok(parse_and_expand('tmux-text "echo ONE"; tmux-ctrl Enter'),
           ["tmux-text echo ONE", "tmux-ctrl Enter"], "quoted text + ctrl")

        # empty text before ;
        ok(parse_and_expand("tmux-text ; tmux-ctrl Enter"),
           ["tmux-text ", "tmux-ctrl Enter"], "empty text before ;")

        total = passed + failed
        print(f"\n  parse_and_expand: {passed}/{total} passed"
              f"{' ✓' if failed == 0 else ''}")
        sys.exit(0 if failed == 0 else 1)

    else:
        src = sys.stdin.read() if len(sys.argv) == 1 else open(sys.argv[1]).read()
        result = parse_and_expand(src)
        if isinstance(result, ParseErr):
            print("ERRORS:")
            for e in result.err:
                print(f"  {e}")
            sys.exit(1)
        else:
            for cmd in result.ok:
                print(cmd)
