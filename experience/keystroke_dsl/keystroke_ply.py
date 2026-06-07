"""
keystroke_ply.py — PLY (Python Lex-Yacc) parser for the keystroke macro DSL.

Grammar (from keystroke-grammar-draft.txt):

    program        := statement_list

    statement_list := ε
                    | statement_list "\\n+" statement

    statement      := COMMENT
                    | K_CTRL    BLANK ctrl_constant
                    | K_TEXT    BLANK text_string
                    | K_VAR     BLANK var_symbol
                    | K_EXPAND  BLANK var_symbol_list
                    | K_MACRO   var_symbol_list K_BEGIN statement_list K_END

    BLANK = one or more spaces (enforced by key token regex, not a token)

Usage:
    from experience.keystroke_dsl.keystroke_ply import parse_keystroke
    ast = parse_keystroke(source_string)

Or standalone:
    python3 keystroke_ply.py --parse <file.ks>
"""

import sys
from typing import List, Optional

# ── PLY preamble ────────────────────────────────────────────────────────

class _LexError(Exception):
    pass

class _ParseError(Exception):
    pass

# Ctrl-key constants (case-sensitive).
_CTRL_KEYS = frozenset({
    "Enter", "Escape", "Return", "Tab", "Space",
    "Backspace", "Delete", "Insert",
    "Up", "Down", "Left", "Right",
    "Home", "End", "PageUp", "PageDown",
    "F1", "F2", "F3", "F4", "F5", "F6",
    "F7", "F8", "F9", "F10", "F11", "F12",
})

# ── Lexer ───────────────────────────────────────────────────────────────

states = (
    ("text", "exclusive"),   # after K_TEXT — everything until newline is text
)

tokens = (
    "K_CTRL", "K_TEXT", "K_VAR", "K_EXPAND", "K_MACRO",
    "K_BEGIN", "K_END",
    "CTRL_CONSTANT", "TEXT_STRING", "VAR_SYMBOL",
    "COMMENT", "NEWLINE",
)

# ── INITIAL state ────────────────────────────────────────────────────

# Newline — one or more.
def t_NEWLINE(t):
    r"\n+"
    t.lexer.lineno += len(t.value)
    return t

# Comments: # to end of line (excludes the newline).
def t_COMMENT(t):
    r"\#[^\n]*"
    return t

# Keywords (hyphenated — must precede t_VAR_SYMBOL).
def t_K_CTRL(t):    r"tmux-ctrl";    return t
def t_K_TEXT(t):
    r"tmux-text"
    t.lexer.begin("text")     # switch to text state for the argument
    return t
def t_K_VAR(t):     r"tmux-var";     return t
def t_K_EXPAND(t):  r"tmux-expand";  return t
def t_K_MACRO(t):   r"tmux-macro";   return t
def t_K_BEGIN(t):   r"begin";        return t
def t_K_END(t):     r"end";          return t

# Ctrl-key constant: C-x and M-x patterns.  Must appear before
# t_VAR_SYMBOL so the more-specific [CM]-[a-zA-Z] takes priority.
def t_CTRL_CONSTANT(t):
    r"[CM]-[a-zA-Z]"
    return t

# Identifier.  Keywords / ctrl-keys are matched above; anything
# reaching here is a plain variable symbol or a known ctrl-key name.
def t_VAR_SYMBOL(t):
    r"[a-zA-Z_][0-9a-zA-Z_]*"
    if t.value in _CTRL_KEYS:
        t.type = "CTRL_CONSTANT"
    return t

# Whitespace — skipped.
t_ignore = " \t"

def t_error(t):
    raise _LexError(f"Illegal character {t.value[0]!r} at line {t.lexer.lineno}")

# ── text state (after K_TEXT) ────────────────────────────────────────

# In the text state, everything until newline is the text argument.
# Leading whitespace is skipped, then the rest of the line (quoted
# or bare) is captured as TEXT_STRING.

t_text_ignore = ""     # no ignore — spaces are part of the text

def t_text_NEWLINE(t):
    r"\n+"
    t.lexer.lineno += len(t.value)
    t.lexer.begin("INITIAL")
    t.type = "TEXT_STRING"
    t.value = ""            # empty text argument
    return t

def t_text_TEXT_STRING(t):
    r'[^\n]+'
    t.lexer.begin("INITIAL")
    v = t.value
    # Strip exactly one leading space (the BLANK separator between
    # keyword and argument).  Any additional spaces are part of the
    # text and are preserved verbatim.
    if v.startswith(" "):
        v = v[1:]
    t.value = v
    return t

def t_text_error(t):
    raise _LexError(f"Illegal character in text: {t.value[0]!r} at line {t.lexer.lineno}")

# ── Parser ──────────────────────────────────────────────────────────────

# Precedence (lowest to highest).
precedence = ()

def p_program(p):
    """program : statement_list opt_nl"""
    p[0] = {"type": "program", "statements": p[1]}

def p_statement_list_one(p):
    """statement_list : statement"""
    p[0] = [p[1]]

def p_statement_list_many(p):
    """statement_list : statement_list NEWLINE statement"""
    p[0] = p[1] + [p[3]]

def p_statement_comment(p):
    """statement : COMMENT"""
    p[0] = {"type": "comment", "text": p[1]}

def p_statement_action(p):
    """statement : action"""
    p[0] = p[1]

def p_statement_macro(p):
    """statement : macro_def"""
    p[0] = p[1]

def p_action_ctrl(p):
    """action : K_CTRL CTRL_CONSTANT"""
    p[0] = {"type": "ctrl", "key": p[2]}

def p_action_text(p):
    """action : K_TEXT TEXT_STRING"""
    p[0] = {"type": "text", "value": p[2]}

def p_action_var(p):
    """action : K_VAR VAR_SYMBOL"""
    p[0] = {"type": "var", "name": p[2]}

def p_action_expand(p):
    """action : K_EXPAND var_symbol_list"""
    p[0] = {"type": "expand", "vars": p[2]}

def p_macro_def(p):
    """macro_def : K_MACRO var_symbol_list NEWLINE K_BEGIN opt_nl statement_list opt_nl K_END"""
    p[0] = {"type": "macro", "params": p[2], "body": p[6]}

def p_opt_nl(p):
    """opt_nl : NEWLINE
              | """
    pass

def p_var_symbol_list_one(p):
    """var_symbol_list : VAR_SYMBOL"""
    p[0] = [p[1]]

def p_var_symbol_list_many(p):
    """var_symbol_list : var_symbol_list VAR_SYMBOL"""
    p[0] = p[1] + [p[2]]

def p_error(p):
    if p:
        raise _ParseError(f"Syntax error at token {p.type!r} (line {p.lineno})")
    raise _ParseError("Syntax error at EOF")

# ── Build ───────────────────────────────────────────────────────────────

import ply.lex as _lex
import ply.yacc as _yacc

_lexer: Optional[_lex.Lexer] = None
_parser: Optional[_yacc.LRParser] = None

def _build():
    global _lexer, _parser
    if _lexer is None:
        _lexer = _lex.lex(module=sys.modules[__name__], debug=False)
    if _parser is None:
        _parser = _yacc.yacc(module=sys.modules[__name__], debug=False, write_tables=False)

# ── Public API ──────────────────────────────────────────────────────────

def parse_keystroke(source: str) -> dict:
    """Parse a keystroke DSL source string, return an AST dict.

    Raises _LexError or _ParseError on failure.
    Returns an empty program for empty or whitespace-only input.
    """
    if not source or not source.strip():
        return {"type": "program", "statements": []}
    _build()
    return _parser.parse(source, lexer=_lexer)

# ── Standalone ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Keystroke DSL parser (PLY)")
    ap.add_argument("file", nargs="?", help="Source file to parse")
    ap.add_argument("--parse", dest="do_parse", action="store_true",
                    default=True, help="Parse input (default)")
    ap.add_argument("--tokens", dest="show_tokens", action="store_true",
                    help="Show token stream instead of parsing")
    args = ap.parse_args()

    if args.file:
        with open(args.file) as f:
            source = f.read()
    else:
        source = sys.stdin.read()

    if not source.strip():
        print("(empty input)")
        sys.exit(0)

    if args.show_tokens:
        _build()
        _lexer.input(source)
        for tok in _lexer:
            print(f"  {tok.type:20s} {tok.value!r}")
    else:
        ast = parse_keystroke(source)
        import json
        print(json.dumps(ast, indent=2))
