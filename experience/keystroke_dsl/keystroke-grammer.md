# Keystroke DSL — Grammar & Semantics

A line-oriented domain-specific language for describing sequences of
keystrokes sent to a tmux pane.  Programs may contain literal text,
control-key commands, variable references, and macro definitions.
Statements are separated by one or more newlines.  The language is
parsed by `experience/keystroke_dsl/keystroke_ply.py` (a PLY-based
parser).

## Grammar (EBNF)

```
program         :=  statement_list

statement_list  :=  ε
                  | statement_list NEWLINE statement

statement       :=  COMMENT
                  | "tmux-ctrl"   BLANK ctrl_constant
                  | "tmux-text"   BLANK text_string
                  | "tmux-var"    BLANK var_symbol
                  | "tmux-expand" BLANK var_symbol_list
                  | "tmux-macro"  var_symbol_list NEWLINE
                    "begin" statement_list "end"

ctrl_constant   :=  [CM]-[a-zA-Z]
                  | Enter | Escape | Return | Tab | Space
                  | Backspace | Delete | Insert
                  | Up | Down | Left | Right
                  | Home | End | PageUp | PageDown
                  | F1 | F2 | F3 | F4 | F5 | F6
                  | F7 | F8 | F9 | F10 | F11 | F12

text_string     :=  [^\n]+                  (rest of line, verbatim)

var_symbol      :=  [a-zA-Z_][0-9a-zA-Z_]*

var_symbol_list :=  var_symbol (BLANK var_symbol)*

COMMENT         :=  "#" [^\n]*

BLANK           :=  [ ]+                    (one or more spaces — enforced implicitly)
NEWLINE         :=  "\n"+                   (one or more newlines)
```

### Notes on whitespace

**BLANK.**  The grammar requires at least one space between a keyword
and its argument (`tmux-ctrl BLANK ctrl_constant`, etc.).  The PLY
lexer enforces this implicitly: the keyword patterns match complete
tokens like `tmux-ctrl`, and the following space is consumed by the
whitespace-skip rule.  Without a space the lexer cannot separate
`tmux-ctrl` from its argument — they would run together as one
identifier — so the mandatory separation is guaranteed.

**Leading whitespace.**  Optional horizontal whitespace (`\s*`) is
allowed at the start of every line — the lexer skips spaces and tabs
before the first token.  This allows macro bodies and multi-statement
blocks to be indented for readability:

```
tmux-macro greet
begin
    tmux-text "hello"
    tmux-ctrl Enter
end
```

## Token Reference

### `ctrl_constant` — control-key names

| Pattern | Example | Semantics |
|---------|---------|-----------|
| `Enter` | `Enter` | Return / Enter key |
| `Escape` | `Escape` | Escape key |
| `Return` | `Return` | Alternate for Enter |
| `Tab` | `Tab` | Tab key |
| `Space` | `Space` | Space bar (as ctrl, not literal) |
| `Backspace` | `Backspace` | Backspace key |
| `Delete` | `Delete` | Delete key |
| `Insert` | `Insert` | Insert key |
| `Up` / `Down` / `Left` / `Right` | `Up` | Arrow keys |
| `Home` / `End` | `Home` | Line navigation |
| `PageUp` / `PageDown` | `PageUp` | Page navigation |
| `F1` – `F12` | `F5` | Function keys |
| `C-`*x* | `C-c` | Ctrl + letter |
| `M-`*x* | `M-x` | Meta / Alt + letter |

### `text_string` — literal text

Bare only.  Everything after the mandatory space until end of line is
taken verbatim as the text to send.  Whitespace within the text is
preserved.

```
tmux-text echo hello        → sends: echo hello
tmux-text :wq               → sends: :wq
tmux-text "hello"           → sends: "hello"   (quotes are literal)
```

The text is always literal — quotes, keywords, and ctrl-key names
have no special meaning after `tmux-text`.

### `var_symbol` — variable name

An identifier matching `[a-zA-Z_][0-9a-zA-Z_]*`.  Used as:

- Arguments to `tmux-var` (expand a single variable).
- Elements of `var_symbol_list` (for `tmux-expand` and macro parameters).
- Macro parameter names in `tmux-macro`.

Reserved words (`tmux-ctrl`, `tmux-text`, `tmux-var`, `tmux-expand`,
`tmux-macro`, `begin`, `end`) and ctrl-key names are NOT valid variable
symbols — the lexer emits them as keyword / ctrl-key tokens before the
identifier rule fires.

## Statement Semantics

### `#` comment

```
# this is a comment
```

Ignored at execution time.  Comments run from `#` to end of line.

### `tmux-ctrl` — send a control key

```
tmux-ctrl Enter
tmux-ctrl C-c
tmux-ctrl Escape
```

Sends a single non-literal keystroke to the tmux pane.  The argument
must be a valid `ctrl_constant`.

### `tmux-text` — send literal text

```
tmux-text "hello world"
tmux-text ':wq'
```

Sends the quoted text as literal keystrokes (`pane.send_keys(text, literal=True)`).

### `tmux-var` — expand a variable

```
tmux-var filename
```

Expands the named variable at execution time.  Only valid inside a
macro body (macros receive their parameter bindings when expanded).

### `tmux-expand` — expand a variable list

```
tmux-expand a b c
```

Expands a space-separated list of variables inline.  Each element in
the list is expanded independently.

### `tmux-macro` — define a macro

```
tmux-macro  name  param1 param2 ... paramN
begin
    statement
    statement
    ...
end
```

Defines a named keystroke macro with zero or more parameters.
`param1 ... paramN` is a `var_symbol_list` (space-separated
identifiers).  A newline is required between the parameter list and
`begin`.  The body between `begin` and `end` is a `statement_list`.
Within the body, `tmux-var` and `tmux-expand` can reference the
macro's parameters.

Macros do NOT execute when defined — they are stored and expanded
later when invoked (macro invocation is a future extension).

## Execution Model

A `program` is a sequence of statements executed top-to-bottom.
Each `tmux-ctrl` or `tmux-text` statement produces one `send_keys`
call to the active tmux pane.

Macro definitions are stored in a symbol table and do not produce
keystrokes at definition time.

## Examples

### Bash command

```
# Run a simple shell command
tmux-text "ls -la"
tmux-ctrl Enter
```

### Vi editing

```
# Create a file in vi
tmux-text "vi /tmp/x.txt"
tmux-ctrl Enter
tmux-text "i"
tmux-text "hello world"
tmux-ctrl Escape
tmux-text ":wq"
tmux-ctrl Enter
```

### Macro definition

```
# Define a macro that types text then saves in vi
tmux-macro vi-save  content
begin
    tmux-text "i"       # enter insert mode
    tmux-var content    # type the parameter
    tmux-ctrl Escape    # exit insert mode
    tmux-text ":wq"     # save and quit
    tmux-ctrl Enter
end
```

## Parser

The canonical parser is `experience/keystroke_dsl/keystroke_ply.py`
(pure-Python PLY implementation).  Import and use:

```python
from experience.keystroke_dsl.keystroke_ply import parse_keystroke

ast = parse_keystroke(source_string)
# => {"type": "program", "statements": [
#       {"type": "ctrl",    "key": "Enter"},
#       {"type": "text",    "value": "hello world"},
#       {"type": "comment", "text": "# a comment"},
#       {"type": "var",     "name": "X"},
#       {"type": "expand",  "vars": ["a", "b"]},
#       {"type": "macro",   "params": ["name"], "body": [...]},
#     ]}
```

The AST node types and their fields:

| `type`     | Fields               |
|------------|----------------------|
| `ctrl`     | `key` (str)          |
| `text`     | `value` (str)        |
| `comment`  | `text` (str)         |
| `var`      | `name` (str)         |
| `expand`   | `vars` (list[str])   |
| `macro`    | `params` (list[str]), `body` (list[statement]) |
| `program`  | `statements` (list[statement]) |

No external build tools are required — PLY generates the lexer and
parser tables at import time.
