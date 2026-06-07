"""
RawLlmEngine: Thin engine that calls the LLM API directly.

The LLM decides WHAT keystrokes to send. A prediction adaptor computes
WHERE the cursor will be and WHAT foveal text to expect — deterministically
for known terminal states (bash, common vi/nano keys), falling back to
LLM-based prediction for complex states.

Architecture:
  1. Extract current_foveal from capture + fixation (deterministic)
  2. Ask the LLM for keystrokes
  3. Adaptor predicts (next_fixation, next_foveal) based on terminal state
  4. Build the KeystrokeNode tree
"""

from typing import List, Tuple, Callable, Optional

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.react.fixation import extract_foveal
from experience.react.react_types import KeystrokeNode


def _parse_fixation(fixation_text: str) -> Tuple[int, int]:
    try:
        parts = fixation_text.strip().split(",")
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return (0, 0)


# ── Prediction adaptors ──────────────────────────────────────────────
#
# Each adaptor takes (capture_text, row, col, keystroke_text, is_ctrl)
# and returns (next_row, next_col, next_foveal).
# Returns None if it can't predict (fall through to next adaptor).


def _bash_adaptor(
    capture: str, row: int, col: int,
    text: str, is_ctrl: bool,
) -> Optional[Tuple[int, int, str]]:
    """Predict for bash shell prompts — text inserts at cursor, Enter → next line."""
    lines = capture.split("\n")
    if is_ctrl:
        if text.lower() == "enter":
            next_row = row + 1
            foveal = extract_foveal(capture, (next_row, 0))
            if not foveal:
                foveal = extract_foveal(capture, (next_row, 0), window=12)
            return (next_row, 0, foveal or "")
        return None  # other ctrl keys — can't predict
    # Text insertion: cursor advances by len(text)
    new_col = col + len(text)
    if row < len(lines):
        simulated = lines.copy()
        simulated[row] = simulated[row][:col] + text + simulated[row][col:]
    else:
        simulated = lines
    pred_screen = "\n".join(simulated)
    foveal = extract_foveal(pred_screen, (row, new_col))
    return (row, new_col, foveal)


def _vi_adaptor(
    capture: str, row: int, col: int,
    text: str, is_ctrl: bool,
) -> Optional[Tuple[int, int, str]]:
    """Predict for vi/vim — common normal-mode and insert-mode commands."""
    lines = capture.split("\n")
    if is_ctrl:
        if text.lower() == "enter":
            return _bash_adaptor(capture, row, col, "enter", True)
        return None

    # Normal mode commands
    if text == "j":
        next_row = min(row + 1, len(lines) - 1)
        next_col = min(col, len(lines[next_row].rstrip()) if next_row < len(lines) else 0)
        foveal = extract_foveal(capture, (next_row, next_col))
        return (next_row, next_col, foveal or "")
    if text == "k":
        next_row = max(row - 1, 0)
        next_col = min(col, len(lines[next_row].rstrip()) if next_row < len(lines) else 0)
        foveal = extract_foveal(capture, (next_row, next_col))
        return (next_row, next_col, foveal or "")
    if text in ("h", "l"):
        delta = -1 if text == "h" else 1
        next_col = max(0, col + delta)
        foveal = extract_foveal(capture, (row, next_col))
        return (row, next_col, foveal or "")
    if text == "w":
        # Move to start of next word on current line
        line = lines[row] if row < len(lines) else ""
        rest = line[col:]
        # Find next word boundary: skip current word, skip spaces, start of next
        i = 0
        while i < len(rest) and rest[i].isalnum():
            i += 1
        while i < len(rest) and not rest[i].isalnum():
            i += 1
        next_col = col + i
        foveal = extract_foveal(capture, (row, next_col))
        return (row, next_col, foveal or "")
    if text == "b":
        # Move to start of previous word
        line = lines[row] if row < len(lines) else ""
        before = line[:col]
        i = len(before) - 1
        while i >= 0 and not before[i].isalnum():
            i -= 1
        while i >= 0 and before[i].isalnum():
            i -= 1
        next_col = max(0, i + 1)
        foveal = extract_foveal(capture, (row, next_col))
        return (row, next_col, foveal or "")
    if text == "0":
        foveal = extract_foveal(capture, (row, 0))
        return (row, 0, foveal or "")
    if text == "$":
        end_col = len(lines[row].rstrip()) if row < len(lines) else 0
        foveal = extract_foveal(capture, (row, end_col))
        return (row, end_col, foveal or "")
    if text in ("i", "a", "o", "O", "I", "A"):
        # Entering insert mode — screen doesn't change, cursor stays/moves slightly
        next_col = col + (1 if text == "a" else 0)
        foveal = extract_foveal(capture, (row, next_col))
        return (row, next_col, foveal or "")
    if text == "x":
        # Delete char under cursor — screen shifts left
        if row < len(lines) and col < len(lines[row]):
            simulated = list(lines)
            simulated[row] = simulated[row][:col] + simulated[row][col+1:]
            pred_screen = "\n".join(simulated)
            foveal = extract_foveal(pred_screen, (row, col))
            return (row, col, foveal or "")
        return None
    if text == "dd":
        # Delete line — cursor moves to next line start
        if row < len(lines):
            next_row = min(row, len(lines) - 2)
            foveal = extract_foveal(capture, (next_row, 0))
            return (next_row, 0, foveal or "")
        return None
    if text == "G":
        last_row = max(0, len(lines) - 1)
        foveal = extract_foveal(capture, (last_row, 0))
        return (last_row, 0, foveal or "")
    if text == "gg":
        foveal = extract_foveal(capture, (0, 0))
        return (0, 0, foveal or "")

    # Insert mode or unknown — fall back to bash-style text insertion
    return _bash_adaptor(capture, row, col, text, is_ctrl)


def _nano_adaptor(
    capture: str, row: int, col: int,
    text: str, is_ctrl: bool,
) -> Optional[Tuple[int, int, str]]:
    """Predict for nano editor — mostly insert-mode, with ctrl shortcuts."""
    if is_ctrl:
        ctrl_map = {
            "c-x": (-1, -1),  # exit — can't predict
            "c-o": (row, col),  # write-out — screen stays
            "c-s": (row, col),  # save — screen stays
        }
        if text.lower() in ctrl_map:
            nr, nc = ctrl_map[text.lower()]
            if nr == -1:
                return None
            foveal = extract_foveal(capture, (nr, nc))
            return (nr, nc, foveal or "")
        return None
    # Insert mode — same as bash text insertion
    return _bash_adaptor(capture, row, col, text, is_ctrl)


def _detect_terminal_state(capture: str) -> str:
    """Detect the current terminal state from screen content."""
    lines = capture.split("\n")
    # Vi insert/visual/replace mode markers — unambiguous
    if any(marker in capture for marker in ("-- INSERT --", "-- VISUAL --", "-- REPLACE --")):
        return "vi"
    # Vi normal mode: ~ lines (empty buffer) or quoted filename at bottom
    has_tilde = any(l.strip() == "~" for l in lines)
    has_file_info = any(l.strip().startswith('"') and ("L," in l or "B" in l or "[New" in l or "[noeol" in l) for l in lines)
    if has_tilde or has_file_info:
        return "vi"
    # Nano/pico: status bar at bottom
    if any("GNU nano" in l or "UW PICO" in l for l in lines):
        return "nano"
    # Nano/pico dialogs: save, exit confirmation
    if any("File Name to write" in l or "Save modified buffer" in l for l in lines):
        return "nano"
    if any("nano" in l.lower() and ("Read" in l or "File:" in l) for l in lines):
        return "nano"
    # Default: bash (don't check for vi/nano by name — command text would false-match)
    return "bash"


# Ordered list of adaptors to try. First match wins.
_ADAPTORS: List[Tuple[str, Callable]] = [
    ("vi", _vi_adaptor),
    ("nano", _nano_adaptor),
    ("bash", _bash_adaptor),
]


def _predict(
    capture: str, row: int, col: int,
    text: str, is_ctrl: bool,
) -> Tuple[int, int, str]:
    """Run adaptors in order until one returns a prediction."""
    state = _detect_terminal_state(capture)
    # Try the detected state's adaptor first, then fall through the rest
    ordered = [(n, a) for n, a in _ADAPTORS if n == state] + \
              [(n, a) for n, a in _ADAPTORS if n != state]
    for name, adaptor in ordered:
        result = adaptor(capture, row, col, text, is_ctrl)
        if result is not None:
            return result
    # Ultimate fallback: assume bash
    return _bash_adaptor(capture, row, col, text, is_ctrl) or (row, col, "")


# ── Keystroke parsing ────────────────────────────────────────────────


def _commands_to_actions(commands: List[str]) -> List[Tuple[bool, str]]:
    """Convert validated DSL command strings to (is_ctrl, text) actions.

    "tmux-ctrl Enter"  -> (True,  "Enter")
    "tmux-text hello"  -> (False, "hello")
    """
    actions = []
    for cmd in commands:
        low = cmd.lower()
        if low.startswith("tmux-ctrl "):
            actions.append((True, cmd[10:].strip()))
        elif low.startswith("tmux-text "):
            actions.append((False, cmd[10:].strip()))
    return actions


def _build_node_tree(
    capture: str,
    row: int, col: int,
    actions: List[Tuple[bool, str]],
) -> KeystrokeNode:
    """Build a KeystrokeNode tree from actions, using adaptors for predictions."""
    if not actions:
        return KeystrokeNode()

    current_row, current_col = row, col
    current_screen = capture

    # Build the root node from the first action
    is_ctrl, text = actions[0]
    current_foveal = extract_foveal(current_screen, (current_row, current_col))
    next_row, next_col, next_foveal = _predict(
        current_screen, current_row, current_col, text, is_ctrl,
    )
    if is_ctrl:
        ks = f"TMUX-CTRL-{text}"
    else:
        ks = f"TMUX-TEXT-{text}"

    root = KeystrokeNode(
        current_fixation=(current_row, current_col),
        current_foveal=current_foveal,
        keystrokes=ks,
        predicted_next_fixation=(next_row, next_col),
        predicted_next_foveal=next_foveal,
    )

    # Simulate screen change for subsequent actions
    if not is_ctrl:
        lines = current_screen.split("\n")
        if current_row < len(lines):
            lines[current_row] = (
                lines[current_row][:current_col] + text +
                lines[current_row][current_col:]
            )
        current_screen = "\n".join(lines)
    current_row, current_col = next_row, next_col

    # Build child nodes for remaining actions
    parent = root
    for is_ctrl, text in actions[1:]:
        current_foveal = extract_foveal(current_screen, (current_row, current_col))
        next_row, next_col, next_foveal = _predict(
            current_screen, current_row, current_col, text, is_ctrl,
        )
        if is_ctrl:
            ks = f"TMUX-CTRL-{text}"
        else:
            ks = f"TMUX-TEXT-{text}"

        child = KeystrokeNode(
            current_fixation=(current_row, current_col),
            current_foveal=current_foveal,
            keystrokes=ks,
            predicted_next_fixation=(next_row, next_col),
            predicted_next_foveal=next_foveal,
        )
        parent.children.append(child)
        parent = child

        if not is_ctrl:
            lines = current_screen.split("\n")
            if current_row < len(lines):
                lines[current_row] = (
                    lines[current_row][:current_col] + text +
                    lines[current_row][current_col:]
                )
            current_screen = "\n".join(lines)
        current_row, current_col = next_row, next_col

    return root


# ── Engine ────────────────────────────────────────────────────────────


def _extract_target(task: str) -> str:
    """Extract the target content from a task description."""
    import re
    # "containing only the word X" or "contain the word X"
    # Match multi-word targets like "hello nano" not just single words.
    m = re.search(r'contain(?:ing)?\s+(?:only\s+)?(?:the\s+)?(?:word\s+)?(.+?)(?:\.\s*(?:Save|Then|$)|$)', task)
    if m:
        return m.group(1).strip().rstrip('.!,;:')
    # "Type: X" or "type X"
    m = re.search(r'(?:T|t)ype:\s*(.+?)(?:\.\s*(?:Save|Then|Press|$)|$)', task)
    if m:
        return m.group(1).strip().rstrip('.!,;:')
    # "Target: X"
    m = re.search(r'Target:\s*(.+?)(?:\.|$)', task)
    if m:
        return m.group(1).strip().rstrip('.!,;:')
    # Last quoted word
    m = re.findall(r"['\"](\w+)['\"]", task)
    if m:
        return m[-1]
    return ""


def engine_step(
    capture: FutureTensor,
    fixation: FutureTensor,
    mail: FutureTensor,
    task: str,
    llm_model: str = None,
) -> FutureTensor:
    shape = list(capture.ft_capacity_shape)
    schema = list(capture.ft_shape_schema)
    relative_to = capture.ft_initial_static_tensor.st_relative_to

    async def _engine_get(coordinates: List[int], trajectory: str):
        if capture.ft_forwarded:
            _, filepath = capture.ft_get_materialized_value(coordinates)
            with open(filepath, "r", encoding="utf-8") as f:
                capture_text = f.read()
        else:
            capture_text, _ = await capture.ft_async_get(coordinates, trajectory)
        if not capture_text:
            capture_text = "(empty terminal)"

        if fixation.ft_forwarded:
            try:
                _, fpath = fixation.ft_get_materialized_value(coordinates)
                with open(fpath, "r", encoding="utf-8") as f:
                    fixation_text = f.read().strip()
            except (IndexError, FileNotFoundError):
                fixation_text = "0,0"
        else:
            fixation_text, _ = await fixation.ft_async_get(coordinates, trajectory)
        if not fixation_text:
            fixation_text = "0,0"

        terminal_state = _detect_terminal_state(capture_text)
        # Clean up vi swap files from previous interrupted sessions
        if terminal_state == "bash":
            import os as _os, glob as _glob
            for _swp in _glob.glob("/tmp/.*.sw*") + _glob.glob("/tmp/.*.swo"):
                try: _os.remove(_swp)
                except OSError: pass
        row, col = _parse_fixation(fixation_text)
        # In vi/nano, the status line at the bottom is not content.
        # Adjust fixation to point at actual file content.
        if terminal_state in ("vi", "nano"):
            lines = capture_text.split("\n")
            content_lines = [
                (i, l) for i, l in enumerate(lines)
                if l and not l.startswith("~") and not l.startswith("-- ")
                and not l.startswith('"') and "GNU nano" not in l
                and not l.startswith("^") and not l.startswith("[")
                and not l.startswith(":")       # vi command-line
                and not l.strip().isdigit()     # vi line-number-only (empty)
            ]
            if content_lines:
                last_i, last_l = content_lines[-1]
                row = last_i
                col = min(col, len(last_l.rstrip()))
        current_foveal = extract_foveal(capture_text, (row, col))
        # Hint for editor modes: what the file should contain
        target_hint = _extract_target(task)
        target_info = f"\nFile should contain: {target_hint}" if target_hint else ""

        # One action per call. The react_loop handles sequencing.
        # Each call asks ONE question based on what's on screen.

        # Translator-produced key tasks — force without LLM
        prompt = None  # default: will be set by terminal state branch if no forced action
        if task.strip() == "TMUX-CTRL-Enter":
            actions = [(True, "Enter")]
        elif task.strip() == "TMUX-CTRL-Escape":
            actions = [(True, "Escape")]
        else:
            prompt = "unset"  # signal to enter terminal state branches

        if prompt is None:
            pass  # forced action taken, skip LLM
        elif terminal_state == "bash":
            prompt = (
                f"Bash terminal:\n{capture_text}\n\n"
                f"Goal: {task}\n"
                f"Reply ONE shell command."
            )
        elif terminal_state == "vi":
            if "Found a swap file" in capture_text or "E325" in capture_text:
                prompt = f"{capture_text}\nVi found swap file. Output: tmux-text d"
            elif "-- INSERT --" in capture_text:
                file_text = "\n".join(l for _, l in content_lines) if content_lines else capture_text
                if target_hint and target_hint in file_text:
                    actions = [(True, "Escape")]
                    prompt = None
                else:
                    prompt = (
                        f"{capture_text}\n\n"
                        f"Vi insert. Output: tmux-text {target_hint or 'the text'}"
                    )
            else:
                has_content = bool(content_lines)
                if not has_content:
                    prompt = f"{capture_text}\n\nVi normal, empty. Output: tmux-text i"
                elif target_hint and target_hint in "\n".join(l for _, l in content_lines):
                    prompt = f"{capture_text}\n\nVi normal, done. Output: tmux-text :wq"
                else:
                    ex = f"tmux-text :%s/.../{target_hint}/g" if target_hint else "tmux-text dd"
                    prompt = f"{capture_text}\n\nVi normal, edit. Output: {ex}"
        elif terminal_state == "nano":
            # Nano/pico dialogs: force known actions without LLM
            if "File Name to write" in capture_text or "File Name:" in capture_text:
                actions = [(True, "Enter")]
                prompt = None
            elif "Save modified buffer" in capture_text:
                actions = [(False, "Y")]
                prompt = None
            elif "^E" in task or "end of line" in task.lower() or "ctrl-e" in task.lower():
                actions = [(True, "C-e")]
                prompt = None
            elif target_hint and target_hint in capture_text:
                # Content already in nano buffer — save and exit
                # Send ^O (save), then the next iteration will detect
                # "File Name to write" and auto-press Enter
                actions = [(True, "C-o")]
                prompt = None
            else:
                # Nano editing mode — type text
                prompt = (
                    f"{capture_text}\n\n"
                    f"Goal: {task}\n"
                    f"Nano editor. Type ONLY the needed text. One line."
                )
        else:
            prompt = (
                f"Terminal:\n{capture_text}\n\n"
                f"Task: {task}\n"
                f"What ONE keystroke? (T text or C key)?"
            )

        import os
        from experience.llm_client.raw_llm_query import raw_llm_query
        from experience.llm_client.agent_config import RawLlmConfig
        from experience.react.gen_keystrokes_until_success import (
            gen_keystrokes_until_success,
        )

        if prompt is not None:
            base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
            if "/anthropic" in base_url:
                base_url = base_url.replace("/anthropic", "/v1")
            config = RawLlmConfig(
                base_url=base_url or os.environ.get("LLM_BASE_URL"),
                api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("LLM_API_KEY"),
                model=llm_model or os.environ.get("ANTHROPIC_MODEL") or os.environ.get("LLM_MODEL"),
            )
            dsl_prompt = (
                f"{prompt}\n\n"
                f"Reply one line: tmux-text <text> OR tmux-ctrl <key>. "
                f"Ctrl keys: Enter Escape Tab C-c C-a C-e C-o C-x "
                f"Backspace Delete Up Down Left Right Home End F1-F12"
            )
            async def _llm_fn(args):
                return await raw_llm_query(args, config=config,
                    extra_body={"thinking": {"type": "disabled"}})
            async def _retry_prompt(errors, args):
                return f"{args}\n[PARSE ERROR: {'; '.join(errors)}. Reply VALID.]"
            try:
                commands = await gen_keystrokes_until_success(
                    llm_fn=_llm_fn, construct_retry_prompt=_retry_prompt,
                    initial_args=dsl_prompt, max_retries=2)
                actions = _commands_to_actions(commands)
            except RuntimeError:
                return ("", Status.self_confidence_but_failed(0.3))

        try:

            if not actions:
                return ("", Status.self_confidence_but_failed(0.3))

            # Only take the first action — multi-action responses cause
            # cascading failures because speculative_keystroke only returns
            # the root node. The LLM must decide one step at a time.
            actions = actions[:1]

            # Vi normal mode safety: if LLM returns multi-char text that
            # doesn't start with ":" (colon command), it's trying to do
            # everything at once (e.g. "ihello<ESC>ZZ"). Truncate to
            # first char — valid vi commands are single-char (i, a, o, x).
            if terminal_state == "vi" and "-- INSERT --" not in capture_text:
                if actions and not actions[0][0] and len(actions[0][1]) > 1 \
                        and not actions[0][1].startswith(":"):
                    actions[0] = (False, actions[0][1][0])

            # Nano: if appending to existing text, prepend a space
            if terminal_state == "nano" and not any(
                kw in capture_text for kw in ("File Name", "Save modified")
            ):
                content_lines = [l for l in capture_text.split("\n")
                                 if l.strip() and not l.startswith(("^","UW","File"))]
                if content_lines and actions and not actions[0][0]:
                    text = actions[0][1]
                    if text and not text.startswith(" "):
                        actions[0] = (False, " " + text)

            # For vi: colon commands need Enter to execute.
            # Merge into the text with \n so the tree has one node.
            if terminal_state == "vi":
                has_enter = any(is_ctrl and t.lower() == "enter" for is_ctrl, t in actions)
                has_colon = any(not is_ctrl and t.startswith(":") for is_ctrl, t in actions)
                if has_colon and not has_enter:
                    for i, (is_ctrl, t) in enumerate(actions):
                        if not is_ctrl and t.startswith(":"):
                            actions[i] = (False, t + "\n")
                            break

            # For bash: text commands need Enter to execute.
            # Merge into a single text keystroke (append \n) so the tree
            # has one node — speculative_keystroke only returns the root.
            if terminal_state == "bash":
                has_text = any(not is_ctrl for is_ctrl, t in actions)
                has_enter = any(is_ctrl and t.lower() == "enter" for is_ctrl, t in actions)
                if has_text and not has_enter:
                    for i, (is_ctrl, t) in enumerate(actions):
                        if not is_ctrl:
                            actions[i] = (False, t + "\n")
                            break

            # Build the KeystrokeNode tree with adaptor-predicted foveals
            node = _build_node_tree(capture_text, row, col, actions)
            return (node.serialize(), Status.confidence(1.0))
        except Exception as e:
            return (f"Error: {e}", Status.self_confidence_but_failed(0.3))

    ft = FutureTensor(relative_to, _engine_get, list(schema))
    ft.ft_capacity_shape = list(shape)
    ft.requires_grad_(True)
    return ft
