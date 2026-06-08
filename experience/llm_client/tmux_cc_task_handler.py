import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

from experience.llm_client.agent_task import AgentTask
from experience.llm_client.agent_config import TmuxCcConfig
from experience.llm_client.agent_config_factory import AgentConfigFactory
from experience.llm_client.agent_binary_provider import AgentBinaryProvider, get_default_provider
from experience.fs_util.pack_dir import pack_dir


def _find_tmux_cc_bin(agent_provider: Optional[AgentBinaryProvider] = None) -> str:
    """Find tmux_cc (agent) binary using the provided or default AgentBinaryProvider.
    
    Args:
        agent_provider: Optional AgentBinaryProvider instance. If None, uses default provider.
    
    Returns:
        Path to agent binary.
    
    Raises:
        FileNotFoundError: If agent binary cannot be found.
    """
    if agent_provider is None:
        agent_provider = get_default_provider()
    
    cli_path = agent_provider.get_cli_path()
    if cli_path is None:
        raise FileNotFoundError(
            "Cannot find tmux_cc (agent) binary. Please either:\n"
            "  1. Set TMUX_CC_BIN environment variable\n"
            "  2. Install baidu-cc extension in ~/.comate/extensions/\n"
            "  3. Add agent binary to your PATH\n"
            "  4. Provide a custom AgentBinaryProvider"
        )
    return cli_path


def _find_settings_json(agent_provider: Optional[AgentBinaryProvider] = None) -> Optional[str]:
    """Find the agent settings.json file using the provided or default AgentBinaryProvider.
    
    Args:
        agent_provider: Optional AgentBinaryProvider instance. If None, uses default provider.
    
    Returns:
        Path to settings.json, or None if not found.
    """
    if agent_provider is None:
        agent_provider = get_default_provider()
    
    return agent_provider.get_settings_path()


def _ensure_settings_json_clean(settings_path: Optional[str]) -> None:
    """Restore settings.json from backup file if previous run left it dirty.
    
    Args:
        settings_path: Path to settings.json file.
    """
    if not settings_path:
        return
    backup_path = settings_path + ".experience_backup"
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, settings_path)
        os.remove(backup_path)
        print(f"[DEBUG] Restored dirty settings.json from backup", file=sys.stderr)


def _patch_settings_json(
    settings_path: Optional[str],
    base_url: str = None,
    api_key: str = None,
    model: str = None,
) -> Optional[dict]:
    """Patch settings.json with experience.json config. Returns backup info for restore.
    
    For internal API (comate): only override base_url, keep token
    For external API (qianfan): override base_url and replace token with api_key
    
    Args:
        settings_path: Path to settings.json file.
        base_url: Base URL for the API.
        api_key: API key for external APIs.
        model: Model name.
    """
    if not settings_path:
        return None
    
    # Save backup to disk (crash-safe)
    backup_path = settings_path + ".experience_backup"
    shutil.copy2(settings_path, backup_path)
    
    with open(settings_path, "r") as f:
        original = json.load(f)
    
    patched = json.loads(json.dumps(original))  # deep copy
    
    is_internal = base_url and ("comate" in base_url or "baidu-int" in base_url)
    
    if base_url:
        patched["env"]["ANTHROPIC_BASE_URL"] = base_url
    
    if not is_internal and api_key:
        # External API: replace token with api_key
        patched["env"].pop("ANTHROPIC_AUTH_TOKEN", None)
        patched["env"].pop("ANTHROPIC_CUSTOM_HEADERS", None)
        patched["env"]["ANTHROPIC_API_KEY"] = api_key
    
    if model:
        patched["env"]["ANTHROPIC_MODEL"] = model
        patched["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
        patched["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
        patched["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
    
    with open(settings_path, "w") as f:
        json.dump(patched, f, indent=2)
    
    print(f"[DEBUG] Patched settings.json: base_url={base_url}, model={model}, is_internal={is_internal}", file=sys.stderr)
    return {"path": settings_path, "backup_path": backup_path}


def _restore_settings_json(backup: dict) -> None:
    """Restore settings.json from backup."""
    if not backup:
        return
    backup_path = backup.get("backup_path")
    settings_path = backup.get("path")
    if backup_path and os.path.exists(backup_path):
        shutil.copy2(backup_path, settings_path)
        os.remove(backup_path)
        print(f"[DEBUG] Restored settings.json", file=sys.stderr)


def _get_workspace_root() -> str:
    """Get the workspace root directory for tmux_cc tasks.

    Priority:
    1. TMUX_CC_WORKSPACE_ROOT environment variable
    2. Default: ~/.tmux_cc_tmp/
    """
    default_root = os.path.expanduser("~/.tmux_cc_tmp")
    return os.environ.get("TMUX_CC_WORKSPACE_ROOT", default_root)


# Workspace root for tmux_cc tasks
TMUX_CC_WORKSPACE_ROOT = _get_workspace_root()

# Default configuration for tmux interactive mode
TMUX_SESSION_PREFIX = "tmux_cc_interactive"
TMUX_CHECK_INTERVAL = 3  # Check interval in seconds
TMUX_IDLE_THRESHOLD = 5  # Number of consecutive unchanged checks to consider idle
TMUX_INITIAL_WAIT = 5    # Initial wait time for tmux_cc to start (seconds)

# Prompt patterns that require auto-confirmation
AUTO_CONFIRM_PATTERNS = {
    "Do you want to proceed": "Enter",
    "Yes, I trust this folder": "Enter",
    "allow all edits during this session": "Down Enter",
    "Press Enter to continue": "Enter",
    "Yes, I accept": "Down Enter",  # Bypass Permissions warning - select "Yes, I accept"
    "No, exit": "Down Enter",  # Same warning - default is "No, exit", need to go down to "Yes"
}


def _flatten_nested(nested) -> list:
    """Flatten a nested list structure into a flat list."""
    if not isinstance(nested, list):
        return [nested]
    result = []
    for item in nested:
        result.extend(_flatten_nested(item))
    return result


def _grep_by_file_content_hint(root_dir: str, todo_file_content_hint: str) -> List[str]:
    """Find all files under root_dir whose content contains the hint string."""
    todo_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if todo_file_content_hint in content:
                    todo_files.append(file_path)
            except (UnicodeDecodeError, OSError):
                continue
    return todo_files


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    import re
    return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)


def _tmux_available() -> bool:
    """Check if tmux command is available on the system."""
    try:
        subprocess.run(["tmux", "-V"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _tmux_session_exists(session_name: str) -> bool:
    """Check if a tmux session exists."""
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _tmux_create_session(session_name: str, working_dir: str) -> None:
    """Create a new tmux session."""
    try:
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, "-c", working_dir],
            check=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "tmux is not installed or not found in PATH. "
            "Please install tmux to use interactive mode."
        ) from e


def _tmux_kill_session(session_name: str) -> None:
    """Kill a tmux session."""
    subprocess.run(
        ["tmux", "kill-session", "-t", session_name],
        capture_output=True,
    )


def _tmux_send_keys(session_name: str, *keys: str) -> None:
    """Send keys to a tmux session."""
    subprocess.run(
        ["tmux", "send-keys", "-t", session_name] + list(keys),
        check=True,
    )


def _tmux_capture_pane(session_name: str) -> str:
    """Capture the content of a tmux pane."""
    result = subprocess.run(
        ["tmux", "capture-pane", "-p", "-e", "-S", "-", "-t", session_name],
        capture_output=True,
        text=True,
    )
    return result.stdout


def _check_and_auto_confirm(session_name: str, patterns: Dict[str, str]) -> bool:
    """Check for auto-confirm patterns and send keys if matched.

    Returns True if a pattern was matched and keys were sent.
    """
    content = _tmux_capture_pane(session_name)
    clean_content = _strip_ansi(content)

    for pattern, keys in patterns.items():
        if pattern in clean_content:
            print(f"[tmux] Detected '{pattern}' -> sending {keys}", file=sys.stderr)
            for key in keys.split():
                _tmux_send_keys(session_name, key)
            return True
    return False


def _wait_for_tmux_cc_idle(
    session_name: str,
    check_interval: float = TMUX_CHECK_INTERVAL,
    idle_threshold: int = TMUX_IDLE_THRESHOLD,
    auto_confirm: bool = True,
    timeout: float = 600,  # 10 minutes default timeout
) -> str:
    """Wait for tmux_cc to become idle (screen content stops changing).

    Args:
        session_name: tmux session name
        check_interval: seconds between checks
        idle_threshold: consecutive unchanged checks to consider idle
        auto_confirm: whether to auto-confirm prompts
        timeout: maximum wait time in seconds

    Returns:
        The final screen content
    """
    last_content = ""
    idle_count = 0
    start_time = time.time()
    task_started = False  # True after we see Claude start processing (e.g., tool calls)
    command_completed = False

    while True:
        if time.time() - start_time > timeout:
            print(f"[tmux] Timeout ({timeout}s), forcing exit", file=sys.stderr)
            break

        if not _tmux_session_exists(session_name):
            print(f"[tmux] Session '{session_name}' has been closed", file=sys.stderr)
            break

        content = _tmux_capture_pane(session_name)
        clean_content = _strip_ansi(content)

        # Auto-confirm if enabled
        if auto_confirm:
            if _check_and_auto_confirm(session_name, AUTO_CONFIRM_PATTERNS):
                time.sleep(1)  # Wait a bit after auto-confirm
                idle_count = 0
                last_content = ""  # Reset to re-check
                continue

        # Check if Claude has started processing the task
        # Look for tool call indicators like "Read(", "Write(", "Edit(", or thinking indicators
        if not task_started:
            task_indicators = ['Read(', 'Write(', 'Edit(', '⏺', 'Thinking', 'Reading', 'Writing']
            for indicator in task_indicators:
                if indicator in clean_content:
                    task_started = True
                    print(f"[tmux_cc] Task started (detected: {indicator})", file=sys.stderr)
                    break

        # Only check for completion after the task has started
        if not task_started:
            print(f"[tmux_cc] Waiting for task to start...", file=sys.stderr)
            last_content = content
            time.sleep(check_interval)
            continue

        # Check if command has completed (shell prompt appears after output)
        # Look for patterns indicating tmux_cc has finished
        lines = clean_content.strip().split('\n')
        last_lines = '\n'.join(lines[-5:]) if len(lines) >= 5 else clean_content

        # tmux_cc completed indicators:
        # 1. TMUX_CC_COMPLETED marker (from our wrapper script)
        # 2. Shell prompt at the end (ends with % or $ after tmux_cc ran)
        # 3. "Done." message from tmux_cc
        # 4. tmux_cc idle state: empty prompt line with ❯ at the end (after processing)
        if not command_completed:
            # Check if Claude is still thinking/working
            # These indicators mean Claude is NOT done yet
            still_working_indicators = ['Proofing', 'Thinking', 'thinking', 'Searching', 'Reading', 'Writing', 'Editing']
            is_still_working = any(ind in clean_content for ind in still_working_indicators)
            
            if is_still_working:
                print(f"[tmux_cc] Claude still working (detected active indicator)", file=sys.stderr)
                last_content = content
                time.sleep(check_interval)
                continue
            
            # Check for tmux_cc idle state: line ends with ❯
            # This indicates tmux_cc has finished processing and is waiting for new input
            tmux_cc_idle = False
            if '❯' in last_lines:
                # tmux_cc shows ❯ when idle, check if it's at the end of a line by itself
                for line in lines[-3:]:
                    stripped = line.strip()
                    # Empty prompt line or line ending with just ❯
                    if stripped == '❯' or (stripped.endswith('❯') and len(stripped) < 5):
                        tmux_cc_idle = True
                        break

            if ('TMUX_CC_COMPLETED' in clean_content or
                'Done.' in clean_content or
                tmux_cc_idle or
                (clean_content.count('%') >= 2 and last_lines.rstrip().endswith('%'))):
                command_completed = True
                print(f"[tmux_cc] Command execution completed", file=sys.stderr)

        if command_completed:
            if content == last_content:
                idle_count += 1
                print(f"[tmux] Screen unchanged ({idle_count}/{idle_threshold})", file=sys.stderr)
                if idle_count >= idle_threshold:
                    break
            else:
                idle_count = 0
                last_content = content
        else:
            print(f"[tmux_cc] Waiting for completion...", file=sys.stderr)
            last_content = content

        time.sleep(check_interval)

    return content if 'content' in dir() else ""


def _create_task_workspace(
    task_idx: int,
    file_idx: int,
    original_workspace_dir: str,
    prompt: str,
    packed_workspace: str,
    todo_file_path: str,
    todo_file_content_hint: str,
) -> str:
    """Create a workspace directory for a single task.

    Structure:
        ~/.tmux_cc_tmp/task_YYYYMMDD_HHMMSS_idx/
            input/
                prompt.txt          # The task prompt
                packed_workspace.txt # The packed directory content
                task_info.txt       # Metadata about the task

    Returns:
        The path to the created workspace directory.
    """
    # Create unique workspace directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_name = f"task_{timestamp}_{task_idx}_{file_idx}"
    workspace_dir = os.path.join(TMUX_CC_WORKSPACE_ROOT, workspace_name)

    # Create directories
    input_dir = os.path.join(workspace_dir, "input")
    os.makedirs(input_dir, exist_ok=True)

    # Write prompt
    prompt_file = os.path.join(input_dir, "prompt.txt")
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    # Write packed workspace
    packed_file = os.path.join(input_dir, "packed_workspace.txt")
    with open(packed_file, "w", encoding="utf-8") as f:
        f.write(packed_workspace)

    # Write task info
    task_info_file = os.path.join(input_dir, "task_info.txt")
    with open(task_info_file, "w", encoding="utf-8") as f:
        f.write(f"original_workspace_dir: {original_workspace_dir}\n")
        f.write(f"todo_file_path: {todo_file_path}\n")
        f.write(f"todo_file_content_hint: {todo_file_content_hint}\n")
        f.write(f"timestamp: {timestamp}\n")

    return workspace_dir


def _build_file_based_prompt(workspace_dir: str, todo_file_path: str, todo_file_content_hint: str) -> str:
    """Build a prompt that instructs Claude to read input from files.

    Args:
        workspace_dir: The task workspace directory (contains input/ and output/)
        todo_file_path: The original path to the TODO file (absolute path to mutable_output_dir/data.txt)
        todo_file_content_hint: The placeholder string to replace

    Returns:
        A short prompt that tells Claude where to find the input files and where to write output.
    """
    prompt = f"""Please complete the following task:

1. Read the task description from: ./input/prompt.txt
2. Read the codebase content from: ./input/packed_workspace.txt
3. The packed_workspace.txt contains the directory structure in repomix format.
4. Follow the instructions in prompt.txt to complete the task.

CRITICAL - Output Location:
- Write your result DIRECTLY to this file: {todo_file_path}
- This is an absolute path, use the Write tool to write to it.
- The file currently contains "{todo_file_content_hint}" - replace it with your answer.

IMPORTANT:
- Output raw text only. Do NOT wrap in markdown code fences (``` or ```lang).
- Do not generate unrelated content.
- Only output the code/content that should replace the placeholder.
"""
    return prompt


class TmuxCcTaskHandler:
    """Handler for running tasks via tmux_cc (Claude) CLI.

    Supports two modes:
    - interactive=False (default): Run tmux_cc via subprocess, non-interactive
    - interactive=True: Run tmux_cc in tmux session for visual observation

    Input is passed via files in workspace directory instead of command line
    to avoid issues with long prompts and special characters.

    Workspace structure:
        ~/.tmux_cc_tmp/task_YYYYMMDD_HHMMSS_idx/
            input/
                prompt.txt          # The task prompt
                packed_workspace.txt # The packed directory content
                task_info.txt       # Metadata about the task

    Note: Output is handled by coding_agent.py via _copy_back_to_storage_view,
    not by this handler directly.
    
    Args:
        config: Optional TmuxCcConfig. If None, creates default config.
    """

    def __init__(self, config: Optional[TmuxCcConfig] = None):
        if config is None:
            config = AgentConfigFactory.create_tmux_cc_config()
        self.config = config
        self.tmux_cc_bin = config.get_cli_path()
        if self.tmux_cc_bin is None:
            raise FileNotFoundError(
                "Cannot find tmux_cc (Claude) binary. Please configure properly."
            )

    def __call__(
        self,
        all_tasks,
        llm_env: Optional[Dict[str, str]] = None,
        interactive: bool = False,
        auto_confirm: bool = True,
        tmux_session: Optional[str] = None,
    ) -> None:
        """Execute tasks via tmux_cc.

        Args:
            all_tasks: List of AgentTask objects
            llm_env: Optional environment variables
            interactive: If True, run in tmux for visual interaction
            auto_confirm: If True (and interactive), auto-confirm prompts
            tmux_session: Custom tmux session name (interactive mode only)
        """
        # Ensure workspace root exists
        workspace_root = self.config.workspace_root or os.path.expanduser("~/.tmux_cc_tmp")
        os.makedirs(workspace_root, exist_ok=True)
        
        # Restore settings.json if previous run left it dirty (crash recovery)
        _ensure_settings_json_clean(self.config.get_settings_path())

        if interactive:
            self._run_interactive(all_tasks, llm_env, auto_confirm, tmux_session)
        else:
            self._run_batch(all_tasks, llm_env)

    def _run_batch(self, all_tasks, llm_env: Optional[Dict[str, str]] = None) -> None:
        """Batch mode: run tmux_cc via subprocess with file-based input."""
        flat_tasks = _flatten_nested(all_tasks)

        for task_idx, task in enumerate(flat_tasks):
            original_workspace_dir = task.workspace_dir
            prompt = task.prompt
            output_relative_dir_or_list = task.output_relative_dir
            todo_file_content_hint = task.todo_file_content_hint

            packed_workspace = pack_dir(original_workspace_dir)

            if isinstance(output_relative_dir_or_list, str):
                output_relative_dirs = [output_relative_dir_or_list]
            else:
                output_relative_dirs = output_relative_dir_or_list

            for output_relative_dir in output_relative_dirs:
                output_root = os.path.join(original_workspace_dir, output_relative_dir)
                todo_file_paths = _grep_by_file_content_hint(output_root, todo_file_content_hint)

                for file_idx, todo_file_path in enumerate(todo_file_paths):
                    # Create workspace with input files
                    task_workspace = _create_task_workspace(
                        task_idx=task_idx,
                        file_idx=file_idx,
                        original_workspace_dir=original_workspace_dir,
                        prompt=prompt,
                        packed_workspace=packed_workspace,
                        todo_file_path=todo_file_path,
                        todo_file_content_hint=todo_file_content_hint,
                    )

                    # Build file-based prompt
                    claude_prompt = _build_file_based_prompt(
                        task_workspace, todo_file_path, todo_file_content_hint
                    )

                    print(f"\n[tmux_cc] Task {task_idx + 1}/{len(flat_tasks)}: {todo_file_path}", file=sys.stderr)
                    print(f"[tmux_cc] Workspace: {task_workspace}", file=sys.stderr)
                    
                    model = os.environ.get("ANTHROPIC_MODEL")
                    base_url = os.environ.get("ANTHROPIC_BASE_URL")
                    api_key = os.environ.get("ANTHROPIC_API_KEY")
                    
                    is_internal_api = base_url and ("comate" in base_url or "baidu-int" in base_url)
                    settings_backup = None
                    
                    if base_url and is_internal_api:
                        # Internal API (comate): patch settings.json to change base_url
                        settings_backup = _patch_settings_json(
                            settings_path=self.config.get_settings_path(),
                            base_url=base_url,
                            model=model
                        )
                    elif base_url and not is_internal_api:
                        # External API (e.g., Qianfan): patch settings.json to replace
                        # comate auth with qianfan api_key
                        settings_backup = _patch_settings_json(
                            settings_path=self.config.get_settings_path(),
                            base_url=base_url,
                            api_key=api_key,
                            model=model
                        )

                    cmd = [
                        self.tmux_cc_bin,
                        "-p", claude_prompt,
                        "--allowedTools", "Read,Edit,Write",
                        "--permission-mode", "bypassPermissions",
                        "--effort", "low",
                    ]

                    if model:
                        cmd.extend(["--model", model])

                    # For external API: disable settings.json, use env vars
                    if base_url and not is_internal_api:
                        cmd.extend(["--setting-sources", "", "--settings", "{}"])
                    
                    # Prepare environment
                    run_env = os.environ.copy()
                    if base_url and not is_internal_api:
                        run_env.pop("ANTHROPIC_AUTH_TOKEN", None)
                        run_env.pop("ANTHROPIC_CUSTOM_HEADERS", None)

                    result = subprocess.run(
                        cmd,
                        cwd=task_workspace,
                        capture_output=True,
                        text=True,
                        env=run_env,
                    )
                    
                    # Restore settings.json after Claude completes
                    _restore_settings_json(settings_backup)

                    if result.returncode != 0:
                        print(f"[tmux_cc] stderr: {result.stderr}", file=sys.stderr)

    def _run_interactive(
        self,
        all_tasks,
        llm_env: Optional[Dict[str, str]] = None,
        auto_confirm: bool = True,
        tmux_session: Optional[str] = None,
    ) -> None:
        """Interactive mode: run tmux_cc in tmux session with file-based input.

        This mode:
        1. Creates a workspace directory with input files
        2. Starts Claude in tmux with a short prompt pointing to the files
        3. Waits for completion (output is handled by coding_agent.py)
        """
        flat_tasks = _flatten_nested(all_tasks)

        for task_idx, task in enumerate(flat_tasks):
            original_workspace_dir = task.workspace_dir
            prompt = task.prompt
            output_relative_dir_or_list = task.output_relative_dir
            todo_file_content_hint = task.todo_file_content_hint

            packed_workspace = pack_dir(original_workspace_dir)

            if isinstance(output_relative_dir_or_list, str):
                output_relative_dirs = [output_relative_dir_or_list]
            else:
                output_relative_dirs = output_relative_dir_or_list

            for output_relative_dir in output_relative_dirs:
                output_root = os.path.join(original_workspace_dir, output_relative_dir)
                todo_file_paths = _grep_by_file_content_hint(output_root, todo_file_content_hint)

                for file_idx, todo_file_path in enumerate(todo_file_paths):
                    # Create workspace with input files
                    task_workspace = _create_task_workspace(
                        task_idx=task_idx,
                        file_idx=file_idx,
                        original_workspace_dir=original_workspace_dir,
                        prompt=prompt,
                        packed_workspace=packed_workspace,
                        todo_file_path=todo_file_path,
                        todo_file_content_hint=todo_file_content_hint,
                    )

                    # Build file-based prompt
                    claude_prompt = _build_file_based_prompt(
                        task_workspace, todo_file_path, todo_file_content_hint
                    )

                    # Create unique session name
                    session_name = tmux_session or f"{TMUX_SESSION_PREFIX}_{task_idx}_{file_idx}"

                    print(f"\n[tmux_cc] Task {task_idx + 1}/{len(flat_tasks)}: {todo_file_path}", file=sys.stderr)
                    print(f"[tmux_cc] Workspace: {task_workspace}", file=sys.stderr)
                    print(f"[tmux_cc] tmux session: {session_name}", file=sys.stderr)
                    print(f"[tmux_cc] Use 'tmux attach -t {session_name}' to watch real-time output", file=sys.stderr)

                    # Kill existing session if any
                    _tmux_kill_session(session_name)

                    # Create new tmux session in task workspace directory
                    _tmux_create_session(session_name, task_workspace)

                    # Step 1: Start tmux_cc with reduced interactions
                    # IS_SANDBOX=1 skips some prompts, --permission-mode bypassPermissions skips permission checks
                    # --effort low for faster response (avoid extended thinking)
                    
                    # Get config from environment (set by setup_env_for_method from ~/.experience.json)
                    model = os.environ.get("ANTHROPIC_MODEL")
                    base_url = os.environ.get("ANTHROPIC_BASE_URL")
                    api_key = os.environ.get("ANTHROPIC_API_KEY")
                    
                    is_internal_api = base_url and ("comate" in base_url or "baidu-int" in base_url)
                    settings_backup = None
                    
                    if base_url and not is_internal_api:
                        # External API (e.g., Qianfan): patch settings.json to replace
                        # comate auth with qianfan api_key, then use env vars to override
                        settings_backup = _patch_settings_json(
                            settings_path=self.config.get_settings_path(),
                            base_url=base_url,
                            api_key=api_key,
                            model=model
                        )
                        tmux_cc_cmd = f'env -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_CUSTOM_HEADERS ANTHROPIC_BASE_URL="{base_url}" ANTHROPIC_API_KEY="{api_key}" {self.tmux_cc_bin} --permission-mode bypassPermissions --allowedTools "Read,Edit,Write" --effort low --setting-sources "" --settings "{{}}"'
                    elif base_url and is_internal_api:
                        # Internal API (comate): patch settings.json to change base_url, keep token auth
                        settings_backup = _patch_settings_json(
                            settings_path=self.config.get_settings_path(),
                            base_url=base_url,
                            model=model
                        )
                        tmux_cc_cmd = f'IS_SANDBOX=1 {self.tmux_cc_bin} --permission-mode bypassPermissions --allowedTools "Read,Edit,Write" --effort low'
                    else:
                        # No config: use settings.json defaults
                        tmux_cc_cmd = f'IS_SANDBOX=1 {self.tmux_cc_bin} --permission-mode bypassPermissions --allowedTools "Read,Edit,Write" --effort low'
                    
                    if model:
                        tmux_cc_cmd += f' --model "{model}"'
                    
                    _tmux_send_keys(session_name, tmux_cc_cmd, "Enter")

                    # Step 2: Wait for tmux_cc to start and show trust prompt
                    print(f"[tmux_cc] Waiting for tmux_cc to start...", file=sys.stderr)
                    time.sleep(3)

                    # Step 3: Auto-confirm trust folder if enabled
                    if auto_confirm:
                        # Wait for and handle the trust folder prompt
                        for _ in range(10):  # Try for up to 10 iterations
                            if _check_and_auto_confirm(session_name, AUTO_CONFIRM_PATTERNS):
                                time.sleep(1)
                                break
                            time.sleep(1)

                    # Step 4: Wait for tmux_cc to be ready for input (show the input prompt)
                    print(f"[tmux_cc] Waiting for tmux_cc to be ready...", file=sys.stderr)
                    time.sleep(2)

                    # Step 5: Send the short file-based prompt (much shorter than before!)
                    _tmux_send_keys(session_name, claude_prompt, "Enter")

                    # Step 6: Wait for tmux_cc to complete the task
                    print(f"[tmux_cc] Waiting for tmux_cc to process task...", file=sys.stderr)
                    _wait_for_tmux_cc_idle(
                        session_name,
                        auto_confirm=auto_confirm,
                    )

                    # Step 7: Verify output file has been written
                    # Wait a bit for any pending file I/O to complete
                    max_wait_for_file = 10  # seconds
                    wait_interval = 0.5
                    waited = 0
                    while waited < max_wait_for_file:
                        if os.path.exists(todo_file_path):
                            try:
                                with open(todo_file_path, "r", encoding="utf-8") as f:
                                    content = f.read()
                                if content.strip() != todo_file_content_hint:
                                    print(f"[tmux_cc] Output file verified: content changed from '{todo_file_content_hint}'", file=sys.stderr)
                                    break
                            except Exception:
                                pass
                        time.sleep(wait_interval)
                        waited += wait_interval
                    else:
                        print(f"[tmux_cc] WARNING: Output file still contains placeholder after {max_wait_for_file}s", file=sys.stderr)

                    print(f"[tmux_cc] Task completed, session '{session_name}' preserved for observation", file=sys.stderr)
                    
                    # Restore settings.json after task completes
                    _restore_settings_json(settings_backup)


if __name__ == "__main__":
    import tempfile

    print("Running TmuxCcTaskHandler tests...\n")

    # Test 1: _flatten_nested
    nested = [[1, 2], [3, [4, 5]]]
    assert _flatten_nested(nested) == [1, 2, 3, 4, 5]
    print("  ok: _flatten_nested works")

    # Test 2: Handler instantiation
    handler = TmuxCcTaskHandler()
    assert callable(handler)
    print("  ok: TmuxCcTaskHandler is callable")

    # Test 3: Workspace creation
    print("\n  Test 3: Workspace creation")
    test_workspace = _create_task_workspace(
        task_idx=0,
        file_idx=0,
        original_workspace_dir="/tmp/test",
        prompt="Test prompt",
        packed_workspace="Test packed content",
        todo_file_path="/tmp/test/output/file.txt",
        todo_file_content_hint="TODO",
    )
    assert os.path.exists(test_workspace)
    assert os.path.exists(os.path.join(test_workspace, "input", "prompt.txt"))
    assert os.path.exists(os.path.join(test_workspace, "input", "packed_workspace.txt"))
    assert os.path.exists(os.path.join(test_workspace, "input", "task_info.txt"))
    print(f"  ok: Workspace created at {test_workspace}")

    # Cleanup test workspace
    shutil.rmtree(test_workspace)
    print("  ok: Test workspace cleaned up")

    print("\nAll tests passed.")
    print(f"\nWorkspace root: {TMUX_CC_WORKSPACE_ROOT}")
