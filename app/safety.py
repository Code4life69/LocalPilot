from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any


RISK_SAFE = "safe"
RISK_MEDIUM = "medium"
RISK_DANGEROUS = "dangerous"
RISK_BLOCKED = "blocked"


DANGEROUS_PATTERNS = [
    r"\bdel\b",
    r"\brmdir\b",
    r"\bformat\b",
    r"\bshutdown\b",
    r"\brestart\b",
    r"\bremove-item\b",
    r"\brd\s*/s\b",
    r"\bgit\s+push\s+--force\b",
]

ADMIN_PATTERNS = [
    r"\brunas\b",
    r"\bstart-process\b.+\brunas\b",
    r"\bset-executionpolicy\b",
    r"\breg\s+add\b",
    r"\bsc\s+config\b",
    r"\bnet\s+user\b",
    r"\bpowershell\b.+-verb\s+runas\b",
]

SENSITIVE_PATTERNS = [
    r"\bpassword\b",
    r"\bcredential\b",
    r"\btoken\b",
    r"\bcookie\b",
    r"\bemail\b",
    r"\bsend message\b",
    r"\bsubmit form\b",
    r"\bspend money\b",
    r"\bchange system settings\b",
]

BROWSER_DANGEROUS_PATTERNS = [
    r"\bsubmit\b",
    r"\blog in\b",
    r"\blogin\b",
    r"\bsign in\b",
    r"\bpassword\b",
    r"\bbuy\b",
    r"\bpurchase\b",
    r"\bcheckout\b",
    r"\bdownload\b",
    r"\binstall\b",
    r"\bsend\b",
    r"\bmessage\b",
]

BROAD_DESTRUCTIVE_PATTERNS = [
    r"\bdelete everything\b",
    r"\bwipe (?:the )?(?:folder|workspace|directory|drive)\b",
    r"\bremove all files\b",
    r"\berase (?:the )?(?:workspace|folder|directory|drive)\b",
    r"\bformat\b",
    r"\brmdir\s*/s\b",
    r"\bremove-item\b",
    r"\brm\s+-rf\b",
    r"\bdelete all files\b",
    r"\bdelete the workspace\b",
]


@dataclass(slots=True)
class SafetyDecision:
    risk_level: str
    approval_required: bool
    allowed: bool
    reason: str


class SafetyManager:
    def __init__(self, approval_callback=None, workspace_root: str | Path | None = None) -> None:
        self.approval_callback = approval_callback
        self.workspace_root = Path(workspace_root or Path("workspace")).resolve()

    def is_command_blocked(self, command: str) -> bool:
        lowered = command.lower()
        return any(re.search(pattern, lowered) for pattern in DANGEROUS_PATTERNS)

    def is_broad_destructive_request(self, text: str) -> bool:
        lowered = text.lower().strip()
        return any(re.search(pattern, lowered) for pattern in BROAD_DESTRUCTIVE_PATTERNS)

    def destructive_refusal_message(self, text: str) -> str:
        return (
            "I will not perform broad destructive actions like deleting everything in a folder. "
            "That request is blocked by safety because it could permanently remove large amounts of local data. "
            "If you need a safer next step, ask me to list the target folder first so you can review it."
        )

    def requires_write_confirmation(self, path: str | Path) -> bool:
        return Path(path).exists()

    def requires_move_confirmation(self, destination: str | Path) -> bool:
        return Path(destination).exists()

    def requires_shell_confirmation(self, command: str) -> bool:
        return True

    def requires_desktop_confirmation(self, action: str) -> bool:
        return action in {"click", "type_text", "hotkey", "move_mouse"}

    def is_path_within_workspace(self, path: str | Path) -> bool:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = (self.workspace_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        try:
            candidate.relative_to(self.workspace_root)
            return True
        except ValueError:
            return False

    def classify_command_risk(self, command: str) -> str:
        lowered = command.lower().strip()
        if self.is_command_blocked(lowered):
            return RISK_BLOCKED
        if any(re.search(pattern, lowered) for pattern in ADMIN_PATTERNS):
            return RISK_DANGEROUS
        return RISK_MEDIUM

    def classify_tool_call(self, tool_name: str, args: dict[str, Any] | None = None) -> SafetyDecision:
        payload = dict(args or {})
        path_value = payload.get("path") or payload.get("cwd") or payload.get("target")

        if tool_name in {"take_screenshot", "analyze_screenshot", "ask_user_approval", "list_checkpoints", "list_sessions", "read_session"}:
            return SafetyDecision(RISK_SAFE, False, True, "Read-only observation tool.")

        if tool_name == "restore_checkpoint":
            return SafetyDecision(RISK_DANGEROUS, True, True, "Restoring a checkpoint modifies files and requires approval.")

        if tool_name in {"list_files", "read_file"}:
            if path_value is None or self.is_path_within_workspace(path_value):
                return SafetyDecision(RISK_SAFE, False, True, "Read-only workspace access.")
            return SafetyDecision(RISK_MEDIUM, True, True, "Path is outside the allowed workspace.")

        if tool_name == "write_file":
            if path_value is None:
                return SafetyDecision(RISK_BLOCKED, False, False, "write_file requires a path.")
            if self.is_path_within_workspace(path_value):
                return SafetyDecision(RISK_MEDIUM, True, True, "Writing inside workspace requires approval.")
            return SafetyDecision(RISK_DANGEROUS, True, True, "Writing outside the allowed workspace requires approval.")

        if tool_name == "run_command":
            command = str(payload.get("command", ""))
            risk = self.classify_command_risk(command)
            if risk == RISK_BLOCKED:
                return SafetyDecision(risk, False, False, "Command is blocked by safety policy.")
            return SafetyDecision(risk, True, True, "Command execution requires approval.")

        if tool_name in {"browser_get_text", "browser_screenshot", "browser_get_page_info"}:
            return SafetyDecision(RISK_SAFE, False, True, "Read-only browser inspection tool.")

        if tool_name in {"browser_close"}:
            return SafetyDecision(RISK_SAFE, False, True, "Browser close is allowed.")

        if tool_name in {"browser_launch", "browser_goto", "browser_search"}:
            return SafetyDecision(RISK_MEDIUM, True, True, "Browser navigation requires approval.")

        if tool_name in {"browser_click_selector", "browser_type_selector", "browser_press_key"}:
            action_text = " ".join(str(value) for value in payload.values()).lower()
            if any(re.search(pattern, action_text) for pattern in BROWSER_DANGEROUS_PATTERNS):
                return SafetyDecision(RISK_DANGEROUS, True, True, "Potentially sensitive browser interaction requires approval.")
            return SafetyDecision(RISK_MEDIUM, True, True, "Browser interaction requires approval.")

        return SafetyDecision(RISK_BLOCKED, False, False, f"Unknown tool for safety classification: {tool_name}")

    def format_approval_request(self, prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, dict):
            tool_calls = prompt.get("tool_calls") or []
            is_plan = prompt.get("type") == "approval_plan"
            lines = ["Allow this browser plan?" if is_plan else "Approval requested."]
            lines.append(str(prompt.get("summary") or "Approval requested."))
            lines.append(f"Risk: {prompt.get('risk', 'unknown')}")
            if tool_calls and is_plan:
                lines.append("Planned tool calls:")
                for call in tool_calls:
                    lines.append(f"- {call.get('tool', 'unknown')} {call.get('args', {})}")
            elif tool_calls:
                first_call = tool_calls[0]
                lines.append(f"Tool: {first_call.get('tool', 'unknown')}")
                lines.append(f"Args: {first_call.get('args', {})}")
            affected_files = prompt.get("affected_files") or []
            if affected_files:
                lines.append("Affected files:")
                for path in affected_files:
                    lines.append(f"- {path}")
            if prompt.get("reason"):
                lines.append(f"Reason: {prompt['reason']}")
            return "\n".join(lines)
        return str(prompt)

    def confirm(self, prompt: Any) -> bool:
        if self.approval_callback is None:
            reply = input(f"{self.format_approval_request(prompt)}\nApprove? y/n: ").strip().lower()
            return reply == "y"
        return bool(self.approval_callback(prompt))
