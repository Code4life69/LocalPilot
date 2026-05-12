from __future__ import annotations

import re
from pathlib import Path


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


class SafetyManager:
    def __init__(self, approval_callback=None) -> None:
        self.approval_callback = approval_callback

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

    def confirm(self, prompt: str) -> bool:
        if self.approval_callback is None:
            reply = input(f"{prompt} Approve? y/n: ").strip().lower()
            return reply == "y"
        return bool(self.approval_callback(prompt))
