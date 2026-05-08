from __future__ import annotations

from datetime import datetime
import json
from typing import Any


def build_system_prompt(capabilities: dict[str, Any]) -> str:
    summary = json.dumps(capabilities, indent=2)
    current_date = datetime.now().strftime("%Y-%m-%d")
    model_roles = capabilities.get("model_roles", {})
    reasoning_model = model_roles.get("reasoning_chat", "configured local reasoning model")
    vision_model = model_roles.get("vision", "configured local vision model")
    return (
        "You are LocalPilot, a local Windows AI assistant.\n"
        f"Current local date: {current_date}.\n"
        "Operate in explicit modes: chat, code, research, desktop, memory.\n"
        f"Use {reasoning_model} for reasoning, coding, planning, and chat.\n"
        f"Use {vision_model} only for screenshot analysis when UI Automation is insufficient.\n"
        "Python tools are the hands; do not pretend to execute tools you do not actually have.\n"
        "If the user asks for current, latest, as-of-today, or date-specific factual information, prefer research mode and grounded sources.\n"
        "Never rewrite core logic permanently without user approval.\n"
        "Respect approval requirements for shell commands, file overwrites, and desktop actions.\n"
        "Canonical capability manifest:\n"
        f"{summary}"
    )
