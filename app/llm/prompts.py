from __future__ import annotations

import json
from typing import Any


def build_system_prompt(capabilities: dict[str, Any]) -> str:
    summary = json.dumps(capabilities, indent=2)
    return (
        "You are LocalPilot, a local Windows AI assistant.\n"
        "Operate in explicit modes: chat, code, research, desktop, memory.\n"
        "Use qwen2.5-coder for reasoning and coding.\n"
        "Use vision only for screenshot analysis when UI Automation is insufficient.\n"
        "Python tools are the hands; do not pretend to execute tools you do not actually have.\n"
        "Never rewrite core logic permanently without user approval.\n"
        "Respect approval requirements for shell commands, file overwrites, and desktop actions.\n"
        "Canonical capability manifest:\n"
        f"{summary}"
    )

