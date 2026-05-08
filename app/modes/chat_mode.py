from __future__ import annotations

from pathlib import Path


class ChatMode:
    def __init__(self, app) -> None:
        self.app = app

    def handle(self, request: dict) -> dict:
        text = request["user_text"].strip()
        lowered = text.lower()
        if lowered in {"help", "/help"}:
            caps = self.app.capabilities
            return {
                "ok": True,
                "message": (
                    f"{caps['name']} modes: {', '.join(caps['modes'])}\n"
                    "Ask me to read/write files, run commands, search the web, save notes, or inspect the desktop."
                ),
            }
        if "what can you do" in lowered or "what do you do" in lowered:
            return {
                "ok": True,
                "message": self.app.describe_capabilities(),
            }
        if lowered == "trust checklist":
            return {
                "ok": True,
                "message": self._load_trust_checklist(),
            }
        response = self.app.ollama.chat(self.app.system_prompt, text)
        return {"ok": True, "message": response}

    def _load_trust_checklist(self) -> str:
        path = Path(self.app.root_dir) / "docs" / "TRUST_GAUNTLET.md"
        if not path.exists():
            return "Trust checklist is missing. Expected docs/TRUST_GAUNTLET.md."
        return path.read_text(encoding="utf-8")
