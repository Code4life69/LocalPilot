from __future__ import annotations

from typing import Any


class AgentMode:
    def __init__(self, app) -> None:
        self.app = app

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self.app.agent.run_task(request["user_text"].strip())
        except Exception as exc:
            return {
                "ok": False,
                "status": "error",
                "message": "",
                "error": str(exc),
                "transcript": [],
                "steps": [],
                "brain_model": self.app.lmstudio.default_text_model,
                "vision_model": self.app.lmstudio.default_vision_model,
                "browser_backend": "Puppeteer",
            }
        payload: dict[str, Any] = {
            "ok": bool(result.get("ok")),
            "status": result.get("status", "error"),
            "message": result.get("message", ""),
            "transcript": result.get("transcript", []),
            "steps": result.get("steps", []),
            "brain_model": self.app.lmstudio.default_text_model,
            "vision_model": self.app.lmstudio.default_vision_model,
            "browser_backend": "Puppeteer",
        }
        if result.get("error"):
            payload["error"] = result["error"]
        if result.get("session_path"):
            payload["session_path"] = result["session_path"]
        if result.get("task_id"):
            payload["task_id"] = result["task_id"]
        return payload
