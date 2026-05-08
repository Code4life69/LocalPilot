from __future__ import annotations

import base64
import subprocess
import time
from pathlib import Path
from typing import Any

import requests


class OllamaClient:
    def __init__(self, host: str, timeout_seconds: int, main_model: str, vision_model: str) -> None:
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.main_model = main_model
        self.vision_model = vision_model
        self.active_main_model = main_model
        self.active_vision_model = vision_model
        self.last_status = "unknown"

    def is_server_available(self) -> bool:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=3)
            response.raise_for_status()
            self.last_status = "running"
            return True
        except Exception:
            self.last_status = "unavailable"
            return False

    def ensure_server(self, auto_start: bool = True, wait_seconds: int = 8) -> tuple[bool, str]:
        if self.is_server_available():
            self.resolve_models()
            return True, self._server_ready_message("Ollama server is running.")

        if not auto_start:
            return False, self.build_unavailable_message(auto_start_attempted=False)

        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except FileNotFoundError:
            self.last_status = "missing_cli"
            return False, "Ollama CLI is not installed or not on PATH."
        except Exception as exc:
            self.last_status = "start_failed"
            return False, f"Failed to start Ollama server automatically: {exc}"

        deadline = time.time() + max(wait_seconds, 1)
        while time.time() < deadline:
            if self.is_server_available():
                self.last_status = "started_by_localpilot"
                self.resolve_models()
                return True, self._server_ready_message("Ollama server started automatically.")
            time.sleep(1)

        self.last_status = "start_timeout"
        return False, self.build_unavailable_message(auto_start_attempted=True)

    def build_unavailable_message(self, auto_start_attempted: bool) -> str:
        lines = [
            "Ollama is not running, so chat and vision are unavailable right now.",
            "Start Ollama and then try again.",
        ]
        if auto_start_attempted:
            lines.append("LocalPilot tried to start `ollama serve` automatically, but the API still did not come up.")
        lines.extend(
            [
                "",
                "Fix:",
                "1. Open PowerShell.",
                "2. Run: ollama serve",
                f"3. Keep it running, then restart LocalPilot or try your message again.",
                f"4. If needed, pull the models: ollama pull {self.main_model} and ollama pull {self.vision_model}",
            ]
        )
        return "\n".join(lines)

    def list_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            return [item.get("name", "") for item in data.get("models", []) if item.get("name")]
        except Exception:
            return []

    def resolve_models(self) -> tuple[str | None, str | None]:
        available = self.list_models()
        self.active_main_model = self._resolve_text_model(available)
        self.active_vision_model = self.vision_model if self.vision_model in available else None
        return self.active_main_model, self.active_vision_model

    def _resolve_text_model(self, available: list[str]) -> str | None:
        if self.main_model in available:
            return self.main_model
        for candidate in ("qwen3:8b", "qwen3:4b", "qwen3:14b"):
            if candidate in available:
                return candidate
        for name in available:
            if "embed" not in name.lower():
                return name
        return None

    def _server_ready_message(self, prefix: str) -> str:
        model_note = f" Active text model: {self.active_main_model or 'none'}."
        if self.active_main_model != self.main_model:
            model_note += (
                f" Preferred model `{self.main_model}` is not installed; using fallback "
                f"`{self.active_main_model}`."
            )
        if self.active_vision_model != self.vision_model:
            model_note += f" Vision model `{self.vision_model}` is not installed yet."
        return prefix + model_note

    def build_model_missing_message(self) -> str:
        available = self.list_models()
        installed = ", ".join(available) if available else "none detected"
        return (
            f"No usable text model is installed for LocalPilot.\n"
            f"Preferred model: {self.main_model}\n"
            f"Installed models: {installed}\n\n"
            f"Fix:\n"
            f"1. Run: ollama pull {self.main_model}\n"
            f"2. Restart LocalPilot or try again.\n"
        )

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        if not self.is_server_available():
            return self.build_unavailable_message(auto_start_attempted=False)
        self.resolve_models()
        if not self.active_main_model:
            return self.build_model_missing_message()

        payload = {
            "model": self.active_main_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "").strip() or "Ollama returned an empty response."
        except Exception as exc:
            return f"Ollama chat request failed: {exc}"

    def analyze_screenshot(self, prompt: str, image_path: str | Path) -> str:
        image_file = Path(image_path)
        if not image_file.exists():
            return f"Image not found: {image_file}"

        if not self.is_server_available():
            return self.build_unavailable_message(auto_start_attempted=False)
        self.resolve_models()
        if not self.active_vision_model:
            return (
                f"Vision model `{self.vision_model}` is not installed yet.\n"
                f"Run: ollama pull {self.vision_model}"
            )

        # TODO: Confirm the exact multimodal request shape supported by the installed Ollama build.
        try:
            encoded = base64.b64encode(image_file.read_bytes()).decode("ascii")
            payload: dict[str, Any] = {
                "model": self.active_vision_model,
                "prompt": prompt,
                "images": [encoded],
                "stream": False,
            }
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "").strip()
            if text:
                return text
            return (
                "Vision call returned no text. TODO: verify the installed Ollama multimodal API "
                "shape for qwen2.5-vl:7b."
            )
        except Exception as exc:
            return (
                "Vision analysis placeholder reached. TODO: verify qwen2.5-vl multimodal Ollama "
                f"request handling. Error: {exc}"
            )
