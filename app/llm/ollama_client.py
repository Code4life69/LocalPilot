from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import requests


class OllamaClient:
    def __init__(self, host: str, timeout_seconds: int, main_model: str, vision_model: str) -> None:
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.main_model = main_model
        self.vision_model = vision_model

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.main_model,
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
            return f"Ollama chat unavailable: {exc}"

    def analyze_screenshot(self, prompt: str, image_path: str | Path) -> str:
        image_file = Path(image_path)
        if not image_file.exists():
            return f"Image not found: {image_file}"

        # TODO: Confirm the exact multimodal request shape supported by the installed Ollama build.
        try:
            encoded = base64.b64encode(image_file.read_bytes()).decode("ascii")
            payload: dict[str, Any] = {
                "model": self.vision_model,
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

