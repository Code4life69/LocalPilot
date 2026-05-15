from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

import requests


class LMStudioClient:
    def __init__(
        self,
        host: str = "http://localhost:1234/v1",
        timeout_seconds: int = 90,
        default_text_model: str = "qwen2.5-coder-14b-instruct",
        default_vision_model: str = "qwen3-vl-8b-instruct",
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.default_text_model = default_text_model
        self.default_vision_model = default_vision_model

    def is_server_available(self) -> bool:
        try:
            response = requests.get(f"{self.host}/models", timeout=min(self.timeout_seconds, 5))
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def encode_image_as_data_url(self, image_path: str | Path) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type:
            mime_type = "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _chat_completions_url(self) -> str:
        return f"{self.host}/chat/completions"

    def _build_text_payload(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        return {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

    def _build_vision_payload(
        self,
        prompt: str,
        image_path: str | Path,
        model: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        return {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self.encode_image_as_data_url(image_path),
                            },
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
        }

    def _extract_text(self, data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("LM Studio returned no choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            text = content.strip()
            if text:
                return text
        raise RuntimeError("LM Studio returned an empty message.")

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = requests.post(
                self._chat_completions_url(),
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise RuntimeError(f"LM Studio request timed out after {self.timeout_seconds} seconds.") from exc
        except requests.RequestException as exc:
            body = ""
            response = getattr(exc, "response", None)
            if response is not None:
                try:
                    body = response.text.strip()
                except Exception:
                    body = ""
            detail = f"LM Studio request failed: {exc}"
            if body:
                detail = f"{detail} | response: {body}"
            raise RuntimeError(detail) from exc

        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("LM Studio returned invalid JSON.") from exc

    def chat_text(
        self,
        messages: list[dict[str, Any]],
        model: str = "qwen2.5-coder-14b-instruct",
        max_tokens: int = 2048,
    ) -> str:
        payload = self._build_text_payload(
            messages=messages,
            model=model or self.default_text_model,
            max_tokens=max_tokens,
        )
        data = self._post_chat(payload)
        return self._extract_text(data)

    def chat_vision(
        self,
        prompt: str,
        image_path: str | Path,
        model: str = "qwen3-vl-8b-instruct",
        max_tokens: int = 2048,
    ) -> str:
        payload = self._build_vision_payload(
            prompt=prompt,
            image_path=image_path,
            model=model or self.default_vision_model,
            max_tokens=max_tokens,
        )
        data = self._post_chat(payload)
        return self._extract_text(data)
