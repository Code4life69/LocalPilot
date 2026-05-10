from __future__ import annotations

import base64
import subprocess
import time
from pathlib import Path
from typing import Any

import requests


class OllamaClient:
    ROLE_NAMES = (
        "main",
        "coder",
        "coder_fallback",
        "vision",
        "router",
        "embedding",
        "quality_slow",
        "reasoning_slow",
        "general_fallback",
    )

    def __init__(
        self,
        host: str,
        timeout_seconds: int,
        model_profiles: dict[str, Any],
        default_role: str = "main",
        performance_profile: dict[str, Any] | None = None,
        performance_profile_name: str = "rtx3060_balanced",
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.default_role = default_role
        self.performance_profile = dict(performance_profile or {})
        self.performance_profile_name = performance_profile_name
        self.model_profiles = {
            key: value
            for key, value in model_profiles.items()
            if isinstance(value, dict) and "model" in value
        }
        self.active_models: dict[str, str | None] = {role: None for role in self.model_profiles}
        self.active_main_model: str | None = self.model_profiles.get("main", {}).get("model")
        self.active_vision_model: str | None = self.model_profiles.get("vision", {}).get("model")
        self.last_status = "unknown"
        self.last_role_used: str | None = None

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
            "Ollama is not running, so chat, coding fallback, vision, and embeddings are unavailable right now.",
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
                "3. Keep it running, then restart LocalPilot or try your message again.",
                "4. Pull the recommended models for this PC:",
            ]
        )
        for role in ("main", "coder", "vision", "router", "embedding"):
            profile = self.model_profiles.get(role, {})
            model_name = profile.get("model")
            if model_name:
                lines.append(f"   - ollama pull {model_name}")
            fallback = profile.get("fallback_model")
            if fallback:
                lines.append(f"   - ollama pull {fallback}")
        quality_model = self.model_profiles.get("quality_slow", {}).get("model")
        if quality_model:
            lines.append(f"5. Optional slow quality mode: ollama pull {quality_model}")
        return "\n".join(lines)

    def list_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            return [item.get("name", "") for item in data.get("models", []) if item.get("name")]
        except Exception:
            return []

    def resolve_models(self) -> dict[str, str | None]:
        available = self.list_models()
        self.active_models = {
            role: self.resolve_model_for_role(role, available)
            for role in self.model_profiles
        }
        self.active_main_model = self.active_models.get("main")
        self.active_vision_model = self.active_models.get("vision")
        return dict(self.active_models)

    def resolve_model_for_role(self, role: str, available: list[str] | None = None) -> str | None:
        profile = self.model_profiles.get(role, {})
        if not profile:
            return None
        available = self.list_models() if available is None else available
        preferred = profile.get("model")
        fallback = profile.get("fallback_model")
        preferred_match = self._find_installed_model_name(preferred, available)
        if preferred_match:
            return preferred_match
        fallback_match = self._find_installed_model_name(fallback, available)
        if fallback_match:
            return fallback_match
        return None

    def get_profile(self, role: str) -> dict[str, Any]:
        profile = dict(self.model_profiles.get(role, {}))
        ctx_key = self._performance_ctx_key(role)
        if ctx_key and ctx_key in self.performance_profile:
            profile["num_ctx"] = self.performance_profile[ctx_key]
        if "keep_alive" in self.performance_profile:
            profile["keep_alive"] = self.performance_profile["keep_alive"]
        return profile

    def is_model_installed(self, model_name: str, available: list[str] | None = None) -> bool | None:
        if available is None and not self.is_server_available():
            return None
        available = self.list_models() if available is None else available
        return self._find_installed_model_name(model_name, available) is not None

    def current_model_for_role(self, role: str) -> str | None:
        if not self.is_server_available():
            return None
        self.resolve_models()
        return self.active_models.get(role)

    def _find_installed_model_name(self, requested: str | None, available: list[str]) -> str | None:
        if not requested:
            return None
        candidates = [requested]
        if ":" not in requested:
            candidates.append(f"{requested}:latest")
        for candidate in candidates:
            if candidate in available:
                return candidate
        return None

    def _performance_ctx_key(self, role: str) -> str | None:
        mapping = {
            "main": "num_ctx_main",
            "coder": "num_ctx_coder",
            "coder_fallback": "num_ctx_coder",
            "vision": "num_ctx_vision",
        }
        return mapping.get(role)

    def _server_ready_message(self, prefix: str) -> str:
        role_notes = []
        for role in ("main", "coder", "vision"):
            selected = self.active_models.get(role)
            preferred = self.model_profiles.get(role, {}).get("model")
            if selected:
                if selected == preferred:
                    role_notes.append(f"{role}: {selected}")
                else:
                    role_notes.append(f"{role}: {selected} (fallback from {preferred})")
            elif preferred:
                role_notes.append(f"{role}: missing ({preferred})")
        return prefix + (" Active roles: " + "; ".join(role_notes) + "." if role_notes else "")

    def build_model_missing_message(self, role: str) -> str:
        profile = self.model_profiles.get(role, {})
        preferred = profile.get("model", "unknown")
        fallback = profile.get("fallback_model")
        available = self.list_models()
        installed = ", ".join(available) if available else "none detected"
        lines = [
            f"No usable model is installed for role `{role}`.",
            f"Preferred model: {preferred}",
        ]
        if fallback:
            lines.append(f"Fallback model: {fallback}")
        lines.extend(
            [
                f"Installed models: {installed}",
                "",
                f"Fix:",
                f"1. Run: ollama pull {preferred}",
            ]
        )
        if fallback:
            lines.append(f"2. Or run: ollama pull {fallback}")
        lines.append("3. Restart LocalPilot or try again.")
        return "\n".join(lines)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        return self.chat_with_role(self.default_role, system_prompt, user_prompt)

    def chat_with_role(self, role: str, system_prompt: str, user_text: str) -> str:
        if not self.is_server_available():
            return self.build_unavailable_message(auto_start_attempted=False)

        available = self.list_models()
        model_name = self.resolve_model_for_role(role, available)
        self.active_models[role] = model_name
        self.active_main_model = self.active_models.get("main")
        self.active_vision_model = self.active_models.get("vision")
        self.last_role_used = role

        if not model_name:
            return self.build_model_missing_message(role)

        profile = self.get_profile(role)
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
            "options": {
                "num_ctx": profile.get("num_ctx", 4096),
                "temperature": profile.get("temperature", 0.2),
            },
        }
        if profile.get("keep_alive"):
            payload["keep_alive"] = profile["keep_alive"]
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
            return f"Ollama chat request failed for role `{role}`: {exc}"

    def analyze_screenshot(self, prompt: str, image_path: str | Path) -> str:
        image_file = Path(image_path)
        if not image_file.exists():
            return f"Image not found: {image_file}"

        if not self.is_server_available():
            return self.build_unavailable_message(auto_start_attempted=False)

        available = self.list_models()
        model_name = self.resolve_model_for_role("vision", available)
        self.active_models["vision"] = model_name
        self.active_vision_model = model_name
        self.last_role_used = "vision"
        if not model_name:
            preferred = self.model_profiles.get("vision", {}).get("model", "vision model")
            return f"Vision model `{preferred}` is not installed yet.\nRun: ollama pull {preferred}"

        try:
            encoded = base64.b64encode(image_file.read_bytes()).decode("ascii")
            payload: dict[str, Any] = {
                "model": model_name,
                "prompt": prompt,
                "images": [encoded],
                "stream": False,
                "options": {
                    "num_ctx": self.model_profiles.get("vision", {}).get("num_ctx", 4096),
                    "temperature": self.model_profiles.get("vision", {}).get("temperature", 0.1),
                },
            }
            vision_profile = self.get_profile("vision")
            if vision_profile.get("keep_alive"):
                payload["keep_alive"] = vision_profile["keep_alive"]
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
                f"shape for {model_name}."
            )
        except Exception as exc:
            return (
                "Vision analysis placeholder reached. TODO: verify qwen2.5-vl multimodal Ollama "
                f"request handling. Error: {exc}"
            )

    def embed_text(self, text: str) -> dict[str, Any]:
        if not self.is_server_available():
            return {"ok": False, "error": self.build_unavailable_message(auto_start_attempted=False)}

        available = self.list_models()
        model_name = self.resolve_model_for_role("embedding", available)
        self.active_models["embedding"] = model_name
        self.last_role_used = "embedding"
        if not model_name:
            return {"ok": False, "error": self.build_model_missing_message("embedding")}

        payload = {"model": model_name, "input": text}
        try:
            response = requests.post(
                f"{self.host}/api/embed",
                json=payload,
                timeout=self.timeout_seconds,
            )
            if response.status_code == 404:
                response = requests.post(
                    f"{self.host}/api/embeddings",
                    json={"model": model_name, "prompt": text},
                    timeout=self.timeout_seconds,
                )
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embeddings") or data.get("embedding") or []
            if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
                embedding = embedding[0]
            return {"ok": True, "model": model_name, "embedding": embedding}
        except Exception as exc:
            return {"ok": False, "error": f"Ollama embedding request failed: {exc}"}

    def build_model_status_report(
        self,
        default_role: str = "main",
        performance_profile_name: str | None = None,
    ) -> str:
        reachable = self.is_server_available()
        available = self.list_models() if reachable else []
        self.resolve_models() if reachable else None

        lines = [
            "Model status",
            f"- Ollama reachable: {'yes' if reachable else 'no'}",
            f"- Default active role: {default_role}",
            f"- Performance profile: {performance_profile_name or self.performance_profile_name}",
        ]
        default_model = self.model_profiles.get(default_role, {}).get("model")
        if default_model == "qwen3:30b":
            lines.append("- Warning: qwen3:30b is selected as the default role model and may be slow on this PC.")

        lines.append("- Configured roles:")
        for role in self.ROLE_NAMES:
            profile = self.model_profiles.get(role)
            if not profile:
                continue
            preferred = profile.get("model", "n/a")
            fallback = profile.get("fallback_model")
            current = self.active_models.get(role) if reachable else None
            installed = self.is_model_installed(preferred, available) if reachable else None
            install_note = "unknown (Ollama unavailable)" if installed is None else ("installed" if installed else "missing")
            detail = f"  - {role}: preferred={preferred} [{install_note}]"
            if fallback:
                fallback_installed = self.is_model_installed(fallback, available)
                fallback_note = "installed" if fallback_installed else "missing"
                detail += f", fallback={fallback} [{fallback_note}]"
            if current:
                detail += f", current={current}"
            else:
                detail += ", current=none"
            lines.append(detail)
        return "\n".join(lines)

    def benchmark_model(
        self,
        model_name: str,
        prompt: str,
        num_ctx: int = 4096,
        temperature: float = 0.2,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        if not self.is_server_available():
            return {"ok": False, "error": self.build_unavailable_message(auto_start_attempted=False), "model": model_name}

        available = self.list_models()
        installed_name = self._find_installed_model_name(model_name, available)
        if not installed_name:
            return {"ok": False, "error": f"Model missing: {model_name}", "model": model_name}

        payload: dict[str, Any] = {
            "model": installed_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": num_ctx,
                "temperature": temperature,
            },
        }
        keep_alive = self.performance_profile.get("keep_alive")
        if keep_alive:
            payload["keep_alive"] = keep_alive
        if images:
            payload["images"] = images

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            eval_count = int(data.get("eval_count") or 0)
            eval_duration = int(data.get("eval_duration") or 0)
            load_duration = int(data.get("load_duration") or 0)
            tokens_per_second = None
            if eval_count > 0 and eval_duration > 0:
                tokens_per_second = eval_count / (eval_duration / 1_000_000_000)
            return {
                "ok": True,
                "model": installed_name,
                "prompt_eval_count": int(data.get("prompt_eval_count") or 0),
                "eval_count": eval_count,
                "eval_duration": eval_duration,
                "load_duration": load_duration,
                "total_duration": int(data.get("total_duration") or 0),
                "tokens_per_second": tokens_per_second,
            }
        except Exception as exc:
            return {"ok": False, "error": f"Benchmark failed for {installed_name}: {exc}", "model": installed_name}

    def build_model_benchmark_report(
        self,
        default_role: str = "main",
        performance_profile_name: str = "rtx3060_balanced",
    ) -> str:
        lines = [
            "Model benchmark",
            f"- Default active role: {default_role}",
            f"- Performance profile: {performance_profile_name}",
        ]
        if not self.is_server_available():
            lines.append("- Warning: Ollama is unavailable, so no benchmark could be run.")
            return "\n".join(lines)

        benchmark_targets = [
            ("main", self.model_profiles.get("main", {}).get("model"), "Say one short sentence about local AI."),
            ("coder", self.model_profiles.get("coder", {}).get("model"), "Write a tiny Python function that adds two numbers."),
            ("coder_fallback", self.model_profiles.get("coder_fallback", {}).get("model"), "Write a tiny Python function that adds two numbers."),
            ("router", self.model_profiles.get("router", {}).get("model"), "Classify this request as chat, code, research, desktop, or memory: show notes"),
        ]

        if self.model_profiles.get(default_role, {}).get("model") == "qwen3:30b":
            lines.append("- Warning: qwen3:30b is selected as the default role model and may be slow on this PC.")
        for role, profile in self.model_profiles.items():
            if role != "quality_slow" and profile.get("model") == "qwen3:30b":
                lines.append(f"- Warning: qwen3:30b is assigned to role `{role}` outside quality_slow.")

        for role, model_name, prompt in benchmark_targets:
            profile = self.get_profile("coder" if role == "coder_fallback" else role)
            result = self.benchmark_model(
                model_name=model_name or "",
                prompt=prompt,
                num_ctx=int(profile.get("num_ctx", 4096)),
                temperature=float(profile.get("temperature", 0.2)),
            )
            if not result.get("ok"):
                lines.append(f"- {role}: warning -> {result.get('error', 'unknown benchmark failure')}")
                continue

            load_seconds = result["load_duration"] / 1_000_000_000 if result["load_duration"] else 0
            tps = result.get("tokens_per_second")
            speed_note = ""
            if tps is not None:
                if tps < 15:
                    speed_note = " (slow)"
                elif tps < 30:
                    speed_note = " (moderate)"
            lines.append(
                f"- {role}: model={result['model']}, "
                f"tps={f'{tps:.2f}' if tps is not None else 'n/a'}{speed_note}, "
                f"load={load_seconds:.2f}s, "
                f"eval_tokens={result['eval_count']}"
            )
        return "\n".join(lines)
