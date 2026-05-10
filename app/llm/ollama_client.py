from __future__ import annotations

import base64
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import requests


class OllamaClient:
    REQUIRED_RECOMMENDED_ROLES = (
        "main",
        "coder",
        "coder_fallback",
        "vision",
        "router",
        "embedding",
    )
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
        lifecycle_settings: dict[str, Any] | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.default_role = default_role
        self.performance_profile = dict(performance_profile or {})
        self.performance_profile_name = performance_profile_name
        self.lifecycle_settings = self._normalize_lifecycle_settings(lifecycle_settings or {})
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
        self.last_heavy_role_used: str | None = None

    def _normalize_lifecycle_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        normalized = {
            "enabled": True,
            "unload_previous_heavy_role": True,
            "keep_lightweight_roles_loaded": True,
            "heavy_roles": ["main", "coder", "vision", "quality_slow"],
        }
        normalized.update(settings)
        normalized["heavy_roles"] = list(normalized.get("heavy_roles") or [])
        return normalized

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

    def detect_model_directory(self) -> tuple[str, str]:
        env_dir = os.environ.get("OLLAMA_MODELS")
        if env_dir:
            return env_dir, "OLLAMA_MODELS"

        candidates: list[Path] = []
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            candidates.append(Path(local_appdata) / "Ollama" / "models")
        candidates.append(Path.home() / ".ollama" / "models")

        for candidate in candidates:
            if candidate.exists():
                return str(candidate), "default"
        return str(candidates[0] if candidates else (Path.home() / ".ollama" / "models")), "default (not found)"

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

    def lifecycle_enabled(self) -> bool:
        return bool(self.lifecycle_settings.get("enabled", True))

    def is_heavy_role(self, role: str | None) -> bool:
        return bool(role and role in set(self.lifecycle_settings.get("heavy_roles", [])))

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

    def find_similar_installed_models(self, requested: str | None, available: list[str]) -> list[str]:
        if not requested:
            return []
        exact = self._find_installed_model_name(requested, available)
        requested_family = self._model_family(requested)
        requested_tag = self._model_tag(requested)
        ranked: list[tuple[int, str]] = []

        for installed in available:
            if installed == exact:
                continue
            score = 0
            installed_family = self._model_family(installed)
            installed_tag = self._model_tag(installed)

            if installed_family == requested_family:
                score += 5
            elif requested_family and (requested_family in installed_family or installed_family in requested_family):
                score += 2

            if requested_tag and installed_tag:
                requested_prefix = requested_tag.split("-")[0]
                installed_prefix = installed_tag.split("-")[0]
                if requested_prefix == installed_prefix:
                    score += 3
                requested_tokens = {token for token in re.split(r"[-_\.]+", requested_tag) if token}
                installed_tokens = {token for token in re.split(r"[-_\.]+", installed_tag) if token}
                score += min(len(requested_tokens & installed_tokens), 2)

            if score > 0:
                ranked.append((score, installed))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [item[1] for item in ranked[:4]]

    def suggested_temporary_fallback(self, role: str, available: list[str]) -> str | None:
        fallback_map = {
            "main": ["llama3.1:8b"],
        }
        for candidate in fallback_map.get(role, []):
            match = self._find_installed_model_name(candidate, available)
            if match:
                return match
        return None

    def get_loaded_models(self) -> list[dict[str, Any]]:
        if not self.is_server_available():
            return []

        try:
            response = requests.get(f"{self.host}/api/ps", timeout=5)
            if response.ok:
                data = response.json()
                models = []
                for item in data.get("models", []):
                    name = item.get("name") or item.get("model")
                    if name:
                        models.append({"name": name, "details": item})
                return models
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["ollama", "ps"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode != 0:
                return []
            models: list[dict[str, Any]] = []
            lines = [line.rstrip() for line in result.stdout.splitlines() if line.strip()]
            for line in lines[1:]:
                parts = re.split(r"\s{2,}", line.strip())
                if parts:
                    models.append({"name": parts[0], "raw": line.strip()})
            return models
        except Exception:
            return []

    def unload_model(self, model_name: str) -> dict[str, Any]:
        if not model_name:
            return {"ok": False, "error": "No model name was provided for unload."}
        if not self.is_server_available():
            return {"ok": False, "error": self.build_unavailable_message(auto_start_attempted=False), "model": model_name}

        loaded_before = {item.get("name") for item in self.get_loaded_models()}
        if model_name not in loaded_before:
            return {"ok": True, "model": model_name, "detail": "Model was not loaded.", "method": "noop"}

        payload = {
            "model": model_name,
            "prompt": "release",
            "stream": False,
            "keep_alive": 0,
            "options": {
                "num_ctx": 32,
                "temperature": 0,
            },
        }
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=min(self.timeout_seconds, 20),
            )
            response.raise_for_status()
            time.sleep(0.5)
            loaded_after_api = {item.get("name") for item in self.get_loaded_models()}
            if model_name not in loaded_after_api:
                return {"ok": True, "model": model_name, "detail": "Unloaded via Ollama API.", "method": "api"}
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["ollama", "stop", model_name],
                capture_output=True,
                text=True,
                timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            loaded_after_cli = {item.get("name") for item in self.get_loaded_models()}
            if model_name not in loaded_after_cli:
                return {"ok": True, "model": model_name, "detail": "Unloaded via ollama stop.", "method": "cli"}
            error_text = (result.stderr or result.stdout or "unknown unload failure").strip()
            return {"ok": False, "model": model_name, "error": error_text}
        except Exception as exc:
            return {"ok": False, "model": model_name, "error": f"Unload failed: {exc}"}

    def unload_role(self, role: str) -> dict[str, Any]:
        if not self.is_server_available():
            return {"ok": False, "error": self.build_unavailable_message(auto_start_attempted=False), "role": role}
        available = self.list_models()
        model_name = self.resolve_model_for_role(role, available)
        if not model_name:
            return {"ok": False, "error": self.build_model_missing_message(role), "role": role}
        result = self.unload_model(model_name)
        result["role"] = role
        if result.get("ok") and self.last_heavy_role_used == role:
            self.last_heavy_role_used = None
        return result

    def unload_all_non_current_models(self, current_role: str | None) -> dict[str, Any]:
        if not self.is_server_available():
            return {"ok": False, "error": self.build_unavailable_message(auto_start_attempted=False)}

        available = self.list_models()
        loaded_entries = self.get_loaded_models()
        managed_models = self._known_localpilot_model_names(available)
        keep_models = self._models_to_keep(current_role, available)
        unloaded: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for entry in loaded_entries:
            model_name = entry.get("name")
            if not model_name or model_name not in managed_models:
                continue
            if model_name in keep_models:
                continue
            result = self.unload_model(model_name)
            if result.get("ok"):
                unloaded.append(result)
            else:
                errors.append(result)

        return {
            "ok": not errors,
            "current_role": current_role,
            "unloaded": unloaded,
            "errors": errors,
            "loaded_after": self.get_loaded_models(),
        }

    def warm_role(self, role: str) -> dict[str, Any]:
        if not self.is_server_available():
            return {"ok": False, "error": self.build_unavailable_message(auto_start_attempted=False), "role": role}

        available = self.list_models()
        model_name = self.resolve_model_for_role(role, available)
        if not model_name:
            return {"ok": False, "error": self.build_model_missing_message(role), "role": role}

        if self.is_heavy_role(role):
            self._prepare_role_activation(role)

        profile = self.get_profile(role)
        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": "Reply with OK.",
            "stream": False,
            "options": {
                "num_ctx": min(int(profile.get("num_ctx", 4096)), 256),
                "temperature": 0,
            },
        }
        if profile.get("keep_alive"):
            payload["keep_alive"] = profile["keep_alive"]
        if role == "vision":
            payload["images"] = [self._tiny_png_base64()]

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            self.active_models[role] = model_name
            self.last_role_used = role
            if self.is_heavy_role(role):
                self.last_heavy_role_used = role
            return {"ok": True, "role": role, "model": model_name, "detail": "Warmed successfully."}
        except Exception as exc:
            return {"ok": False, "role": role, "model": model_name, "error": f"Warmup failed: {exc}"}

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

    def _model_family(self, model_name: str | None) -> str:
        if not model_name:
            return ""
        return model_name.split(":", 1)[0].lower()

    def _model_tag(self, model_name: str | None) -> str:
        if not model_name or ":" not in model_name:
            return ""
        return model_name.split(":", 1)[1].lower()

    def _performance_ctx_key(self, role: str) -> str | None:
        mapping = {
            "main": "num_ctx_main",
            "coder": "num_ctx_coder",
            "coder_fallback": "num_ctx_coder",
            "vision": "num_ctx_vision",
        }
        return mapping.get(role)

    def _tiny_png_base64(self) -> str:
        return (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7ZxY4AAAAASUVORK5CYII="
        )

    def _known_localpilot_model_names(self, available: list[str]) -> set[str]:
        names: set[str] = set()
        for profile in self.model_profiles.values():
            preferred = self._find_installed_model_name(profile.get("model"), available)
            fallback = self._find_installed_model_name(profile.get("fallback_model"), available)
            if preferred:
                names.add(preferred)
            if fallback:
                names.add(fallback)
        return names

    def _models_to_keep(self, current_role: str | None, available: list[str]) -> set[str]:
        keep_models: set[str] = set()
        if current_role:
            current_model = self.resolve_model_for_role(current_role, available)
            if current_model:
                keep_models.add(current_model)
        if current_role and self.lifecycle_settings.get("keep_lightweight_roles_loaded", True):
            for role in self.model_profiles:
                if role == current_role or self.is_heavy_role(role):
                    continue
                model_name = self.resolve_model_for_role(role, available)
                if model_name:
                    keep_models.add(model_name)
        return keep_models

    def _prepare_role_activation(self, role: str) -> None:
        if not self.lifecycle_enabled():
            return
        if not self.is_heavy_role(role):
            return
        if not self.lifecycle_settings.get("unload_previous_heavy_role", True):
            return
        if self.last_heavy_role_used and self.last_heavy_role_used != role:
            self.unload_all_non_current_models(role)

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

        self._prepare_role_activation(role)
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
            if self.is_heavy_role(role):
                self.last_heavy_role_used = role
            return data.get("message", {}).get("content", "").strip() or "Ollama returned an empty response."
        except Exception as exc:
            return f"Ollama chat request failed for role `{role}`: {exc}"

    def analyze_screenshot(self, prompt: str, image_path: str | Path) -> str:
        image_file = Path(image_path)
        if not image_file.exists():
            return f"Image not found: {image_file}"

        if not self.is_server_available():
            return self.build_unavailable_message(auto_start_attempted=False)

        self._prepare_role_activation("vision")
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
            self.last_heavy_role_used = "vision"
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
            temp_fallback = None
            if reachable and not current:
                temp_fallback = self.suggested_temporary_fallback(role, available)
            if temp_fallback:
                lines.append(f"    possible temporary fallback available: {temp_fallback}")
        lifecycle = self.lifecycle_settings
        lines.append("- Lifecycle:")
        lines.append(f"  - enabled: {'yes' if lifecycle.get('enabled', True) else 'no'}")
        lines.append(
            f"  - unload_previous_heavy_role: {'yes' if lifecycle.get('unload_previous_heavy_role', True) else 'no'}"
        )
        lines.append(
            f"  - keep_lightweight_roles_loaded: {'yes' if lifecycle.get('keep_lightweight_roles_loaded', True) else 'no'}"
        )
        lines.append(f"  - heavy_roles: {', '.join(lifecycle.get('heavy_roles', []))}")
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
            self._prepare_role_activation("coder" if role == "coder_fallback" else role)
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
            if self.is_heavy_role("coder" if role == "coder_fallback" else role):
                self.last_heavy_role_used = "coder" if role == "coder_fallback" else role
        if self.lifecycle_enabled() and self.is_heavy_role(default_role):
            self.unload_all_non_current_models(default_role)
        return "\n".join(lines)

    def build_model_doctor_report(
        self,
        default_role: str = "main",
        performance_profile_name: str | None = None,
    ) -> str:
        reachable = self.is_server_available()
        available = self.list_models() if reachable else []
        self.resolve_models() if reachable else None
        model_dir, model_dir_source = self.detect_model_directory()
        ollama_models_env = os.environ.get("OLLAMA_MODELS") or "not set"

        lines = [
            "Model doctor",
            f"- Ollama reachable: {'yes' if reachable else 'no'}",
            f"- Ollama model directory: {model_dir} ({model_dir_source})",
            f"- OLLAMA_MODELS: {ollama_models_env}",
            f"- Default active role: {default_role}",
            f"- Performance profile: {performance_profile_name or self.performance_profile_name}",
        ]

        if not reachable:
            lines.append("- Installed models: unknown because Ollama is unavailable.")
            lines.append("- Recommended repair commands:")
            for command in self.recommended_repair_commands([]):
                lines.append(f"  {command}")
            return "\n".join(lines)

        lines.append("- Installed models:")
        if available:
            for model in available:
                lines.append(f"  - {model}")
        else:
            lines.append("  - none reported by `ollama list`")

        if ollama_models_env != "not set" and not Path(model_dir).exists():
            lines.append("- Warning: OLLAMA_MODELS is set, but the directory does not exist.")
        elif not available:
            lines.append("- Warning: Ollama is reachable but returned no installed models. Check OLLAMA_MODELS and your model path.")

        lines.append("- Configured roles:")
        missing_models: list[str] = []
        for role in self.ROLE_NAMES:
            profile = self.model_profiles.get(role)
            if not profile:
                continue
            preferred = profile.get("model")
            fallback = profile.get("fallback_model")
            exact = self._find_installed_model_name(preferred, available)
            current = self.active_models.get(role)
            status = "installed" if exact else "missing"
            detail = f"  - {role}: preferred={preferred} [{status}]"
            if current:
                detail += f", current={current}"
            else:
                detail += ", current=none"
            if fallback:
                fallback_exact = self._find_installed_model_name(fallback, available)
                detail += f", fallback={fallback} [{'installed' if fallback_exact else 'missing'}]"
            lines.append(detail)

            if not exact and preferred:
                missing_models.append(preferred)
                similar = self.find_similar_installed_models(preferred, available)
                if similar:
                    lines.append(f"    similar installed models: {', '.join(similar)}")
                    lines.append("    Similar model found, but exact configured tag is missing.")
                temp_fallback = self.suggested_temporary_fallback(role, available)
                if temp_fallback:
                    lines.append(f"    possible temporary fallback available: {temp_fallback}")

        lines.append("- Missing configured models:")
        if missing_models:
            for model in missing_models:
                lines.append(f"  - {model}")
        else:
            lines.append("  - none")

        lines.append("- Recommended repair commands:")
        for command in self.recommended_repair_commands(available):
            lines.append(f"  {command}")
        return "\n".join(lines)

    def recommended_repair_commands(self, available: list[str]) -> list[str]:
        commands: list[str] = []
        for role in self.REQUIRED_RECOMMENDED_ROLES:
            profile = self.model_profiles.get(role, {})
            model_name = profile.get("model")
            if not model_name:
                continue
            status = "already installed" if self._find_installed_model_name(model_name, available) else "missing"
            commands.append(f"ollama pull {model_name}  # {status}")
        quality_model = self.model_profiles.get("quality_slow", {}).get("model")
        if quality_model:
            status = "already installed" if self._find_installed_model_name(quality_model, available) else "optional"
            commands.append(f"ollama pull {quality_model}  # {status}")
        return commands

    def build_model_repair_plan(self) -> str:
        reachable = self.is_server_available()
        available = self.list_models() if reachable else []
        lines = [
            "Model repair plan",
            "- This plan does not pull automatically.",
            "- Run these commands in PowerShell:",
        ]
        for command in self.recommended_repair_commands(available):
            lines.append(f"  {command}")
        lines.extend(
            [
                "- After pulling:",
                "  1. Fully quit Ollama.",
                "  2. Start Ollama again.",
                "  3. Run: powershell -ExecutionPolicy Bypass -File scripts/check_models.ps1",
            ]
        )
        return "\n".join(lines)

    def build_model_unload_report(self) -> str:
        lines = ["Model unload"]
        if not self.is_server_available():
            lines.append("- Warning: Ollama is unavailable, so no unload could be performed.")
            return "\n".join(lines)

        result = self.unload_all_non_current_models(None)
        unloaded = result.get("unloaded", [])
        errors = result.get("errors", [])
        if unloaded:
            for item in unloaded:
                lines.append(f"- Unloaded: {item.get('model')} via {item.get('method', 'unknown')}")
        else:
            lines.append("- No loaded LocalPilot models needed unloading.")
        for item in errors:
            lines.append(f"- Warning: could not unload {item.get('model', 'unknown')}: {item.get('error', 'unknown error')}")

        loaded_after = result.get("loaded_after", [])
        lines.append("- Loaded models now:")
        if loaded_after:
            for item in loaded_after:
                lines.append(f"  - {item.get('name')}")
        else:
            lines.append("  - none")
        return "\n".join(lines)

    def build_model_warmup_report(self, roles: tuple[str, ...] = ("router", "main")) -> str:
        lines = ["Model warmup"]
        if not self.is_server_available():
            lines.append("- Warning: Ollama is unavailable, so no warmup could be run.")
            return "\n".join(lines)

        for role in roles:
            result = self.warm_role(role)
            if result.get("ok"):
                lines.append(f"- Warmed {role}: {result.get('model')}")
            else:
                lines.append(f"- Warning: {role} warmup failed -> {result.get('error', 'unknown error')}")

        loaded_after = self.get_loaded_models()
        lines.append("- Loaded models now:")
        if loaded_after:
            for item in loaded_after:
                lines.append(f"  - {item.get('name')}")
        else:
            lines.append("  - none")
        return "\n".join(lines)
