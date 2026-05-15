from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sys
import threading
import tkinter as tk
import atexit
import time
import uuid
from pathlib import Path
from tkinter import scrolledtext, ttk
from typing import Any

from app.agent import LocalPilotAgent
from app.git_sync import GitSyncManager
from app.lmstudio_client import LMStudioClient
from app.llm.ollama_client import OllamaClient
from app.llm.prompts import build_system_prompt
from app.logger import AppLogger
from app.memory import MemoryStore
from app.modes.chat_mode import ChatMode
from app.modes.code_mode import CodeMode
from app.modes.desktop_mode import DesktopMode
from app.modes.research_mode import ResearchMode
from app.router import KeywordRouter
from app.safety import SafetyManager
from app.system_doctor import build_system_doctor_report
from app.task_state import TaskStateStore
from app.tools.desktop_lessons import DesktopLessonStore
from app.tools.screen import take_screenshot
from app.tools.test_runner import TestRunner
from app.tool_registry import ToolRegistry


def load_settings(root_dir: str | Path) -> dict[str, Any]:
    path = Path(root_dir) / "config" / "settings.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_lmstudio_vision_test(root_dir: str | Path) -> tuple[int, str]:
    root_path = Path(root_dir)
    settings = load_settings(root_path)
    lmstudio_settings = settings.get("lmstudio", {})
    host = lmstudio_settings.get("host", "http://localhost:1234/v1")
    model = lmstudio_settings.get("vision_model", "qwen3-vl-8b-instruct")
    screenshot_dir = root_path / lmstudio_settings.get("screenshot_dir", "logs/screenshots")
    client = LMStudioClient(
        host=host,
        timeout_seconds=int(lmstudio_settings.get("timeout_seconds", 90)),
        default_text_model=lmstudio_settings.get("text_model", "qwen2.5-coder-14b-instruct"),
        default_vision_model=model,
    )

    lines = [
        "LM Studio screenshot vision test",
        f"- LM Studio URL: {host}",
        f"- model used: {model}",
    ]

    if not client.is_server_available():
        lines.append(f"LM Studio is not reachable at {host}")
        lines.append(f"Start LM Studio server and load {model}.")
        return 2, "\n".join(lines)

    screenshot = take_screenshot(screenshot_dir)
    if not screenshot.get("ok"):
        lines.append(f"Screenshot failed: {screenshot.get('error', 'unknown error')}")
        return 1, "\n".join(lines)

    screenshot_path = str(screenshot["path"])
    prompt = "Describe this screenshot in one sentence and mention any obvious visible text."
    lines.append(f"- screenshot path: {screenshot_path}")
    lines.append(f"- prompt: {prompt}")

    try:
        description = client.chat_vision(
            prompt=prompt,
            image_path=screenshot_path,
            model=model,
            max_tokens=512,
        )
    except Exception as exc:
        lines.append(f"LM Studio vision failed: {exc}")
        return 1, "\n".join(lines)

    lines.append(f"- vision response: {description}")
    return 0, "\n".join(lines)


def run_agent_cli(root_dir: str | Path) -> int:
    root_path = Path(root_dir)
    settings = load_settings(root_path)
    logger = AppLogger(root_path / settings.get("logs_dir", "logs"))
    lmstudio_settings = settings.get("lmstudio", {})
    lmstudio_client = LMStudioClient(
        host=lmstudio_settings.get("host", "http://localhost:1234/v1"),
        timeout_seconds=int(lmstudio_settings.get("timeout_seconds", 90)),
        default_text_model=lmstudio_settings.get("text_model", "qwen2.5-coder-14b-instruct"),
        default_vision_model=lmstudio_settings.get("vision_model", "qwen3-vl-8b-instruct"),
    )
    safety = SafetyManager(workspace_root=root_path / "workspace")
    tool_registry = ToolRegistry(
        root_dir=root_path,
        safety=safety,
        logger=logger,
        lmstudio_client=lmstudio_client,
    )
    agent = LocalPilotAgent(
        llm_client=lmstudio_client,
        tool_registry=tool_registry,
        planner_model=lmstudio_client.default_text_model,
    )

    safe_console_print("LocalPilot Agent CLI")
    safe_console_print("Type 'exit' to quit.")

    while True:
        try:
            user_text = input("\nUser: ").strip()
        except EOFError:
            print()
            return 0
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            return 0

        result = agent.run_task(user_text)
        for step in result.get("transcript", []):
            if step["type"] == "tool_call":
                safe_console_print("\nAI tool call:")
                safe_console_print(json.dumps(step["payload"], indent=2))
            elif step["type"] == "tool_result":
                safe_console_print("\nTool result:")
                safe_console_print(json.dumps(step["payload"], indent=2))
            elif step["type"] == "question":
                safe_console_print(f"\nAI question:\n{step['payload']['message']}")
            elif step["type"] == "final":
                safe_console_print(f"\nAI final:\n{step['payload']['message']}")

        if not result.get("ok"):
            safe_console_print(f"\nAgent error:\n{result.get('error', 'Unknown agent error.')}")


class LocalPilotApp:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.settings = load_settings(self.root_dir)
        self.model_profiles = self._load_json(self.root_dir / "config" / "model_profiles.json")
        self.performance_profiles = self._load_json(self.root_dir / "config" / "performance_profiles.json")
        self.operating_profiles = self._load_json(self.root_dir / "config" / "operating_profiles.json")
        self.logger = AppLogger(self.root_dir / self.settings["logs_dir"])
        lmstudio_settings = self.settings.get("lmstudio", {})
        self.lmstudio = LMStudioClient(
            host=lmstudio_settings.get("host", "http://localhost:1234/v1"),
            timeout_seconds=int(lmstudio_settings.get("timeout_seconds", 90)),
            default_text_model=lmstudio_settings.get("text_model", "qwen2.5-coder-14b-instruct"),
            default_vision_model=lmstudio_settings.get("vision_model", "qwen3-vl-8b-instruct"),
        )
        self.git_sync = GitSyncManager(self.root_dir, self.settings, self.logger)
        self.memory = MemoryStore(
            self.root_dir / self.settings["memory_dir"],
            self.root_dir / "config" / "capabilities.json",
        )
        self.desktop_lessons = DesktopLessonStore(self.root_dir / self.settings["memory_dir"] / "desktop_lessons.jsonl")
        self.capabilities = self.memory.load_capabilities()
        self.system_prompt = build_system_prompt(self.capabilities)
        self.router = KeywordRouter()
        self.ollama = OllamaClient(
            host=self.model_profiles["ollama"]["host"],
            timeout_seconds=self.model_profiles["ollama"]["timeout_seconds"],
            model_profiles=self.model_profiles,
            default_role=self.settings.get("active_model_role", self.model_profiles.get("default_role", "main")),
            performance_profile=self._selected_performance_profile(),
            performance_profile_name=self._active_performance_profile_name(),
            lifecycle_settings=self.settings.get("model_lifecycle", {}),
            debug_views_dir=self.root_dir / "workspace" / "debug_views",
            log_event_callback=self.logger.event,
        )
        self.task_state = TaskStateStore(
            self.root_dir / "workspace" / "runtime" / "task_state.json",
            safety_constraints=self._build_task_state_safety_constraints(),
            event_callback=self.logger.event,
        )
        self.test_runner = TestRunner(self)
        self._apply_operating_profile(self._active_operating_profile_name())
        self._initialize_ollama()
        self.safety = SafetyManager(approval_callback=self._approval_callback)
        self.gui: LocalPilotGUI | None = None
        self._shutdown_complete = False
        self.pending_followup: dict[str, Any] | None = None
        self.modes = {
            "chat": ChatMode(self),
            "code": CodeMode(self),
            "research": ResearchMode(self),
            "desktop": DesktopMode(self),
        }
        self._run_git_sync("startup")

    def _load_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def attach_gui(self, gui: "LocalPilotGUI") -> None:
        self.gui = gui
        self.logger.register_callback(gui.on_event)

    def _active_performance_profile_name(self) -> str:
        return self.settings.get(
            "active_performance_profile",
            self.performance_profiles.get("default_profile", "rtx3060_balanced"),
        )

    def _active_operating_profile_name(self) -> str:
        return self.task_state.snapshot().get(
            "operating_profile",
            self.settings.get("active_operating_profile", self.operating_profiles.get("default_profile", "reliable_stack")),
        )

    def _selected_operating_profile(self) -> dict[str, Any]:
        name = self._active_operating_profile_name()
        return dict(self.operating_profiles.get("profiles", {}).get(name, {}))

    def _selected_performance_profile(self) -> dict[str, Any]:
        profile_name = self._active_performance_profile_name()
        return dict(self.performance_profiles.get("profiles", {}).get(profile_name, {}))

    def _build_task_state_safety_constraints(self) -> dict[str, Any]:
        return {
            "desktop_requires_confirmation": bool(self.settings.get("approvals", {}).get("desktop_requires_confirmation", True)),
            "shell_requires_confirmation": bool(self.settings.get("approvals", {}).get("shell_requires_confirmation", True)),
            "overwrite_requires_confirmation": bool(self.settings.get("approvals", {}).get("overwrite_requires_confirmation", True)),
            "page_confidence_threshold": float(self.settings.get("page_understanding", {}).get("confidence_threshold", 0.85)),
            "auto_submit_allowed": False,
        }

    def _apply_operating_profile(self, profile_name: str) -> None:
        profile = dict(self.operating_profiles.get("profiles", {}).get(profile_name, {}))
        self.ollama.set_role_overrides(profile.get("role_overrides", {}))
        self.task_state.update(
            operating_profile=profile_name,
            active_model=self.resolve_runtime_model_for_role("main"),
            safety_constraints=self._build_task_state_safety_constraints(),
        )

    def switch_operating_profile(self, profile_name: str) -> dict[str, Any]:
        normalized = re.sub(r"[^a-z0-9]+", "_", profile_name.strip().lower()).strip("_")
        profiles = self.operating_profiles.get("profiles", {})
        if normalized not in profiles:
            return {
                "ok": False,
                "error": (
                    f"Unknown operating profile: {profile_name}. "
                    f"Available: {', '.join(sorted(profiles))}"
                ),
            }
        self._apply_operating_profile(normalized)
        profile = profiles[normalized]
        self.logger.event("OperatingMode", f"switched to {normalized}")
        return {
            "ok": True,
            "message": (
                f"Operating profile set to {normalized}.\n"
                f"{profile.get('description', '')}".strip()
            ),
            "profile": normalized,
            "role_overrides": profile.get("role_overrides", {}),
        }

    def resolve_runtime_model_for_role(self, role: str) -> str:
        profile = self.ollama.get_profile(role)
        return str(profile.get("model", ""))

    def _initialize_ollama(self) -> None:
        ollama_settings = self.settings.get("ollama", {})
        ok, message = self.ollama.ensure_server(
            auto_start=bool(ollama_settings.get("auto_start_server", True)),
            wait_seconds=int(ollama_settings.get("startup_wait_seconds", 8)),
        )
        role = "Reasoner" if ok else "Ollama"
        self.logger.event(role, message)

    def _run_git_sync(self, trigger: str) -> None:
        ok, message = self.git_sync.sync(trigger)
        role = "GitSync" if ok else "GitSyncWarning"
        self.logger.event(role, message, persist=False, trigger=trigger)

    def _approval_callback(self, prompt: str) -> bool:
        self.logger.event("Safety", "Approval pending", prompt=prompt)
        if self.gui is not None:
            approved = self.gui.request_approval(prompt)
        else:
            approved = self._cli_approval(prompt)
        self.logger.event("Safety", "Approval accepted" if approved else "Approval denied", prompt=prompt)
        return approved

    def _cli_approval(self, prompt: str) -> bool:
        reply = input(f"{prompt}\nApprove? y/n: ").strip().lower()
        return reply == "y"

    def ask_approval(self, prompt: str) -> bool:
        return self.safety.confirm(prompt)

    def describe_capabilities(self) -> str:
        caps = self.capabilities
        return (
            f"{caps['name']}: {caps['purpose']}\n"
            f"Modes: {', '.join(caps['modes'])}\n"
            f"Safety: {'; '.join(caps['safety_rules'])}\n"
            f"Known limits: {'; '.join(caps['known_limits'])}"
        )

    def describe_model_status(self) -> str:
        default_role = self.settings.get("active_model_role", self.model_profiles.get("default_role", "main"))
        return self.ollama.build_model_status_report(
            default_role=default_role,
            performance_profile_name=self._active_performance_profile_name(),
            operating_profile_name=self._active_operating_profile_name(),
        )

    def describe_model_benchmark(self) -> str:
        default_role = self.settings.get("active_model_role", self.model_profiles.get("default_role", "main"))
        return self.ollama.build_model_benchmark_report(
            default_role=default_role,
            performance_profile_name=self._active_performance_profile_name(),
            operating_profile_name=self._active_operating_profile_name(),
        )

    def describe_model_compare(self, target: str) -> str:
        if target.strip().lower() == "operating-modes":
            return self.ollama.build_operating_modes_compare_report(
                operating_profiles=self.operating_profiles,
                active_profile_name=self._active_operating_profile_name(),
            )
        return self.ollama.build_model_compare_report(target)

    def describe_model_doctor(self) -> str:
        default_role = self.settings.get("active_model_role", self.model_profiles.get("default_role", "main"))
        return self.ollama.build_model_doctor_report(
            default_role=default_role,
            performance_profile_name=self._active_performance_profile_name(),
        )

    def describe_model_repair_plan(self) -> str:
        return self.ollama.build_model_repair_plan()

    def describe_model_unload(self) -> str:
        return self.ollama.build_model_unload_report()

    def describe_model_warmup(self) -> str:
        return self.ollama.build_model_warmup_report()

    def describe_vision_test(self) -> str:
        return self.ollama.build_vision_test_report()

    def describe_lmstudio_screenshot(self) -> str:
        _exit_code, output = run_lmstudio_vision_test(self.root_dir)
        return output

    def describe_system_doctor(self) -> str:
        return build_system_doctor_report(
            root_dir=self.root_dir,
            ollama_reachable=self.ollama.is_server_available(),
        )

    def describe_log_tail(self) -> str:
        return self.logger.format_event_tail(limit=80)

    def start_project_tests(self) -> dict[str, Any]:
        return self.test_runner.start()

    def cancel_project_tests(self) -> dict[str, Any]:
        return self.test_runner.cancel()

    def process_user_input(self, user_text: str) -> dict[str, Any]:
        followup_request = self._process_pending_followup(user_text)
        if followup_request is not None:
            return followup_request

        if self.safety.is_broad_destructive_request(user_text):
            self.logger.event("Safety", "Destructive request blocked", user_text=user_text)
            return {
                "user_text": user_text,
                "mode": "safety",
                "requires_confirmation": False,
                "approved": False,
                "result": {
                    "ok": False,
                    "error": self.safety.destructive_refusal_message(user_text),
                },
                "events": [{"role": "Safety", "message": "Destructive request blocked"}],
            }

        request: dict[str, Any] = {
            "request_id": uuid.uuid4().hex[:12],
            "user_text": user_text,
            "mode": self.router.classify(user_text),
            "requires_confirmation": False,
            "approved": None,
            "result": None,
            "events": [],
        }
        self.logger.event("Router", f"classified as {request['mode']}", user_text=user_text)
        request["events"].append({"role": "Router", "message": f"classified as {request['mode']}"})
        self.logger.event("Reasoner", f"dispatching mode {request['mode']}")
        self.logger.event(f"Mode:{request['mode']}", "activated")
        active_role = self._role_for_mode(request["mode"])
        active_model = self.resolve_runtime_model_for_role(active_role)
        task_state_loaded = hasattr(self, "task_state")
        if hasattr(self, "task_state"):
            self.task_state.update(
                current_goal=user_text,
                active_mode=request["mode"],
                active_model=active_model,
                last_action=f"dispatch:{request['mode']}",
                next_recommended_action=f"Handle request in {request['mode']} mode.",
            )
        self.logger.event(
            "Request",
            "started",
            request_id=request["request_id"],
            user_text=user_text,
            classified_mode=request["mode"],
            operating_profile=self._active_operating_profile_name(),
            active_model_role=active_role,
            active_model=active_model,
            task_state_loaded=task_state_loaded,
            task_state_updated=task_state_loaded,
            current_goal=user_text,
            objective_verified=None,
            confidence_score=None,
            safety_state="guarded" if request["mode"] == "desktop" else "idle",
            final_result_status="started",
        )

        if request["mode"] == "memory":
            result = self._handle_memory_request(request)
        else:
            handler = self.modes.get(request["mode"], self.modes["chat"])
            result = handler.handle(request)
        request["result"] = result
        if hasattr(self, "task_state"):
            self._update_task_state_after_result(request, result)
        self.logger.event(
            "Request",
            "completed",
            request_id=request["request_id"],
            user_text=user_text,
            classified_mode=request["mode"],
            operating_profile=self._active_operating_profile_name(),
            active_model_role=active_role,
            active_model=active_model,
            task_state_loaded=task_state_loaded,
            task_state_updated=hasattr(self, "task_state"),
            current_goal=user_text,
            objective_verified=result.get("objective_verified"),
            confidence_score=result.get("confidence_score"),
            safety_state=self._safety_state_for_result(request["mode"], result),
            final_result_status=self._result_status_for_logging(result),
        )
        return request

    def _role_for_mode(self, mode: str) -> str:
        mapping = {
            "chat": "main",
            "code": "coder",
            "research": "main",
            "desktop": "main",
            "memory": "main",
            "safety": "main",
        }
        return mapping.get(mode, "main")

    def _update_task_state_after_result(self, request: dict[str, Any], result: dict[str, Any]) -> None:
        mode = request.get("mode", "chat")
        updates: dict[str, Any] = {
            "active_mode": mode,
            "active_model": self.resolve_runtime_model_for_role(self._role_for_mode(mode)),
            "last_action": f"completed:{mode}" if result.get("ok") else f"failed:{mode}",
            "last_failure": "" if result.get("ok") else result.get("error", "Unknown failure"),
            "confidence_score": result.get("confidence_score"),
            "next_recommended_action": self._next_action_from_result(mode, result),
        }
        if "page_state" in result:
            updates["page_state"] = result.get("page_state", {})
        elif any(key in result for key in ("active_window", "focused_control", "visible_controls", "mouse_position")):
            updates["page_state"] = {
                "active_window": result.get("active_window", {}),
                "focused_control": result.get("focused_control", {}),
                "visible_controls": result.get("visible_controls", {}),
                "mouse_position": result.get("mouse_position", {}),
                "ocr": result.get("ocr", {}),
            }
        if "objective_state" in result:
            updates["objective_state"] = result.get("objective_state", {})
        elif any(key in result for key in ("objective_match_confidence", "objective_verified", "reason")):
            updates["objective_state"] = {
                "objective_match_confidence": result.get("objective_match_confidence"),
                "objective_verified": result.get("objective_verified"),
                "reason": result.get("reason", ""),
            }
        if any(key in result for key in ("project_path", "verification", "acceptance_checklist", "status")):
            updates["build_state"] = {
                "project_path": result.get("project_path", ""),
                "status": result.get("status", "unknown"),
                "verification": result.get("verification", {}),
                "acceptance_checklist": result.get("acceptance_checklist", []),
            }
        if any(key in result for key in ("query", "results", "note_saved")):
            updates["research_state"] = {
                "query": result.get("query", ""),
                "result_count": len(result.get("results", []) or []),
                "note_saved": bool(result.get("note_saved", False)),
            }
        files_changed = result.get("files") or result.get("files_changed") or []
        if files_changed:
            updates["files_changed"] = files_changed
        tests_run = result.get("tests_run") or []
        if tests_run:
            updates["tests_run"] = tests_run
        self.task_state.update(**updates)

    def _next_action_from_result(self, mode: str, result: dict[str, Any]) -> str:
        if result.get("ok"):
            if mode == "code" and result.get("project_path"):
                return "Review the generated project and run any remaining manual checks."
            if mode == "desktop" and not result.get("objective_verified", True):
                return "Inspect the page state and retry only with higher confidence."
            return "Wait for the next user instruction."
        if mode == "code":
            return "Inspect verification failures, apply a fix, and rerun checks."
        if mode == "desktop":
            return "Stop, inspect the current page state, and ask for clarification if the target is uncertain."
        return "Review the last failure before taking another action."

    def _result_status_for_logging(self, result: dict[str, Any]) -> str:
        if "status" in result:
            return str(result.get("status"))
        if "result" in result:
            return str(result.get("result"))
        if result.get("running"):
            return "running"
        return "ok" if result.get("ok") else "error"

    def _safety_state_for_result(self, mode: str, result: dict[str, Any]) -> str:
        if mode == "safety":
            return "blocked"
        if mode == "desktop":
            return "guarded"
        if mode == "code" and self.test_runner.is_running():
            return "running-tests"
        return "idle"

    def set_pending_followup(self, mode: str, prompt: str, callback) -> None:
        self.pending_followup = {
            "mode": mode,
            "prompt": prompt,
            "callback": callback,
        }
        self.logger.event("Pending", prompt)

    def clear_pending_followup(self) -> None:
        self.pending_followup = None

    def _process_pending_followup(self, user_text: str) -> dict[str, Any] | None:
        lowered = user_text.strip().lower()
        if not self._is_followup_phrase(lowered):
            return None

        if self.pending_followup is None:
            message = (
                "No pending task to continue."
                if self._is_affirmative_followup(lowered)
                else "No pending task to cancel."
            )
            return {
                "user_text": user_text,
                "mode": "chat",
                "requires_confirmation": False,
                "approved": None,
                "result": {"ok": True, "message": message},
                "events": [],
            }

        pending = self.pending_followup
        self.pending_followup = None
        if self._is_negative_followup(lowered):
            self.logger.event("Pending", "cancelled by user")
            return {
                "user_text": user_text,
                "mode": pending["mode"],
                "requires_confirmation": False,
                "approved": False,
                "result": {"ok": False, "error": "Pending task cancelled by user."},
                "events": [],
            }

        self.logger.event("Pending", "continuing pending task")
        result = pending["callback"]()
        return {
            "user_text": user_text,
            "mode": pending["mode"],
            "requires_confirmation": False,
            "approved": True,
            "result": result,
            "events": [],
        }

    def _is_followup_phrase(self, lowered: str) -> bool:
        return self._is_affirmative_followup(lowered) or self._is_negative_followup(lowered)

    def _is_affirmative_followup(self, lowered: str) -> bool:
        return lowered in {"yes", "y", "do it", "go ahead"}

    def _is_negative_followup(self, lowered: str) -> bool:
        return lowered in {"no", "cancel"}

    def _handle_memory_request(self, request: dict[str, Any]) -> dict[str, Any]:
        text = request["user_text"].strip()
        lowered = text.lower()
        self.logger.event("Memory", f"Handling memory request: {text}")
        if hasattr(self, "task_state"):
            self.task_state.snapshot()

        if lowered.startswith("save note") or lowered.startswith("remember"):
            note_text = text.split(" ", 2)[-1] if " " in text else ""
            return {"ok": True, "message": self.memory.save_note(note_text)}

        if lowered.startswith("search notes"):
            keyword = text.split(" ", 2)[-1] if " " in text else ""
            matches = self.memory.search_notes(keyword)
            return {"ok": True, "matches": matches}

        if lowered.startswith("show notes") or lowered == "notes":
            return {"ok": True, "content": self.memory.show_notes()}

        if lowered.startswith("save fact"):
            parts = text.split(" ", 3)
            if len(parts) < 4:
                return {"ok": False, "error": "Use: save fact <key> <value>"}
            return {"ok": True, "message": self.memory.save_fact(parts[2], parts[3])}

        return {
            "ok": True,
            "message": (
                "Memory mode supports: save note ..., search notes ..., show notes, save fact <key> <value>."
            ),
        }

    def shutdown(self) -> None:
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        self._run_git_sync("shutdown")

    def run_guarded_desktop_action(self, action_name: str, action):
        self.logger.event("DesktopGuard", f"starting {action_name}")
        if self.gui is not None:
            self.gui.show_desktop_busy_overlay(action_name)
        try:
            return action()
        finally:
            if self.gui is not None:
                self.gui.hide_desktop_busy_overlay()
            self.logger.event("DesktopGuard", f"finished {action_name}")


class LocalPilotGUI:
    def __init__(self, app: LocalPilotApp) -> None:
        self.app = app
        self.root = tk.Tk()
        self.root.title("LocalPilot")
        self.root.geometry("1240x820")
        self.root.minsize(1100, 760)
        self.event_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self.desktop_overlay: tk.Toplevel | None = None
        self.desktop_overlay_action_label: tk.Label | None = None
        self.desktop_overlay_shown_at: float | None = None
        self.approval_window: tk.Toplevel | None = None
        self.memory_text: scrolledtext.ScrolledText | None = None
        self.last_debug_image_path: Path | None = None
        self._build_widgets()
        self._refresh_status_bar()
        self.root.after(150, self._drain_events)

    def _build_widgets(self) -> None:
        theme = self.app.settings.get("ui", {}).get("theme", "dark")
        colors = self._theme_colors(theme)
        self.colors = colors
        self.root.configure(bg=colors["bg"])
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Status.TFrame", background=colors["panel"])
        self.style.configure("StatusLabel.TLabel", background=colors["panel"], foreground=colors["text"], font=("Segoe UI", 10))
        self.style.configure("StatusValue.TLabel", background=colors["panel"], foreground=colors["accent"], font=("Segoe UI", 10, "bold"))
        self.style.configure("Tabs.TNotebook", background=colors["bg"], borderwidth=0)
        self.style.configure("Tabs.TNotebook.Tab", font=("Segoe UI", 10), padding=(14, 8), background=colors["panel"], foreground=colors["muted"])
        self.style.map("Tabs.TNotebook.Tab", background=[("selected", colors["surface"])], foreground=[("selected", colors["text"])])
        self.style.configure("Action.TButton", font=("Segoe UI", 10), padding=(10, 8))

        header = ttk.Frame(self.root, style="Status.TFrame", padding=(14, 12))
        header.pack(fill="x", padx=14, pady=(14, 10))

        self.mode_var = tk.StringVar(value="idle")
        self.role_var = tk.StringVar(value="idle")
        self.ollama_var = tk.StringVar(value="unknown")
        self.main_model_var = tk.StringVar(value="n/a")
        self.vision_model_var = tk.StringVar(value="n/a")
        self.safety_var = tk.StringVar(value="Guarded")
        self.running_var = tk.StringVar(value="idle")

        status_items = [
            ("Mode", self.mode_var),
            ("Role", self.role_var),
            ("Ollama", self.ollama_var),
            ("Main", self.main_model_var),
            ("Vision", self.vision_model_var),
            ("Safety", self.safety_var),
            ("Running", self.running_var),
        ]
        for index, (label, variable) in enumerate(status_items):
            item = ttk.Frame(header, style="Status.TFrame")
            item.grid(row=0, column=index, sticky="w", padx=(0, 18))
            ttk.Label(item, text=f"{label}:", style="StatusLabel.TLabel").pack(side="left")
            ttk.Label(item, textvariable=variable, style="StatusValue.TLabel").pack(side="left", padx=(6, 0))

        body = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            sashrelief=tk.RAISED,
            bg=colors["bg"],
            sashwidth=8,
            bd=0,
        )
        body.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        left = tk.Frame(body, bg=colors["bg"])
        right = tk.Frame(body, bg=colors["bg"])
        body.add(left, stretch="always", minsize=720)
        body.add(right, minsize=360)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        tk.Label(
            left,
            text="Conversation",
            font=("Segoe UI", 11, "bold"),
            bg=colors["bg"],
            fg=colors["text"],
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.output = scrolledtext.ScrolledText(
            left,
            wrap=tk.WORD,
            height=20,
            font=("Segoe UI", 11),
            bg=colors["surface"],
            fg=colors["text"],
            insertbackground=colors["text"],
            relief="flat",
            bd=0,
            padx=14,
            pady=14,
            spacing1=6,
            spacing3=10,
        )
        self.output.grid(row=1, column=0, sticky="nsew")
        self.output.configure(state="disabled")
        self.output.tag_configure("user", font=("Segoe UI", 11, "bold"), foreground=colors["accent"])
        self.output.tag_configure("assistant", font=("Segoe UI", 11, "bold"), foreground=colors["success"])
        self.output.tag_configure("body", font=("Segoe UI", 11), foreground=colors["text"])
        self.output.tag_configure("error", font=("Segoe UI", 11), foreground=colors["danger"])

        input_frame = tk.Frame(left, bg=colors["bg"])
        input_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        input_frame.grid_columnconfigure(0, weight=1)
        self.input_entry = tk.Entry(
            input_frame,
            font=("Segoe UI", 11),
            bg=colors["surface"],
            fg=colors["text"],
            insertbackground=colors["text"],
            relief="flat",
            bd=0,
        )
        self.input_entry.grid(row=0, column=0, sticky="ew")
        self.input_entry.bind("<Return>", lambda _event: self.submit_input())
        send_button = ttk.Button(input_frame, text="Send", command=self.submit_input, style="Action.TButton")
        send_button.grid(row=0, column=1, padx=(10, 0))

        notebook = ttk.Notebook(right, style="Tabs.TNotebook")
        notebook.grid(row=0, column=0, sticky="nsew")

        activity_tab = tk.Frame(notebook, bg=colors["bg"])
        logs_tab = tk.Frame(notebook, bg=colors["bg"])
        memory_tab = tk.Frame(notebook, bg=colors["bg"])
        tools_tab = tk.Frame(notebook, bg=colors["bg"])
        notebook.add(activity_tab, text="Activity")
        notebook.add(logs_tab, text="Logs")
        notebook.add(memory_tab, text="Memory")
        notebook.add(tools_tab, text="Tools")

        self.timeline = self._make_panel_text(activity_tab, height=28)
        self.logs = self._make_panel_text(logs_tab, height=28)
        self.memory_text = self._make_panel_text(memory_tab, height=28)
        self._load_memory_panel()
        self._build_tools_tab(tools_tab)
        self._load_default_panels()

        for widget in (self.output, self.timeline, self.logs, self.memory_text):
            widget.bind("<Key>", lambda _event: "break")
            widget.bind("<<Paste>>", lambda _event: "break")
            widget.bind("<Button-3>", lambda _event: "break")

    def submit_input(self) -> None:
        text = self.input_entry.get().strip()
        if not text:
            return
        self.input_entry.delete(0, tk.END)
        self.submit_text(text)

    def submit_text(self, text: str) -> None:
        self._append_chat_message("You", text, speaker_tag="user")
        request = self.app.process_user_input(text)
        self._remember_debug_image(request["result"])
        rendered = format_result(request["result"])
        speaker_tag = "assistant"
        body_tag = "body"
        if isinstance(request["result"], dict) and request["result"].get("error"):
            body_tag = "error"
        self._append_chat_message("LocalPilot", rendered, speaker_tag=speaker_tag, body_tag=body_tag)
        self._refresh_status_bar()
        self._maybe_refresh_memory(request["result"])

    def on_event(self, event: dict[str, Any]) -> None:
        self.event_queue.put(event)

    def _drain_events(self) -> None:
        while not self.event_queue.empty():
            event = self.event_queue.get()
            role = event["role"]
            message = event["message"]
            self.role_var.set(role)
            if role.startswith("Mode:"):
                self.mode_var.set(role.replace("Mode:", "").strip())
            if role == "Safety":
                self._update_safety_state(message)
            if role == "Tests":
                self._update_running_state(message, event.get("extra", {}))
            line = f"[{event['timestamp']}] {role} -> {message}\n"
            self._append_readonly(self.timeline, line)
            self._append_readonly(self.logs, line)
            self._refresh_status_bar()
        self.root.after(150, self._drain_events)

    def request_approval(self, prompt: str) -> bool:
        approved = {"value": False}
        done = threading.Event()

        def show_dialog() -> None:
            self.safety_var.set("Waiting for approval")
            dialog = self._build_approval_window(prompt, approved, done)
            self.approval_window = dialog

        if threading.current_thread() is threading.main_thread():
            show_dialog()
            if self.approval_window is not None and self.approval_window.winfo_exists():
                self.root.wait_window(self.approval_window)
        else:
            self.root.after(0, show_dialog)
            done.wait()
        return approved["value"]

    def run(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def on_close(self) -> None:
        self.app.shutdown()
        self.root.destroy()

    def _append_readonly(self, widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.configure(state="disabled")

    def _append_chat_message(self, speaker: str, text: str, speaker_tag: str, body_tag: str = "body") -> None:
        widget = self.output
        widget.configure(state="normal")
        widget.insert(tk.END, f"{speaker}\n", (speaker_tag,))
        widget.insert(tk.END, f"{text}\n\n", (body_tag,))
        widget.see(tk.END)
        widget.configure(state="disabled")

    def _make_panel_text(self, parent: tk.Frame, height: int) -> scrolledtext.ScrolledText:
        text = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            height=height,
            font=("Segoe UI", 10),
            bg=self.colors["surface"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            relief="flat",
            bd=0,
            padx=12,
            pady=12,
        )
        text.pack(fill="both", expand=True, padx=2, pady=2)
        text.configure(state="disabled")
        return text

    def _build_tools_tab(self, parent: tk.Frame) -> None:
        container = tk.Frame(parent, bg=self.colors["bg"])
        container.pack(fill="both", expand=True, padx=8, pady=8)

        actions = [
            ("Show Notes", lambda: self.submit_text("show notes")),
            ("Take Screenshot", lambda: self.submit_text("take screenshot")),
            ("Mouse Position", lambda: self.submit_text("get mouse position")),
            ("Run Pytest", lambda: self.submit_text("run pytest")),
            ("Cancel Tests", lambda: self.submit_text("cancel tests")),
            ("Open Last Debug Image", self.open_last_debug_image),
            ("Clear Chat", self.clear_chat),
        ]
        for label, command in actions:
            button = ttk.Button(container, text=label, command=command, style="Action.TButton")
            button.pack(fill="x", pady=6)

    def clear_chat(self) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.configure(state="disabled")
        self._append_chat_message(
            "LocalPilot",
            "Chat cleared. Ask me something or use the quick tools on the right.",
            speaker_tag="assistant",
        )

    def _refresh_status_bar(self) -> None:
        self.ollama_var.set(self.app.ollama.last_status.replace("_", " "))
        self.main_model_var.set(
            self.app.ollama.active_main_model
            or self.app.model_profiles.get("main", {}).get("model", "n/a")
        )
        self.vision_model_var.set(
            self.app.ollama.active_vision_model
            or self.app.model_profiles.get("vision", {}).get("model", "n/a")
        )
        if not self.safety_var.get():
            self.safety_var.set("Guarded")
        if hasattr(self, "running_var") and not self.running_var.get():
            self.running_var.set("idle")

    def _load_memory_panel(self) -> None:
        if self.memory_text is None:
            return
        content = self.app.memory.show_notes()
        self.memory_text.configure(state="normal")
        self.memory_text.delete("1.0", tk.END)
        self.memory_text.insert(tk.END, content)
        self.memory_text.configure(state="disabled")

    def _maybe_refresh_memory(self, result: dict[str, Any]) -> None:
        if any(key in result for key in ("content", "matches", "message")):
            self._load_memory_panel()

    def _remember_debug_image(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            return
        path_value = result.get("path")
        if not path_value:
            return
        path = Path(path_value)
        if "debug_views" not in path.as_posix():
            return
        if not path.is_absolute():
            path = self.app.root_dir / path
        self.last_debug_image_path = path

    def open_last_debug_image(self) -> None:
        if self.last_debug_image_path is None:
            self._append_chat_message(
                "LocalPilot",
                "No desktop understanding image is available yet. Run `visualize desktop understanding` first.",
                speaker_tag="assistant",
            )
            return
        if not self.last_debug_image_path.exists():
            self._append_chat_message(
                "LocalPilot",
                f"Last debug image not found: {self.last_debug_image_path}",
                speaker_tag="assistant",
                body_tag="error",
            )
            return
        try:
            os.startfile(str(self.last_debug_image_path))
            self._append_chat_message(
                "LocalPilot",
                f"Opened debug image:\n{self.last_debug_image_path}",
                speaker_tag="assistant",
            )
        except OSError as exc:
            self._append_chat_message(
                "LocalPilot",
                f"Could not open debug image: {exc}",
                speaker_tag="assistant",
                body_tag="error",
            )

    def _load_default_panels(self) -> None:
        self._append_chat_message(
            "LocalPilot",
            "Ready. Try `what can you do`, `show notes`, or use the Tools tab for quick actions.",
            speaker_tag="assistant",
        )
        self._append_readonly(self.logs, "Recent logs will appear here as the session runs.\n")

    def _build_approval_window(self, prompt: str, approved: dict[str, bool], done: threading.Event) -> tk.Toplevel:
        if self.approval_window is not None and self.approval_window.winfo_exists():
            try:
                self.approval_window.destroy()
            except tk.TclError:
                pass

        dialog = tk.Toplevel(self.root)
        dialog.title("LocalPilot Approval")
        dialog.configure(bg=self.colors["panel"])
        dialog.resizable(False, False)
        dialog.attributes("-topmost", True)
        dialog.transient(self.root)
        dialog.protocol("WM_DELETE_WINDOW", lambda: finish(False))
        dialog.grid_columnconfigure(0, weight=1)

        width = 640
        height = 380
        try:
            root_x = self.root.winfo_rootx()
            root_y = self.root.winfo_rooty()
            root_w = self.root.winfo_width()
            root_h = self.root.winfo_height()
            pos_x = root_x + max((root_w - width) // 2, 40)
            pos_y = root_y + max((root_h - height) // 2, 40)
            dialog.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
        except tk.TclError:
            dialog.geometry(f"{width}x{height}+420+220")

        header = tk.Label(
            dialog,
            text="Approval Required",
            font=("Segoe UI", 18, "bold"),
            fg=self.colors["text"],
            bg=self.colors["panel"],
        )
        header.grid(row=0, column=0, sticky="w", padx=20, pady=(18, 10))

        instruction = tk.Label(
            dialog,
            text="Choose Allow to continue or Deny to cancel.",
            font=("Segoe UI", 10),
            fg=self.colors["muted"],
            bg=self.colors["panel"],
        )
        instruction.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 10))

        body = scrolledtext.ScrolledText(
            dialog,
            wrap=tk.WORD,
            height=8,
            font=("Segoe UI", 11),
            bg=self.colors["surface"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            relief="flat",
            bd=0,
            padx=12,
            pady=12,
        )
        body.grid(row=2, column=0, sticky="ew", padx=20)
        body.insert("1.0", prompt)
        body.configure(state="disabled")

        button_row = tk.Frame(dialog, bg=self.colors["panel"])
        button_row.grid(row=3, column=0, sticky="ew", padx=20, pady=(16, 18))

        def finish(value: bool) -> None:
            approved["value"] = value
            self.safety_var.set("Guarded")
            if dialog.winfo_exists():
                try:
                    dialog.grab_release()
                except Exception:
                    pass
                dialog.destroy()
            self.approval_window = None
            done.set()

        deny = ttk.Button(button_row, text="Deny", command=lambda: finish(False), style="Action.TButton")
        deny.pack(side="right")
        allow = ttk.Button(button_row, text="Allow", command=lambda: finish(True), style="Action.TButton")
        allow.pack(side="right", padx=(0, 10))
        dialog.bind("<Return>", lambda _event: finish(True))
        dialog.bind("<Escape>", lambda _event: finish(False))

        try:
            dialog.grab_set()
        except Exception:
            pass
        dialog.deiconify()
        dialog.lift()
        try:
            dialog.focus_force()
        except tk.TclError:
            pass
        self.root.after(100, lambda: self._refresh_approval_window(dialog))
        allow.focus_set()
        return dialog

    def _refresh_approval_window(self, dialog: tk.Toplevel) -> None:
        if dialog is None or not dialog.winfo_exists():
            return
        dialog.attributes("-topmost", True)
        dialog.lift()
        try:
            dialog.focus_force()
        except tk.TclError:
            pass

    def _update_safety_state(self, message: str) -> None:
        lowered = message.lower()
        if "approval pending" in lowered:
            self.safety_var.set("Waiting for approval")
        elif "approval accepted" in lowered or "approval denied" in lowered:
            self.safety_var.set("Guarded")

    def _update_running_state(self, message: str, extra: dict[str, Any]) -> None:
        if not hasattr(self, "running_var"):
            return
        lowered = message.lower()
        if lowered == "started":
            self.running_var.set(f"Running: {extra.get('command', '')}")
        elif lowered in {"passed", "failed", "cancelled"}:
            self.running_var.set("idle")

    def _theme_colors(self, theme: str) -> dict[str, str]:
        if theme == "light":
            return {
                "bg": "#edf1f5",
                "panel": "#dce5ef",
                "surface": "#ffffff",
                "text": "#16212e",
                "muted": "#5b6774",
                "accent": "#185ea8",
                "success": "#18794e",
                "danger": "#b42318",
            }
        return {
            "bg": "#0f141b",
            "panel": "#18202a",
            "surface": "#1f2935",
            "text": "#f3f6fb",
            "muted": "#a2afbf",
            "accent": "#61b0ff",
            "success": "#6fd3a5",
            "danger": "#ff8f8f",
        }

    def show_desktop_busy_overlay(self, action_name: str) -> None:
        settings = self.app.settings.get("desktop_guard", {})
        if not settings.get("show_overlay", True):
            return

        if threading.current_thread() is threading.main_thread():
            self._build_or_refresh_desktop_overlay(action_name)
            self._flush_root_updates()
        else:
            ready = threading.Event()

            def build_overlay() -> None:
                try:
                    self._build_or_refresh_desktop_overlay(action_name)
                    self._flush_root_updates()
                finally:
                    ready.set()

            self.root.after(0, build_overlay)
            ready.wait(timeout=2.0)

    def _build_or_refresh_desktop_overlay(self, action_name: str) -> None:
        settings = self.app.settings.get("desktop_guard", {})
        if self.desktop_overlay is not None and self.desktop_overlay.winfo_exists():
            if self.desktop_overlay_action_label is not None and self.desktop_overlay_action_label.winfo_exists():
                self.desktop_overlay_action_label.configure(text=f"Current action: {action_name}")
            self.desktop_overlay.deiconify()
            self.desktop_overlay.lift()
            self.desktop_overlay.update_idletasks()
            return

        overlay = tk.Toplevel(self.root)
        overlay.title(settings.get("title", "LocalPilot Is Using Your PC"))
        overlay.attributes("-topmost", True)
        overlay.geometry("560x220+480+220")
        overlay.configure(bg="#101820")
        overlay.resizable(False, False)
        overlay.protocol("WM_DELETE_WINDOW", lambda: None)

        title = tk.Label(
            overlay,
            text=settings.get("title", "LocalPilot Is Using Your PC"),
            font=("Segoe UI", 18, "bold"),
            fg="#f4f6f8",
            bg="#101820",
        )
        title.pack(pady=(24, 12))

        body = tk.Label(
            overlay,
            text=settings.get(
                "message",
                "Please do not touch your mouse or keyboard until this action is finished.",
            ),
            font=("Segoe UI", 12),
            fg="#f4f6f8",
            bg="#101820",
            wraplength=500,
            justify="center",
        )
        body.pack(padx=24)

        action = tk.Label(
            overlay,
            text=f"Current action: {action_name}",
            font=("Segoe UI", 11, "bold"),
            fg="#86d0ff",
            bg="#101820",
        )
        action.pack(pady=(14, 8))

        footer = tk.Label(
            overlay,
            text=settings.get(
                "footer",
                "LocalPilot will remove this notice as soon as it is safe again.",
            ),
            font=("Segoe UI", 10),
            fg="#b8c4cc",
            bg="#101820",
            wraplength=500,
            justify="center",
        )
        footer.pack(padx=24, pady=(0, 18))

        self.desktop_overlay = overlay
        self.desktop_overlay_action_label = action
        self.desktop_overlay_shown_at = time.monotonic()
        overlay.deiconify()
        overlay.lift()
        overlay.update_idletasks()
        try:
            overlay.update()
        except tk.TclError:
            pass

    def _flush_root_updates(self) -> None:
        self.root.update_idletasks()
        try:
            self.root.update()
        except tk.TclError:
            pass

    def hide_desktop_busy_overlay(self) -> None:
        def destroy_overlay() -> None:
            if self.desktop_overlay is not None and self.desktop_overlay.winfo_exists():
                shown_at = self.desktop_overlay_shown_at or time.monotonic()
                elapsed_ms = (time.monotonic() - shown_at) * 1000
                min_visible_ms = 800
                remaining_ms = max(0, int(min_visible_ms - elapsed_ms))
                if remaining_ms > 0:
                    self.root.after(remaining_ms, destroy_overlay)
                    self.desktop_overlay_shown_at = None
                    return
                self.desktop_overlay.destroy()
            self.desktop_overlay = None
            self.desktop_overlay_action_label = None
            self.desktop_overlay_shown_at = None

        self.root.after(0, destroy_overlay)


def format_result(result: dict[str, Any]) -> str:
    if result.get("ok") and result.get("path") and "debug_views" in str(result.get("path", "")).replace("\\", "/"):
        return f"Desktop understanding image saved:\n{result['path']}"
    if "message" in result:
        return str(result["message"])
    if "content" in result:
        return str(result["content"])
    if result.get("ok") and "x" in result and "y" in result:
        return f"Mouse position: ({result['x']}, {result['y']})"
    if result.get("ok") and "path" in result and len(result) <= 3:
        return str(result["path"])
    if "matches" in result:
        matches = result.get("matches") or []
        if not matches:
            return "No matching notes found."
        return "\n".join(f"- {match}" for match in matches)
    if result.get("results"):
        lines = [f"Research results for: {result.get('query', '')}"]
        for item in result["results"]:
            lines.append(f"- {item.get('title', '')}")
            lines.append(f"  {item.get('url', '')}")
            if item.get("snippet"):
                lines.append(f"  {item['snippet']}")
        return "\n".join(lines)
    if "error" in result:
        return f"Error: {result['error']}"
    return json.dumps(result, indent=2)


def safe_console_print(text: str = "") -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        sanitized = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(sanitized)


def run_cli(app: LocalPilotApp) -> None:
    def on_event(event: dict[str, Any]) -> None:
        if event.get("role") != "Tests":
            return
        message = event.get("message", "")
        extra = event.get("extra") or {}
        if message == "started":
            safe_console_print(f"[Tests] Running: {extra.get('command', '')}")
        elif message in {"passed", "failed", "cancelled"}:
            safe_console_print(
                f"[Tests] {message} | exit={extra.get('exit_code')} | duration={extra.get('duration_seconds')}s | {extra.get('summary', '')}"
            )
        else:
            stream = extra.get("stream", "stdout")
            safe_console_print(f"[Tests:{stream}] {message}")

    app.logger.register_callback(on_event)
    safe_console_print("LocalPilot CLI started. Type 'exit' to quit.")
    safe_console_print(app.describe_capabilities())
    if app.ollama.last_status not in {"running", "started_by_localpilot"}:
        safe_console_print()
        safe_console_print(
            app.ollama.build_unavailable_message(auto_start_attempted=app.ollama.last_status == "start_timeout")
        )
    while True:
        try:
            user_text = input("\nYou> ").strip()
        except EOFError:
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break
        request = app.process_user_input(user_text)
        safe_console_print(f"\nLocalPilot> {format_result(request['result'])}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localpilot.py", add_help=True)
    parser.add_argument(
        "--model-status",
        action="store_true",
        help="Print model role status and exit without starting the GUI.",
    )
    parser.add_argument(
        "--model-doctor",
        action="store_true",
        help="Print model doctor diagnostics and exit without starting the GUI.",
    )
    parser.add_argument(
        "--vision-test",
        action="store_true",
        help="Run a minimal vision probe and exit without starting the GUI.",
    )
    parser.add_argument(
        "--lmstudio-vision-test",
        action="store_true",
        help="Take a real screenshot, send it to LM Studio vision, and print the description.",
    )
    parser.add_argument(
        "--agent-cli",
        action="store_true",
        help="Run the lightweight AI-driven agent CLI without starting the GUI.",
    )
    parser.add_argument(
        "--system-doctor",
        action="store_true",
        help="Print dependency and runtime diagnostics and exit without starting the GUI.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Alias for --system-doctor.",
    )
    parser.add_argument(
        "--task-state",
        action="store_true",
        help="Print the current shared runtime task state and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    root_dir = Path(__file__).resolve().parent.parent

    if args.lmstudio_vision_test:
        exit_code, output = run_lmstudio_vision_test(root_dir)
        safe_console_print(output)
        return exit_code

    if args.agent_cli:
        return run_agent_cli(root_dir)

    app = LocalPilotApp(root_dir)
    atexit.register(app.shutdown)

    if args.model_status:
        safe_console_print(app.describe_model_status())
        app.shutdown()
        return 0

    if args.model_doctor:
        safe_console_print(app.describe_model_doctor())
        app.shutdown()
        return 0

    if args.vision_test:
        safe_console_print(app.describe_vision_test())
        app.shutdown()
        return 0

    if args.system_doctor or args.doctor:
        safe_console_print(app.describe_system_doctor())
        app.shutdown()
        return 0

    if args.task_state:
        safe_console_print(json.dumps(app.task_state.snapshot(), indent=2))
        app.shutdown()
        return 0

    enable_gui = bool(app.settings.get("enable_gui", True))

    if enable_gui:
        try:
            gui = LocalPilotGUI(app)
            app.attach_gui(gui)
            if bool(app.settings.get("enable_cli_thread_with_gui", False)):
                cli_thread = threading.Thread(target=run_cli, args=(app,), daemon=True)
                cli_thread.start()
            gui.run()
            return 0
        except Exception as exc:
            app.logger.event("GUI", f"GUI unavailable, falling back to CLI: {exc}")

    run_cli(app)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
