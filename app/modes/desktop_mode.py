from __future__ import annotations

import re

from app.tools.desktop_flow import DesktopExecutionFlow
from app.tools.desktop_visualizer import visualize_desktop_understanding
from app.tools.mouse_keyboard import click, hotkey, move_mouse, type_text
from app.tools.ocr import read_screenshot
from app.tools.page_understanding import PageUnderstandingEngine
from app.tools.screen import get_active_window_basic, get_mouse_position, take_screenshot
from app.tools.windows_ui import get_active_window_title, get_control_at_point, get_focused_control, list_visible_controls


class DesktopMode:
    def __init__(self, app) -> None:
        self.app = app
        self.execution_flow = DesktopExecutionFlow(app)
        self.page_understanding = PageUnderstandingEngine(app)

    def handle(self, request: dict) -> dict:
        text = request["user_text"].strip()
        lowered = text.lower()
        if hasattr(self.app, "task_state"):
            self.app.task_state.snapshot()
            self.app.task_state.update(
                active_mode="desktop",
                active_model=self.app.resolve_runtime_model_for_role("main") if hasattr(self.app, "resolve_runtime_model_for_role") else "",
                last_action="desktop:handle",
            )

        flow_result = self.execution_flow.execute(text) if self.execution_flow.can_handle(text) else None
        if flow_result is not None:
            return flow_result

        if lowered in {"visualize desktop", "visualize desktop understanding", "show me what you see"}:
            debug_dir = str(self.app.root_dir / "workspace" / "debug_views")
            return visualize_desktop_understanding(self.app.settings["screenshots_dir"], debug_dir)

        if lowered == "page inspect":
            return self._page_inspect(include_vision=False)

        if lowered == "page confidence":
            return self._page_inspect(include_vision=True, assess=True)

        if lowered == "show page understanding":
            return self._page_inspect(include_vision=True, assess=True, heading="Page understanding")

        if lowered == "show desktop lessons":
            return {"ok": True, "content": self.app.desktop_lessons.render_recent()}

        if lowered in {"ocr screenshot", "read screen text", "page ocr"}:
            return self._ocr_screenshot()

        if lowered == "inspect desktop":
            return self._inspect_desktop()

        if lowered in {"what window am i on", "what window am i in"}:
            return self._describe_active_window()

        if lowered in {"what is under my mouse", "what is under my cursor"}:
            return self._describe_under_mouse()

        if "screenshot" in lowered:
            return take_screenshot(self.app.settings["screenshots_dir"])

        if "mouse position" in lowered or "cursor" in lowered:
            return get_mouse_position()

        if "active window title" in lowered:
            return get_active_window_title()

        if "active window" in lowered:
            ui_result = get_active_window_title()
            if ui_result.get("ok"):
                return ui_result
            return get_active_window_basic()

        if lowered in {"get focused control", "focused control"} or "focused control" in lowered:
            return self._describe_focused_control()

        if lowered in {"list visible controls", "show visible controls"} or "visible controls" in lowered or "list controls" in lowered:
            return self._describe_visible_controls()

        if lowered.startswith("move mouse"):
            coords = self._extract_coords(text)
            if coords is None:
                return {"ok": False, "error": "Provide x and y coordinates."}
            if not self.app.ask_approval(f"Move mouse to {coords[0]}, {coords[1]}?"):
                return {"ok": False, "error": "Mouse move cancelled by user."}
            return self.app.run_guarded_desktop_action(
                f"move mouse to {coords[0]}, {coords[1]}",
                lambda: move_mouse(*coords),
            )

        if lowered.startswith("click"):
            coords = self._extract_coords(text)
            blocked_result, gate_snapshot = self._guard_action(
                action_kind="click",
                action_name=f"click at {coords[0]}, {coords[1]}" if coords else "click current cursor position",
                request_text=text,
                target_point=coords,
            )
            if blocked_result is not None:
                return blocked_result
            if not self.app.ask_approval(f"Approve click action? {coords if coords else 'current cursor position'}"):
                return {"ok": False, "error": "Click cancelled by user."}
            if coords:
                result = self.app.run_guarded_desktop_action(
                    f"click at {coords[0]}, {coords[1]}",
                    lambda: click(*coords),
                )
                return self._finalize_guarded_action(result, gate_snapshot, "click", text, coords)
            result = self.app.run_guarded_desktop_action("click current cursor position", click)
            return self._finalize_guarded_action(result, gate_snapshot, "click", text, coords)

        if lowered.startswith("type "):
            payload = text[5:]
            blocked_result, gate_snapshot = self._guard_action(
                action_kind="type_text",
                action_name="type text",
                request_text=text,
            )
            if blocked_result is not None:
                return blocked_result
            if not self.app.ask_approval(f"Approve typing this text?\n{payload}"):
                return {"ok": False, "error": "Typing cancelled by user."}
            result = self.app.run_guarded_desktop_action(
                "type text",
                lambda: type_text(payload),
            )
            return self._finalize_guarded_action(result, gate_snapshot, "type_text", text)

        if lowered.startswith("hotkey"):
            keys = [part.strip() for part in re.split(r"[+, ]+", text[6:].strip()) if part.strip()]
            if not keys:
                return {"ok": False, "error": "No hotkey keys provided."}
            blocked_result, gate_snapshot = self._guard_action(
                action_kind="hotkey",
                action_name=f"hotkey {' + '.join(keys)}",
                request_text=text,
            )
            if blocked_result is not None:
                return blocked_result
            if not self.app.ask_approval(f"Approve hotkey: {' + '.join(keys)}?"):
                return {"ok": False, "error": "Hotkey cancelled by user."}
            result = self.app.run_guarded_desktop_action(
                f"hotkey {' + '.join(keys)}",
                lambda: hotkey(*keys),
            )
            return self._finalize_guarded_action(result, gate_snapshot, "hotkey", text)

        if "analyze screenshot" in lowered:
            screenshot = take_screenshot(self.app.settings["screenshots_dir"])
            if not screenshot.get("ok"):
                return screenshot
            prompt = text.split("analyze screenshot", 1)[-1].strip() or "Describe this screenshot."
            analysis = self.app.ollama.analyze_screenshot(prompt, screenshot["path"])
            return {"ok": True, "path": screenshot["path"], "analysis": analysis}

        return {
            "ok": True,
            "message": (
                "Desktop mode can take screenshots, report mouse position, inspect active windows, "
                "inspect focused controls, list visible controls, and perform guarded input actions."
            ),
        }

    def _extract_coords(self, text: str) -> tuple[int, int] | None:
        match = re.search(r"(-?\d+)[,\s]+(-?\d+)", text)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    def _inspect_desktop(self) -> dict:
        active_window = self._best_active_window()
        focused_control = get_focused_control()
        mouse_position = get_mouse_position()
        visible_controls = list_visible_controls(max_depth=1)
        under_mouse = self._control_under_mouse(mouse_position)

        title = active_window.get("title") or "Unknown"
        focused_summary = self._control_summary(focused_control)
        under_mouse_summary = self._control_summary(under_mouse)
        visible_count = len(visible_controls.get("controls", [])) if visible_controls.get("ok") else 0

        lines = [
            f"Active window: {title}",
            f"Focused control: {focused_summary}",
            f"Mouse position: ({mouse_position.get('x', 'unknown')}, {mouse_position.get('y', 'unknown')})",
            f"Under mouse: {under_mouse_summary}",
            f"Visible controls: {visible_count}",
        ]
        dependency_warning = self._dependency_warning(focused_control) or self._dependency_warning(under_mouse) or self._dependency_warning(visible_controls)
        if dependency_warning:
            lines.append(dependency_warning)
        elif visible_controls.get("error"):
            lines.append(f"Visible control scan warning: {visible_controls['error']}")
        return {
            "ok": True,
            "content": "\n".join(lines),
            "active_window": active_window,
            "focused_control": focused_control,
            "mouse_position": mouse_position,
            "under_mouse": under_mouse,
            "visible_controls": visible_controls,
        }

    def _describe_active_window(self) -> dict:
        active_window = self._best_active_window()
        title = active_window.get("title") or "Unknown"
        return {
            "ok": True,
            "content": f"Active window: {title}",
            "active_window": active_window,
        }

    def _describe_under_mouse(self) -> dict:
        mouse_position = get_mouse_position()
        under_mouse = self._control_under_mouse(mouse_position)
        lines = [
            f"Mouse position: ({mouse_position.get('x', 'unknown')}, {mouse_position.get('y', 'unknown')})",
            f"Under mouse: {self._control_summary(under_mouse)}",
        ]
        dependency_warning = self._dependency_warning(under_mouse)
        if dependency_warning:
            lines.append(dependency_warning)
        elif under_mouse.get("error"):
            lines.append(f"UI Automation warning: {under_mouse['error']}")
        return {
            "ok": True,
            "content": "\n".join(lines),
            "mouse_position": mouse_position,
            "under_mouse": under_mouse,
        }

    def _describe_focused_control(self) -> dict:
        focused_control = get_focused_control()
        lines = [f"Focused control: {self._control_summary(focused_control)}"]
        dependency_warning = self._dependency_warning(focused_control)
        if dependency_warning:
            lines.append(dependency_warning)
        elif focused_control.get("error"):
            lines.append(f"UI Automation warning: {focused_control['error']}")
        return {
            "ok": True,
            "content": "\n".join(lines),
            "focused_control": focused_control,
        }

    def _describe_visible_controls(self) -> dict:
        visible_controls = list_visible_controls(max_depth=1)
        controls = visible_controls.get("controls", []) if visible_controls.get("ok") else []
        lines = [f"Visible controls: {len(controls)}"]
        for control in controls[:5]:
            lines.append(
                f"- {control.get('control_type') or 'Control'}: {control.get('name') or '(unnamed)'}"
            )
        dependency_warning = self._dependency_warning(visible_controls)
        if dependency_warning:
            lines.append(dependency_warning)
        elif visible_controls.get("error"):
            lines.append(f"UI Automation warning: {visible_controls['error']}")
        return {
            "ok": True,
            "content": "\n".join(lines),
            "visible_controls": visible_controls,
        }

    def _page_inspect(
        self,
        *,
        include_vision: bool,
        assess: bool = False,
        heading: str = "Page inspect",
    ) -> dict:
        if assess:
            snapshot = self.page_understanding.assess(
                action_kind="inspect",
                action_text=heading,
                include_vision=include_vision,
                vision_prompt="Describe this page and whether the main visible target is clearly identifiable.",
            )
        else:
            snapshot = self.page_understanding.snapshot(
                capture_screenshot=True,
                include_vision=include_vision,
                vision_prompt="Describe this page in one short paragraph.",
            )
            snapshot["confidence_score"] = 0.0
            snapshot["confidence_threshold"] = self.page_understanding.threshold
            snapshot["confidence_allowed"] = True
            snapshot["confidence_reason"] = "Inspection only."
            snapshot["candidate_targets_count"] = len(snapshot.get("candidate_targets", []))
        return {
            "ok": True,
            "content": self.page_understanding.render(snapshot, heading=heading),
            **snapshot,
        }

    def _ocr_screenshot(self) -> dict:
        result = read_screenshot(
            self.app.settings["screenshots_dir"],
            self.app.root_dir / "workspace" / "debug_views",
        )
        if not result.get("ok"):
            return {
                **result,
                "content": (
                    f"OCR unavailable: {result.get('error', 'OCR backend unavailable.')}\n"
                    f"Fix: {result.get('install_hint', 'Install pytesseract and Tesseract OCR.')}"
                ),
            }

        lines = [
            f"OCR backend: {result.get('backend', 'unknown')}",
            f"Source image: {result.get('source_image', '')}",
            f"Confidence: {result.get('confidence', 0.0)}",
            "Visible text:",
            result.get("text", "(no text detected)") or "(no text detected)",
        ]
        return {
            **result,
            "content": "\n".join(lines),
        }

    def _guard_action(
        self,
        *,
        action_kind: str,
        action_name: str,
        request_text: str,
        target_point: tuple[int, int] | None = None,
    ) -> tuple[dict | None, dict]:
        snapshot = self.page_understanding.assess(
            action_kind=action_kind,
            action_text=request_text,
            target_point=target_point,
            include_vision=True,
            vision_prompt="Describe whether the intended target or page is clearly present for this action.",
        )
        if snapshot.get("confidence_allowed"):
            return None, snapshot
        self.app.desktop_lessons.record(
            "confidence_gate_refusal",
            action_name,
            snapshot.get("confidence_reason", "Desktop action blocked by confidence gate."),
            confidence_score=snapshot.get("confidence_score", 0.0),
            confidence_threshold=snapshot.get("confidence_threshold", self.page_understanding.threshold),
            active_window_title=snapshot.get("active_window", {}).get("title", ""),
            vision_summary=snapshot.get("vision_summary", ""),
        )
        return self.page_understanding.build_refusal_payload(snapshot, action_name=action_name), snapshot

    def _finalize_guarded_action(
        self,
        action_result: dict,
        gate_snapshot: dict,
        action_kind: str,
        request_text: str,
        target_point: tuple[int, int] | None = None,
    ) -> dict:
        if not action_result.get("ok"):
            self.app.desktop_lessons.record(
                "action_failure",
                request_text,
                action_result.get("error", "Desktop action failed."),
            )
            return action_result

        post_snapshot = self.page_understanding.post_action_verification(
            gate_snapshot,
            action_kind=action_kind,
            action_text=request_text,
            target_point=target_point,
        )
        verification = post_snapshot.get("verification", {})
        if not verification.get("verified", False):
            self.app.desktop_lessons.record(
                "post_action_verification_failure",
                request_text,
                verification.get("reason", "Post-action verification failed."),
                active_window_title=verification.get("active_window_title", ""),
                vision_summary=verification.get("vision_summary", ""),
            )

        content = self.page_understanding.render(gate_snapshot, heading="Pre-action page understanding")
        content += "\n\n"
        content += self.page_understanding.render(post_snapshot, heading="Post-action verification")

        return {
            **action_result,
            "content": content,
            "verified": verification.get("verified", False),
            "verification_source": verification.get("verification_source", "page_snapshot"),
            "reason": verification.get("reason", ""),
            "active_window_title": verification.get("active_window_title", ""),
            "vision_summary": verification.get("vision_summary", ""),
            "pre_action": gate_snapshot,
            "post_action": post_snapshot,
        }

    def _best_active_window(self) -> dict:
        ui_result = get_active_window_title()
        if ui_result.get("ok") and ui_result.get("title"):
            return ui_result
        return get_active_window_basic()

    def _control_under_mouse(self, mouse_position: dict) -> dict:
        if not mouse_position.get("ok"):
            return {"ok": False, "error": mouse_position.get("error", "Mouse position unavailable.")}
        return get_control_at_point(mouse_position["x"], mouse_position["y"])

    def _control_summary(self, control: dict) -> str:
        if not control.get("ok"):
            if control.get("reason") == "dependency_missing":
                return f"Unavailable (dependency_missing: {control.get('dependency')})"
            return f"Unavailable ({control.get('error', 'unknown reason')})"
        control_type = control.get("control_type") or "Control"
        name = control.get("name") or "(unnamed)"
        return f"{control_type}: {name}"

    def _dependency_warning(self, payload: dict) -> str | None:
        if payload.get("reason") != "dependency_missing":
            return None
        dependency = payload.get("dependency", "unknown")
        fix = payload.get("fix", "Install the missing dependency into .venv.")
        return f"UI Automation status: dependency_missing ({dependency}). Fix: {fix}"
