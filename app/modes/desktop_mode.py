from __future__ import annotations

import re

from app.tools.desktop_flow import DesktopExecutionFlow
from app.tools.desktop_visualizer import visualize_desktop_understanding
from app.tools.mouse_keyboard import click, hotkey, move_mouse, type_text
from app.tools.screen import get_active_window_basic, get_mouse_position, take_screenshot
from app.tools.windows_ui import get_active_window_title, get_control_at_point, get_focused_control, list_visible_controls


class DesktopMode:
    def __init__(self, app) -> None:
        self.app = app
        self.execution_flow = DesktopExecutionFlow(app)

    def handle(self, request: dict) -> dict:
        text = request["user_text"].strip()
        lowered = text.lower()

        flow_result = self.execution_flow.execute(text) if self.execution_flow.can_handle(text) else None
        if flow_result is not None:
            return flow_result

        if lowered in {"visualize desktop", "visualize desktop understanding", "show me what you see"}:
            debug_dir = str(self.app.root_dir / "workspace" / "debug_views")
            return visualize_desktop_understanding(self.app.settings["screenshots_dir"], debug_dir)

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
            if not self.app.ask_approval(f"Approve click action? {coords if coords else 'current cursor position'}"):
                return {"ok": False, "error": "Click cancelled by user."}
            if coords:
                return self.app.run_guarded_desktop_action(
                    f"click at {coords[0]}, {coords[1]}",
                    lambda: click(*coords),
                )
            return self.app.run_guarded_desktop_action("click current cursor position", click)

        if lowered.startswith("type "):
            payload = text[5:]
            if not self.app.ask_approval(f"Approve typing this text?\n{payload}"):
                return {"ok": False, "error": "Typing cancelled by user."}
            return self.app.run_guarded_desktop_action(
                "type text",
                lambda: type_text(payload),
            )

        if lowered.startswith("hotkey"):
            keys = [part.strip() for part in re.split(r"[+, ]+", text[6:].strip()) if part.strip()]
            if not keys:
                return {"ok": False, "error": "No hotkey keys provided."}
            if not self.app.ask_approval(f"Approve hotkey: {' + '.join(keys)}?"):
                return {"ok": False, "error": "Hotkey cancelled by user."}
            return self.app.run_guarded_desktop_action(
                f"hotkey {' + '.join(keys)}",
                lambda: hotkey(*keys),
            )

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
