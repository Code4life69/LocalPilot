from __future__ import annotations

import re

from app.tools.desktop_flow import DesktopExecutionFlow
from app.tools.mouse_keyboard import click, hotkey, move_mouse, type_text
from app.tools.screen import get_active_window_basic, get_mouse_position, take_screenshot
from app.tools.windows_ui import get_active_window_title, get_focused_control, list_visible_controls


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

        if "focused control" in lowered:
            return get_focused_control()

        if "visible controls" in lowered or "list controls" in lowered:
            return list_visible_controls()

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
