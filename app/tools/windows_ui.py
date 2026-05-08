from __future__ import annotations

from typing import Any


def get_focused_control() -> dict:
    try:
        import uiautomation as auto

        control = auto.GetFocusedControl()
        return {
            "ok": True,
            "name": getattr(control, "Name", ""),
            "control_type": getattr(control, "ControlTypeName", ""),
            "automation_id": getattr(control, "AutomationId", ""),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def get_active_window_title() -> dict:
    try:
        import uiautomation as auto

        window = auto.GetForegroundWindow()
        return {"ok": True, "title": window.Name}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def list_visible_controls(max_depth: int = 2) -> dict:
    try:
        import uiautomation as auto

        window = auto.GetForegroundWindow()
        controls: list[dict[str, Any]] = []
        for child, depth in _walk_controls(window, 0, max_depth):
            controls.append(
                {
                    "depth": depth,
                    "name": getattr(child, "Name", ""),
                    "control_type": getattr(child, "ControlTypeName", ""),
                    "automation_id": getattr(child, "AutomationId", ""),
                }
            )
        return {"ok": True, "controls": controls}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _walk_controls(control, depth: int, max_depth: int):
    if depth > max_depth:
        return
    for child in control.GetChildren():
        yield child, depth
        yield from _walk_controls(child, depth + 1, max_depth)
