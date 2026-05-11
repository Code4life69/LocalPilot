from __future__ import annotations

from typing import Any

from app.system_doctor import dependency_missing_payload


def get_focused_control() -> dict:
    try:
        import uiautomation as auto

        control = auto.GetFocusedControl()
        return {
            "ok": True,
            "name": getattr(control, "Name", ""),
            "control_type": getattr(control, "ControlTypeName", ""),
            "automation_id": getattr(control, "AutomationId", ""),
            "bounds": _extract_bounds(control),
        }
    except ModuleNotFoundError as exc:
        if _is_uiautomation_missing(exc):
            return dependency_missing_payload("uiautomation")
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def get_active_window_title() -> dict:
    try:
        import uiautomation as auto

        window = _get_foreground_control(auto)
        return {
            "ok": True,
            "title": getattr(window, "Name", ""),
            "control_type": getattr(window, "ControlTypeName", ""),
            "automation_id": getattr(window, "AutomationId", ""),
            "bounds": _extract_bounds(window),
        }
    except ModuleNotFoundError as exc:
        if _is_uiautomation_missing(exc):
            return dependency_missing_payload("uiautomation")
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def list_visible_controls(max_depth: int = 2) -> dict:
    try:
        import uiautomation as auto

        window = _get_foreground_control(auto)
        controls: list[dict[str, Any]] = []
        for child, depth in _walk_controls(window, 0, max_depth):
            controls.append(
                {
                    "depth": depth,
                    "name": getattr(child, "Name", ""),
                    "control_type": getattr(child, "ControlTypeName", ""),
                    "automation_id": getattr(child, "AutomationId", ""),
                    "bounds": _extract_bounds(child),
                }
            )
        return {"ok": True, "controls": controls}
    except ModuleNotFoundError as exc:
        if _is_uiautomation_missing(exc):
            return dependency_missing_payload("uiautomation")
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def get_control_at_point(x: int, y: int) -> dict:
    try:
        import uiautomation as auto

        control = auto.ControlFromPoint(x, y)
        if control is None:
            return {"ok": False, "error": f"No control found at {x}, {y}.", "x": x, "y": y}
        return {
            "ok": True,
            "x": x,
            "y": y,
            "name": getattr(control, "Name", ""),
            "control_type": getattr(control, "ControlTypeName", ""),
            "automation_id": getattr(control, "AutomationId", ""),
            "bounds": _extract_bounds(control),
        }
    except ModuleNotFoundError as exc:
        if _is_uiautomation_missing(exc):
            payload = dependency_missing_payload("uiautomation")
            payload["x"] = x
            payload["y"] = y
            return payload
        return {"ok": False, "error": str(exc), "x": x, "y": y}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "x": x, "y": y}


def _get_foreground_control(auto):
    handle = auto.GetForegroundWindow()
    control = auto.ControlFromHandle(handle)
    if control is None:
        raise RuntimeError(f"Could not resolve foreground window handle: {handle}")
    return control


def _walk_controls(control, depth: int, max_depth: int):
    if depth > max_depth:
        return
    for child in control.GetChildren():
        yield child, depth
        yield from _walk_controls(child, depth + 1, max_depth)


def _extract_bounds(control) -> dict[str, int] | None:
    try:
        rect = getattr(control, "BoundingRectangle", None)
        if rect is None:
            return None
        left = int(getattr(rect, "left", getattr(rect, "Left", 0)))
        top = int(getattr(rect, "top", getattr(rect, "Top", 0)))
        right = int(getattr(rect, "right", getattr(rect, "Right", 0)))
        bottom = int(getattr(rect, "bottom", getattr(rect, "Bottom", 0)))
        if right <= left or bottom <= top:
            return None
        return {"left": left, "top": top, "right": right, "bottom": bottom}
    except Exception:
        return None


def _is_uiautomation_missing(exc: ModuleNotFoundError) -> bool:
    target = (exc.name or "").lower()
    return target == "uiautomation" or "uiautomation" in str(exc).lower()
