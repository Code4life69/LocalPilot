from __future__ import annotations

from datetime import datetime
from pathlib import Path


def take_screenshot(output_dir: str) -> dict:
    try:
        import mss
    except ImportError as exc:
        return {"ok": False, "error": f"mss not installed: {exc}"}

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    filename = folder / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    session_factory = getattr(mss, "MSS", None) or getattr(mss, "mss", None)
    if session_factory is None:
        return {"ok": False, "error": "mss does not expose MSS() or mss() on this installation."}
    with session_factory() as sct:
        sct.shot(output=str(filename))
    return {"ok": True, "path": str(filename)}


def get_mouse_position() -> dict:
    try:
        import pyautogui
    except ImportError as exc:
        return {"ok": False, "error": f"pyautogui not installed: {exc}"}

    x, y = pyautogui.position()
    return {"ok": True, "x": x, "y": y}


def get_active_window_basic() -> dict:
    try:
        import pyautogui
    except ImportError as exc:
        return {"ok": False, "error": f"pyautogui not installed: {exc}"}

    title = ""
    try:
        window = pyautogui.getActiveWindow()
        if window is not None:
            title = window.title or ""
    except Exception:
        title = ""
    return {"ok": True, "title": title}
