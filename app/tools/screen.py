from __future__ import annotations

from datetime import datetime
from pathlib import Path


DEFAULT_SCREENSHOT_DIR = Path("logs") / "screenshots"


def take_screenshot(output_dir: str | Path | None = None) -> dict:
    try:
        import mss
    except ImportError as exc:
        return {"ok": False, "error": f"mss not installed: {exc}"}

    folder = Path(output_dir) if output_dir is not None else DEFAULT_SCREENSHOT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    filename = folder / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    session_factory = getattr(mss, "MSS", None) or getattr(mss, "mss", None)
    if session_factory is None:
        return {"ok": False, "error": "mss does not expose MSS() or mss() on this installation."}
    with session_factory() as sct:
        sct.shot(output=str(filename))
    return {"ok": True, "path": str(filename)}


def latest_screenshot(output_dir: str | Path | None = None) -> str | None:
    folder = Path(output_dir) if output_dir is not None else DEFAULT_SCREENSHOT_DIR
    if not folder.exists():
        return None
    screenshots = sorted(folder.glob("screenshot_*.png"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not screenshots:
        return None
    return str(screenshots[0])


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
