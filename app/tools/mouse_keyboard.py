from __future__ import annotations

def move_mouse(x: int, y: int) -> dict:
    try:
        import pyautogui
    except ImportError as exc:
        return {"ok": False, "error": f"pyautogui not installed: {exc}"}

    pyautogui.moveTo(x, y)
    return {"ok": True, "x": x, "y": y}


def click(x: int | None = None, y: int | None = None) -> dict:
    try:
        import pyautogui
    except ImportError as exc:
        return {"ok": False, "error": f"pyautogui not installed: {exc}"}

    pyautogui.click(x=x, y=y)
    return {"ok": True, "x": x, "y": y}


def type_text(text: str) -> dict:
    try:
        import pyautogui
    except ImportError as exc:
        return {"ok": False, "error": f"pyautogui not installed: {exc}"}

    pyautogui.write(text)
    return {"ok": True, "text": text}


def hotkey(*keys: str) -> dict:
    try:
        import pyautogui
    except ImportError as exc:
        return {"ok": False, "error": f"pyautogui not installed: {exc}"}

    pyautogui.hotkey(*keys)
    return {"ok": True, "keys": list(keys)}
