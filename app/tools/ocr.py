from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from app.tools.screen import take_screenshot


TESSERACT_INSTALL_HINT = (
    "Install pytesseract into LocalPilot and install Tesseract OCR for Windows. "
    "Then make sure tesseract.exe is on PATH or installed in C:\\Program Files\\Tesseract-OCR."
)

KNOWN_TESSERACT_PATHS = (
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
)


def get_ocr_backend_status() -> dict[str, Any]:
    try:
        import pytesseract
    except ModuleNotFoundError:
        return {
            "available": False,
            "backend": "pytesseract",
            "error": "OCR backend unavailable: pytesseract is not installed.",
            "install_hint": r".\.venv\Scripts\python.exe -m pip install pytesseract",
            "tesseract_path": "",
        }
    except Exception as exc:
        return {
            "available": False,
            "backend": "pytesseract",
            "error": f"OCR backend unavailable: {exc}",
            "install_hint": TESSERACT_INSTALL_HINT,
            "tesseract_path": "",
        }

    tesseract_path = _find_tesseract_executable(pytesseract)
    if not tesseract_path:
        return {
            "available": False,
            "backend": "pytesseract",
            "error": "OCR backend unavailable: Tesseract executable was not found.",
            "install_hint": TESSERACT_INSTALL_HINT,
            "tesseract_path": "",
        }

    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    return {
        "available": True,
        "backend": "pytesseract",
        "error": "",
        "install_hint": "",
        "tesseract_path": tesseract_path,
    }


def preprocess_image(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    crop_bounds: dict[str, int] | None = None,
    max_width: int = 1600,
) -> dict[str, Any]:
    from PIL import Image, ImageEnhance, ImageOps

    source = Path(image_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as image:
        image = image.convert("RGB")
        if crop_bounds:
            left = max(0, int(crop_bounds.get("left", 0)))
            top = max(0, int(crop_bounds.get("top", 0)))
            right = max(left + 1, int(crop_bounds.get("right", image.width)))
            bottom = max(top + 1, int(crop_bounds.get("bottom", image.height)))
            image = image.crop((left, top, right, bottom))

        if image.width < 1200:
            scale = min(2.0, 1600 / max(image.width, 1))
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
        elif image.width > max_width:
            scale = max_width / image.width
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))

        image = ImageOps.grayscale(image)
        image = ImageOps.autocontrast(image)
        image = ImageEnhance.Contrast(image).enhance(1.4)

        filename = (
            f"{source.stem}_ocr_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{source.suffix or '.png'}"
        )
        processed_path = out_dir / filename
        image.save(processed_path)

    return {
        "ok": True,
        "source_image": str(source),
        "processed_image": str(processed_path),
        "mode": "L",
    }


def read_image(
    image_path: str | Path,
    *,
    output_dir: str | Path,
    crop_bounds: dict[str, int] | None = None,
) -> dict[str, Any]:
    status = get_ocr_backend_status()
    if not status["available"]:
        return {
            "ok": False,
            "backend": status["backend"],
            "source_image": str(image_path),
            "text": "",
            "blocks": [],
            "confidence": 0.0,
            "error": status["error"],
            "install_hint": status["install_hint"],
        }

    import pytesseract
    from PIL import Image
    from pytesseract import Output

    try:
        processed = preprocess_image(image_path, output_dir, crop_bounds=crop_bounds)
        processed_path = processed["processed_image"]
        with Image.open(processed_path) as image:
            text = pytesseract.image_to_string(image)
            data = pytesseract.image_to_data(image, output_type=Output.DICT)
    except Exception as exc:
        return {
            "ok": False,
            "backend": status["backend"],
            "source_image": str(image_path),
            "text": "",
            "blocks": [],
            "confidence": 0.0,
            "error": f"OCR preprocessing failed: {exc}",
            "install_hint": status["install_hint"],
        }

    blocks: list[dict[str, Any]] = []
    confidences: list[float] = []
    for index, raw_text in enumerate(data.get("text", [])):
        content = (raw_text or "").strip()
        raw_conf = str(data.get("conf", [""])[index]).strip()
        try:
            confidence = float(raw_conf)
        except ValueError:
            confidence = -1.0
        if not content:
            continue
        block = {
            "text": content,
            "confidence": confidence if confidence >= 0 else None,
            "left": int(data.get("left", [0])[index]),
            "top": int(data.get("top", [0])[index]),
            "width": int(data.get("width", [0])[index]),
            "height": int(data.get("height", [0])[index]),
        }
        blocks.append(block)
        if confidence >= 0:
            confidences.append(confidence)

    return {
        "ok": True,
        "backend": status["backend"],
        "source_image": str(image_path),
        "processed_image": processed_path,
        "text": text.strip(),
        "blocks": blocks,
        "confidence": round(sum(confidences) / len(confidences), 2) if confidences else 0.0,
        "tesseract_path": status["tesseract_path"],
    }


def read_screenshot(screenshots_dir: str, debug_views_dir: str | Path) -> dict[str, Any]:
    screenshot = take_screenshot(screenshots_dir)
    if not screenshot.get("ok"):
        return {
            "ok": False,
            "backend": "pytesseract",
            "source_image": "",
            "text": "",
            "blocks": [],
            "confidence": 0.0,
            "error": screenshot.get("error", "Screenshot capture failed."),
            "install_hint": "",
        }
    result = read_image(screenshot["path"], output_dir=debug_views_dir)
    result["source_image"] = screenshot["path"]
    return result


def _find_tesseract_executable(pytesseract_module) -> str:
    configured = getattr(pytesseract_module.pytesseract, "tesseract_cmd", "") or ""
    if configured and configured != "tesseract" and Path(configured).exists():
        return configured

    discovered = shutil.which("tesseract")
    if discovered:
        return discovered

    for candidate in KNOWN_TESSERACT_PATHS:
        if candidate.exists():
            return str(candidate)
    return ""
