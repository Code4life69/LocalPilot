from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from app.tools.screen import get_mouse_position, take_screenshot
from app.tools.windows_ui import get_active_window_title, get_focused_control, list_visible_controls


def visualize_desktop_understanding(screenshots_dir: str, output_dir: str) -> dict[str, Any]:
    screenshot = take_screenshot(screenshots_dir)
    if not screenshot.get("ok"):
        return screenshot

    active_window = get_active_window_title()
    focused_control = get_focused_control()
    visible_controls = list_visible_controls(max_depth=2)
    mouse = get_mouse_position()

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"desktop_understanding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    display_path = output_path
    try:
        display_path = output_path.relative_to(Path.cwd())
    except ValueError:
        display_path = output_path

    annotate_desktop_understanding(
        screenshot_path=screenshot["path"],
        output_path=output_path,
        active_window=active_window,
        focused_control=focused_control,
        visible_controls=visible_controls,
        mouse=mouse,
    )

    controls = visible_controls.get("controls", []) if visible_controls.get("ok") else []
    focused_name = focused_control.get("name") or focused_control.get("control_type") or "Unknown"
    return {
        "ok": True,
        "message": str(display_path),
        "path": str(display_path),
        "active_window_title": active_window.get("title", "Unknown"),
        "focused_control": focused_name,
        "visible_control_count": len(controls),
    }


def annotate_desktop_understanding(
    screenshot_path: str | Path,
    output_path: str | Path,
    active_window: dict[str, Any],
    focused_control: dict[str, Any],
    visible_controls: dict[str, Any],
    mouse: dict[str, Any],
) -> None:
    screenshot_path = Path(screenshot_path)
    output_path = Path(output_path)
    image = Image.open(screenshot_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    if active_window.get("ok"):
        _draw_labeled_box(
            draw,
            active_window.get("bounds"),
            "Window",
            outline=(74, 163, 255, 255),
            fill=(74, 163, 255, 36),
            font=font,
        )

    visible_items = visible_controls.get("controls", []) if visible_controls.get("ok") else []
    for item in visible_items:
        _draw_labeled_box(
            draw,
            item.get("bounds"),
            _label_for_control(item),
            outline=(0, 212, 170, 255),
            fill=(0, 212, 170, 20),
            font=font,
        )

    if focused_control.get("ok"):
        _draw_labeled_box(
            draw,
            focused_control.get("bounds"),
            f"Focused { _label_for_control(focused_control) }",
            outline=(255, 197, 61, 255),
            fill=(255, 197, 61, 28),
            font=font,
            width=3,
        )

    if mouse.get("ok"):
        _draw_mouse_marker(draw, mouse["x"], mouse["y"], font)

    header_lines = [
        f"Active window: {active_window.get('title', 'Unknown')}" if active_window.get("ok") else "Active window: unavailable",
        f"Focused: {focused_control.get('name', '') or focused_control.get('control_type', 'Unknown')}" if focused_control.get("ok") else "Focused: unavailable",
        f"Mouse: ({mouse.get('x', '?')}, {mouse.get('y', '?')})" if mouse.get("ok") else "Mouse: unavailable",
        f"Visible controls: {len(visible_items)}" if visible_controls.get("ok") else "Visible controls: unavailable",
    ]
    _draw_header(draw, header_lines, font)

    annotated = Image.alpha_composite(image, overlay).convert("RGB")
    annotated.save(output_path)


def _draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    bounds: dict[str, int] | None,
    label: str,
    outline: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
    font,
    width: int = 2,
) -> None:
    if not bounds:
        return
    box = [bounds["left"], bounds["top"], bounds["right"], bounds["bottom"]]
    draw.rectangle(box, outline=outline, fill=fill, width=width)
    _draw_label(draw, bounds["left"], max(4, bounds["top"] - 16), label, outline, font)


def _draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color: tuple[int, int, int, int], font) -> None:
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle([left - 3, top - 2, right + 3, bottom + 2], fill=(10, 16, 24, 220))
    draw.text((x, y), text, fill=color, font=font)


def _draw_header(draw: ImageDraw.ImageDraw, lines: list[str], font) -> None:
    y = 8
    for line in lines:
        _draw_label(draw, 8, y, line, (255, 255, 255, 255), font)
        y += 16


def _draw_mouse_marker(draw: ImageDraw.ImageDraw, x: int, y: int, font) -> None:
    radius = 14
    color = (255, 92, 92, 255)
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, width=3)
    draw.line([x - 20, y, x + 20, y], fill=color, width=2)
    draw.line([x, y - 20, x, y + 20], fill=color, width=2)
    _draw_label(draw, x + 16, y + 8, "Mouse", color, font)


def _label_for_control(control: dict[str, Any]) -> str:
    control_type = (control.get("control_type") or "").replace("Control", "").strip()
    if control_type:
        return control_type
    name = (control.get("name") or "").strip()
    return name or "Control"
