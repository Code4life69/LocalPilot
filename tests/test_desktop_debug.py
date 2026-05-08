from pathlib import Path

from PIL import Image

from app.tools.desktop_debug import annotate_debug_view


def test_annotate_debug_view_saves_output_with_minimal_data(tmp_path):
    screenshot = tmp_path / "screen.png"
    output = tmp_path / "annotated.png"
    Image.new("RGB", (320, 240), color=(32, 32, 32)).save(screenshot)

    annotate_debug_view(
        screenshot_path=screenshot,
        output_path=output,
        active_window={
            "ok": True,
            "title": "Example Window",
            "bounds": {"left": 10, "top": 10, "right": 300, "bottom": 200},
        },
        focused_control={
            "ok": True,
            "name": "Search box",
            "control_type": "EditControl",
            "bounds": {"left": 20, "top": 40, "right": 200, "bottom": 80},
        },
        visible_controls={
            "ok": True,
            "controls": [
                {
                    "name": "Submit",
                    "control_type": "ButtonControl",
                    "bounds": {"left": 210, "top": 40, "right": 280, "bottom": 80},
                }
            ],
        },
        mouse={"ok": True, "x": 100, "y": 120},
    )

    assert output.exists()
    assert output.stat().st_size > 0
