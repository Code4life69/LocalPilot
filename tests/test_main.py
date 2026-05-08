from pathlib import Path
from types import SimpleNamespace

from app.main import LocalPilotGUI, format_result


def test_format_result_for_desktop_understanding_image():
    result = {
        "ok": True,
        "path": r"workspace\debug_views\desktop_understanding_20260508_161500.png",
        "active_window_title": "Example",
    }
    assert format_result(result) == (
        "Desktop understanding image saved:\n"
        r"workspace\debug_views\desktop_understanding_20260508_161500.png"
    )


def test_gui_remembers_last_debug_image_path():
    gui = LocalPilotGUI.__new__(LocalPilotGUI)
    gui.app = SimpleNamespace(root_dir=Path(r"C:\LocalPilot"))
    gui.last_debug_image_path = None

    gui._remember_debug_image({"path": r"workspace\debug_views\desktop_understanding_20260508_161500.png"})

    assert gui.last_debug_image_path == Path(r"C:\LocalPilot\workspace\debug_views\desktop_understanding_20260508_161500.png")
