from pathlib import Path
from types import SimpleNamespace

from app import main as main_module
from app.main import LocalPilotApp, LocalPilotGUI, format_result


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


def test_yes_with_no_pending_task_is_safe():
    app = LocalPilotApp.__new__(LocalPilotApp)
    app.pending_followup = None

    request = app.process_user_input("yes")

    assert request["mode"] == "chat"
    assert request["result"]["message"] == "No pending task to continue."


def test_affirmative_followup_continues_pending_task():
    app = LocalPilotApp.__new__(LocalPilotApp)
    app.pending_followup = {
        "mode": "code",
        "prompt": "Continue pending scaffold",
        "callback": lambda: {"ok": True, "message": "Pending task completed."},
    }
    app.logger = SimpleNamespace(event=lambda *args, **kwargs: None)

    request = app.process_user_input("go ahead")

    assert request["mode"] == "code"
    assert request["result"]["message"] == "Pending task completed."
    assert app.pending_followup is None


def test_negative_followup_cancels_pending_task():
    app = LocalPilotApp.__new__(LocalPilotApp)
    app.pending_followup = {
        "mode": "code",
        "prompt": "Continue pending scaffold",
        "callback": lambda: {"ok": True, "message": "should not run"},
    }
    app.logger = SimpleNamespace(event=lambda *args, **kwargs: None)

    request = app.process_user_input("cancel")

    assert request["mode"] == "code"
    assert request["result"]["error"] == "Pending task cancelled by user."
    assert app.pending_followup is None


def test_main_model_status_flag_prints_without_gui(monkeypatch):
    calls = {"shutdown": False, "output": []}

    class FakeApp:
        def __init__(self, root_dir):
            self.root_dir = root_dir

        def describe_model_status(self):
            return "Model status\n- Ollama reachable: no"

        def shutdown(self):
            calls["shutdown"] = True

    monkeypatch.setattr(main_module, "LocalPilotApp", FakeApp)
    monkeypatch.setattr(main_module, "safe_console_print", lambda text="": calls["output"].append(text))

    exit_code = main_module.main(["--model-status"])

    assert exit_code == 0
    assert calls["shutdown"] is True
    assert calls["output"] == ["Model status\n- Ollama reachable: no"]


def test_main_model_doctor_flag_prints_without_gui(monkeypatch):
    calls = {"shutdown": False, "output": []}

    class FakeApp:
        def __init__(self, root_dir):
            self.root_dir = root_dir

        def describe_model_doctor(self):
            return "Model doctor\n- Ollama reachable: no"

        def shutdown(self):
            calls["shutdown"] = True

    monkeypatch.setattr(main_module, "LocalPilotApp", FakeApp)
    monkeypatch.setattr(main_module, "safe_console_print", lambda text="": calls["output"].append(text))

    exit_code = main_module.main(["--model-doctor"])

    assert exit_code == 0
    assert calls["shutdown"] is True
    assert calls["output"] == ["Model doctor\n- Ollama reachable: no"]
