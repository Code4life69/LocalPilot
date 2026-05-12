from pathlib import Path
from types import SimpleNamespace
import threading

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


def test_main_vision_test_flag_prints_without_gui(monkeypatch):
    calls = {"shutdown": False, "output": []}

    class FakeApp:
        def __init__(self, root_dir):
            self.root_dir = root_dir

        def describe_vision_test(self):
            return "Vision test\nVision unavailable: Ollama is not running."

        def shutdown(self):
            calls["shutdown"] = True

    monkeypatch.setattr(main_module, "LocalPilotApp", FakeApp)
    monkeypatch.setattr(main_module, "safe_console_print", lambda text="": calls["output"].append(text))

    exit_code = main_module.main(["--vision-test"])

    assert exit_code == 0
    assert calls["shutdown"] is True
    assert calls["output"] == ["Vision test\nVision unavailable: Ollama is not running."]


def test_main_system_doctor_flag_prints_without_gui(monkeypatch):
    calls = {"shutdown": False, "output": []}

    class FakeApp:
        def __init__(self, root_dir):
            self.root_dir = root_dir

        def describe_system_doctor(self):
            return "System doctor\n- UI Automation: dependency_missing"

        def shutdown(self):
            calls["shutdown"] = True

    monkeypatch.setattr(main_module, "LocalPilotApp", FakeApp)
    monkeypatch.setattr(main_module, "safe_console_print", lambda text="": calls["output"].append(text))

    exit_code = main_module.main(["--system-doctor"])

    assert exit_code == 0
    assert calls["shutdown"] is True
    assert calls["output"] == ["System doctor\n- UI Automation: dependency_missing"]


def test_approval_callback_logs_pending_and_acceptance():
    events = []
    app = LocalPilotApp.__new__(LocalPilotApp)
    app.logger = SimpleNamespace(event=lambda role, message, **extra: events.append((role, message, extra)))
    app.gui = SimpleNamespace(request_approval=lambda prompt: True)

    approved = app._approval_callback("Approve desktop execution?")

    assert approved is True
    assert events[0][0:2] == ("Safety", "Approval pending")
    assert events[1][0:2] == ("Safety", "Approval accepted")


def test_approval_callback_logs_denial():
    events = []
    app = LocalPilotApp.__new__(LocalPilotApp)
    app.logger = SimpleNamespace(event=lambda role, message, **extra: events.append((role, message, extra)))
    app.gui = SimpleNamespace(request_approval=lambda prompt: False)

    approved = app._approval_callback("Approve desktop execution?")

    assert approved is False
    assert events[0][0:2] == ("Safety", "Approval pending")
    assert events[1][0:2] == ("Safety", "Approval denied")


def test_gui_safety_state_updates_from_events():
    values = []

    class FakeVar:
        def set(self, value):
            values.append(value)

    gui = LocalPilotGUI.__new__(LocalPilotGUI)
    gui.safety_var = FakeVar()

    gui._update_safety_state("Approval pending")
    gui._update_safety_state("Approval accepted")
    gui._update_safety_state("Approval denied")

    assert values == ["Waiting for approval", "Guarded", "Guarded"]


def test_refresh_status_bar_preserves_waiting_approval_state():
    class FakeVar:
        def __init__(self, value=""):
            self.value = value

        def set(self, value):
            self.value = value

        def get(self):
            return self.value

    gui = LocalPilotGUI.__new__(LocalPilotGUI)
    gui.app = SimpleNamespace(
        ollama=SimpleNamespace(last_status="running", active_main_model="qwen3:8b", active_vision_model="qwen2.5vl:7b"),
        model_profiles={"main": {"model": "qwen3:8b"}, "vision": {"model": "qwen2.5vl:7b"}},
    )
    gui.ollama_var = FakeVar()
    gui.main_model_var = FakeVar()
    gui.vision_model_var = FakeVar()
    gui.safety_var = FakeVar("Waiting for approval")

    gui._refresh_status_bar()

    assert gui.safety_var.get() == "Waiting for approval"


def test_show_desktop_busy_overlay_waits_for_background_thread_build(monkeypatch):
    gui = LocalPilotGUI.__new__(LocalPilotGUI)
    gui.app = SimpleNamespace(settings={"desktop_guard": {"show_overlay": True}})
    gui.desktop_overlay = None
    gui.desktop_overlay_action_label = None
    gui.desktop_overlay_shown_at = None
    called = {"count": 0}

    class FakeRoot:
        def after(self, _delay, callback):
            callback()

        def update_idletasks(self):
            return None

        def update(self):
            return None

    gui.root = FakeRoot()
    gui._build_or_refresh_desktop_overlay = lambda action_name: called.__setitem__("count", called["count"] + 1)

    fake_main_thread = object()
    fake_worker_thread = object()
    monkeypatch.setattr(threading, "main_thread", lambda: fake_main_thread)
    monkeypatch.setattr(threading, "current_thread", lambda: fake_worker_thread)

    gui.show_desktop_busy_overlay("type text")

    assert called["count"] == 1


def test_main_does_not_start_cli_thread_with_gui_when_disabled(monkeypatch):
    thread_calls = {"created": 0}

    class FakeApp:
        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.settings = {"enable_gui": True, "enable_cli_thread_with_gui": False}

        def attach_gui(self, gui):
            self.gui = gui

        def shutdown(self):
            return None

    class FakeGUI:
        def __init__(self, app):
            self.app = app

        def run(self):
            return None

    class FakeThread:
        def __init__(self, *args, **kwargs):
            thread_calls["created"] += 1

        def start(self):
            thread_calls["created"] += 100

    monkeypatch.setattr(main_module, "LocalPilotApp", FakeApp)
    monkeypatch.setattr(main_module, "LocalPilotGUI", FakeGUI)
    monkeypatch.setattr(main_module.threading, "Thread", FakeThread)

    exit_code = main_module.main([])

    assert exit_code == 0
    assert thread_calls["created"] == 0


def test_main_starts_cli_thread_with_gui_when_enabled(monkeypatch):
    thread_calls = {"created": 0, "started": 0}

    class FakeApp:
        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.settings = {"enable_gui": True, "enable_cli_thread_with_gui": True}

        def attach_gui(self, gui):
            self.gui = gui

        def shutdown(self):
            return None

    class FakeGUI:
        def __init__(self, app):
            self.app = app

        def run(self):
            return None

    class FakeThread:
        def __init__(self, *args, **kwargs):
            thread_calls["created"] += 1

        def start(self):
            thread_calls["started"] += 1

    monkeypatch.setattr(main_module, "LocalPilotApp", FakeApp)
    monkeypatch.setattr(main_module, "LocalPilotGUI", FakeGUI)
    monkeypatch.setattr(main_module.threading, "Thread", FakeThread)

    exit_code = main_module.main([])

    assert exit_code == 0
    assert thread_calls["created"] == 1
    assert thread_calls["started"] == 1
