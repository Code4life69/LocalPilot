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


def test_broad_destructive_request_routes_through_safety_refusal():
    events = []
    app = LocalPilotApp.__new__(LocalPilotApp)
    app.pending_followup = None
    app.safety = SimpleNamespace(
        is_broad_destructive_request=lambda text: True,
        destructive_refusal_message=lambda text: "Blocked by safety.",
    )
    app.logger = SimpleNamespace(event=lambda role, message, **extra: events.append((role, message, extra)))

    request = app.process_user_input(r"delete everything in C:\LocalPilot\workspace")

    assert request["mode"] == "safety"
    assert request["approved"] is False
    assert request["result"]["ok"] is False
    assert request["result"]["error"] == "Blocked by safety."
    assert request["events"] == [{"role": "Safety", "message": "Destructive request blocked"}]
    assert events == [("Safety", "Destructive request blocked", {"user_text": r"delete everything in C:\LocalPilot\workspace"})]


def test_process_user_input_logs_request_summary():
    events = []
    app = LocalPilotApp.__new__(LocalPilotApp)
    app.pending_followup = None
    app.safety = SimpleNamespace(is_broad_destructive_request=lambda text: False)
    app.router = SimpleNamespace(classify=lambda text: "chat")
    app.modes = {"chat": SimpleNamespace(handle=lambda request: {"ok": True, "message": "done"})}
    app.logger = SimpleNamespace(event=lambda role, message, **extra: events.append((role, message, extra)))
    app.task_state = SimpleNamespace(update=lambda **kwargs: kwargs)
    app._active_operating_profile_name = lambda: "reliable_stack"
    app.resolve_runtime_model_for_role = lambda role: "qwen3:8b"
    app._role_for_mode = lambda mode: "main"
    app._update_task_state_after_result = lambda request, result: None
    app._safety_state_for_result = lambda mode, result: "idle"
    app._result_status_for_logging = lambda result: "ok"

    request = app.process_user_input("hello")

    assert request["result"]["message"] == "done"
    request_events = [event for event in events if event[0] == "Request"]
    assert request_events[0][1] == "started"
    assert request_events[0][2]["classified_mode"] == "chat"
    assert request_events[1][1] == "completed"
    assert request_events[1][2]["final_result_status"] == "ok"


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


def test_main_doctor_alias_prints_without_gui(monkeypatch):
    calls = {"shutdown": False, "output": []}

    class FakeApp:
        def __init__(self, root_dir):
            self.root_dir = root_dir

        def describe_system_doctor(self):
            return "System doctor\n- OCR backend: unavailable"

        def shutdown(self):
            calls["shutdown"] = True

    monkeypatch.setattr(main_module, "LocalPilotApp", FakeApp)
    monkeypatch.setattr(main_module, "safe_console_print", lambda text="": calls["output"].append(text))

    exit_code = main_module.main(["--doctor"])

    assert exit_code == 0
    assert calls["shutdown"] is True
    assert calls["output"] == ["System doctor\n- OCR backend: unavailable"]


def test_main_task_state_flag_prints_without_gui(monkeypatch):
    calls = {"shutdown": False, "output": []}

    class FakeApp:
        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.task_state = SimpleNamespace(snapshot=lambda: {"current_goal": "demo", "active_mode": "code"})

        def shutdown(self):
            calls["shutdown"] = True

    monkeypatch.setattr(main_module, "LocalPilotApp", FakeApp)
    monkeypatch.setattr(main_module, "safe_console_print", lambda text="": calls["output"].append(text))

    exit_code = main_module.main(["--task-state"])

    assert exit_code == 0
    assert calls["shutdown"] is True
    assert '"current_goal": "demo"' in calls["output"][0]


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
        ollama=SimpleNamespace(last_status="running", active_main_model="gemma4:31b", active_vision_model="gemma4:31b"),
        model_profiles={"main": {"model": "gemma4:31b"}, "vision": {"model": "gemma4:31b"}},
    )
    gui.ollama_var = FakeVar()
    gui.main_model_var = FakeVar()
    gui.vision_model_var = FakeVar()
    gui.safety_var = FakeVar("Waiting for approval")

    gui._refresh_status_bar()

    assert gui.safety_var.get() == "Waiting for approval"


def test_refresh_status_bar_shows_agent_models_when_agent_mode_selected():
    class FakeVar:
        def __init__(self, value=""):
            self.value = value

        def set(self, value):
            self.value = value

        def get(self):
            return self.value

    gui = LocalPilotGUI.__new__(LocalPilotGUI)
    gui.app = SimpleNamespace(
        ollama=SimpleNamespace(last_status="running", active_main_model="gemma4:e4b", active_vision_model="gemma4:e4b"),
        model_profiles={"main": {"model": "gemma4:e4b"}, "vision": {"model": "gemma4:e4b"}},
        lmstudio=SimpleNamespace(default_text_model="qwen2.5-coder-14b-instruct", default_vision_model="qwen3-vl-8b-instruct"),
    )
    gui.mode_var = FakeVar("agent")
    gui.input_mode_var = FakeVar("agent")
    gui.ollama_var = FakeVar()
    gui.main_model_var = FakeVar()
    gui.vision_model_var = FakeVar()
    gui.browser_var = FakeVar()
    gui.safety_var = FakeVar("Guarded")
    gui.running_var = FakeVar("idle")

    gui._refresh_status_bar()

    assert gui.ollama_var.get() == "lm studio"
    assert gui.main_model_var.get() == "qwen2.5-coder-14b"
    assert gui.vision_model_var.get() == "qwen3-vl-8b"
    assert gui.browser_var.get() == "Puppeteer"


def test_process_user_input_respects_requested_agent_mode():
    events = []
    app = LocalPilotApp.__new__(LocalPilotApp)
    app.pending_followup = None
    app.safety = SimpleNamespace(is_broad_destructive_request=lambda text: False)
    app.router = SimpleNamespace(classify=lambda text: "chat")
    app.modes = {
        "chat": SimpleNamespace(handle=lambda request: {"ok": True, "message": "chat"}),
        "agent": SimpleNamespace(handle=lambda request: {"ok": True, "message": "agent", "transcript": []}),
    }
    app.logger = SimpleNamespace(event=lambda role, message, **extra: events.append((role, message, extra)))
    app.task_state = SimpleNamespace(update=lambda **kwargs: kwargs)
    app._active_operating_profile_name = lambda: "reliable_stack"
    app._active_model_for_mode = lambda mode: "qwen2.5-coder-14b-instruct" if mode == "agent" else "gemma4:e4b"
    app._role_for_mode = lambda mode: "agent" if mode == "agent" else "main"
    app._update_task_state_after_result = lambda request, result: None
    app._safety_state_for_result = lambda mode, result: "guarded" if mode == "agent" else "idle"
    app._result_status_for_logging = lambda result: "ok"
    app._resolve_mode = main_module.LocalPilotApp._resolve_mode.__get__(app, LocalPilotApp)

    request = app.process_user_input("describe my screen", requested_mode="agent")

    assert request["mode"] == "agent"
    assert request["result"]["message"] == "agent"
    request_events = [event for event in events if event[0] == "Request"]
    assert request_events[0][2]["classified_mode"] == "agent"


def test_gui_submit_text_uses_agent_mode_when_selected():
    calls = []
    gui = LocalPilotGUI.__new__(LocalPilotGUI)
    gui.input_mode_var = SimpleNamespace(get=lambda: "agent")
    gui.app = SimpleNamespace(
        process_user_input=lambda text, requested_mode=None: calls.append((text, requested_mode)) or {
            "mode": "agent",
            "result": {"ok": True, "message": "done", "transcript": []},
        }
    )
    gui._append_chat_message = lambda *args, **kwargs: None
    gui._remember_debug_image = lambda result: None
    gui._render_agent_result = lambda result: calls.append(("render_agent", result["message"]))
    gui._refresh_status_bar = lambda: None
    gui._maybe_refresh_memory = lambda result: None

    gui.submit_text("Describe my current screen briefly.")

    assert calls[0] == ("Describe my current screen briefly.", "agent")
    assert calls[1] == ("render_agent", "done")


def test_gui_submit_text_keeps_old_route_when_auto_mode_selected():
    calls = []
    gui = LocalPilotGUI.__new__(LocalPilotGUI)
    gui.input_mode_var = SimpleNamespace(get=lambda: "auto")
    gui.app = SimpleNamespace(
        process_user_input=lambda text, requested_mode=None: calls.append((text, requested_mode)) or {
            "mode": "chat",
            "result": {"ok": True, "message": "done"},
        }
    )
    gui._append_chat_message = lambda *args, **kwargs: None
    gui._remember_debug_image = lambda result: None
    gui._refresh_status_bar = lambda: None
    gui._maybe_refresh_memory = lambda result: None

    gui.submit_text("hello")

    assert calls[0] == ("hello", None)


def test_load_memory_panel_includes_recent_sessions():
    class FakeText:
        def __init__(self):
            self.value = ""
            self.states = []

        def configure(self, **kwargs):
            self.states.append(kwargs.get("state"))

        def delete(self, *_args):
            self.value = ""

        def insert(self, *_args):
            self.value += _args[1]

    gui = LocalPilotGUI.__new__(LocalPilotGUI)
    gui.app = SimpleNamespace(
        memory=SimpleNamespace(
            show_notes=lambda: "# LocalPilot Notes\n\n- saved note",
            list_session_summaries=lambda limit=6: [
                {
                    "session_id": "2026_demo",
                    "user_task": "describe my screen",
                    "status": "final",
                    "final_answer": "The screen is visible.",
                    "files_changed": ["C:\\LocalPilot\\workspace\\demo.py"],
                    "browser_actions": [{"tool": "browser_search", "args": {"query": "cats"}}],
                    "errors": [],
                }
            ],
        )
    )
    gui.memory_text = FakeText()

    gui._load_memory_panel()

    assert "Recent Sessions" in gui.memory_text.value
    assert "describe my screen" in gui.memory_text.value
    assert "The screen is visible." in gui.memory_text.value


def test_build_approval_window_keeps_buttons_visible_and_binds_shortcuts(monkeypatch):
    labels = []
    frames = []
    text_areas = []
    buttons = []

    class FakeVar:
        def __init__(self, value=""):
            self.value = value

        def set(self, value):
            self.value = value

        def get(self):
            return self.value

    class FakeRoot:
        def winfo_rootx(self):
            return 100

        def winfo_rooty(self):
            return 120

        def winfo_width(self):
            return 900

        def winfo_height(self):
            return 700

        def after(self, _delay, callback):
            callback()

    class FakeDialog:
        def __init__(self):
            self.bindings = {}
            self.protocols = {}
            self.geometry_value = None
            self.exists = True

        def title(self, _value):
            return None

        def configure(self, **_kwargs):
            return None

        def resizable(self, *_args):
            return None

        def attributes(self, *_args):
            return None

        def transient(self, _root):
            return None

        def protocol(self, name, callback):
            self.protocols[name] = callback

        def grid_columnconfigure(self, *_args, **_kwargs):
            return None

        def geometry(self, value):
            self.geometry_value = value

        def winfo_exists(self):
            return self.exists

        def grab_set(self):
            return None

        def grab_release(self):
            return None

        def deiconify(self):
            return None

        def lift(self):
            return None

        def focus_force(self):
            return None

        def bind(self, name, callback):
            self.bindings[name] = callback

        def destroy(self):
            self.exists = False

    class FakeWidget:
        def __init__(self, _parent=None, **kwargs):
            self.kwargs = kwargs
            self.grid_calls = []
            self.pack_calls = []
            self.inserted = None
            self.focused = False

        def grid(self, **kwargs):
            self.grid_calls.append(kwargs)

        def pack(self, **kwargs):
            self.pack_calls.append(kwargs)

        def insert(self, *args):
            self.inserted = args

        def configure(self, **kwargs):
            self.kwargs.update(kwargs)

        def focus_set(self):
            self.focused = True

    class FakeLabel(FakeWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            labels.append(self)

    class FakeFrame(FakeWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            frames.append(self)

    class FakeText(FakeWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            text_areas.append(self)

    class FakeButton(FakeWidget):
        def __init__(self, *args, **kwargs):
            self.command = kwargs.get("command")
            super().__init__(*args, **kwargs)
            buttons.append(self)

    dialog = FakeDialog()
    monkeypatch.setattr(main_module.tk, "Toplevel", lambda _root: dialog)
    monkeypatch.setattr(main_module.tk, "Label", FakeLabel)
    monkeypatch.setattr(main_module.tk, "Frame", FakeFrame)
    monkeypatch.setattr(main_module.scrolledtext, "ScrolledText", FakeText)
    monkeypatch.setattr(main_module.ttk, "Button", FakeButton)

    gui = LocalPilotGUI.__new__(LocalPilotGUI)
    gui.root = FakeRoot()
    gui.colors = {"panel": "#111111", "text": "#ffffff", "muted": "#bbbbbb", "surface": "#222222"}
    gui.safety_var = FakeVar("Waiting for approval")
    gui.approval_window = None

    approved = {"value": False}
    done = threading.Event()
    window = gui._build_approval_window("Approve desktop execution?", approved, done)

    assert window is dialog
    assert "x380" in dialog.geometry_value
    assert any(widget.kwargs.get("text") == "Choose Allow to continue or Deny to cancel." for widget in labels)
    assert text_areas[0].kwargs["height"] == 8
    assert text_areas[0].grid_calls
    assert not text_areas[0].pack_calls
    assert frames[0].grid_calls
    assert "<Return>" in dialog.bindings
    assert "<Escape>" in dialog.bindings

    dialog.bindings["<Return>"](None)

    assert approved["value"] is True
    assert done.is_set() is True
    assert gui.safety_var.get() == "Guarded"
    assert dialog.exists is False
    assert any(button.kwargs.get("text") == "Allow" for button in buttons)
    assert any(button.kwargs.get("text") == "Deny" for button in buttons)


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


def test_describe_lmstudio_screenshot_returns_description(monkeypatch, tmp_path):
    app = LocalPilotApp.__new__(LocalPilotApp)
    app.root_dir = tmp_path
    monkeypatch.setattr(
        main_module,
        "run_lmstudio_vision_test",
        lambda root_dir: (0, "LM Studio screenshot vision test\n- vision response: A desktop with a browser open."),
    )

    report = app.describe_lmstudio_screenshot()

    assert "LM Studio screenshot vision test" in report
    assert "A desktop with a browser open." in report


def test_run_lmstudio_vision_test_returns_clean_unreachable_error(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        """
{
  "lmstudio": {
    "host": "http://localhost:1234/v1",
    "timeout_seconds": 90,
    "text_model": "qwen2.5-coder-14b-instruct",
    "vision_model": "qwen3-vl-8b-instruct",
    "screenshot_dir": "logs/screenshots"
  }
}
""".strip(),
        encoding="utf-8",
    )

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def is_server_available(self):
            return False

    monkeypatch.setattr(main_module, "LMStudioClient", FakeClient)

    exit_code, output = main_module.run_lmstudio_vision_test(tmp_path)

    assert exit_code == 2
    assert "LM Studio URL: http://localhost:1234/v1" in output
    assert "model used: qwen3-vl-8b-instruct" in output
    assert "LM Studio is not reachable at http://localhost:1234/v1" in output


def test_main_handles_lmstudio_vision_test_before_app_bootstrap(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module.Path, "resolve", lambda self: tmp_path / "app" / "main.py")
    monkeypatch.setattr(main_module, "run_lmstudio_vision_test", lambda root_dir: (0, "vision ok"))

    class ExplodingApp:
        def __init__(self, root_dir):
            raise AssertionError("LocalPilotApp should not be constructed for lmstudio vision test")

    printed = []
    monkeypatch.setattr(main_module, "LocalPilotApp", ExplodingApp)
    monkeypatch.setattr(main_module, "safe_console_print", lambda text="": printed.append(text))

    exit_code = main_module.main(["--lmstudio-vision-test"])

    assert exit_code == 0
    assert printed == ["vision ok"]


def test_main_handles_agent_cli_before_app_bootstrap(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module.Path, "resolve", lambda self: tmp_path / "app" / "main.py")
    monkeypatch.setattr(main_module, "run_agent_cli", lambda root_dir: 0)

    class ExplodingApp:
        def __init__(self, root_dir):
            raise AssertionError("LocalPilotApp should not be constructed for agent CLI")

    monkeypatch.setattr(main_module, "LocalPilotApp", ExplodingApp)

    exit_code = main_module.main(["--agent-cli"])

    assert exit_code == 0
