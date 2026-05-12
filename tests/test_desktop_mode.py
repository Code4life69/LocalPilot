from app.modes.desktop_mode import DesktopMode


class DummyLogger:
    def event(self, *args, **kwargs):
        return None


class DummyLessonStore:
    def __init__(self):
        self.entries = []

    def record(self, lesson_type, task, reason, **extra):
        self.entries.append(
            {
                "type": lesson_type,
                "task": task,
                "reason": reason,
                "extra": extra,
            }
        )

    def render_recent(self, limit=20):
        return "Desktop lessons:\n- sample lesson"


class DummyApp:
    def __init__(self):
        self.logger = DummyLogger()
        self.settings = {
            "screenshots_dir": "workspace/screenshots",
            "page_understanding": {"confidence_threshold": 0.85},
        }
        self.root_dir = "."
        self.desktop_lessons = DummyLessonStore()
        self.ollama = type("Ollama", (), {"analyze_screenshot": staticmethod(lambda prompt, path: "Vision says the page is present.")})()

    def ask_approval(self, prompt):
        return True

    def run_guarded_desktop_action(self, _action_name, action):
        return action()


def test_inspect_desktop_returns_real_observation_summary(monkeypatch):
    monkeypatch.setattr(
        "app.modes.desktop_mode.get_active_window_title",
        lambda: {"ok": True, "title": "Visual Studio Code"},
    )
    monkeypatch.setattr(
        "app.modes.desktop_mode.get_active_window_basic",
        lambda: {"ok": True, "title": "Fallback Window"},
    )
    monkeypatch.setattr(
        "app.modes.desktop_mode.get_focused_control",
        lambda: {"ok": True, "control_type": "Edit", "name": "Editor"},
    )
    monkeypatch.setattr(
        "app.modes.desktop_mode.get_mouse_position",
        lambda: {"ok": True, "x": 100, "y": 200},
    )
    monkeypatch.setattr(
        "app.modes.desktop_mode.get_control_at_point",
        lambda x, y: {"ok": True, "control_type": "Button", "name": "Run", "x": x, "y": y},
    )
    monkeypatch.setattr(
        "app.modes.desktop_mode.list_visible_controls",
        lambda max_depth=1: {"ok": True, "controls": [{"control_type": "Edit", "name": "Editor"}]},
    )

    mode = DesktopMode(DummyApp())
    result = mode.handle({"user_text": "inspect desktop"})

    assert result["ok"]
    assert "Active window: Visual Studio Code" in result["content"]
    assert "Focused control: Edit: Editor" in result["content"]
    assert "Mouse position: (100, 200)" in result["content"]
    assert "Under mouse: Button: Run" in result["content"]


def test_what_window_am_i_on_uses_window_tools(monkeypatch):
    monkeypatch.setattr(
        "app.modes.desktop_mode.get_active_window_title",
        lambda: {"ok": True, "title": "Google Chrome"},
    )
    mode = DesktopMode(DummyApp())

    result = mode.handle({"user_text": "what window am I on"})

    assert result["ok"]
    assert result["content"] == "Active window: Google Chrome"


def test_desktop_observation_reports_dependency_missing_clearly(monkeypatch):
    monkeypatch.setattr(
        "app.modes.desktop_mode.get_active_window_title",
        lambda: {"ok": True, "title": "Explorer"},
    )
    monkeypatch.setattr(
        "app.modes.desktop_mode.get_mouse_position",
        lambda: {"ok": True, "x": 10, "y": 20},
    )
    missing_payload = {
        "ok": False,
        "reason": "dependency_missing",
        "dependency": "uiautomation",
        "error": "uiautomation is not installed in the active Python environment.",
        "fix": r".\.venv\Scripts\python.exe -m pip install uiautomation",
    }
    monkeypatch.setattr("app.modes.desktop_mode.get_focused_control", lambda: dict(missing_payload))
    monkeypatch.setattr("app.modes.desktop_mode.get_control_at_point", lambda x, y: dict(missing_payload, x=x, y=y))
    monkeypatch.setattr("app.modes.desktop_mode.list_visible_controls", lambda max_depth=1: dict(missing_payload))

    mode = DesktopMode(DummyApp())
    result = mode.handle({"user_text": "inspect desktop"})

    assert result["ok"]
    assert "dependency_missing" in result["content"]
    assert "uiautomation" in result["content"]


def test_page_inspect_returns_snapshot_summary(monkeypatch):
    monkeypatch.setattr(
        "app.tools.page_understanding.get_active_window_title",
        lambda: {"ok": True, "title": "Google Chrome"},
    )
    monkeypatch.setattr(
        "app.tools.page_understanding.get_focused_control",
        lambda: {"ok": True, "control_type": "Document", "name": "Search results", "bounds": {"left": 1, "top": 1, "right": 10, "bottom": 10}},
    )
    monkeypatch.setattr(
        "app.tools.page_understanding.get_mouse_position",
        lambda: {"ok": True, "x": 40, "y": 50},
    )
    monkeypatch.setattr(
        "app.tools.page_understanding.get_control_at_point",
        lambda x, y: {"ok": True, "control_type": "Link", "name": "Result", "bounds": {"left": x, "top": y, "right": x + 10, "bottom": y + 10}},
    )
    monkeypatch.setattr(
        "app.tools.page_understanding.list_visible_controls",
        lambda max_depth=1: {"ok": True, "controls": [{"control_type": "Link", "name": "Result", "bounds": {"left": 1, "top": 1, "right": 10, "bottom": 10}}]},
    )
    monkeypatch.setattr(
        "app.tools.page_understanding.take_screenshot",
        lambda _output_dir: {"ok": True, "path": "workspace/screenshots/page.png"},
    )

    mode = DesktopMode(DummyApp())
    result = mode.handle({"user_text": "page inspect"})

    assert result["ok"] is True
    assert "Page inspect" in result["content"]
    assert "Active window: Google Chrome" in result["content"]
    assert "Screenshot: workspace/screenshots/page.png" in result["content"]


def test_show_desktop_lessons_reads_local_store():
    mode = DesktopMode(DummyApp())

    result = mode.handle({"user_text": "show desktop lessons"})

    assert result["ok"] is True
    assert "Desktop lessons" in result["content"]


def test_click_is_blocked_below_confidence_threshold(monkeypatch):
    monkeypatch.setattr(
        "app.tools.page_understanding.get_active_window_title",
        lambda: {"ok": True, "title": "Google Chrome"},
    )
    monkeypatch.setattr(
        "app.tools.page_understanding.get_focused_control",
        lambda: {"ok": False, "reason": "dependency_missing", "dependency": "uiautomation", "error": "missing", "fix": "install"},
    )
    monkeypatch.setattr(
        "app.tools.page_understanding.get_mouse_position",
        lambda: {"ok": True, "x": 40, "y": 50},
    )
    monkeypatch.setattr(
        "app.tools.page_understanding.get_control_at_point",
        lambda x, y: {"ok": False, "reason": "dependency_missing", "dependency": "uiautomation", "error": "missing", "fix": "install", "x": x, "y": y},
    )
    monkeypatch.setattr(
        "app.tools.page_understanding.list_visible_controls",
        lambda max_depth=1: {"ok": False, "reason": "dependency_missing", "dependency": "uiautomation", "error": "missing", "fix": "install"},
    )
    monkeypatch.setattr(
        "app.tools.page_understanding.take_screenshot",
        lambda _output_dir: {"ok": True, "path": "workspace/screenshots/page.png"},
    )

    app = DummyApp()
    mode = DesktopMode(app)
    result = mode.handle({"user_text": "click 10 20"})

    assert result["ok"] is False
    assert result["verification_source"] == "confidence_gate"
    assert "Refused click at 10, 20" in result["error"]
    assert app.desktop_lessons.entries[0]["type"] == "confidence_gate_refusal"
