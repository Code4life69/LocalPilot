from app.modes.desktop_mode import DesktopMode


class DummyLogger:
    def event(self, *args, **kwargs):
        return None


class DummyApp:
    def __init__(self):
        self.logger = DummyLogger()
        self.settings = {"screenshots_dir": "workspace/screenshots"}
        self.root_dir = "."

    def ask_approval(self, prompt):
        return True


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
