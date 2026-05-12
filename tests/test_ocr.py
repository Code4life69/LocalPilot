from pathlib import Path
import subprocess

from PIL import Image

from app.modes.desktop_mode import DesktopMode
from app.tools.ocr import preprocess_image, read_image
from app.tools.page_understanding import PageUnderstandingEngine


class DummyLogger:
    def event(self, *args, **kwargs):
        return None


class DummyLessonStore:
    def __init__(self):
        self.entries = []

    def record(self, lesson_type, task, reason, **extra):
        self.entries.append({"type": lesson_type, "task": task, "reason": reason, "extra": extra})

    def render_recent(self, limit=20):
        return "No desktop lessons recorded yet."


class DummyApp:
    def __init__(self, root_dir: Path):
        self.logger = DummyLogger()
        self.settings = {
            "screenshots_dir": str(root_dir / "workspace" / "screenshots"),
            "page_understanding": {"confidence_threshold": 0.85},
        }
        self.root_dir = root_dir
        self.desktop_lessons = DummyLessonStore()
        self.ollama = type("Ollama", (), {"analyze_screenshot": staticmethod(lambda prompt, path: "Vision says the page is present.")})()

    def ask_approval(self, prompt):
        return True

    def run_guarded_desktop_action(self, _action_name, action):
        return action()


def test_ocr_unavailable_path_does_not_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.tools.ocr.get_ocr_backend_status",
        lambda: {
            "available": False,
            "backend": "pytesseract",
            "error": "OCR backend unavailable",
            "install_hint": "Install Tesseract and pytesseract",
            "tesseract_path": "",
        },
    )
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 40), "white").save(image_path)

    result = read_image(image_path, output_dir=tmp_path)

    assert result["ok"] is False
    assert result["backend"] == "pytesseract"
    assert "OCR backend unavailable" in result["error"]
    assert "Install Tesseract and pytesseract" in result["install_hint"]


def test_preprocess_image_creates_grayscale_processed_image(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (400, 200), "white").save(image_path)

    result = preprocess_image(image_path, tmp_path)

    assert result["ok"] is True
    processed_path = Path(result["processed_image"])
    assert processed_path.exists()
    with Image.open(processed_path) as processed:
        assert processed.mode == "L"
        assert processed.width > 400
        assert processed.width <= 1600


def test_ocr_command_returns_clear_unavailable_message(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.modes.desktop_mode.read_screenshot",
        lambda screenshots_dir, debug_views_dir: {
            "ok": False,
            "backend": "pytesseract",
            "source_image": "",
            "text": "",
            "blocks": [],
            "confidence": 0.0,
            "error": "OCR backend unavailable",
            "install_hint": "Install Tesseract and pytesseract",
        },
    )
    mode = DesktopMode(DummyApp(tmp_path))

    result = mode.handle({"user_text": "ocr screenshot"})

    assert result["ok"] is False
    assert "OCR unavailable" in result["content"]
    assert "Install Tesseract and pytesseract" in result["content"]


def test_page_state_includes_ocr_fields(monkeypatch, tmp_path):
    monkeypatch.setattr("app.tools.page_understanding.get_active_window_title", lambda: {"ok": True, "title": "Google Chrome"})
    monkeypatch.setattr("app.tools.page_understanding.get_focused_control", lambda: {"ok": True, "control_type": "Edit", "name": "Search box", "bounds": {"left": 1, "top": 1, "right": 10, "bottom": 10}})
    monkeypatch.setattr("app.tools.page_understanding.list_visible_controls", lambda max_depth=1: {"ok": True, "controls": []})
    monkeypatch.setattr("app.tools.page_understanding.get_mouse_position", lambda: {"ok": True, "x": 10, "y": 20})
    monkeypatch.setattr("app.tools.page_understanding.get_control_at_point", lambda x, y: {"ok": True, "control_type": "Edit", "name": "Search box", "bounds": {"left": 1, "top": 1, "right": 10, "bottom": 10}})
    monkeypatch.setattr("app.tools.page_understanding.take_screenshot", lambda output_dir: {"ok": True, "path": str(tmp_path / "screen.png")})
    monkeypatch.setattr(
        "app.tools.page_understanding.read_image",
        lambda image_path, output_dir: {
            "ok": True,
            "backend": "pytesseract",
            "source_image": str(image_path),
            "processed_image": str(tmp_path / "processed.png"),
            "text": "Google dolphins images",
            "blocks": [{"text": "Google", "confidence": 92.0}],
            "confidence": 92.0,
        },
    )

    app = DummyApp(tmp_path)
    snapshot = PageUnderstandingEngine(app).snapshot(capture_screenshot=True)

    assert snapshot["ocr_available"] is True
    assert snapshot["ocr_backend"] == "pytesseract"
    assert "Google dolphins images" in snapshot["ocr_text"]
    assert snapshot["ocr_blocks"]


def test_confidence_can_use_ocr_text_but_cannot_click_from_ocr_alone(monkeypatch, tmp_path):
    monkeypatch.setattr("app.tools.page_understanding.get_active_window_title", lambda: {"ok": True, "title": "Google Chrome"})
    monkeypatch.setattr("app.tools.page_understanding.get_focused_control", lambda: {"ok": False, "error": "focused control unavailable"})
    monkeypatch.setattr("app.tools.page_understanding.list_visible_controls", lambda max_depth=1: {"ok": True, "controls": []})
    monkeypatch.setattr("app.tools.page_understanding.get_mouse_position", lambda: {"ok": True, "x": 10, "y": 20})
    monkeypatch.setattr("app.tools.page_understanding.get_control_at_point", lambda x, y: {"ok": False, "error": "no bounds"})
    monkeypatch.setattr("app.tools.page_understanding.take_screenshot", lambda output_dir: {"ok": True, "path": str(tmp_path / "screen.png")})
    monkeypatch.setattr(
        "app.tools.page_understanding.read_image",
        lambda image_path, output_dir: {
            "ok": True,
            "backend": "pytesseract",
            "source_image": str(image_path),
            "processed_image": str(tmp_path / "processed.png"),
            "text": "Dolphins images Google Search",
            "blocks": [{"text": "Dolphins", "confidence": 91.0}],
            "confidence": 91.0,
        },
    )

    app = DummyApp(tmp_path)
    assessment = PageUnderstandingEngine(app).assess(
        action_kind="click",
        action_text="click dolphins images result in the browser",
        include_vision=False,
    )

    assert any("OCR matched expected text" in item for item in assessment["confidence_evidence"])
    assert assessment["confidence_allowed"] is False
    assert "no target bounds exist" in assessment["confidence_reason"]


def test_known_good_google_results_page_scores_above_threshold(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.tools.page_understanding.get_active_window_title",
        lambda: {"ok": True, "title": "Code4life69 LocalPilot issue 4 - Google Search - Google Chrome"},
    )
    monkeypatch.setattr(
        "app.tools.page_understanding.get_focused_control",
        lambda: {
            "ok": True,
            "control_type": "DocumentControl",
            "name": "Code4life69 LocalPilot issue 4 - Google Search",
            "bounds": {"left": 1, "top": 1, "right": 10, "bottom": 10},
        },
    )
    monkeypatch.setattr("app.tools.page_understanding.list_visible_controls", lambda max_depth=1: {"ok": True, "controls": [{"control_type": "PaneControl", "name": "Chrome", "bounds": {"left": 1, "top": 1, "right": 10, "bottom": 10}}]})
    monkeypatch.setattr("app.tools.page_understanding.get_mouse_position", lambda: {"ok": True, "x": 10, "y": 20})
    monkeypatch.setattr("app.tools.page_understanding.get_control_at_point", lambda x, y: {"ok": True, "control_type": "GroupControl", "name": "(unnamed)", "bounds": {"left": 1, "top": 1, "right": 10, "bottom": 10}})
    monkeypatch.setattr("app.tools.page_understanding.take_screenshot", lambda output_dir: {"ok": True, "path": str(tmp_path / "screen.png")})
    monkeypatch.setattr(
        "app.tools.page_understanding.read_image",
        lambda image_path, output_dir: {
            "ok": True,
            "backend": "pytesseract",
            "source_image": str(image_path),
            "processed_image": str(tmp_path / "processed.png"),
            "text": "Code4life69 LocalPilot issue 4 Google Search results",
            "blocks": [{"text": "Code4life69", "confidence": 92.0}],
            "confidence": 92.0,
        },
    )

    app = DummyApp(tmp_path)
    assessment = PageUnderstandingEngine(app).assess(
        action_kind="inspect",
        action_text="page confidence",
        include_vision=False,
    )

    assert assessment["confidence_score"] >= 0.85
    assert assessment["confidence_allowed"] is True


def test_discord_active_window_scores_below_threshold(monkeypatch, tmp_path):
    monkeypatch.setattr("app.tools.page_understanding.get_active_window_title", lambda: {"ok": True, "title": "Chatting - Discord"})
    monkeypatch.setattr(
        "app.tools.page_understanding.get_focused_control",
        lambda: {"ok": True, "control_type": "DocumentControl", "name": "Discord chat", "bounds": {"left": 1, "top": 1, "right": 10, "bottom": 10}},
    )
    monkeypatch.setattr("app.tools.page_understanding.list_visible_controls", lambda max_depth=1: {"ok": True, "controls": [{"control_type": "PaneControl", "name": "Discord", "bounds": {"left": 1, "top": 1, "right": 10, "bottom": 10}}]})
    monkeypatch.setattr("app.tools.page_understanding.get_mouse_position", lambda: {"ok": True, "x": 10, "y": 20})
    monkeypatch.setattr("app.tools.page_understanding.get_control_at_point", lambda x, y: {"ok": True, "control_type": "GroupControl", "name": "Discord", "bounds": {"left": 1, "top": 1, "right": 10, "bottom": 10}})
    monkeypatch.setattr("app.tools.page_understanding.take_screenshot", lambda output_dir: {"ok": True, "path": str(tmp_path / "screen.png")})
    monkeypatch.setattr(
        "app.tools.page_understanding.read_image",
        lambda image_path, output_dir: {
            "ok": True,
            "backend": "pytesseract",
            "source_image": str(image_path),
            "processed_image": str(tmp_path / "processed.png"),
            "text": "Discord chat window",
            "blocks": [{"text": "Discord", "confidence": 92.0}],
            "confidence": 92.0,
        },
    )

    app = DummyApp(tmp_path)
    assessment = PageUnderstandingEngine(app).assess(
        action_kind="inspect",
        action_text="page confidence",
        include_vision=False,
    )

    assert assessment["confidence_score"] < 0.85
    assert assessment["confidence_allowed"] is False


def test_focused_document_title_increases_confidence(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.tools.page_understanding.get_active_window_title",
        lambda: {"ok": True, "title": "Code4life69 LocalPilot issue 4 - Google Search - Google Chrome"},
    )
    monkeypatch.setattr("app.tools.page_understanding.list_visible_controls", lambda max_depth=1: {"ok": True, "controls": []})
    monkeypatch.setattr("app.tools.page_understanding.get_mouse_position", lambda: {"ok": True, "x": 10, "y": 20})
    monkeypatch.setattr("app.tools.page_understanding.get_control_at_point", lambda x, y: {"ok": True, "control_type": "GroupControl", "name": "(unnamed)", "bounds": {"left": 1, "top": 1, "right": 10, "bottom": 10}})
    monkeypatch.setattr("app.tools.page_understanding.take_screenshot", lambda output_dir: {"ok": True, "path": str(tmp_path / "screen.png")})
    monkeypatch.setattr(
        "app.tools.page_understanding.read_image",
        lambda image_path, output_dir: {
            "ok": True,
            "backend": "pytesseract",
            "source_image": str(image_path),
            "processed_image": str(tmp_path / "processed.png"),
            "text": "Code4life69 LocalPilot issue 4 Google Search results",
            "blocks": [{"text": "Code4life69", "confidence": 92.0}],
            "confidence": 92.0,
        },
    )

    app = DummyApp(tmp_path)
    engine = PageUnderstandingEngine(app)

    monkeypatch.setattr(
        "app.tools.page_understanding.get_focused_control",
        lambda: {"ok": True, "control_type": "ButtonControl", "name": "Search", "bounds": {"left": 1, "top": 1, "right": 10, "bottom": 10}},
    )
    lower = engine.assess(action_kind="inspect", action_text="page confidence", include_vision=False)

    monkeypatch.setattr(
        "app.tools.page_understanding.get_focused_control",
        lambda: {
            "ok": True,
            "control_type": "DocumentControl",
            "name": "Code4life69 LocalPilot issue 4 - Google Search",
            "bounds": {"left": 1, "top": 1, "right": 10, "bottom": 10},
        },
    )
    higher = engine.assess(action_kind="inspect", action_text="page confidence", include_vision=False)

    assert higher["confidence_score"] > lower["confidence_score"]


def test_negative_vision_overrides_positive_term_presence(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.tools.page_understanding.get_active_window_title",
        lambda: {"ok": True, "title": "Code4life69 LocalPilot issue 4 - Google Search - Google Chrome"},
    )
    monkeypatch.setattr(
        "app.tools.page_understanding.get_focused_control",
        lambda: {
            "ok": True,
            "control_type": "DocumentControl",
            "name": "Code4life69 LocalPilot issue 4 - Google Search",
            "bounds": {"left": 1, "top": 1, "right": 10, "bottom": 10},
        },
    )
    monkeypatch.setattr("app.tools.page_understanding.list_visible_controls", lambda max_depth=1: {"ok": True, "controls": []})
    monkeypatch.setattr("app.tools.page_understanding.get_mouse_position", lambda: {"ok": True, "x": 10, "y": 20})
    monkeypatch.setattr("app.tools.page_understanding.get_control_at_point", lambda x, y: {"ok": True, "control_type": "GroupControl", "name": "(unnamed)", "bounds": {"left": 1, "top": 1, "right": 10, "bottom": 10}})
    monkeypatch.setattr("app.tools.page_understanding.take_screenshot", lambda output_dir: {"ok": True, "path": str(tmp_path / "screen.png")})
    monkeypatch.setattr(
        "app.tools.page_understanding.read_image",
        lambda image_path, output_dir: {
            "ok": True,
            "backend": "pytesseract",
            "source_image": str(image_path),
            "processed_image": str(tmp_path / "processed.png"),
            "text": "Code4life69 LocalPilot issue 4 Google Search results",
            "blocks": [{"text": "Code4life69", "confidence": 92.0}],
            "confidence": 92.0,
        },
    )

    app = DummyApp(tmp_path)
    assessment = PageUnderstandingEngine(app).assess(
        action_kind="inspect",
        action_text="search for Code4life69 LocalPilot issue 4 in the browser",
        include_vision=False,
    )
    assessment["vision_summary"] = "No, this is not the expected Google results page."
    assessment["confidence_allowed"] = False

    app = DummyApp(tmp_path)
    monkeypatch.setattr(
        PageUnderstandingEngine,
        "snapshot",
        lambda self, **kwargs: {
            "ok": True,
            "active_window": {"ok": True, "title": "Code4life69 LocalPilot issue 4 - Google Search - Google Chrome"},
            "focused_control": {
                "ok": True,
                "control_type": "DocumentControl",
                "name": "Code4life69 LocalPilot issue 4 - Google Search",
                "bounds": {"left": 1, "top": 1, "right": 10, "bottom": 10},
            },
            "visible_controls": {"ok": True, "controls": []},
            "mouse_position": {"ok": True, "x": 10, "y": 20},
            "target_control": {"ok": True, "control_type": "GroupControl", "name": "(unnamed)", "bounds": {"left": 1, "top": 1, "right": 10, "bottom": 10}},
            "screenshot": {"ok": True, "path": str(tmp_path / "screen.png")},
            "screenshot_path": str(tmp_path / "screen.png"),
            "candidate_targets": [],
            "ocr_available": True,
            "ocr_text": "Code4life69 LocalPilot issue 4 Google Search results",
            "ocr_blocks": [],
            "ocr_backend": "pytesseract",
            "ocr_confidence": 92.0,
            "vision_summary": "No, this is not the expected Google results page.",
        },
    )
    blocked = PageUnderstandingEngine(app).assess(
        action_kind="inspect",
        action_text="search for Code4life69 LocalPilot issue 4 in the browser",
        include_vision=True,
    )

    assert blocked["confidence_allowed"] is False
    assert blocked["confidence_reason"] == "vision says the target or page is not present"


def test_workspace_debug_view_artifacts_are_git_ignored(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    artifact = repo_root / "workspace" / "debug_views" / "ocr_test_artifact.png"
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(artifact)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
