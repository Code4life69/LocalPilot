from pathlib import Path

from app.desktop_tool import (
    DesktopSuggestionStore,
    _is_sensitive_desktop_context,
    execute_suggestion_click,
    get_mouse_position,
    get_screen_size,
    move_mouse_preview,
    suggest_action_from_screenshot,
)


class FakePyAutoGUI:
    def __init__(self):
        self.moves = []
        self.clicks = []

    def size(self):
        return (1920, 1080)

    def position(self):
        return (320, 240)

    def moveTo(self, x, y, duration=0.0):
        self.moves.append((x, y, duration))

    def click(self, x, y):
        self.clicks.append((x, y))


class FakeVisionClient:
    default_vision_model = "qwen3-vl-8b-instruct"

    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    def chat_vision(self, prompt, image_path, model, max_tokens):
        self.calls.append(
            {
                "prompt": prompt,
                "image_path": str(image_path),
                "model": model,
                "max_tokens": max_tokens,
            }
        )
        return self.response_text


def test_get_screen_size_reads_pyautogui_dimensions():
    result = get_screen_size(pyautogui_module=FakePyAutoGUI())

    assert result == {"ok": True, "width": 1920, "height": 1080}


def test_get_mouse_position_reads_pyautogui_position():
    result = get_mouse_position(pyautogui_module=FakePyAutoGUI())

    assert result == {"ok": True, "x": 320, "y": 240}


def test_move_mouse_preview_moves_without_click():
    fake_gui = FakePyAutoGUI()

    result = move_mouse_preview(400, 300, target="Search field", confidence=0.88, pyautogui_module=fake_gui)

    assert result["ok"] is True
    assert result["preview_only"] is True
    assert result["message"] == "Mouse moved for preview only. No click was performed."
    assert fake_gui.moves == [(400, 300, 0.15)]


def test_suggest_action_from_screenshot_returns_structured_schema(tmp_path):
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"png")
    client = FakeVisionClient(
        """
        {
          "action": "click",
          "target": "Google search bar",
          "x": 735,
          "y": 410,
          "confidence": 0.86,
          "risk": "medium",
          "reason": "The user wants to search, and this appears to be the main search field."
        }
        """
    )

    result = suggest_action_from_screenshot(image_path, "Tell me what you would click next.", client)

    assert result["ok"] is True
    assert result["action"] == "click"
    assert result["target"] == "Google search bar"
    assert result["x"] == 735
    assert result["y"] == 410
    assert result["confidence"] == 0.86
    assert result["requires_approval_to_execute"] is True
    assert result["executed"] is False
    assert result["desktop_action"]["mode"] == "dry_run"


def test_suggest_action_from_screenshot_handles_json_parse_failure(tmp_path):
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"png")
    client = FakeVisionClient("not json")

    result = suggest_action_from_screenshot(image_path, "Suggest the next action.", client)

    assert result["ok"] is False
    assert "could not parse model json" in result["error"].lower()


def test_low_confidence_suggestion_is_marked_not_executable(tmp_path):
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"png")
    client = FakeVisionClient(
        """
        {
          "action": "click",
          "target": "Unclear icon",
          "x": 100,
          "y": 200,
          "confidence": 0.42,
          "risk": "medium",
          "reason": "This might be the right control, but it is hard to read."
        }
        """
    )

    result = suggest_action_from_screenshot(image_path, "Tell me what you would click next.", client)

    assert result["ok"] is True
    assert result["can_execute"] is False
    assert result["can_preview_move"] is False
    assert result["next_step"] == "ask_for_clarification"
    assert "below 0.60" in result["warning"]


def test_suggestion_store_saves_suggestion_id_and_runtime_record(tmp_path):
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"png")
    store = DesktopSuggestionStore(tmp_path / "memory" / "runtime" / "desktop_suggestions.json")

    record = store.create_suggestion(
        task_id="task123",
        suggestion={
            "action": "click",
            "target": "Google search bar",
            "x": 735,
            "y": 410,
            "confidence": 0.86,
            "risk": "medium",
            "reason": "The user wants to search.",
        },
        screenshot_path=image_path,
    )

    saved_payload = store.path.read_text(encoding="utf-8")
    assert record["suggestion_id"].startswith("desk_suggest_")
    assert "Google search bar" in saved_payload
    assert record["screenshot_hash"]


def test_execute_suggestion_click_blocks_expired_suggestion(tmp_path):
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"png")
    store = DesktopSuggestionStore(tmp_path / "memory" / "runtime" / "desktop_suggestions.json", ttl_seconds=30)
    record = store.create_suggestion(
        task_id="task123",
        suggestion={
            "action": "click",
            "target": "Google search bar",
            "x": 735,
            "y": 410,
            "confidence": 0.86,
            "risk": "medium",
            "reason": "The user wants to search.",
        },
        screenshot_path=image_path,
    )
    suggestions = store._load()
    suggestions[0]["expires_at"] = "2000-01-01T00:00:00"
    store._save(suggestions)

    result = execute_suggestion_click(record["suggestion_id"], store, pyautogui_module=FakePyAutoGUI(), pre_click_delay_seconds=0.0)

    assert result["ok"] is False
    assert "expired" in result["error"].lower()


def test_execute_suggestion_click_blocks_already_executed_suggestion(tmp_path):
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"png")
    store = DesktopSuggestionStore(tmp_path / "memory" / "runtime" / "desktop_suggestions.json")
    record = store.create_suggestion(
        task_id="task123",
        suggestion={
            "action": "click",
            "target": "Google search bar",
            "x": 735,
            "y": 410,
            "confidence": 0.86,
            "risk": "medium",
            "reason": "The user wants to search.",
        },
        screenshot_path=image_path,
    )
    store.mark_executed(record["suggestion_id"])

    result = execute_suggestion_click(record["suggestion_id"], store, pyautogui_module=FakePyAutoGUI(), pre_click_delay_seconds=0.0)

    assert result["ok"] is False
    assert "already executed" in result["error"].lower()


def test_execute_suggestion_click_blocks_low_confidence_suggestion(tmp_path):
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"png")
    store = DesktopSuggestionStore(tmp_path / "memory" / "runtime" / "desktop_suggestions.json")
    record = store.create_suggestion(
        task_id="task123",
        suggestion={
            "action": "click",
            "target": "Unclear icon",
            "x": 100,
            "y": 120,
            "confidence": 0.79,
            "risk": "medium",
            "reason": "This might be the right thing.",
        },
        screenshot_path=image_path,
    )

    result = execute_suggestion_click(record["suggestion_id"], store, pyautogui_module=FakePyAutoGUI(), pre_click_delay_seconds=0.0)

    assert result["ok"] is False
    assert "below 0.80" in result["error"]


def test_execute_suggestion_click_blocks_sensitive_target(tmp_path):
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"png")
    store = DesktopSuggestionStore(tmp_path / "memory" / "runtime" / "desktop_suggestions.json")
    record = store.create_suggestion(
        task_id="task123",
        suggestion={
            "action": "click",
            "target": "Send email button",
            "x": 400,
            "y": 300,
            "confidence": 0.95,
            "risk": "dangerous",
            "reason": "This would send the email.",
        },
        screenshot_path=image_path,
    )

    result = execute_suggestion_click(record["suggestion_id"], store, pyautogui_module=FakePyAutoGUI(), pre_click_delay_seconds=0.0)

    assert result["ok"] is False
    assert "sensitive" in result["error"].lower()


def test_execute_suggestion_click_blocks_out_of_bounds_coordinates(tmp_path):
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"png")
    store = DesktopSuggestionStore(tmp_path / "memory" / "runtime" / "desktop_suggestions.json")
    record = store.create_suggestion(
        task_id="task123",
        suggestion={
            "action": "click",
            "target": "Far away control",
            "x": 9999,
            "y": 9999,
            "confidence": 0.95,
            "risk": "medium",
            "reason": "Test bad coordinates.",
        },
        screenshot_path=image_path,
    )

    result = execute_suggestion_click(record["suggestion_id"], store, pyautogui_module=FakePyAutoGUI(), pre_click_delay_seconds=0.0)

    assert result["ok"] is False
    assert "outside the screen bounds" in result["error"]


def test_execute_suggestion_click_calls_pyautogui_with_expected_coordinates(tmp_path):
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"png")
    store = DesktopSuggestionStore(tmp_path / "memory" / "runtime" / "desktop_suggestions.json")
    record = store.create_suggestion(
        task_id="task123",
        suggestion={
            "action": "click",
            "target": "Google search bar",
            "x": 735,
            "y": 410,
            "confidence": 0.86,
            "risk": "medium",
            "reason": "The user wants to search.",
        },
        screenshot_path=image_path,
    )
    fake_gui = FakePyAutoGUI()

    result = execute_suggestion_click(record["suggestion_id"], store, pyautogui_module=fake_gui, pre_click_delay_seconds=0.0)

    assert result["ok"] is True
    assert fake_gui.clicks == [(735, 410)]
    saved = store.get_suggestion(record["suggestion_id"])
    assert saved is not None
    assert saved["executed"] is True


def test_safe_test_button_target_is_not_classified_as_sensitive():
    assert _is_sensitive_desktop_context("SAFE TEST BUTTON Click the safe local test button.") is False
