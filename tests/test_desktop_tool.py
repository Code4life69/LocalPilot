from pathlib import Path

from app.desktop_tool import (
    get_mouse_position,
    get_screen_size,
    move_mouse_preview,
    suggest_action_from_screenshot,
)


class FakePyAutoGUI:
    def __init__(self):
        self.moves = []

    def size(self):
        return (1920, 1080)

    def position(self):
        return (320, 240)

    def moveTo(self, x, y, duration=0.0):
        self.moves.append((x, y, duration))


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
