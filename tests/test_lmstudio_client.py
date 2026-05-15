import base64
from pathlib import Path

import pytest
import requests

from app.lmstudio_client import LMStudioClient


def test_encode_image_as_data_url_returns_base64_png_prefix(tmp_path):
    image_path = tmp_path / "sample.png"
    raw = b"\x89PNG\r\n\x1a\nfakepng"
    image_path.write_bytes(raw)
    client = LMStudioClient()

    encoded = client.encode_image_as_data_url(image_path)

    assert encoded.startswith("data:image/png;base64,")
    assert encoded.split(",", 1)[1] == base64.b64encode(raw).decode("ascii")


def test_encode_image_as_data_url_raises_for_missing_file(tmp_path):
    client = LMStudioClient()

    with pytest.raises(FileNotFoundError):
        client.encode_image_as_data_url(tmp_path / "missing.png")


def test_build_vision_payload_uses_qwen_vl_image_url_format(tmp_path):
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nscreen")
    client = LMStudioClient()

    payload = client._build_vision_payload(
        prompt="Describe the screenshot.",
        image_path=image_path,
        model="qwen3-vl-8b-instruct",
        max_tokens=256,
    )

    assert payload["model"] == "qwen3-vl-8b-instruct"
    assert payload["max_tokens"] == 256
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Describe the screenshot."}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_chat_vision_posts_chat_completions_payload(tmp_path, monkeypatch):
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nscreen")
    client = LMStudioClient(host="http://localhost:1234/v1", timeout_seconds=12)
    seen = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "A browser window with a search box."}}]}

    def fake_post(url, json=None, timeout=None):
        seen["url"] = url
        seen["json"] = json
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("app.lmstudio_client.requests.post", fake_post)

    result = client.chat_vision("Describe this screenshot.", image_path, max_tokens=111)

    assert result == "A browser window with a search box."
    assert seen["url"] == "http://localhost:1234/v1/chat/completions"
    assert seen["timeout"] == 12
    assert seen["json"]["max_tokens"] == 111
    assert seen["json"]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_is_server_available_returns_false_on_request_error(monkeypatch):
    client = LMStudioClient()

    def fake_get(*args, **kwargs):
        raise requests.RequestException("boom")

    monkeypatch.setattr("app.lmstudio_client.requests.get", fake_get)

    assert client.is_server_available() is False
