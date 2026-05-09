import json
from pathlib import Path

from app.llm.ollama_client import OllamaClient


def load_model_profiles() -> dict:
    return json.loads(Path("config/model_profiles.json").read_text(encoding="utf-8"))


def test_model_profiles_default_main_is_qwen3_8b():
    profiles = load_model_profiles()

    assert profiles["main"]["model"] == "qwen3:8b"
    assert profiles["main"]["model"] != "qwen3:30b"


def test_model_profiles_keep_quality_slow_role():
    profiles = load_model_profiles()

    assert profiles["quality_slow"]["model"] == "qwen3:30b"


def test_coder_role_falls_back_when_primary_missing():
    profiles = load_model_profiles()
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
    )

    resolved = client.resolve_model_for_role(
        "coder",
        available=["qwen2.5-coder:7b", "qwen3:8b"],
    )

    assert resolved == "qwen2.5-coder:7b"


def test_model_status_report_handles_ollama_unavailable():
    profiles = load_model_profiles()
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
    )
    client.is_server_available = lambda: False

    report = client.build_model_status_report(default_role="main")

    assert "Model status" in report
    assert "- Ollama reachable: no" in report
    assert "main: preferred=qwen3:8b" in report
