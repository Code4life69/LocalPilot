import json
from pathlib import Path

from PIL import Image

from app.llm.ollama_client import OllamaClient


def load_model_profiles() -> dict:
    return json.loads(Path("config/model_profiles.json").read_text(encoding="utf-8"))


def load_performance_profiles() -> dict:
    return json.loads(Path("config/performance_profiles.json").read_text(encoding="utf-8"))


def load_settings() -> dict:
    return json.loads(Path("config/settings.json").read_text(encoding="utf-8"))


def load_install_script() -> str:
    return Path("scripts/install_recommended_models.ps1").read_text(encoding="utf-8")


def load_optional_gemma_install_script() -> str:
    return Path("scripts/install_optional_gemma4.ps1").read_text(encoding="utf-8")


def test_model_profiles_default_main_is_gemma4_31b():
    profiles = load_model_profiles()

    assert profiles["main"]["model"] == "gemma4:31b"
    assert profiles["main"]["model"] != "qwen3:30b"


def test_model_profiles_keep_quality_slow_role():
    profiles = load_model_profiles()

    assert profiles["quality_slow"]["model"] == "qwen3:30b"


def test_model_profiles_include_optional_gemma4_comparison_roles():
    profiles = load_model_profiles()

    assert profiles["gemma4_fast"]["model"] == "gemma4:e4b"
    assert profiles["gemma4_quality"]["model"] == "gemma4:latest"


def test_performance_profiles_default_is_rtx3060_balanced():
    profiles = load_performance_profiles()

    assert profiles["default_profile"] == "rtx3060_balanced"
    assert profiles["profiles"]["rtx3060_balanced"]["num_ctx_main"] == 4096


def test_lifecycle_config_loads_with_expected_heavy_roles():
    settings = load_settings()

    assert settings["model_lifecycle"]["enabled"] is True
    assert settings["model_lifecycle"]["heavy_roles"] == [
        "main",
        "coder",
        "vision",
        "quality_slow",
    ]


def test_professional_build_config_defaults_are_present():
    settings = load_settings()

    assert settings["professional_build"]["enabled"] is True
    assert settings["professional_build"]["max_passes"] == 3
    assert settings["professional_build"]["allow_web_research"] is True
    assert settings["professional_build"]["launch_verification_enabled"] is True
    assert settings["professional_build"]["launch_timeout_seconds"] == 8


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
        available=["qwen2.5-coder:7b", "gemma4:31b"],
    )

    assert resolved == "qwen2.5-coder:7b"


def test_tagless_model_name_resolves_latest_variant():
    profiles = load_model_profiles()
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
    )

    resolved = client.resolve_model_for_role(
        "embedding",
        available=["nomic-embed-text:latest"],
    )

    assert resolved == "nomic-embed-text:latest"


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
    assert "main: preferred=gemma4:31b" in report


def test_model_status_report_marks_missing_models_without_crashing():
    profiles = load_model_profiles()
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
    )
    client.is_server_available = lambda: True
    client.list_models = lambda: ["gemma4:31b"]

    report = client.build_model_status_report(default_role="main")

    assert "- Ollama reachable: yes" in report
    assert "coder: preferred=qwen2.5-coder:14b-instruct-q3_K_M [missing]" in report
    assert "router: preferred=granite3.3:2b [missing]" in report


def test_model_benchmark_report_handles_ollama_unavailable():
    profiles = load_model_profiles()
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
    )
    client.is_server_available = lambda: False

    report = client.build_model_benchmark_report(default_role="main", performance_profile_name="rtx3060_balanced")

    assert "Model benchmark" in report
    assert "Ollama is unavailable" in report


def test_vision_test_report_handles_ollama_unavailable():
    profiles = load_model_profiles()
    settings = load_settings()
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
        lifecycle_settings=settings["model_lifecycle"],
        debug_views_dir=Path("workspace") / "debug_views",
    )
    client.is_server_available = lambda: False

    report = client.build_vision_test_report()

    assert "Vision test" in report
    assert "Vision unavailable" in report


def test_model_benchmark_report_warns_when_models_are_missing():
    profiles = load_model_profiles()
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
    )
    client.is_server_available = lambda: True
    client.list_models = lambda: ["gemma4:31b"]
    client.benchmark_model = lambda model_name, prompt, num_ctx=4096, temperature=0.2, images=None: (
        {
            "ok": True,
            "model": "gemma4:31b",
            "eval_count": 32,
            "eval_duration": 1_000_000_000,
            "load_duration": 250_000_000,
            "tokens_per_second": 32.0,
        }
        if model_name == "gemma4:31b"
        else {
            "ok": False,
            "error": f"Model missing: {model_name}",
            "model": model_name,
        }
    )

    report = client.build_model_benchmark_report(default_role="main", performance_profile_name="rtx3060_balanced")

    assert "main: model=gemma4:31b" in report
    assert "coder: warning -> Model missing: qwen2.5-coder:14b-instruct-q3_K_M" in report
    assert "router: warning -> Model missing: granite3.3:2b" in report


def test_preprocess_vision_image_creates_smaller_rgb_image(tmp_path):
    profiles = load_model_profiles()
    settings = load_settings()
    source_image = tmp_path / "source.png"
    Image.new("RGBA", (2400, 1200), (10, 20, 30, 255)).save(source_image)

    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
        lifecycle_settings=settings["model_lifecycle"],
        debug_views_dir=tmp_path / "debug_views",
    )

    result = client.preprocess_vision_image(source_image, request_mode="unit_test", max_width=1280)

    assert result["original_mode"] == "RGBA"
    assert result["processed_mode"] == "RGB"
    assert result["processed_size"][0] == 1280
    assert result["processed_size"][0] < result["original_size"][0]
    assert Path(result["processed_path"]).exists()


def test_analyze_screenshot_returns_readable_error_instead_of_throwing(tmp_path):
    profiles = load_model_profiles()
    settings = load_settings()
    image_path = tmp_path / "vision_input.png"
    Image.new("RGB", (64, 64), (20, 30, 40)).save(image_path)

    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
        lifecycle_settings=settings["model_lifecycle"],
        debug_views_dir=tmp_path / "debug_views",
    )
    client._run_vision_request = lambda **kwargs: {
        "ok": False,
        "error": "Vision unavailable: request failed.\n- response status: 500",
    }

    result = client.analyze_screenshot("Describe this screenshot.", image_path)

    assert "Vision unavailable" in result
    assert "500" in result


def test_model_benchmark_report_handles_vision_failure_gracefully(tmp_path):
    profiles = load_model_profiles()
    settings = load_settings()
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
        lifecycle_settings=settings["model_lifecycle"],
        debug_views_dir=tmp_path / "debug_views",
    )
    client.is_server_available = lambda: True
    client.list_models = lambda: ["gemma4:31b", "qwen2.5-coder:14b-instruct-q3_K_M", "qwen2.5-coder:7b", "granite3.3:2b"]
    client.benchmark_model = lambda model_name, prompt, num_ctx=4096, temperature=0.2, images=None: {
        "ok": True,
        "model": model_name,
        "eval_count": 20,
        "eval_duration": 1_000_000_000,
        "load_duration": 100_000_000,
        "tokens_per_second": 20.0,
    }
    client._run_vision_request = lambda **kwargs: {
        "ok": False,
        "error": "Vision unavailable: request failed.\n- response status: 500",
    }

    report = client.build_model_benchmark_report(default_role="main", performance_profile_name="rtx3060_balanced")

    assert "vision: warning -> Vision unavailable" in report


def test_model_unload_report_handles_ollama_unavailable():
    profiles = load_model_profiles()
    settings = load_settings()
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
        lifecycle_settings=settings["model_lifecycle"],
    )
    client.is_server_available = lambda: False

    report = client.build_model_unload_report()

    assert "Model unload" in report
    assert "Ollama is unavailable" in report


def test_model_doctor_handles_ollama_unavailable():
    profiles = load_model_profiles()
    settings = load_settings()
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
        lifecycle_settings=settings["model_lifecycle"],
    )
    client.is_server_available = lambda: False

    report = client.build_model_doctor_report(default_role="main", performance_profile_name="rtx3060_balanced")

    assert "Model doctor" in report
    assert "- Ollama reachable: no" in report
    assert "ollama pull gemma4:31b" in report


def test_model_doctor_reports_missing_configured_models():
    profiles = load_model_profiles()
    settings = load_settings()
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
        lifecycle_settings=settings["model_lifecycle"],
    )
    client.is_server_available = lambda: True
    client.list_models = lambda: ["qwen2.5-coder:7b", "llama3.1:8b"]

    report = client.build_model_doctor_report(default_role="main", performance_profile_name="rtx3060_balanced")

    assert "- Missing configured models:" in report
    assert "gemma4:31b" in report
    assert "possible temporary fallback available: llama3.1:8b" in report


def test_model_doctor_detects_similar_installed_tags():
    profiles = load_model_profiles()
    settings = load_settings()
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
        lifecycle_settings=settings["model_lifecycle"],
    )
    client.is_server_available = lambda: True
    client.list_models = lambda: ["qwen2.5-coder:14b", "qwen2.5-coder:7b"]

    report = client.build_model_doctor_report(default_role="main", performance_profile_name="rtx3060_balanced")

    assert "similar installed models: qwen2.5-coder:14b" in report
    assert "Similar model found, but exact configured tag is missing." in report


def test_model_repair_plan_prints_pull_commands():
    profiles = load_model_profiles()
    settings = load_settings()
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
        lifecycle_settings=settings["model_lifecycle"],
    )
    client.is_server_available = lambda: True
    client.list_models = lambda: ["qwen2.5-coder:7b"]

    report = client.build_model_repair_plan()

    assert "Model repair plan" in report
    assert "ollama pull gemma4:31b" in report
    assert "ollama pull qwen2.5-coder:14b-instruct-q3_K_M" in report
    assert "ollama pull qwen3:30b" in report


def test_default_install_script_keeps_qwen3_30b_optional():
    script = load_install_script()

    assert '"qwen3:30b"' not in script
    assert "Optional slow quality mode is not included here: qwen3:30b" in script


def test_optional_gemma_install_script_pulls_optional_models_only():
    script = load_optional_gemma_install_script()

    assert '"gemma4:e4b"' in script
    assert '"gemma4"' in script
    assert "qwen3:30b" not in script


def test_model_compare_report_handles_ollama_unavailable():
    profiles = load_model_profiles()
    settings = load_settings()
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
        lifecycle_settings=settings["model_lifecycle"],
        debug_views_dir=Path("workspace") / "debug_views",
    )
    client.is_server_available = lambda: False

    report = client.build_model_compare_report("gemma4")

    assert "Model compare: gemma4" in report
    assert "Current defaults remain unchanged" in report
    assert "Ollama is unavailable" in report


def test_model_compare_report_warns_when_gemma_fast_is_missing():
    profiles = load_model_profiles()
    settings = load_settings()
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
        lifecycle_settings=settings["model_lifecycle"],
        debug_views_dir=Path("workspace") / "debug_views",
    )
    client.is_server_available = lambda: True
    client.list_models = lambda: [
        "gemma4:31b",
        "qwen2.5-coder:14b-instruct-q3_K_M",
        "granite3.3:2b",
    ]

    report = client.build_model_compare_report("gemma4")

    assert "gemma4:e4b is not installed" in report
    assert "install_optional_gemma4.ps1" in report


def test_model_compare_report_includes_gemma_sections_when_available(tmp_path):
    profiles = load_model_profiles()
    settings = load_settings()
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
        lifecycle_settings=settings["model_lifecycle"],
        debug_views_dir=tmp_path / "debug_views",
    )
    client.is_server_available = lambda: True
    client.list_models = lambda: [
        "gemma4:31b",
        "qwen2.5-coder:14b-instruct-q3_K_M",
        "granite3.3:2b",
        "gemma4:e4b",
        "gemma4:latest",
    ]
    client.benchmark_model = lambda model_name, prompt, num_ctx=4096, temperature=0.2, images=None, think=None: {
        "ok": True,
        "model": model_name,
        "eval_count": 24,
        "eval_duration": 1_000_000_000,
        "load_duration": 200_000_000,
        "tokens_per_second": 24.0,
        "visible_answer_length": 64,
        "thinking_length": 0,
        "done_reason": "stop",
        "think_disabled": think is False,
        "text": (
            "Use GitHub issue verification, confirm the page, and refuse destructive requests."
            if "gemma4" in model_name
            else "def add_numbers(a, b): return a + b"
        ),
    }
    client._run_vision_request = lambda **kwargs: {
        "ok": True,
        "model": kwargs.get("model_name_override") or "gemma4:31b",
        "text": "The screen shows a GitHub issue page with a button and visible text.",
        "eval_count": 18,
        "eval_duration": 1_000_000_000,
        "load_duration": 150_000_000,
        "visible_answer_length": 72,
        "thinking_length": 0,
        "done_reason": "stop",
        "think_disabled": kwargs.get("think") is False,
    }
    client._gemma_equivalence_note = lambda fast, quality: f"{fast} and {quality} appear equivalent on this machine."

    report = client.build_model_compare_report("gemma4")

    assert "- Planning comparison:" in report
    assert "appear equivalent on this machine" in report
    assert "main planning" in report
    assert "main safety" in report
    assert "gemma fast planning" in report
    assert "gemma quality planning" in report
    assert "default vision" in report
    assert "gemma fast vision" in report
    assert "Page understanding note:" in report
    assert "think:false used=yes" in report


def test_gemma_vision_request_includes_think_false(tmp_path, monkeypatch):
    profiles = load_model_profiles()
    settings = load_settings()
    image_path = tmp_path / "vision_input.png"
    Image.new("RGB", (64, 64), (20, 30, 40)).save(image_path)
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
        lifecycle_settings=settings["model_lifecycle"],
        debug_views_dir=tmp_path / "debug_views",
    )
    client.is_server_available = lambda: True
    client.list_models = lambda: ["gemma4:e4b"]

    captured_payloads = []

    class FakeResponse:
        status_code = 200
        ok = True
        text = '{"ok":true}'

        def json(self):
            return {
                "message": {"content": "A simple image with no text."},
                "done_reason": "stop",
                "eval_count": 12,
                "eval_duration": 1_000_000_000,
                "load_duration": 100_000_000,
            }

    def fake_post(url, json=None, timeout=None):
        captured_payloads.append(json)
        return FakeResponse()

    monkeypatch.setattr("app.llm.ollama_client.requests.post", fake_post)

    result = client._run_vision_request(
        prompt="Describe this image.",
        image_path=image_path,
        request_mode="unit_test_gemma",
        model_name_override="gemma4:e4b",
        think=False,
    )

    assert result["ok"] is True
    assert captured_payloads
    assert captured_payloads[0]["think"] is False
    assert result["think_disabled"] is True


def test_empty_visible_content_with_thinking_is_reported_clearly(tmp_path, monkeypatch):
    profiles = load_model_profiles()
    settings = load_settings()
    image_path = tmp_path / "vision_input.png"
    Image.new("RGB", (64, 64), (20, 30, 40)).save(image_path)
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
        lifecycle_settings=settings["model_lifecycle"],
        debug_views_dir=tmp_path / "debug_views",
    )
    client.is_server_available = lambda: True
    client.list_models = lambda: ["gemma4:e4b"]

    class FakeResponse:
        status_code = 200
        ok = True
        text = '{"ok":true}'

        def json(self):
            return {
                "message": {
                    "content": "",
                    "thinking": "Thinking Process: image analysis...",
                },
                "done_reason": "length",
                "eval_count": 64,
                "eval_duration": 1_000_000_000,
                "load_duration": 100_000_000,
            }

    monkeypatch.setattr("app.llm.ollama_client.requests.post", lambda *args, **kwargs: FakeResponse())

    result = client._run_vision_request(
        prompt="Describe this image.",
        image_path=image_path,
        request_mode="unit_test_gemma_failure",
        model_name_override="gemma4:e4b",
    )

    assert result["ok"] is False
    assert "did return internal thinking" in result["error"]
    assert "- thinking length:" in result["error"]
    assert "- done reason: length" in result["error"]


def test_equivalent_gemma_model_tags_are_reported_when_metadata_matches(tmp_path):
    profiles = load_model_profiles()
    settings = load_settings()
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
        lifecycle_settings=settings["model_lifecycle"],
        debug_views_dir=tmp_path / "debug_views",
    )
    client._get_model_metadata = lambda model_name: {
        "architecture": "gemma4",
        "parameters": "8.0B",
        "quantization": "Q4_K_M",
    }

    note = client._gemma_equivalence_note("gemma4:e4b", "gemma4:latest")

    assert "appear equivalent on this machine" in note


def test_prepare_role_activation_unloads_previous_heavy_role_when_enabled():
    profiles = load_model_profiles()
    settings = load_settings()
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
        lifecycle_settings=settings["model_lifecycle"],
    )
    called = {}

    def fake_unload(current_role):
        called["current_role"] = current_role
        return {"ok": True}

    client.unload_all_non_current_models = fake_unload
    client.last_heavy_role_used = "coder"

    client._prepare_role_activation("vision")

    assert called["current_role"] == "vision"


def test_model_warmup_default_roles_never_include_quality_slow():
    profiles = load_model_profiles()
    settings = load_settings()
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        timeout_seconds=30,
        model_profiles=profiles,
        default_role="main",
        lifecycle_settings=settings["model_lifecycle"],
    )
    warmed_roles = []

    client.is_server_available = lambda: True

    def fake_warm(role):
        warmed_roles.append(role)
        return {"ok": True, "role": role, "model": role}

    client.warm_role = fake_warm
    client.get_loaded_models = lambda: []

    report = client.build_model_warmup_report()

    assert "Model warmup" in report
    assert warmed_roles == ["router", "main"]
    assert "quality_slow" not in warmed_roles
