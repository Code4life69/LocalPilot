from pathlib import Path
from types import SimpleNamespace

from app.modes.chat_mode import ChatMode


def test_trust_checklist_command_reads_gauntlet_doc(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "TRUST_GAUNTLET.md").write_text("# LocalPilot Trust Gauntlet\n\n- Test 1\n", encoding="utf-8")

    app = SimpleNamespace(root_dir=tmp_path, capabilities={"name": "LocalPilot", "modes": ["chat"]})
    mode = ChatMode(app)
    result = mode.handle({"user_text": "trust checklist"})

    assert result["ok"]
    assert "# LocalPilot Trust Gauntlet" in result["message"]


def test_model_status_command_returns_status_text(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    app = SimpleNamespace(
        root_dir=tmp_path,
        capabilities={"name": "LocalPilot", "modes": ["chat"]},
        describe_model_status=lambda: "Model status\n- Ollama reachable: no",
        ollama=SimpleNamespace(chat_with_role=lambda *args, **kwargs: "unused"),
        system_prompt="system",
    )
    mode = ChatMode(app)

    result = mode.handle({"user_text": "model status"})

    assert result["ok"]
    assert "Model status" in result["message"]


def test_model_benchmark_command_returns_status_text(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    app = SimpleNamespace(
        root_dir=tmp_path,
        capabilities={"name": "LocalPilot", "modes": ["chat"]},
        describe_model_status=lambda: "unused",
        describe_model_benchmark=lambda: "Model benchmark\n- main: warning -> Ollama unavailable",
        ollama=SimpleNamespace(chat_with_role=lambda *args, **kwargs: "unused"),
        system_prompt="system",
    )
    mode = ChatMode(app)

    result = mode.handle({"user_text": "model benchmark"})

    assert result["ok"]
    assert "Model benchmark" in result["message"]


def test_model_unload_command_returns_status_text(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    app = SimpleNamespace(
        root_dir=tmp_path,
        capabilities={"name": "LocalPilot", "modes": ["chat"]},
        describe_model_status=lambda: "unused",
        describe_model_benchmark=lambda: "unused",
        describe_model_unload=lambda: "Model unload\n- No loaded LocalPilot models needed unloading.",
        describe_model_warmup=lambda: "unused",
        ollama=SimpleNamespace(chat_with_role=lambda *args, **kwargs: "unused"),
        system_prompt="system",
    )
    mode = ChatMode(app)

    result = mode.handle({"user_text": "model unload"})

    assert result["ok"]
    assert "Model unload" in result["message"]


def test_model_warmup_command_returns_status_text(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    app = SimpleNamespace(
        root_dir=tmp_path,
        capabilities={"name": "LocalPilot", "modes": ["chat"]},
        describe_model_status=lambda: "unused",
        describe_model_benchmark=lambda: "unused",
        describe_model_unload=lambda: "unused",
        describe_model_warmup=lambda: "Model warmup\n- Warmed router: granite3.3:2b",
        ollama=SimpleNamespace(chat_with_role=lambda *args, **kwargs: "unused"),
        system_prompt="system",
    )
    mode = ChatMode(app)

    result = mode.handle({"user_text": "model warmup"})

    assert result["ok"]
    assert "Model warmup" in result["message"]


def test_model_doctor_command_returns_status_text(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    app = SimpleNamespace(
        root_dir=tmp_path,
        capabilities={"name": "LocalPilot", "modes": ["chat"]},
        describe_model_status=lambda: "unused",
        describe_model_benchmark=lambda: "unused",
        describe_model_doctor=lambda: "Model doctor\n- Ollama reachable: yes",
        describe_model_repair_plan=lambda: "unused",
        describe_model_unload=lambda: "unused",
        describe_model_warmup=lambda: "unused",
        ollama=SimpleNamespace(chat_with_role=lambda *args, **kwargs: "unused"),
        system_prompt="system",
    )
    mode = ChatMode(app)

    result = mode.handle({"user_text": "model doctor"})

    assert result["ok"]
    assert "Model doctor" in result["message"]


def test_model_repair_plan_command_returns_status_text(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    app = SimpleNamespace(
        root_dir=tmp_path,
        capabilities={"name": "LocalPilot", "modes": ["chat"]},
        describe_model_status=lambda: "unused",
        describe_model_benchmark=lambda: "unused",
        describe_model_doctor=lambda: "unused",
        describe_model_repair_plan=lambda: "Model repair plan\n- This plan does not pull automatically.",
        describe_model_unload=lambda: "unused",
        describe_model_warmup=lambda: "unused",
        ollama=SimpleNamespace(chat_with_role=lambda *args, **kwargs: "unused"),
        system_prompt="system",
    )
    mode = ChatMode(app)

    result = mode.handle({"user_text": "model repair plan"})

    assert result["ok"]
    assert "Model repair plan" in result["message"]


def test_vision_test_command_returns_status_text(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    app = SimpleNamespace(
        root_dir=tmp_path,
        capabilities={"name": "LocalPilot", "modes": ["chat"]},
        describe_model_status=lambda: "unused",
        describe_model_benchmark=lambda: "unused",
        describe_model_doctor=lambda: "unused",
        describe_model_repair_plan=lambda: "unused",
        describe_model_unload=lambda: "unused",
        describe_model_warmup=lambda: "unused",
        describe_vision_test=lambda: "Vision test\nVision unavailable: Ollama is not running.",
        ollama=SimpleNamespace(chat_with_role=lambda *args, **kwargs: "unused"),
        system_prompt="system",
    )
    mode = ChatMode(app)

    result = mode.handle({"user_text": "vision test"})

    assert result["ok"]
    assert "Vision test" in result["message"]


def test_chat_mode_uses_main_role_for_llm_calls(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    captured = {}

    def chat_with_role(role, system_prompt, user_text):
        captured["role"] = role
        captured["system_prompt"] = system_prompt
        captured["user_text"] = user_text
        return "main role response"

    app = SimpleNamespace(
        root_dir=tmp_path,
        capabilities={"name": "LocalPilot", "modes": ["chat"]},
        describe_model_status=lambda: "status",
        ollama=SimpleNamespace(chat_with_role=chat_with_role),
        system_prompt="base system prompt",
    )
    mode = ChatMode(app)

    result = mode.handle({"user_text": "hello there"})

    assert result["message"] == "main role response"
    assert captured["role"] == "main"
    assert "base system prompt" in captured["system_prompt"]
    assert captured["user_text"] == "hello there"
    assert "Do not introduce yourself unless the user asks who you are." in captured["system_prompt"]


def test_small_talk_prompt_adds_natural_style_rules(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    captured = {}

    def chat_with_role(role, system_prompt, user_text):
        captured["role"] = role
        captured["system_prompt"] = system_prompt
        captured["user_text"] = user_text
        return "small talk response"

    app = SimpleNamespace(
        root_dir=tmp_path,
        capabilities={"name": "LocalPilot", "modes": ["chat"]},
        describe_model_status=lambda: "status",
        ollama=SimpleNamespace(chat_with_role=chat_with_role),
        system_prompt="base system prompt",
    )
    mode = ChatMode(app)

    result = mode.handle({"user_text": "How are you doing?"})

    assert result["message"] == "small talk response"
    assert captured["role"] == "main"
    assert "ordinary human conversation" in captured["system_prompt"].lower()
    assert "Do not pivot into a capabilities overview." in captured["system_prompt"]


def test_chat_mode_strips_emoji_when_user_did_not_use_any(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)

    app = SimpleNamespace(
        root_dir=tmp_path,
        capabilities={"name": "LocalPilot", "modes": ["chat"]},
        describe_model_status=lambda: "status",
        ollama=SimpleNamespace(chat_with_role=lambda *args, **kwargs: "I'm doing well! 😊"),
        system_prompt="base system prompt",
    )
    mode = ChatMode(app)

    result = mode.handle({"user_text": "How are you doing?"})

    assert result["message"] == "I'm doing well!"
