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
