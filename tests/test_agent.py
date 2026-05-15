import json
from pathlib import Path

import pytest

from app.agent import LocalPilotAgent
from app.memory import MemoryStore


class FakeRegistry:
    def __init__(self):
        self.calls = []

    def list_tools(self):
        return [
            {
                "name": "take_screenshot",
                "description": "Capture the current screen.",
                "argument_schema": {"type": "object", "properties": {}},
                "risk_level": "safe",
                "approval_required": False,
            },
            {
                "name": "browser_search",
                "description": "Search the web in the Puppeteer-controlled browser.",
                "argument_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                "risk_level": "medium",
                "approval_required": True,
            }
        ]

    def execute_tool_call(self, tool_call):
        self.calls.append(tool_call)
        return {"ok": True, "tool": tool_call["tool"], "result": {"path": "logs/screenshots/demo.png"}}


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_text(self, messages, model, max_tokens):
        self.calls.append({"messages": messages, "model": model, "max_tokens": max_tokens})
        return self.responses.pop(0)


def test_agent_parses_tool_json_from_fenced_block():
    agent = LocalPilotAgent(llm_client=FakeLLM([]), tool_registry=FakeRegistry())

    parsed = agent.parse_agent_response(
        """```json
{"type":"tool_call","tool":"take_screenshot","args":{},"reason":"inspect screen"}
```"""
    )

    assert parsed["type"] == "tool_call"
    assert parsed["tool"] == "take_screenshot"


def test_agent_parses_tool_name_in_type_field():
    agent = LocalPilotAgent(llm_client=FakeLLM([]), tool_registry=FakeRegistry())

    parsed = agent.parse_agent_response('{"type":"take_screenshot","args":{}}')

    assert parsed["type"] == "tool_call"
    assert parsed["tool"] == "take_screenshot"


def test_agent_rejects_invalid_json():
    agent = LocalPilotAgent(llm_client=FakeLLM([]), tool_registry=FakeRegistry())

    with pytest.raises(ValueError):
        agent.parse_agent_response("not json")


def test_agent_loop_sends_tool_results_back_to_ai():
    llm = FakeLLM(
        [
            json.dumps({"type": "tool_call", "tool": "take_screenshot", "args": {}, "reason": "inspect screen"}),
            json.dumps({"type": "final", "message": "The screen has been described."}),
        ]
    )
    registry = FakeRegistry()
    agent = LocalPilotAgent(llm_client=llm, tool_registry=registry)

    result = agent.run_task("describe my screen")

    assert result["ok"] is True
    assert result["status"] == "final"
    assert registry.calls[0]["tool"] == "take_screenshot"
    assert registry.calls[0]["task_id"]
    assert registry.calls[0]["tool_call_id"]
    assert len(llm.calls) == 2
    tool_feedback_message = llm.calls[1]["messages"][-1]["content"]
    assert '"tool": "take_screenshot"' in tool_feedback_message
    assert '"path": "logs/screenshots/demo.png"' in tool_feedback_message


def test_agent_returns_question_without_executing_tool():
    llm = FakeLLM([json.dumps({"type": "question", "message": "Which folder should I inspect?"})])
    registry = FakeRegistry()
    agent = LocalPilotAgent(llm_client=llm, tool_registry=registry)

    result = agent.run_task("inspect files")

    assert result["status"] == "question"
    assert registry.calls == []


def test_agent_executes_mocked_browser_tool_call():
    llm = FakeLLM(
        [
            json.dumps({"type": "tool_call", "tool": "browser_search", "args": {"query": "cats"}, "reason": "search the web"}),
            json.dumps({"type": "final", "message": "Google search results for cats are open."}),
        ]
    )
    registry = FakeRegistry()
    agent = LocalPilotAgent(llm_client=llm, tool_registry=registry)

    result = agent.run_task("Open Google and search cats")

    assert result["ok"] is True
    assert registry.calls[0]["tool"] == "browser_search"


def test_agent_loads_pilot_rules_into_prompt(tmp_path):
    (tmp_path / ".pilotrules").write_text("The AI must choose tools.", encoding="utf-8")
    agent = LocalPilotAgent(llm_client=FakeLLM([]), tool_registry=FakeRegistry(), root_dir=tmp_path)

    prompt = agent._build_system_prompt()

    assert "Pilot rules:" in prompt
    assert "The AI must choose tools." in prompt


def test_agent_writes_session_record(tmp_path):
    memory_dir = tmp_path / "memory"
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "capabilities.json").write_text("{}", encoding="utf-8")
    memory = MemoryStore(memory_dir, config_dir / "capabilities.json")
    llm = FakeLLM(
        [
            json.dumps({"type": "tool_call", "tool": "take_screenshot", "args": {}, "reason": "inspect screen"}),
            json.dumps({"type": "final", "message": "Done."}),
        ]
    )
    agent = LocalPilotAgent(llm_client=llm, tool_registry=FakeRegistry(), memory_store=memory, root_dir=tmp_path)

    result = agent.run_task("describe my screen")

    assert result["session_path"]
    session = json.loads(Path(result["session_path"]).read_text(encoding="utf-8"))
    assert session["user_task"] == "describe my screen"
    assert session["mode"] == "agent"
    assert session["tool_calls"][0]["tool"] == "take_screenshot"


def test_no_hardcoded_google_cats_flow_exists():
    source = Path("app/main.py").read_text(encoding="utf-8") + Path("app/tool_registry.py").read_text(encoding="utf-8")

    assert "open google and search cats" not in source.lower()
