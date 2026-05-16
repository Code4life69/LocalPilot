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
            },
            {
                "name": "desktop_suggest_action",
                "description": "Suggest the next desktop action without executing it.",
                "argument_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "instruction": {"type": "string"}},
                },
                "risk_level": "safe",
                "approval_required": False,
            },
            {
                "name": "desktop_execute_suggestion",
                "description": "Execute one previously suggested desktop click after explicit approval.",
                "argument_schema": {
                    "type": "object",
                    "properties": {"suggestion_id": {"type": "string"}},
                },
                "risk_level": "dangerous",
                "approval_required": True,
            },
        ]

    def execute_tool_call(self, tool_call):
        self.calls.append(tool_call)
        return {"ok": True, "tool": tool_call["tool"], "result": {"path": "logs/screenshots/demo.png"}}


class RichRegistry(FakeRegistry):
    def execute_tool_call(self, tool_call):
        self.calls.append(tool_call)
        if tool_call["tool"] == "take_screenshot":
            return {"ok": True, "tool": "take_screenshot", "result": {"path": "logs/screenshots/demo.png"}}
        if tool_call["tool"] == "analyze_screenshot":
            return {
                "ok": True,
                "tool": "analyze_screenshot",
                "result": {
                    "path": "logs/screenshots/demo.png",
                    "description": "The screenshot shows the ChatGPT web interface with a left sidebar of recent chats.",
                },
            }
        return super().execute_tool_call(tool_call)


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_text(self, messages, model, max_tokens):
        self.calls.append({"messages": messages, "model": model, "max_tokens": max_tokens})
        return self.responses.pop(0)


class OverflowThenSuccessLLM(FakeLLM):
    def __init__(self, responses):
        super().__init__(responses)

    def chat_text(self, messages, model, max_tokens):
        self.calls.append({"messages": messages, "model": model, "max_tokens": max_tokens})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


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


def test_agent_parses_last_json_object_when_model_returns_multiple_objects():
    agent = LocalPilotAgent(llm_client=FakeLLM([]), tool_registry=FakeRegistry())

    parsed = agent.parse_agent_response(
        """```json
{"type":"final","message":"approved"}
```
```json
{"type":"tool_call","tool":"desktop_execute_suggestion","args":{"suggestion_id":"desk_suggest_demo"},"reason":"Execute the saved click."}
```"""
    )

    assert parsed["type"] == "tool_call"
    assert parsed["tool"] == "desktop_execute_suggestion"


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


def test_agent_executes_mocked_desktop_suggest_action_tool_call():
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "tool": "desktop_suggest_action",
                    "args": {"path": "logs/screenshots/demo.png", "instruction": "Tell me what you would click next. Do not click anything."},
                    "reason": "Suggest the next desktop action.",
                }
            ),
            json.dumps({"type": "final", "message": "I would click the highlighted search field next."}),
        ]
    )
    registry = FakeRegistry()
    agent = LocalPilotAgent(llm_client=llm, tool_registry=registry)

    result = agent.run_task("Look at my screen and tell me what you would click next.")

    assert result["ok"] is True
    assert registry.calls[0]["tool"] == "desktop_suggest_action"


def test_agent_can_execute_mocked_desktop_suggestion_by_id():
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "tool": "desktop_execute_suggestion",
                    "args": {"suggestion_id": "desk_suggest_demo"},
                    "reason": "The user approved the suggested click.",
                }
            ),
            json.dumps({"type": "final", "message": "The approved click was executed."}),
        ]
    )
    registry = FakeRegistry()
    agent = LocalPilotAgent(llm_client=llm, tool_registry=registry)

    result = agent.run_task("approve the suggested click")

    assert result["ok"] is True
    assert registry.calls[0]["tool"] == "desktop_execute_suggestion"


def test_agent_loads_pilot_rules_into_prompt(tmp_path):
    (tmp_path / ".pilotrules").write_text("The AI must choose tools.", encoding="utf-8")
    agent = LocalPilotAgent(llm_client=FakeLLM([]), tool_registry=FakeRegistry(), root_dir=tmp_path)

    prompt = agent._build_system_prompt()

    assert "Rules:" in prompt
    assert "The AI must choose tools." in prompt


def test_agent_reports_low_planner_context_warning():
    agent = LocalPilotAgent(
        llm_client=FakeLLM([]),
        tool_registry=FakeRegistry(),
        planner_context_length=4096,
    )

    warning = agent.planner_context_warning()

    assert warning is not None
    assert "too small" in warning.lower()
    assert "8192" in warning


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
    assert "summary" in session
    assert "Tools: take_screenshot" in session["summary"]


def test_agent_followup_yes_uses_active_task_context(tmp_path):
    memory_dir = tmp_path / "memory"
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "capabilities.json").write_text("{}", encoding="utf-8")
    memory = MemoryStore(memory_dir, config_dir / "capabilities.json")
    memory.save_current_task(
        {
            "active_task_id": "task123",
            "original_user_task": "make a basic website locally",
            "latest_user_message": "waiting on approval",
            "mode": "agent",
            "status": "waiting_for_approval",
            "last_approval_request": {"summary": "Write files into workspace/basic_website."},
        }
    )
    llm = FakeLLM([json.dumps({"type": "final", "message": "Continuing the task."})])
    agent = LocalPilotAgent(llm_client=llm, tool_registry=FakeRegistry(), memory_store=memory, root_dir=tmp_path)

    result = agent.run_task("yes")

    assert result["ok"] is True
    first_user_message = llm.calls[0]["messages"][1]["content"]
    assert "approved the pending request" in first_user_message.lower()
    assert "make a basic website locally" in first_user_message


def test_agent_followup_approve_phrase_uses_active_task_context(tmp_path):
    memory_dir = tmp_path / "memory"
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "capabilities.json").write_text("{}", encoding="utf-8")
    memory = MemoryStore(memory_dir, config_dir / "capabilities.json")
    memory.save_current_task(
        {
            "active_task_id": "task123",
            "original_user_task": "Look at my screen and tell me what you would click next.",
            "latest_user_message": "waiting on approval",
            "mode": "agent",
            "status": "active",
            "last_desktop_suggestion_id": "desk_suggest_demo",
            "last_desktop_suggestion_target": "Google search bar",
            "last_desktop_suggestion_executed": False,
        }
    )
    llm = FakeLLM([json.dumps({"type": "final", "message": "Continuing the task."})])
    agent = LocalPilotAgent(llm_client=llm, tool_registry=FakeRegistry(), memory_store=memory, root_dir=tmp_path)

    result = agent.run_task("approve the suggested click")

    assert result["ok"] is True
    first_user_message = llm.calls[0]["messages"][1]["content"]
    system_prompt = llm.calls[0]["messages"][0]["content"]
    assert "approved the pending request" in first_user_message.lower()
    assert "desktop_execute_suggestion tool call" in first_user_message
    assert "desk_suggest_demo" in system_prompt


def test_agent_followup_continue_uses_active_task_context(tmp_path):
    memory_dir = tmp_path / "memory"
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "capabilities.json").write_text("{}", encoding="utf-8")
    memory = MemoryStore(memory_dir, config_dir / "capabilities.json")
    memory.save_current_task(
        {
            "active_task_id": "task123",
            "original_user_task": "open Google and search cats",
            "latest_user_message": "continue later",
            "mode": "agent",
            "status": "active",
        }
    )
    llm = FakeLLM([json.dumps({"type": "final", "message": "Retrying."})])
    agent = LocalPilotAgent(llm_client=llm, tool_registry=FakeRegistry(), memory_store=memory, root_dir=tmp_path)

    result = agent.run_task("continue")

    assert result["ok"] is True
    first_user_message = llm.calls[0]["messages"][1]["content"]
    assert "continue the active task" in first_user_message.lower()
    assert "open Google and search cats" in first_user_message


def test_agent_short_clarification_uses_active_task_context(tmp_path):
    memory_dir = tmp_path / "memory"
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "capabilities.json").write_text("{}", encoding="utf-8")
    memory = MemoryStore(memory_dir, config_dir / "capabilities.json")
    memory.save_current_task(
        {
            "active_task_id": "task123",
            "original_user_task": "Describe my screen briefly.",
            "latest_user_message": "Describe my screen briefly.",
            "mode": "agent",
            "status": "completed",
            "last_tool_result_summary": "Screenshot analysis: ChatGPT with recent chats in the left sidebar.",
        }
    )
    llm = FakeLLM([json.dumps({"type": "final", "message": "The sidebar shows recent chats."})])
    agent = LocalPilotAgent(llm_client=llm, tool_registry=FakeRegistry(), memory_store=memory, root_dir=tmp_path)

    result = agent.run_task("What about the sidebar?")

    assert result["ok"] is True
    first_user_message = llm.calls[0]["messages"][1]["content"]
    assert "continue the active task" in first_user_message.lower()
    assert "what about the sidebar?" in first_user_message.lower()


def test_agent_persists_tool_result_summary_and_recent_turns_for_followups(tmp_path):
    memory_dir = tmp_path / "memory"
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "capabilities.json").write_text("{}", encoding="utf-8")
    memory = MemoryStore(memory_dir, config_dir / "capabilities.json")
    first_llm = FakeLLM(
        [
            json.dumps({"type": "tool_call", "tool": "take_screenshot", "args": {}, "reason": "capture the screen"}),
            json.dumps(
                {
                    "type": "tool_call",
                    "tool": "analyze_screenshot",
                    "args": {"path": "logs/screenshots/demo.png"},
                    "reason": "understand the screen",
                }
            ),
            json.dumps({"type": "final", "message": "The screen shows ChatGPT with recent chats in the sidebar."}),
        ]
    )
    registry = RichRegistry()
    agent = LocalPilotAgent(llm_client=first_llm, tool_registry=registry, memory_store=memory, root_dir=tmp_path)

    first_result = agent.run_task("Describe my screen briefly.")

    assert first_result["ok"] is True
    current_task = memory.load_current_task()
    assert current_task is not None
    assert "Screenshot analysis:" in current_task["last_tool_result_summary"]
    assert any(message["role"] == "assistant" for message in current_task["recent_messages"])

    second_llm = FakeLLM([json.dumps({"type": "final", "message": "Continuing from the same screen context."})])
    followup_agent = LocalPilotAgent(llm_client=second_llm, tool_registry=registry, memory_store=memory, root_dir=tmp_path)

    followup_agent.run_task("continue")

    system_prompt = second_llm.calls[0]["messages"][0]["content"]
    assert "Screenshot analysis:" in system_prompt
    assert "ChatGPT web interface" in system_prompt
    assert "Recent messages:" in system_prompt


def test_agent_retries_once_with_ultra_compact_prompt_after_context_overflow(tmp_path):
    memory_dir = tmp_path / "memory"
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "capabilities.json").write_text("{}", encoding="utf-8")
    memory = MemoryStore(memory_dir, config_dir / "capabilities.json")
    memory.save_current_task(
        {
            "active_task_id": "task123",
            "original_user_task": "Open Google and search cats.",
            "latest_user_message": "continue",
            "mode": "agent",
            "status": "active",
            "recent_tool_calls": [{"tool": "browser_search", "args": {"query": "cats"}}],
            "last_tool_result_summary": "browser_search -> cats - Google Search",
        }
    )
    llm = OverflowThenSuccessLLM(
        [
            RuntimeError(
                'LM Studio request failed: 400 Client Error | response: {"error":"The number of tokens to keep from the initial prompt is greater than the context length (n_keep: 4240>= n_ctx: 4096)."}'
            ),
            json.dumps({"type": "final", "message": "Still on the cats search results."}),
        ]
    )
    agent = LocalPilotAgent(llm_client=llm, tool_registry=FakeRegistry(), memory_store=memory, root_dir=tmp_path)

    result = agent.run_task("continue")

    assert result["ok"] is True
    assert result["message"] == "Still on the cats search results."
    assert len(llm.calls) == 2
    system_prompt = llm.calls[1]["messages"][0]["content"]
    assert "browser_search" in system_prompt
    assert "- desktop_suggest_action(" not in system_prompt


def test_agent_failed_followup_keeps_current_task_retryable_after_context_overflow(tmp_path):
    memory_dir = tmp_path / "memory"
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "capabilities.json").write_text("{}", encoding="utf-8")
    memory = MemoryStore(memory_dir, config_dir / "capabilities.json")
    memory.save_current_task(
        {
            "active_task_id": "task123",
            "original_user_task": "Open Google and search cats.",
            "latest_user_message": "continue",
            "mode": "agent",
            "status": "active",
            "recent_tool_calls": [{"tool": "browser_search", "args": {"query": "cats"}}],
            "last_tool_result_summary": "browser_search -> cats - Google Search",
        }
    )
    overflow_error = RuntimeError(
        'LM Studio request failed: 400 Client Error | response: {"error":"The number of tokens to keep from the initial prompt is greater than the context length (n_keep: 4240>= n_ctx: 4096)."}'
    )
    llm = OverflowThenSuccessLLM([overflow_error, overflow_error])
    agent = LocalPilotAgent(llm_client=llm, tool_registry=FakeRegistry(), memory_store=memory, root_dir=tmp_path)

    result = agent.run_task("what did you look up?")

    assert result["ok"] is False
    assert result["status"] == "failed_retryable"
    assert "Increase the planner model context" in result["error"]
    current_task = memory.load_current_task()
    assert current_task is not None
    assert current_task["active_task_id"] == "task123"
    assert current_task["status"] == "failed_retryable"
    assert "Increase the planner context" in current_task["retry_suggestion"]
    assert "planner prompt exceeded" in current_task["last_error"]


def test_agent_writes_planner_telemetry(tmp_path):
    memory_dir = tmp_path / "memory"
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "capabilities.json").write_text("{}", encoding="utf-8")
    memory = MemoryStore(memory_dir, config_dir / "capabilities.json")
    llm = FakeLLM([json.dumps({"type": "final", "message": "Done."})])
    agent = LocalPilotAgent(llm_client=llm, tool_registry=FakeRegistry(), memory_store=memory, root_dir=tmp_path)

    result = agent.run_task("describe my screen")

    assert result["ok"] is True
    telemetry_path = tmp_path / "logs" / "agent_planner.log"
    entry = json.loads(telemetry_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert entry["task_id"] == result["task_id"]
    assert entry["model"] == "qwen2.5-coder-14b-instruct"
    assert entry["tool_count"] >= 1
    assert entry["prompt_char_count"] > 0
    assert "take_screenshot" in entry["tool_names"]


def test_no_hardcoded_google_cats_flow_exists():
    source = Path("app/main.py").read_text(encoding="utf-8") + Path("app/tool_registry.py").read_text(encoding="utf-8")

    assert "open google and search cats" not in source.lower()
