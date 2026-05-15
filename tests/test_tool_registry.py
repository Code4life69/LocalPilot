import json
from pathlib import Path

from app.browser_tool import BrowserToolBridge
from app.logger import AppLogger
from app.safety import RISK_BLOCKED, RISK_MEDIUM, RISK_SAFE, SafetyManager
from app.tool_registry import ToolRegistry


class FakeVisionClient:
    default_vision_model = "qwen3-vl-8b-instruct"

    def __init__(self):
        self.calls = []

    def chat_vision(self, prompt, image_path, model):
        self.calls.append({"prompt": prompt, "image_path": str(image_path), "model": model})
        return "A screenshot of a code editor."


class FakeBrowserBridge:
    def __init__(self):
        self.calls = []

    def run(self, action, **kwargs):
        self.calls.append({"action": action, **kwargs})
        return {
            "ok": True,
            "action": action,
            "title": "Google",
            "url": "https://www.google.com/search?q=cats",
            "text_preview": "cats search results",
            "screenshot_path": "C:\\LocalPilot\\logs\\browser\\latest.png",
        }


def build_registry(tmp_path, approval_callback=lambda prompt: True):
    root_dir = tmp_path / "repo"
    workspace = root_dir / "workspace"
    logs = root_dir / "logs"
    workspace.mkdir(parents=True)
    logs.mkdir(parents=True)
    logger = AppLogger(logs)
    safety = SafetyManager(approval_callback=approval_callback, workspace_root=workspace)
    vision = FakeVisionClient()
    browser = FakeBrowserBridge()
    registry = ToolRegistry(root_dir=root_dir, safety=safety, logger=logger, lmstudio_client=vision, browser_bridge=browser)
    return registry, root_dir, workspace, vision, browser


def test_tool_registry_registers_expected_builtin_tools(tmp_path):
    registry, _root_dir, _workspace, _vision, _browser = build_registry(tmp_path)

    names = {tool["name"] for tool in registry.list_tools()}

    assert {
        "list_files",
        "read_file",
        "write_file",
        "run_command",
        "take_screenshot",
        "analyze_screenshot",
        "ask_user_approval",
        "list_checkpoints",
        "restore_checkpoint",
        "list_sessions",
        "read_session",
        "browser_launch",
        "browser_close",
        "browser_goto",
        "browser_search",
        "browser_click_selector",
        "browser_type_selector",
        "browser_press_key",
        "browser_get_text",
        "browser_screenshot",
        "browser_get_page_info",
    }.issubset(names)


def test_unknown_tool_returns_structured_error(tmp_path):
    registry, _root_dir, _workspace, _vision, _browser = build_registry(tmp_path)

    result = registry.execute_tool_call({"tool": "missing_tool", "args": {}, "reason": "test"})

    assert result == {"ok": False, "tool": "missing_tool", "error": "Unknown tool: missing_tool"}


def test_tool_result_schema_for_list_files(tmp_path):
    registry, _root_dir, workspace, _vision, _browser = build_registry(tmp_path)
    (workspace / "note.txt").write_text("hi", encoding="utf-8")

    result = registry.execute_tool_call({"tool": "list_files", "args": {"path": "."}, "reason": "inspect workspace"})

    assert result["ok"] is True
    assert result["tool"] == "list_files"
    assert "result" in result
    assert result["result"]["items"][0]["name"] == "note.txt"


def test_registry_requires_approval_for_risky_write(tmp_path):
    prompts = []
    registry, _root_dir, _workspace, _vision, _browser = build_registry(tmp_path, approval_callback=lambda prompt: prompts.append(prompt) or False)

    result = registry.execute_tool_call(
        {"tool": "write_file", "args": {"path": "note.txt", "content": "hello"}, "reason": "write a note"}
    )

    assert result["ok"] is False
    assert result["error"] == "User denied approval."
    assert prompts[0]["risk"] == "medium"
    assert prompts[0]["tool_calls"][0]["tool"] == "write_file"


def test_registry_analyze_screenshot_uses_lmstudio_vision_client(tmp_path):
    registry, _root_dir, workspace, vision, _browser = build_registry(tmp_path)
    image_path = workspace / "screen.png"
    image_path.write_bytes(b"png")

    result = registry.execute_tool_call(
        {"tool": "analyze_screenshot", "args": {"path": str(image_path)}, "reason": "understand the screen"}
    )

    assert result["ok"] is True
    assert result["result"]["description"] == "A screenshot of a code editor."
    assert vision.calls[0]["model"] == "qwen3-vl-8b-instruct"


def test_safety_classifies_workspace_and_commands(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    safety = SafetyManager(workspace_root=workspace)

    assert safety.classify_tool_call("list_files", {"path": "."}).risk_level == RISK_SAFE
    assert safety.classify_tool_call("write_file", {"path": "note.txt"}).risk_level == RISK_MEDIUM
    assert safety.classify_tool_call("run_command", {"command": "git status"}).risk_level == RISK_MEDIUM
    assert safety.classify_tool_call("run_command", {"command": "del note.txt"}).risk_level == RISK_BLOCKED
    assert safety.classify_tool_call("browser_get_text", {}).risk_level == RISK_SAFE
    assert safety.classify_tool_call("browser_click_selector", {"selector": "#submit"}).approval_required is True


def test_outside_workspace_requires_approval(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    safety = SafetyManager(workspace_root=workspace)

    decision = safety.classify_tool_call("read_file", {"path": str(outside)})

    assert decision.risk_level == RISK_MEDIUM
    assert decision.approval_required is True


def test_registry_browser_get_page_info_returns_structured_json(tmp_path):
    registry, _root_dir, _workspace, _vision, browser = build_registry(tmp_path)

    result = registry.execute_tool_call({"tool": "browser_get_page_info", "args": {}, "reason": "inspect page"})

    assert result["ok"] is True
    assert result["tool"] == "browser_get_page_info"
    assert result["result"]["title"] == "Google"
    assert browser.calls[0]["action"] == "get_page_info"


def test_browser_click_requires_approval(tmp_path):
    prompts = []
    registry, _root_dir, _workspace, _vision, _browser = build_registry(tmp_path, approval_callback=lambda prompt: prompts.append(prompt) or False)

    result = registry.execute_tool_call(
        {"tool": "browser_click_selector", "args": {"selector": "#search"}, "reason": "click the page control"}
    )

    assert result["ok"] is False
    assert result["error"] == "User denied approval."
    assert prompts[0]["tool_calls"][0]["tool"] == "browser_click_selector"


def test_write_file_creates_checkpoint_and_returns_checkpoint_id(tmp_path):
    registry, _root_dir, workspace, _vision, _browser = build_registry(tmp_path)
    target = workspace / "note.txt"
    target.write_text("before", encoding="utf-8")

    result = registry.execute_tool_call(
        {
            "tool": "write_file",
            "args": {"path": "note.txt", "content": "after"},
            "reason": "update note",
        }
    )

    assert result["ok"] is True
    assert result["result"]["checkpoint_id"]
    manifest_path = Path(result["result"]["checkpoint_manifest_path"])
    assert manifest_path.exists()


def test_checkpoint_manifest_records_original_file_state(tmp_path):
    registry, _root_dir, workspace, _vision, _browser = build_registry(tmp_path)
    target = workspace / "demo.txt"
    target.write_text("before", encoding="utf-8")

    result = registry.execute_tool_call(
        {
            "tool": "write_file",
            "args": {"path": "demo.txt", "content": "after"},
            "reason": "update demo file",
        }
    )

    manifest = json.loads(Path(result["result"]["checkpoint_manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["checkpoint_id"] == result["result"]["checkpoint_id"]
    assert manifest["files"][0]["original_path"] == str(target.resolve())
    assert manifest["files"][0]["file_existed_before"] is True


def test_denied_approval_returns_structured_approval_payload(tmp_path):
    registry, _root_dir, _workspace, _vision, _browser = build_registry(tmp_path, approval_callback=lambda prompt: False)

    result = registry.execute_tool_call(
        {"tool": "browser_search", "args": {"query": "cats"}, "reason": "search the web"}
    )

    assert result["ok"] is False
    assert result["approval"]["granted"] is False
    assert result["approval"]["risk"] == "medium"


def test_approval_summary_includes_risk_tool_and_args(tmp_path):
    registry, _root_dir, _workspace, _vision, _browser = build_registry(tmp_path)
    decision = registry.safety.classify_tool_call("browser_goto", {"url": "https://example.com"})

    approval = registry.build_approval_request(
        "browser_goto",
        {"url": "https://example.com"},
        "Need to open the requested website.",
        decision,
    )
    rendered = registry.safety.format_approval_request(approval)

    assert approval["risk"] == "medium"
    assert approval["tool_calls"][0]["tool"] == "browser_goto"
    assert "browser_goto" in rendered
    assert "https://example.com" in rendered


def test_list_checkpoints_tool_returns_structured_result(tmp_path):
    registry, _root_dir, workspace, _vision, _browser = build_registry(tmp_path)
    (workspace / "demo.txt").write_text("before", encoding="utf-8")
    write_result = registry.execute_tool_call(
        {"tool": "write_file", "args": {"path": "demo.txt", "content": "after"}, "reason": "update demo"}
    )

    result = registry.execute_tool_call({"tool": "list_checkpoints", "args": {}, "reason": "inspect checkpoints"})

    assert result["ok"] is True
    assert result["tool"] == "list_checkpoints"
    assert result["result"]["checkpoints"][0]["checkpoint_id"] == write_result["result"]["checkpoint_id"]


def test_restore_checkpoint_requires_approval(tmp_path):
    prompts = []
    registry, _root_dir, workspace, _vision, _browser = build_registry(tmp_path, approval_callback=lambda prompt: prompts.append(prompt) or False)
    target = workspace / "demo.txt"
    target.write_text("before", encoding="utf-8")
    checkpoint = registry.checkpoint_manager.create_file_checkpoint(target)

    result = registry.execute_tool_call(
        {"tool": "restore_checkpoint", "args": {"checkpoint_id": checkpoint["checkpoint_id"]}, "reason": "undo the edit"}
    )

    assert result["ok"] is False
    assert result["error"] == "User denied approval."
    assert prompts[0]["type"] == "approval_request"
    assert prompts[0]["risk"] == "dangerous"


def test_restore_checkpoint_returns_structured_result(tmp_path):
    registry, _root_dir, workspace, _vision, _browser = build_registry(tmp_path)
    target = workspace / "demo.txt"
    target.write_text("before", encoding="utf-8")
    checkpoint = registry.checkpoint_manager.create_file_checkpoint(target)
    target.write_text("after", encoding="utf-8")

    result = registry.execute_tool_call(
        {"tool": "restore_checkpoint", "args": {"checkpoint_id": checkpoint["checkpoint_id"]}, "reason": "undo the edit"}
    )

    assert result["ok"] is True
    assert result["result"]["checkpoint_id"] == checkpoint["checkpoint_id"]
    assert str(target) in result["result"]["restored_files"]
    assert target.read_text(encoding="utf-8") == "before"


def test_list_sessions_tool_returns_saved_sessions(tmp_path):
    registry, _root_dir, _workspace, _vision, _browser = build_registry(tmp_path)
    registry.memory_store.save_session(
        {
            "task_id": "task123",
            "user_task": "describe my screen",
            "mode": "agent",
            "start_time": "2026-05-15T17:30:00",
            "end_time": "2026-05-15T17:30:01",
            "status": "final",
            "final_answer": "done",
            "browser_actions": [],
            "files_changed": [],
            "errors": [],
        }
    )

    result = registry.execute_tool_call({"tool": "list_sessions", "args": {}, "reason": "inspect recent sessions"})

    assert result["ok"] is True
    assert result["result"]["sessions"][0]["task_id"] == "task123"


def test_read_session_tool_returns_saved_session(tmp_path):
    registry, _root_dir, _workspace, _vision, _browser = build_registry(tmp_path)
    session_path = registry.memory_store.save_session(
        {
            "task_id": "task123",
            "user_task": "describe my screen",
            "mode": "agent",
            "start_time": "2026-05-15T17:30:00",
            "end_time": "2026-05-15T17:30:01",
            "status": "final",
            "final_answer": "done",
            "browser_actions": [],
            "files_changed": [],
            "errors": [],
        }
    )

    result = registry.execute_tool_call(
        {"tool": "read_session", "args": {"session_id": Path(session_path).stem}, "reason": "inspect prior session"}
    )

    assert result["ok"] is True
    assert result["result"]["session"]["task_id"] == "task123"


def test_grouped_approval_plan_is_structured_and_reused(tmp_path):
    prompts = []
    registry, _root_dir, _workspace, _vision, _browser = build_registry(tmp_path, approval_callback=lambda prompt: prompts.append(prompt) or True)
    approval_plan = {
        "type": "approval_plan",
        "summary": "Open Google and search cats.",
        "tool_calls": [
            {"tool": "browser_launch", "args": {}},
            {"tool": "browser_goto", "args": {"url": "https://www.google.com"}},
            {"tool": "browser_search", "args": {"query": "cats"}},
        ],
    }

    first = registry.execute_tool_call(
        {
            "tool": "browser_launch",
            "args": {},
            "reason": "Start a visible browser session.",
            "approval_plan": approval_plan,
        }
    )
    second = registry.execute_tool_call(
        {"tool": "browser_goto", "args": {"url": "https://www.google.com"}, "reason": "Open Google."}
    )
    third = registry.execute_tool_call(
        {"tool": "browser_search", "args": {"query": "cats"}, "reason": "Search for cats."}
    )

    assert first["ok"] is True
    assert first["approval"]["type"] == "approval_plan"
    assert second["ok"] is True
    assert third["ok"] is True
    assert prompts[0]["type"] == "approval_plan"
    assert len(prompts) == 1


def test_dangerous_actions_cannot_be_grouped(tmp_path):
    prompts = []
    registry, _root_dir, _workspace, _vision, _browser = build_registry(tmp_path, approval_callback=lambda prompt: prompts.append(prompt) or True)
    approval_plan = {
        "type": "approval_plan",
        "summary": "Search and type a password.",
        "tool_calls": [
            {"tool": "browser_click_selector", "args": {"selector": "#password"}},
            {"tool": "browser_search", "args": {"query": "cats"}},
        ],
    }

    result = registry.execute_tool_call(
        {
            "tool": "browser_click_selector",
            "args": {"selector": "#password"},
            "reason": "Interact with a password field.",
            "approval_plan": approval_plan,
        }
    )

    assert result["ok"] is True
    assert prompts[0]["type"] == "approval_request"
    assert prompts[0]["risk"] == "dangerous"


def test_denied_approval_plan_returns_structured_result(tmp_path):
    registry, _root_dir, _workspace, _vision, _browser = build_registry(tmp_path, approval_callback=lambda prompt: False)
    approval_plan = {
        "type": "approval_plan",
        "summary": "Open Google and search cats.",
        "tool_calls": [
            {"tool": "browser_launch", "args": {}},
            {"tool": "browser_goto", "args": {"url": "https://www.google.com"}},
        ],
    }

    result = registry.execute_tool_call(
        {
            "tool": "browser_launch",
            "args": {},
            "reason": "Start a browser session.",
            "approval_plan": approval_plan,
        }
    )

    assert result["ok"] is False
    assert result["approval"]["type"] == "approval_plan"
    assert result["approval"]["granted"] is False
