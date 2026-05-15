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
    assert prompts


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
    assert prompts
