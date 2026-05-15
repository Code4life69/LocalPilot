from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from app.browser_tool import BrowserToolBridge
from app.lmstudio_client import LMStudioClient
from app.logger import AppLogger
from app.safety import RISK_BLOCKED, SafetyDecision, SafetyManager
from app.tools.files import list_folder, read_file, write_file
from app.tools.screen import take_screenshot


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    argument_schema: dict[str, Any]
    risk_level: str
    approval_required: bool
    handler: ToolHandler


class ToolRegistry:
    def __init__(
        self,
        root_dir: str | Path,
        safety: SafetyManager,
        logger: AppLogger | None = None,
        lmstudio_client: LMStudioClient | None = None,
        browser_bridge: BrowserToolBridge | None = None,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.workspace_root = (self.root_dir / "workspace").resolve()
        self.logs_dir = (self.root_dir / "logs").resolve()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.actions_log_path = self.logs_dir / "actions.log"
        self.errors_log_path = self.logs_dir / "errors.log"
        self.safety = safety
        self.logger = logger
        self.lmstudio_client = lmstudio_client or LMStudioClient()
        self.browser_bridge = browser_bridge or BrowserToolBridge(self.root_dir)
        self._tools: dict[str, ToolDefinition] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        self.register(
            ToolDefinition(
                name="list_files",
                description="List files and folders under a target path.",
                argument_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                risk_level="safe",
                approval_required=False,
                handler=self._handle_list_files,
            )
        )
        self.register(
            ToolDefinition(
                name="read_file",
                description="Read a UTF-8 text file.",
                argument_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                risk_level="safe",
                approval_required=False,
                handler=self._handle_read_file,
            )
        )
        self.register(
            ToolDefinition(
                name="write_file",
                description="Write UTF-8 text to a file.",
                argument_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                },
                risk_level="medium",
                approval_required=True,
                handler=self._handle_write_file,
            )
        )
        self.register(
            ToolDefinition(
                name="run_command",
                description="Run a shell command in the LocalPilot root directory unless cwd is provided.",
                argument_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "cwd": {"type": "string"},
                        "timeout_seconds": {"type": "integer"},
                    },
                    "required": ["command"],
                },
                risk_level="medium",
                approval_required=True,
                handler=self._handle_run_command,
            )
        )
        self.register(
            ToolDefinition(
                name="take_screenshot",
                description="Capture the current screen and save it to logs/screenshots.",
                argument_schema={"type": "object", "properties": {}},
                risk_level="safe",
                approval_required=False,
                handler=self._handle_take_screenshot,
            )
        )
        self.register(
            ToolDefinition(
                name="analyze_screenshot",
                description="Send a screenshot to the LM Studio vision model and return a description.",
                argument_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["path"],
                },
                risk_level="safe",
                approval_required=False,
                handler=self._handle_analyze_screenshot,
            )
        )
        self.register(
            ToolDefinition(
                name="ask_user_approval",
                description="Ask the user to explicitly approve a requested action.",
                argument_schema={
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"],
                },
                risk_level="safe",
                approval_required=False,
                handler=self._handle_ask_user_approval,
            )
        )
        self.register(
            ToolDefinition(
                name="browser_launch",
                description="Launch the Puppeteer-controlled browser in a visible window when possible.",
                argument_schema={"type": "object", "properties": {"headless": {"type": "boolean"}}},
                risk_level="medium",
                approval_required=True,
                handler=self._handle_browser_launch,
            )
        )
        self.register(
            ToolDefinition(
                name="browser_close",
                description="Close the active Puppeteer-controlled browser session.",
                argument_schema={"type": "object", "properties": {}},
                risk_level="safe",
                approval_required=False,
                handler=self._handle_browser_close,
            )
        )
        self.register(
            ToolDefinition(
                name="browser_goto",
                description="Navigate the Puppeteer-controlled browser to a URL.",
                argument_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
                risk_level="medium",
                approval_required=True,
                handler=self._handle_browser_goto,
            )
        )
        self.register(
            ToolDefinition(
                name="browser_search",
                description="Search the web in the Puppeteer-controlled browser.",
                argument_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "engine": {"type": "string"}},
                    "required": ["query"],
                },
                risk_level="medium",
                approval_required=True,
                handler=self._handle_browser_search,
            )
        )
        self.register(
            ToolDefinition(
                name="browser_click_selector",
                description="Click a DOM selector in the Puppeteer-controlled browser.",
                argument_schema={"type": "object", "properties": {"selector": {"type": "string"}}, "required": ["selector"]},
                risk_level="medium",
                approval_required=True,
                handler=self._handle_browser_click_selector,
            )
        )
        self.register(
            ToolDefinition(
                name="browser_type_selector",
                description="Type text into a DOM selector in the Puppeteer-controlled browser.",
                argument_schema={
                    "type": "object",
                    "properties": {"selector": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["selector", "text"],
                },
                risk_level="medium",
                approval_required=True,
                handler=self._handle_browser_type_selector,
            )
        )
        self.register(
            ToolDefinition(
                name="browser_press_key",
                description="Press a keyboard key in the Puppeteer-controlled browser.",
                argument_schema={"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
                risk_level="medium",
                approval_required=True,
                handler=self._handle_browser_press_key,
            )
        )
        self.register(
            ToolDefinition(
                name="browser_get_text",
                description="Read the current page text from the Puppeteer-controlled browser.",
                argument_schema={"type": "object", "properties": {}},
                risk_level="safe",
                approval_required=False,
                handler=self._handle_browser_get_text,
            )
        )
        self.register(
            ToolDefinition(
                name="browser_screenshot",
                description="Capture a screenshot from the Puppeteer-controlled browser.",
                argument_schema={"type": "object", "properties": {}},
                risk_level="safe",
                approval_required=False,
                handler=self._handle_browser_screenshot,
            )
        )
        self.register(
            ToolDefinition(
                name="browser_get_page_info",
                description="Get the current page title, URL, text preview, and screenshot path from the Puppeteer-controlled browser.",
                argument_schema={"type": "object", "properties": {}},
                risk_level="safe",
                approval_required=False,
                handler=self._handle_browser_get_page_info,
            )
        )

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "argument_schema": tool.argument_schema,
                "risk_level": tool.risk_level,
                "approval_required": tool.approval_required,
            }
            for tool in self._tools.values()
        ]

    def execute_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        tool_name = str(tool_call.get("tool", "")).strip()
        args = tool_call.get("args") or {}
        reason = str(tool_call.get("reason", "")).strip()
        tool = self.get(tool_name)
        if tool is None:
            result = {"ok": False, "tool": tool_name or "(missing)", "error": f"Unknown tool: {tool_name}"}
            self._log_tool_event(tool_name or "(missing)", args, None, False, None, result, reason)
            return result
        if not isinstance(args, dict):
            result = {"ok": False, "tool": tool_name, "error": "Tool args must be a JSON object."}
            self._log_tool_event(tool_name, args, None, False, None, result, reason)
            return result

        decision = self.safety.classify_tool_call(tool_name, args)
        if not decision.allowed or decision.risk_level == RISK_BLOCKED:
            result = {"ok": False, "tool": tool_name, "error": decision.reason}
            self._log_tool_event(tool_name, args, decision, False, None, result, reason)
            return result

        approval_result: bool | None = None
        if decision.approval_required:
            prompt = self._build_approval_prompt(tool_name, args, reason, decision)
            approval_result = self.safety.confirm(prompt)
            if not approval_result:
                result = {"ok": False, "tool": tool_name, "error": "User denied approval."}
                self._log_tool_event(tool_name, args, decision, True, approval_result, result, reason)
                return result

        try:
            raw_result = tool.handler(args)
            if raw_result.get("ok"):
                result = {"ok": True, "tool": tool_name, "result": {k: v for k, v in raw_result.items() if k != "ok"}}
            else:
                result = {"ok": False, "tool": tool_name, "error": raw_result.get("error", "Tool failed.")}
        except Exception as exc:
            result = {"ok": False, "tool": tool_name, "error": str(exc)}

        self._log_tool_event(tool_name, args, decision, decision.approval_required, approval_result, result, reason)
        return result

    def _build_approval_prompt(self, tool_name: str, args: dict[str, Any], reason: str, decision: SafetyDecision) -> str:
        return (
            f"Approve tool `{tool_name}`?\n"
            f"Risk: {decision.risk_level}\n"
            f"Reason: {reason or 'No reason provided.'}\n"
            f"Args: {json.dumps(args, ensure_ascii=True)}"
        )

    def _log_tool_event(
        self,
        tool_name: str,
        args: Any,
        decision: SafetyDecision | None,
        approval_required: bool,
        approval_result: bool | None,
        result: dict[str, Any],
        reason: str,
    ) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tool": tool_name,
            "args": args,
            "reason": reason,
            "risk_level": decision.risk_level if decision else "unknown",
            "approval_required": approval_required,
            "approval_result": approval_result,
            "ok": bool(result.get("ok")),
            "error": result.get("error"),
        }
        with self.actions_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
        if not result.get("ok"):
            with self.errors_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
        if self.logger is not None:
            role = "Tool" if result.get("ok") else "ToolError"
            self.logger.event(role, tool_name, **entry)

    def _resolve_user_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path.resolve()
        return (self.workspace_root / path).resolve()

    def _handle_list_files(self, args: dict[str, Any]) -> dict[str, Any]:
        return list_folder(str(self._resolve_user_path(str(args["path"]))))

    def _handle_read_file(self, args: dict[str, Any]) -> dict[str, Any]:
        return read_file(str(self._resolve_user_path(str(args["path"]))))

    def _handle_write_file(self, args: dict[str, Any]) -> dict[str, Any]:
        return write_file(str(self._resolve_user_path(str(args["path"]))), str(args["content"]))

    def _handle_run_command(self, args: dict[str, Any]) -> dict[str, Any]:
        cwd_value = args.get("cwd")
        cwd = self.root_dir if cwd_value is None else self._resolve_user_path(str(cwd_value))
        timeout_seconds = int(args.get("timeout_seconds", 30))
        completed = subprocess.run(
            str(args["command"]),
            cwd=str(cwd),
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return {
            "ok": completed.returncode == 0,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "command": str(args["command"]),
            "cwd": str(cwd),
        }

    def _handle_take_screenshot(self, args: dict[str, Any]) -> dict[str, Any]:
        screenshots_dir = self.logs_dir / "screenshots"
        return take_screenshot(screenshots_dir)

    def _handle_analyze_screenshot(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_user_path(str(args["path"])) if not Path(str(args["path"])).is_absolute() else Path(str(args["path"])).resolve()
        prompt = str(args.get("prompt") or "Describe this screenshot in one sentence and mention any obvious visible text.")
        description = self.lmstudio_client.chat_vision(prompt=prompt, image_path=path, model=self.lmstudio_client.default_vision_model)
        return {"ok": True, "path": str(path), "description": description}

    def _handle_ask_user_approval(self, args: dict[str, Any]) -> dict[str, Any]:
        prompt = str(args["prompt"])
        approved = self.safety.confirm(prompt)
        return {"ok": True, "approved": approved, "prompt": prompt}

    def _handle_browser_launch(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.browser_bridge.run("launch_browser", headless=bool(args.get("headless", False)))

    def _handle_browser_close(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.browser_bridge.run("close_browser")

    def _handle_browser_goto(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.browser_bridge.run("goto_url", url=str(args["url"]))

    def _handle_browser_search(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.browser_bridge.run("search_web", query=str(args["query"]), engine=str(args.get("engine", "google")))

    def _handle_browser_click_selector(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.browser_bridge.run("click_selector", selector=str(args["selector"]))

    def _handle_browser_type_selector(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.browser_bridge.run("type_selector", selector=str(args["selector"]), text=str(args["text"]))

    def _handle_browser_press_key(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.browser_bridge.run("press_key", key=str(args["key"]))

    def _handle_browser_get_text(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.browser_bridge.run("get_page_text")

    def _handle_browser_screenshot(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.browser_bridge.run("screenshot")

    def _handle_browser_get_page_info(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.browser_bridge.run("get_page_info")
