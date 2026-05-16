from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from app.checkpoints import CheckpointManager
from app.browser_tool import BrowserToolBridge
from app.desktop_tool import (
    DesktopSuggestionStore,
    execute_suggestion_click,
    get_mouse_position as desktop_get_mouse_position,
    get_screen_size as desktop_get_screen_size,
    move_mouse_preview,
    suggest_action_from_screenshot,
)
from app.lmstudio_client import LMStudioClient
from app.logger import AppLogger
from app.memory import MemoryStore
from app.safety import RISK_BLOCKED, RISK_DANGEROUS, RISK_MEDIUM, SafetyDecision, SafetyManager
from app.timer_tool import TimerManager
from app.tools.files import list_folder, read_file, write_file
from app.tools.screen import take_screenshot


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]
GROUPABLE_APPROVAL_TOOLS = {
    "browser_launch",
    "browser_goto",
    "browser_search",
    "browser_click_selector",
    "browser_type_selector",
    "browser_press_key",
}
INTERNAL_ARG_KEYS = {"task_id", "tool_call_id"}


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
        checkpoint_manager: CheckpointManager | None = None,
        memory_store: MemoryStore | None = None,
        timer_manager: TimerManager | None = None,
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
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(self.root_dir / "memory" / "checkpoints")
        self.memory_store = memory_store or MemoryStore(self.root_dir / "memory", self.root_dir / "config" / "capabilities.json")
        self.timer_manager = timer_manager or TimerManager(self.root_dir / "memory" / "timers.json")
        self.desktop_suggestion_store = DesktopSuggestionStore(self.root_dir / "memory" / "runtime" / "desktop_suggestions.json")
        self._approved_plans: dict[str, dict[str, Any]] = {}
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
                name="list_checkpoints",
                description="List saved file checkpoints and the files they can restore.",
                argument_schema={"type": "object", "properties": {"limit": {"type": "integer"}}},
                risk_level="safe",
                approval_required=False,
                handler=self._handle_list_checkpoints,
            )
        )
        self.register(
            ToolDefinition(
                name="restore_checkpoint",
                description="Restore files from a saved checkpoint.",
                argument_schema={
                    "type": "object",
                    "properties": {"checkpoint_id": {"type": "string"}},
                    "required": ["checkpoint_id"],
                },
                risk_level="dangerous",
                approval_required=True,
                handler=self._handle_restore_checkpoint,
            )
        )
        self.register(
            ToolDefinition(
                name="list_sessions",
                description="List recent saved agent sessions.",
                argument_schema={"type": "object", "properties": {"limit": {"type": "integer"}}},
                risk_level="safe",
                approval_required=False,
                handler=self._handle_list_sessions,
            )
        )
        self.register(
            ToolDefinition(
                name="read_session",
                description="Read a saved agent session by session id or task id.",
                argument_schema={
                    "type": "object",
                    "properties": {"session_id": {"type": "string"}},
                    "required": ["session_id"],
                },
                risk_level="safe",
                approval_required=False,
                handler=self._handle_read_session,
            )
        )
        self.register(
            ToolDefinition(
                name="get_current_task",
                description="Read the current active agent task state.",
                argument_schema={"type": "object", "properties": {}},
                risk_level="safe",
                approval_required=False,
                handler=self._handle_get_current_task,
            )
        )
        self.register(
            ToolDefinition(
                name="update_current_task",
                description="Update the current active task state with new status or notes.",
                argument_schema={"type": "object", "properties": {"updates": {"type": "object"}}, "required": ["updates"]},
                risk_level="safe",
                approval_required=False,
                handler=self._handle_update_current_task,
            )
        )
        self.register(
            ToolDefinition(
                name="clear_current_task",
                description="Clear the saved current active task state.",
                argument_schema={"type": "object", "properties": {}},
                risk_level="safe",
                approval_required=False,
                handler=self._handle_clear_current_task,
            )
        )
        self.register(
            ToolDefinition(
                name="summarize_recent_sessions",
                description="Summarize the most recent agent sessions.",
                argument_schema={"type": "object", "properties": {"limit": {"type": "integer"}}},
                risk_level="safe",
                approval_required=False,
                handler=self._handle_summarize_recent_sessions,
            )
        )
        self.register(
            ToolDefinition(
                name="set_timer",
                description="Set a real local PC timer that notifies the user without blocking the agent.",
                argument_schema={
                    "type": "object",
                    "properties": {
                        "duration_seconds": {"type": "integer"},
                        "label": {"type": "string"},
                        "notify": {"type": "boolean"},
                    },
                    "required": ["duration_seconds"],
                },
                risk_level="safe",
                approval_required=False,
                handler=self._handle_set_timer,
            )
        )
        self.register(
            ToolDefinition(
                name="list_timers",
                description="List active or recent local timers.",
                argument_schema={"type": "object", "properties": {"include_inactive": {"type": "boolean"}}},
                risk_level="safe",
                approval_required=False,
                handler=self._handle_list_timers,
            )
        )
        self.register(
            ToolDefinition(
                name="cancel_timer",
                description="Cancel a previously scheduled local timer.",
                argument_schema={
                    "type": "object",
                    "properties": {"timer_id": {"type": "string"}},
                    "required": ["timer_id"],
                },
                risk_level="safe",
                approval_required=False,
                handler=self._handle_cancel_timer,
            )
        )
        self.register(
            ToolDefinition(
                name="desktop_get_screen_size",
                description="Read the current primary screen size in pixels.",
                argument_schema={"type": "object", "properties": {}},
                risk_level="safe",
                approval_required=False,
                handler=self._handle_desktop_get_screen_size,
            )
        )
        self.register(
            ToolDefinition(
                name="desktop_get_mouse_position",
                description="Read the current mouse cursor position on the Windows desktop.",
                argument_schema={"type": "object", "properties": {}},
                risk_level="safe",
                approval_required=False,
                handler=self._handle_desktop_get_mouse_position,
            )
        )
        self.register(
            ToolDefinition(
                name="desktop_suggest_action",
                description="Analyze a desktop screenshot and suggest the next desktop action without executing it.",
                argument_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "instruction": {"type": "string"},
                    },
                    "required": ["path", "instruction"],
                },
                risk_level="safe",
                approval_required=False,
                handler=self._handle_desktop_suggest_action,
            )
        )
        self.register(
            ToolDefinition(
                name="desktop_move_mouse_preview",
                description="Move the mouse to preview a suggested desktop target without clicking.",
                argument_schema={
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                        "target": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["x", "y"],
                },
                risk_level="medium",
                approval_required=True,
                handler=self._handle_desktop_move_mouse_preview,
            )
        )
        self.register(
            ToolDefinition(
                name="desktop_execute_suggestion",
                description="Execute one previously suggested desktop click by suggestion_id after explicit approval.",
                argument_schema={
                    "type": "object",
                    "properties": {"suggestion_id": {"type": "string"}},
                    "required": ["suggestion_id"],
                },
                risk_level="dangerous",
                approval_required=True,
                handler=self._handle_desktop_execute_suggestion,
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

    def list_tools(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        allowed = set(names or [])
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "argument_schema": tool.argument_schema,
                "risk_level": tool.risk_level,
                "approval_required": tool.approval_required,
            }
            for tool in self._tools.values()
            if not allowed or tool.name in allowed
        ]

    def execute_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        tool_name = str(tool_call.get("tool", "")).strip()
        args = tool_call.get("args") or {}
        execution_args = dict(args) if isinstance(args, dict) else args
        if isinstance(execution_args, dict):
            if tool_call.get("task_id") and "task_id" not in execution_args:
                execution_args["task_id"] = tool_call.get("task_id")
            if tool_call.get("tool_call_id") and "tool_call_id" not in execution_args:
                execution_args["tool_call_id"] = tool_call.get("tool_call_id")
        reason = str(tool_call.get("reason", "")).strip()
        tool = self.get(tool_name)
        if tool is None:
            result = {"ok": False, "tool": tool_name or "(missing)", "error": f"Unknown tool: {tool_name}"}
            self._log_tool_event(tool_name or "(missing)", args, None, False, None, result, reason)
            return result
        if not isinstance(execution_args, dict):
            result = {"ok": False, "tool": tool_name, "error": "Tool args must be a JSON object."}
            self._log_tool_event(tool_name, args, None, False, None, result, reason)
            return result

        decision = self.safety.classify_tool_call(tool_name, execution_args)
        if not decision.allowed or decision.risk_level == RISK_BLOCKED:
            result = {"ok": False, "tool": tool_name, "error": decision.reason}
            self._log_tool_event(tool_name, args, decision, False, None, result, reason)
            return result

        approval_result: bool | None = None
        approval_request: dict[str, Any] | None = None
        cached_approval = self._consume_matching_approval_plan(tool_name, execution_args)
        if cached_approval is not None:
            approval_result = True
            approval_request = {**cached_approval, "granted": True, "reused": True}
        elif decision.approval_required:
            approval_request = self.build_approval_request(
                tool_name,
                execution_args,
                reason,
                decision,
                tool_call.get("approval_plan"),
            )
            approval_result = self.safety.confirm(approval_request)
            if not approval_result:
                result = {
                    "ok": False,
                    "tool": tool_name,
                    "error": "User denied approval.",
                    "approval": {**approval_request, "granted": False},
                }
                self._log_tool_event(tool_name, args, decision, True, approval_result, result, reason)
                return result
            if approval_request.get("type") == "approval_plan":
                self._activate_approval_plan(approval_request, tool_name, execution_args)

        try:
            raw_result = tool.handler(execution_args)
            if raw_result.get("ok"):
                result = {"ok": True, "tool": tool_name, "result": {k: v for k, v in raw_result.items() if k != "ok"}}
            else:
                result = {"ok": False, "tool": tool_name, "error": raw_result.get("error", "Tool failed.")}
        except Exception as exc:
            result = {"ok": False, "tool": tool_name, "error": str(exc)}
        if approval_request is not None:
            result["approval"] = {**approval_request, "granted": bool(approval_result)}

        self._log_tool_event(tool_name, args, decision, decision.approval_required, approval_result, result, reason)
        return result

    def build_approval_request(
        self,
        tool_name: str,
        args: dict[str, Any],
        reason: str,
        decision: SafetyDecision,
        proposed_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        grouped = self._build_grouped_approval_request(tool_name, args, reason, proposed_plan)
        if grouped is not None:
            return grouped
        summary = f"The agent wants to run `{tool_name}`."
        affected_files: list[str] = []
        if tool_name == "restore_checkpoint":
            checkpoint = self.checkpoint_manager.get_checkpoint(str(args.get("checkpoint_id", "")))
            if checkpoint is not None:
                affected_files = [str(entry.get("original_path")) for entry in checkpoint.get("files", []) if entry.get("original_path")]
                summary = f"Restore checkpoint `{args.get('checkpoint_id', '')}` for {len(affected_files)} file(s)."
        elif tool_name == "desktop_move_mouse_preview":
            target_suffix = f" near {args.get('target')!r}" if args.get("target") else ""
            summary = (
                "Move the mouse for preview only"
                f" to ({args.get('x')}, {args.get('y')})"
                f"{target_suffix}."
                " No click will be performed."
            )
        elif tool_name == "desktop_execute_suggestion":
            suggestion = self.desktop_suggestion_store.get_suggestion(str(args.get("suggestion_id", "")))
            if suggestion is not None:
                summary = (
                    "Execute one approved desktop click"
                    f" on {suggestion.get('target', 'unknown target')!r}"
                    f" at ({suggestion.get('x')}, {suggestion.get('y')})."
                )
            else:
                summary = f"Execute desktop suggestion `{args.get('suggestion_id', '')}`."
        request = {
            "approval_id": uuid.uuid4().hex[:12],
            "type": "approval_request",
            "risk": decision.risk_level,
            "summary": summary,
            "reason": reason or "No reason provided.",
            "tool_calls": [{"tool": tool_name, "args": self._public_args(args)}],
        }
        if affected_files:
            request["affected_files"] = affected_files
        return request

    def _build_grouped_approval_request(
        self,
        tool_name: str,
        args: dict[str, Any],
        reason: str,
        proposed_plan: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(proposed_plan, dict):
            return None
        tool_calls = proposed_plan.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            return None
        normalized_calls: list[dict[str, Any]] = []
        for index, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                return None
            plan_tool = str(call.get("tool", "")).strip()
            plan_args = call.get("args") or {}
            if not plan_tool or not isinstance(plan_args, dict):
                return None
            if plan_tool not in GROUPABLE_APPROVAL_TOOLS:
                return None
            decision = self.safety.classify_tool_call(plan_tool, plan_args)
            if not decision.allowed or decision.risk_level != RISK_MEDIUM or not decision.approval_required:
                return None
            normalized_call = {"tool": plan_tool, "args": plan_args}
            normalized_calls.append(normalized_call)
            if index == 0 and not self._tool_call_matches(plan_tool, plan_args, tool_name, self._public_args(args)):
                return None
        return {
            "approval_id": str(proposed_plan.get("approval_id") or uuid.uuid4().hex[:12]),
            "type": "approval_plan",
            "risk": RISK_MEDIUM,
            "summary": str(proposed_plan.get("summary") or f"Allow this browser plan for `{tool_name}`?"),
            "reason": str(proposed_plan.get("reason") or reason or "No reason provided."),
            "tool_calls": normalized_calls,
        }

    def _activate_approval_plan(self, approval_request: dict[str, Any], current_tool_name: str, current_args: dict[str, Any]) -> None:
        remaining_signatures = [
            self._tool_signature(call["tool"], call["args"])
            for call in approval_request.get("tool_calls", [])
            if not self._tool_call_matches(call["tool"], call["args"], current_tool_name, self._public_args(current_args))
        ]
        self._approved_plans[str(approval_request["approval_id"])] = {
            "approval": approval_request,
            "remaining_signatures": remaining_signatures,
        }

    def _consume_matching_approval_plan(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        signature = self._tool_signature(tool_name, self._public_args(args))
        for approval_id in list(self._approved_plans):
            cached = self._approved_plans[approval_id]
            remaining = cached.get("remaining_signatures", [])
            if remaining and remaining[0] == signature:
                cached["remaining_signatures"] = remaining[1:]
                approval = dict(cached["approval"])
                if not cached["remaining_signatures"]:
                    self._approved_plans.pop(approval_id, None)
                return approval
            if not remaining:
                self._approved_plans.pop(approval_id, None)
        return None

    def _tool_signature(self, tool_name: str, args: dict[str, Any]) -> str:
        return json.dumps({"tool": tool_name, "args": args}, sort_keys=True, ensure_ascii=True)

    def _tool_call_matches(self, left_tool: str, left_args: dict[str, Any], right_tool: str, right_args: dict[str, Any]) -> bool:
        return self._tool_signature(left_tool, left_args) == self._tool_signature(right_tool, right_args)

    def _public_args(self, args: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in args.items() if key not in INTERNAL_ARG_KEYS}

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
            "approval_id": (result.get("approval") or {}).get("approval_id") if isinstance(result, dict) else None,
            "approval_type": (result.get("approval") or {}).get("type") if isinstance(result, dict) else None,
            "approval_summary": (result.get("approval") or {}).get("summary") if isinstance(result, dict) else None,
            "affected_files": ((result.get("result") or {}).get("restored_files") if isinstance(result, dict) else None),
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
        path = self._resolve_user_path(str(args["path"]))
        checkpoint = self.checkpoint_manager.create_file_checkpoint(
            path,
            task_id=str(args.get("task_id") or "") or None,
            tool_call_id=str(args.get("tool_call_id") or "") or None,
        )
        result = write_file(str(path), str(args["content"]))
        if result.get("ok"):
            result["checkpoint_id"] = checkpoint["checkpoint_id"]
            result["checkpoint_manifest_path"] = checkpoint["manifest_path"]
            result["file_existed_before"] = checkpoint["file_existed_before"]
        return result

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

    def _handle_list_checkpoints(self, args: dict[str, Any]) -> dict[str, Any]:
        limit = max(int(args.get("limit", 20)), 1)
        manifests = self.checkpoint_manager.list_checkpoints()[:limit]
        checkpoints = [
            {
                "checkpoint_id": manifest.get("checkpoint_id"),
                "timestamp": manifest.get("timestamp"),
                "task_id": manifest.get("task_id"),
                "tool_call_id": manifest.get("tool_call_id"),
                "files": [entry.get("original_path") for entry in manifest.get("files", [])],
            }
            for manifest in manifests
        ]
        return {"ok": True, "checkpoints": checkpoints}

    def _handle_restore_checkpoint(self, args: dict[str, Any]) -> dict[str, Any]:
        checkpoint_id = str(args["checkpoint_id"])
        restored = self.checkpoint_manager.restore_checkpoint(checkpoint_id)
        if not restored.get("ok"):
            return restored
        return {
            "ok": True,
            "checkpoint_id": checkpoint_id,
            "restored_files": restored.get("restored_paths", []),
        }

    def _handle_list_sessions(self, args: dict[str, Any]) -> dict[str, Any]:
        limit = max(int(args.get("limit", 10)), 1)
        return {"ok": True, "sessions": self.memory_store.list_session_summaries(limit=limit)}

    def _handle_read_session(self, args: dict[str, Any]) -> dict[str, Any]:
        session = self.memory_store.read_session(str(args["session_id"]))
        if session is None:
            return {"ok": False, "error": f"Session not found: {args['session_id']}"}
        return {"ok": True, "session": session}

    def _handle_get_current_task(self, args: dict[str, Any]) -> dict[str, Any]:
        task = self.memory_store.load_current_task()
        return {"ok": True, "current_task": task}

    def _handle_update_current_task(self, args: dict[str, Any]) -> dict[str, Any]:
        updates = args.get("updates") or {}
        if not isinstance(updates, dict):
            return {"ok": False, "error": "updates must be an object."}
        current_task = self.memory_store.update_current_task(**updates)
        return {"ok": True, "current_task": current_task}

    def _handle_clear_current_task(self, args: dict[str, Any]) -> dict[str, Any]:
        message = self.memory_store.clear_current_task()
        return {"ok": True, "message": message}

    def _handle_summarize_recent_sessions(self, args: dict[str, Any]) -> dict[str, Any]:
        limit = max(int(args.get("limit", 3)), 1)
        return {"ok": True, "summary": self.memory_store.summarize_recent_sessions(limit=limit)}

    def _handle_set_timer(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.timer_manager.set_timer(
            duration_seconds=int(args["duration_seconds"]),
            label=str(args.get("label", "Timer")),
            notify=bool(args.get("notify", True)),
        )

    def _handle_list_timers(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.timer_manager.list_timers(include_inactive=bool(args.get("include_inactive", False)))

    def _handle_cancel_timer(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.timer_manager.cancel_timer(str(args["timer_id"]))

    def _handle_desktop_get_screen_size(self, args: dict[str, Any]) -> dict[str, Any]:
        return desktop_get_screen_size()

    def _handle_desktop_get_mouse_position(self, args: dict[str, Any]) -> dict[str, Any]:
        return desktop_get_mouse_position()

    def _handle_desktop_suggest_action(self, args: dict[str, Any]) -> dict[str, Any]:
        raw_path = str(args["path"])
        path = self._resolve_user_path(raw_path) if not Path(raw_path).is_absolute() else Path(raw_path).resolve()
        result = suggest_action_from_screenshot(
            screenshot_path=path,
            instruction=str(args["instruction"]),
            lmstudio_client=self.lmstudio_client,
            model=self.lmstudio_client.default_vision_model,
        )
        if not result.get("ok"):
            return result
        suggestion_record = self.desktop_suggestion_store.create_suggestion(
            task_id=str(args.get("task_id") or "").strip() or None,
            suggestion=result,
            screenshot_path=path,
        )
        desktop_action = dict(result.get("desktop_action") or {})
        desktop_action.update(
            {
                "suggestion_id": suggestion_record["suggestion_id"],
                "target": suggestion_record["target"],
                "x": suggestion_record["x"],
                "y": suggestion_record["y"],
                "confidence": suggestion_record["confidence"],
                "approved": False,
            }
        )
        result.update(
            {
                "suggestion_id": suggestion_record["suggestion_id"],
                "timestamp": suggestion_record["timestamp"],
                "task_id": suggestion_record["task_id"],
                "screenshot_path": suggestion_record["screenshot_path"],
                "screenshot_hash": suggestion_record["screenshot_hash"],
                "expires_at": suggestion_record["expires_at"],
                "desktop_action": desktop_action,
            }
        )
        return result

    def _handle_desktop_move_mouse_preview(self, args: dict[str, Any]) -> dict[str, Any]:
        return move_mouse_preview(
            int(args["x"]),
            int(args["y"]),
            target=str(args.get("target", "")),
            confidence=float(args["confidence"]) if args.get("confidence") is not None else None,
        )

    def _handle_desktop_execute_suggestion(self, args: dict[str, Any]) -> dict[str, Any]:
        suggestion_id = str(args.get("suggestion_id", "")).strip()
        if not suggestion_id:
            return {"ok": False, "error": "suggestion_id is required for desktop_execute_suggestion."}
        return execute_suggestion_click(
            suggestion_id=suggestion_id,
            suggestion_store=self.desktop_suggestion_store,
        )

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
