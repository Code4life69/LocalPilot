from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


CONTEXT_MINIMUM_USABLE = 8192
CONTEXT_RECOMMENDED = 16384
CONTEXT_BAD_FOLLOWUPS = 4096

COMPACT_RULE_FALLBACK = [
    "User chats with AI agent, not Python.",
    "AI chooses tools; Python validates, executes, and logs.",
    "Never fake tool success.",
    "Use Puppeteer browser tools for websites.",
    "Use Qwen3-VL for screenshots and visual analysis.",
    "Require approval for risky actions.",
    "Do not execute desktop click, type, or hotkey actions yet.",
]

TOOL_GROUPS: dict[str, list[str]] = {
    "timer": [
        "set_timer",
        "list_timers",
        "cancel_timer",
        "get_current_task",
        "summarize_recent_sessions",
    ],
    "desktop": [
        "take_screenshot",
        "analyze_screenshot",
        "desktop_suggest_action",
        "desktop_get_screen_size",
        "desktop_get_mouse_position",
    ],
    "browser": [
        "browser_launch",
        "browser_goto",
        "browser_search",
        "browser_get_text",
        "browser_get_page_info",
        "browser_screenshot",
    ],
    "file": [
        "list_files",
        "read_file",
        "write_file",
        "list_checkpoints",
        "restore_checkpoint",
    ],
    "memory": [
        "get_current_task",
        "summarize_recent_sessions",
        "list_sessions",
        "read_session",
    ],
    "general": [
        "get_current_task",
        "summarize_recent_sessions",
        "take_screenshot",
        "analyze_screenshot",
        "desktop_suggest_action",
        "browser_launch",
        "browser_goto",
        "browser_search",
        "browser_get_page_info",
        "set_timer",
        "list_timers",
        "cancel_timer",
        "list_files",
        "read_file",
        "write_file",
    ],
}

ULTRA_COMPACT_TOOL_GROUPS: dict[str, list[str]] = {
    "timer": ["set_timer", "list_timers", "cancel_timer"],
    "desktop": ["take_screenshot", "analyze_screenshot", "desktop_suggest_action"],
    "browser": ["browser_search", "browser_get_page_info", "browser_screenshot"],
    "file": ["read_file", "write_file", "list_checkpoints", "restore_checkpoint"],
    "memory": ["get_current_task", "list_sessions", "read_session"],
    "general": ["get_current_task", "take_screenshot", "browser_get_page_info", "set_timer"],
}


@dataclass(slots=True)
class PromptBuild:
    system_prompt: str
    working_memory: str
    tool_names: list[str]
    tool_count: int
    task_category: str
    prompt_mode: str
    planner_context_warning: str | None = None


class PromptBuilder:
    def __init__(
        self,
        *,
        planner_context_length: int = CONTEXT_RECOMMENDED,
        minimum_context_length: int = CONTEXT_MINIMUM_USABLE,
        recommended_context_length: int = CONTEXT_RECOMMENDED,
    ) -> None:
        self.planner_context_length = max(int(planner_context_length or 0), 0)
        self.minimum_context_length = max(int(minimum_context_length or 0), 0)
        self.recommended_context_length = max(int(recommended_context_length or 0), 0)

    def planner_context_warning(self) -> str | None:
        if self.planner_context_length and self.planner_context_length < self.minimum_context_length:
            return (
                "Planner context is too small for reliable agent follow-ups. "
                f"Use {self.minimum_context_length} or {self.recommended_context_length}."
            )
        return None

    def build(
        self,
        *,
        user_message: str,
        current_task: dict[str, Any] | None,
        available_tools: list[dict[str, Any]],
        rules_text: str,
        prompt_mode: str = "standard",
    ) -> PromptBuild:
        task_category = self.infer_task_category(user_message, current_task)
        tool_names = self.select_tool_names(task_category, current_task, prompt_mode=prompt_mode)
        tools = self.filter_tools(available_tools, tool_names)
        compact_rules = self._compact_rules(rules_text)
        working_memory = self._build_working_memory(
            user_message=user_message,
            current_task=current_task,
            task_category=task_category,
            prompt_mode=prompt_mode,
        )
        tools_block = self._format_tools_block(tools)
        system_prompt = (
            "You are LocalPilot, an AI agent.\n"
            "Reply with JSON only.\n"
            "Allowed response formats:\n"
            '{"type":"tool_call","tool":"tool_name","args":{},"reason":"why this tool is needed"}\n'
            '{"type":"final","message":"final answer for the user"}\n'
            '{"type":"question","message":"one clarification question"}\n'
            "Rules:\n"
            f"{compact_rules}\n"
            "Available tools:\n"
            f"{tools_block}"
        )
        if working_memory:
            system_prompt += f"\nWorking memory:\n{working_memory}"
        warning = self.planner_context_warning()
        return PromptBuild(
            system_prompt=system_prompt,
            working_memory=working_memory,
            tool_names=[tool["name"] for tool in tools],
            tool_count=len(tools),
            task_category=task_category,
            prompt_mode=prompt_mode,
            planner_context_warning=warning,
        )

    def infer_task_category(self, user_message: str, current_task: dict[str, Any] | None) -> str:
        lowered = user_message.strip().lower()
        current_tool_names = {
            str(call.get("tool", "")).strip()
            for call in (current_task or {}).get("recent_tool_calls", [])
            if isinstance(call, dict)
        }
        if self._contains_any(lowered, ("timer", "remind me", "reminder", "minutes", "minute", "seconds", "second", "hour")):
            return "timer"
        if self._contains_any(lowered, ("google", "browser", "website", "web", "url", "search", "search results", "page")):
            return "browser"
        if self._contains_any(lowered, ("screen", "sidebar", "desktop", "window", "click next", "what would click", "visible ui", "icon")):
            return "desktop"
        if self._contains_any(lowered, ("file", "folder", "workspace", "write", "read", "checkpoint", "restore", "undo")):
            return "file"
        if self._contains_any(lowered, ("what happened", "current task", "previous", "session", "remember", "memory")):
            return "memory"
        if lowered in {"continue", "try again", "what did you look up?"}:
            return self._category_from_recent_tools(current_tool_names)
        if current_tool_names:
            return self._category_from_recent_tools(current_tool_names)
        return "general"

    def select_tool_names(self, task_category: str, current_task: dict[str, Any] | None, *, prompt_mode: str = "standard") -> list[str]:
        tool_groups = ULTRA_COMPACT_TOOL_GROUPS if prompt_mode == "ultra_compact" else TOOL_GROUPS
        selected = list(tool_groups.get(task_category, tool_groups["general"]))
        current_tool_names = [
            str(call.get("tool", "")).strip()
            for call in (current_task or {}).get("recent_tool_calls", [])
            if isinstance(call, dict) and call.get("tool")
        ]
        for tool_name in current_tool_names[-2:]:
            if tool_name and tool_name not in selected:
                selected.append(tool_name)
        return selected

    def filter_tools(self, available_tools: list[dict[str, Any]], tool_names: list[str]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        allowed = set(tool_names)
        for tool in available_tools:
            if tool.get("name") in allowed:
                selected.append(tool)
        return selected

    def _build_working_memory(
        self,
        *,
        user_message: str,
        current_task: dict[str, Any] | None,
        task_category: str,
        prompt_mode: str,
    ) -> str:
        if not current_task:
            return f"Task category: {task_category}"
        lines: list[str] = [f"Task category: {task_category}"]
        active_task_summary = self._build_active_task_summary(current_task)
        if active_task_summary:
            lines.append(f"Active task: {active_task_summary}")
        pending_approval = self._summarize_pending_approval(current_task)
        if pending_approval:
            lines.append(f"Pending approval: {pending_approval}")
        last_tool_result = self._truncate(str(current_task.get("last_tool_result_summary", "")).strip(), 800)
        if last_tool_result:
            lines.append(f"Last tool result: {last_tool_result}")
        last_final_answer = self._truncate(str(current_task.get("last_final_answer", "")).strip(), 600)
        if last_final_answer:
            lines.append(f"Last final answer: {last_final_answer}")
        recent_messages = current_task.get("recent_messages") or []
        rendered_recent_messages: list[str] = []
        for message in recent_messages[-3:]:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "message")).strip()
            content = self._truncate(str(message.get("content", "")).strip(), 300)
            if content:
                rendered_recent_messages.append(f"- {role}: {content}")
        if rendered_recent_messages:
            lines.append("Recent messages:")
            lines.extend(rendered_recent_messages)
        recent_results = current_task.get("recent_tool_result_summaries") or []
        rendered_recent_results: list[str] = []
        seen: set[str] = set()
        for summary in recent_results[-3:]:
            normalized = self._truncate(str(summary).strip(), 400)
            if normalized and normalized not in seen and normalized != last_tool_result:
                seen.add(normalized)
                rendered_recent_results.append(f"- {normalized}")
        if rendered_recent_results and prompt_mode != "ultra_compact":
            lines.append("Recent tool results:")
            lines.extend(rendered_recent_results)
        last_error = self._truncate(str(current_task.get("last_error", "")).strip(), 400)
        if last_error:
            lines.append(f"Last error: {last_error}")
        retry_suggestion = self._truncate(str(current_task.get("retry_suggestion", "")).strip(), 200)
        if retry_suggestion:
            lines.append(f"Retry suggestion: {retry_suggestion}")
        lines.append(f"Current user message: {self._truncate(user_message.strip(), 300)}")
        return "\n".join(line for line in lines if line)

    def _build_active_task_summary(self, current_task: dict[str, Any]) -> str:
        parts = [
            f"id={current_task.get('active_task_id', '')}",
            f"original={self._truncate(str(current_task.get('original_user_task', '')).strip(), 250)}",
            f"latest={self._truncate(str(current_task.get('latest_user_message', '')).strip(), 180)}",
            f"status={current_task.get('status', '')}",
        ]
        return self._truncate(" | ".join(part for part in parts if part and not part.endswith("=")), 800)

    def _summarize_pending_approval(self, current_task: dict[str, Any]) -> str:
        approval = current_task.get("last_approval_request") or {}
        if not isinstance(approval, dict):
            return ""
        summary = str(approval.get("summary", "")).strip()
        risk = str(approval.get("risk", "")).strip()
        if not summary:
            return ""
        combined = summary if not risk else f"{summary} | risk={risk}"
        return self._truncate(combined, 300)

    def _compact_rules(self, rules_text: str) -> str:
        selected: list[str] = []
        for raw_line in rules_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if not line.startswith("-"):
                if not selected and rules_text.strip():
                    selected.append(line)
                continue
            normalized = line.lstrip("- ").strip()
            lowered = normalized.lower()
            if any(
                key in lowered
                for key in (
                    "user chats with the ai agent",
                    "python is only",
                    "ai must make the plan",
                    "prefer puppeteer",
                    "prefer qwen3-vl",
                    "require approval",
                    "never fake tool success",
                    "desktop click",
                    "typing",
                    "key press",
                    "hotkey",
                )
            ):
                selected.append(normalized)
        if not selected:
            selected = list(COMPACT_RULE_FALLBACK)
        deduped: list[str] = []
        seen: set[str] = set()
        for line in selected:
            if line not in seen:
                deduped.append(line)
                seen.add(line)
        return "\n".join(f"- {line}" for line in deduped[:7])

    def _format_tools_block(self, tools: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for tool in tools:
            name = str(tool.get("name", "")).strip()
            description = self._truncate(str(tool.get("description", "")).strip(), 120)
            risk = str(tool.get("risk_level", "")).strip()
            approval = "yes" if tool.get("approval_required") else "no"
            signature = self._tool_signature(tool)
            lines.append(f"- {name}{signature} [risk={risk}, approval={approval}]: {description}")
        return "\n".join(lines)

    def _tool_signature(self, tool: dict[str, Any]) -> str:
        schema = tool.get("argument_schema") or {}
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        parts: list[str] = []
        for key, value in properties.items():
            type_name = str((value or {}).get("type", "any"))
            suffix = "" if key in required else "?"
            parts.append(f"{key}{suffix}:{type_name}")
        if not parts:
            return "()"
        return f"({', '.join(parts)})"

    def _category_from_recent_tools(self, current_tool_names: set[str]) -> str:
        if any(name.startswith("browser_") for name in current_tool_names):
            return "browser"
        if any(name.startswith("desktop_") for name in current_tool_names) or {"take_screenshot", "analyze_screenshot"} & current_tool_names:
            return "desktop"
        if {"set_timer", "list_timers", "cancel_timer"} & current_tool_names:
            return "timer"
        if {"write_file", "read_file", "list_files", "restore_checkpoint"} & current_tool_names:
            return "file"
        if {"get_current_task", "list_sessions", "read_session"} & current_tool_names:
            return "memory"
        return "general"

    def _contains_any(self, text: str, tokens: tuple[str, ...]) -> bool:
        return any(token in text for token in tokens)

    def _truncate(self, text: str, max_chars: int) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= max_chars:
            return cleaned
        if max_chars <= 1:
            return cleaned[:max_chars]
        return cleaned[: max_chars - 3].rstrip() + "..."

    def compact_json(self, payload: Any, *, string_limit: int = 240, list_limit: int = 6, depth: int = 0) -> Any:
        if depth >= 4:
            return self._truncate(json.dumps(payload, ensure_ascii=True), string_limit)
        if isinstance(payload, dict):
            compacted: dict[str, Any] = {}
            for key, value in list(payload.items())[:12]:
                compacted[str(key)] = self.compact_json(value, string_limit=string_limit, list_limit=list_limit, depth=depth + 1)
            return compacted
        if isinstance(payload, list):
            compacted_items = [
                self.compact_json(item, string_limit=string_limit, list_limit=list_limit, depth=depth + 1)
                for item in payload[:list_limit]
            ]
            if len(payload) > list_limit:
                compacted_items.append(f"... ({len(payload) - list_limit} more)")
            return compacted_items
        if isinstance(payload, str):
            return self._truncate(payload, string_limit)
        return payload
