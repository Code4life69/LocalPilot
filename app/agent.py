from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.lmstudio_client import LMStudioClient
from app.memory import MemoryStore
from app.prompt_builder import CONTEXT_MINIMUM_USABLE, CONTEXT_RECOMMENDED, PromptBuild, PromptBuilder
from app.tool_registry import ToolRegistry


@dataclass(slots=True)
class AgentStep:
    kind: str
    payload: dict[str, Any]


class LocalPilotAgent:
    def __init__(
        self,
        llm_client: LMStudioClient,
        tool_registry: ToolRegistry,
        planner_model: str = "qwen2.5-coder-14b-instruct",
        max_steps: int = 12,
        memory_store: MemoryStore | None = None,
        root_dir: str | Path | None = None,
        planner_context_length: int = CONTEXT_RECOMMENDED,
        minimum_context_length: int = CONTEXT_MINIMUM_USABLE,
        recommended_context_length: int = CONTEXT_RECOMMENDED,
        planner_timeout_seconds: int = 120,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.planner_model = planner_model
        self.max_steps = max_steps
        self.memory_store = memory_store
        self.root_dir = Path(root_dir).resolve() if root_dir is not None else None
        self.planner_timeout_seconds = max(int(planner_timeout_seconds or 120), 1)
        self.prompt_builder = PromptBuilder(
            planner_context_length=planner_context_length,
            minimum_context_length=minimum_context_length,
            recommended_context_length=recommended_context_length,
        )
        if self.root_dir is not None:
            self.logs_dir = self.root_dir / "logs"
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            self.planner_log_path = self.logs_dir / "agent_planner.log"
        else:
            self.logs_dir = None
            self.planner_log_path = None

    def parse_agent_response(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        stripped = re.sub(r"```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            payloads = self._extract_json_objects(stripped)
            if not payloads:
                raise ValueError(f"Agent response was not valid JSON: {exc}") from exc
            payload = payloads[-1]
        if not isinstance(payload, dict):
            raise ValueError("Agent response JSON must be an object.")
        response_type = payload.get("type")
        tool_names = {tool["name"] for tool in self.tool_registry.list_tools()}
        if "tool" in payload and response_type not in {"tool_call", "final", "question"}:
            payload["type"] = "tool_call"
            payload.setdefault("reason", "No reason provided.")
            payload.setdefault("args", {})
            return payload
        if isinstance(response_type, str) and response_type in tool_names:
            payload["type"] = "tool_call"
            payload["tool"] = response_type
            payload.setdefault("reason", "No reason provided.")
            payload.setdefault("args", {})
            return payload
        if response_type is None and "message" in payload:
            payload["type"] = "final"
            return payload
        if response_type not in {"tool_call", "final", "question"}:
            raise ValueError("Agent response type must be one of: tool_call, final, question.")
        return payload

    def _extract_json_objects(self, text: str) -> list[dict[str, Any]]:
        decoder = json.JSONDecoder()
        payloads: list[dict[str, Any]] = []
        index = 0
        length = len(text)
        while index < length:
            while index < length and text[index].isspace():
                index += 1
            if index >= length:
                break
            try:
                payload, end_index = decoder.raw_decode(text, index)
            except json.JSONDecodeError:
                next_object = text.find("{", index + 1)
                if next_object == -1:
                    break
                index = next_object
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
            index = end_index
        return payloads

    def planner_context_warning(self) -> str | None:
        return self.prompt_builder.planner_context_warning()

    def _build_prompt(self, user_message: str, current_task: dict[str, Any] | None, prompt_mode: str = "standard") -> PromptBuild:
        return self.prompt_builder.build(
            user_message=user_message,
            current_task=current_task,
            available_tools=self.tool_registry.list_tools(),
            rules_text=self._load_pilot_rules(),
            prompt_mode=prompt_mode,
        )

    def _build_system_prompt(
        self,
        working_memory: str = "",
        user_message: str = "",
        current_task: dict[str, Any] | None = None,
        prompt_mode: str = "standard",
    ) -> str:
        prompt_build = self._build_prompt(user_message or "Plan the next best step.", current_task, prompt_mode=prompt_mode)
        if working_memory:
            base_prompt = prompt_build.system_prompt.split("\nWorking memory:\n", 1)[0]
            return f"{base_prompt}\nWorking memory:\n{working_memory}"
        return prompt_build.system_prompt

    def _build_working_memory(self, current_task: dict[str, Any] | None, user_message: str = "") -> str:
        return self._build_prompt(user_message or "Continue the active task.", current_task).working_memory

    def _load_pilot_rules(self) -> str:
        if self.root_dir is None:
            return ""
        rules_path = self.root_dir / ".pilotrules"
        if not rules_path.exists():
            return ""
        return rules_path.read_text(encoding="utf-8").strip()

    def _prepare_user_task(self, user_task: str, current_task: dict[str, Any] | None) -> tuple[str, str | None]:
        if self.memory_store is None or current_task is None:
            return user_task, None
        followup_kind = self.memory_store.followup_kind(user_task)
        if followup_kind is None and self._should_continue_active_task(user_task, current_task):
            active_task_id = str(current_task.get("active_task_id") or "") or None
            original_task = str(current_task.get("original_user_task") or current_task.get("latest_user_message") or "")
            return (
                "Continue the active task using the saved context.\n"
                f"Original task: {original_task}\n"
                f"User follow-up: {user_task}\n"
                "Answer the follow-up from the existing task context unless a new tool is required.",
                active_task_id,
            )
        if followup_kind is None:
            return user_task, None
        active_task_id = str(current_task.get("active_task_id") or "") or None
        original_task = str(current_task.get("original_user_task") or current_task.get("latest_user_message") or "")
        last_approval = current_task.get("last_approval_request") or {}
        approval_summary = last_approval.get("summary") if isinstance(last_approval, dict) else ""
        last_tool = current_task.get("last_tool_call") or {}
        last_tool_name = last_tool.get("tool") if isinstance(last_tool, dict) else None
        desktop_suggestion_id = str(current_task.get("last_desktop_suggestion_id") or "").strip()
        desktop_suggestion_hint = (
            f"Saved desktop suggestion_id: {desktop_suggestion_id}\n"
            "If the user is approving a previously suggested desktop click, use desktop_execute_suggestion with that saved suggestion_id.\n"
            "If the saved suggestion is still valid and safe, your next reply should be a desktop_execute_suggestion tool call instead of a final answer.\n"
            if desktop_suggestion_id and not current_task.get("last_desktop_suggestion_executed")
            else ""
        )
        if followup_kind == "approve":
            return (
                "The user approved the pending request or continuation for the active task.\n"
                f"Original task: {original_task}\n"
                f"Pending approval: {approval_summary or 'No explicit approval summary recorded.'}\n"
                f"Last tool call: {last_tool_name or 'none'}\n"
                f"{desktop_suggestion_hint}"
                "Continue the task and avoid restarting from scratch.",
                active_task_id,
            )
        if followup_kind == "deny":
            return (
                "The user denied the pending request for the active task.\n"
                f"Original task: {original_task}\n"
                f"Pending approval: {approval_summary or 'No explicit approval summary recorded.'}\n"
                f"{desktop_suggestion_hint}"
                "Do not execute the denied action. Explain the outcome and propose a safer next step.",
                active_task_id,
            )
        if followup_kind == "status":
            return (
                "The user asked for a status update on the active task.\n"
                f"Original task: {original_task}\n"
                "Summarize what happened, what tools ran, and the current outcome.",
                active_task_id,
            )
        return (
            "Continue the active task using the saved context.\n"
            f"Original task: {original_task}\n"
            f"User follow-up: {user_task}\n"
            "Use the working memory to decide the next step.",
            active_task_id,
        )

    def _should_continue_active_task(self, user_task: str, current_task: dict[str, Any]) -> bool:
        active_task_id = str(current_task.get("active_task_id") or "").strip()
        if not active_task_id:
            return False
        lowered = user_task.strip().lower()
        if not lowered:
            return False
        followup_prefixes = (
            "what about",
            "what did",
            "and ",
            "also ",
            "continue",
            "try again",
            "now what",
            "what happened",
            "what loaded",
        )
        if any(lowered.startswith(prefix) for prefix in followup_prefixes):
            return True
        if len(lowered.split()) <= 6 and any(token in lowered for token in ("sidebar", "loaded", "that", "it", "there", "them")):
            return True
        return False

    def _current_task_status_for_question(self, question_text: str) -> str:
        lowered = question_text.lower()
        if any(token in lowered for token in ("approve", "approval", "allow", "confirm")):
            return "waiting_for_approval"
        return "active"

    def _append_recent_message(
        self,
        recent_messages: list[dict[str, str]],
        *,
        role: str,
        content: str,
    ) -> None:
        cleaned = content.strip()
        if not cleaned:
            return
        candidate = {"role": role, "content": cleaned}
        if recent_messages and recent_messages[-1] == candidate:
            return
        recent_messages.append(candidate)

    def _truncate_text(self, text: str, max_chars: int) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= max_chars:
            return cleaned
        if max_chars <= 1:
            return cleaned[:max_chars]
        return cleaned[: max_chars - 3].rstrip() + "..."

    def _summarize_tool_result_for_memory(
        self,
        tool_payload: dict[str, Any],
        tool_result: dict[str, Any],
    ) -> str:
        tool_name = str(tool_payload.get("tool", "tool"))
        if not tool_result.get("ok"):
            return self._truncate_text(f"{tool_name} failed: {tool_result.get('error', 'unknown error')}", 800)
        result_payload = tool_result.get("result") or {}
        if not isinstance(result_payload, dict):
            return f"{tool_name} succeeded."
        if tool_name == "analyze_screenshot":
            description = self._truncate_text(str(result_payload.get("description", "")).strip(), 760)
            return f"Screenshot analysis: {description}" if description else "Screenshot analysis completed."
        if tool_name == "desktop_suggest_action":
            action = str(result_payload.get("action", "unknown"))
            target = str(result_payload.get("target", "unknown"))
            confidence = result_payload.get("confidence")
            risk = str(result_payload.get("risk", "unknown"))
            suggestion_id = str(result_payload.get("suggestion_id", "")).strip()
            confidence_text = ""
            if isinstance(confidence, (int, float)):
                confidence_text = f" at {confidence:.0%} confidence"
            suggestion_text = f" | suggestion_id={suggestion_id}" if suggestion_id else ""
            return self._truncate_text(f"Suggested desktop action: {action} {target}{confidence_text}, risk={risk}{suggestion_text}.", 400)
        if tool_name == "desktop_execute_suggestion":
            action = str(result_payload.get("action", "click"))
            target = str(result_payload.get("target", "unknown"))
            suggestion_id = str(result_payload.get("suggestion_id", "")).strip()
            executed = bool(result_payload.get("executed"))
            status = "executed" if executed else "not executed"
            suggestion_text = f" | suggestion_id={suggestion_id}" if suggestion_id else ""
            return self._truncate_text(f"Desktop {action} {status}: {target}{suggestion_text}.", 400)
        if tool_name == "take_screenshot":
            path = str(result_payload.get("path", "")).strip()
            return self._truncate_text(f"Screenshot saved: {path}" if path else "Screenshot captured.", 400)
        if tool_name.startswith("browser_"):
            title = str(result_payload.get("title", "")).strip()
            url = str(result_payload.get("url", "")).strip()
            text_preview = self._truncate_text(str(result_payload.get("text_preview", "")).strip(), 260)
            parts = [part for part in [title, url, text_preview] if part]
            return self._truncate_text(f"{tool_name} -> {' | '.join(parts)}" if parts else f"{tool_name} succeeded.", 400)
        if tool_name == "set_timer":
            label = str(result_payload.get("label", "Timer")).strip()
            fires_at = str(result_payload.get("fires_at", "")).strip()
            return self._truncate_text(f"Timer set: {label} for {fires_at}.", 400)
        if tool_name == "write_file":
            path = str(result_payload.get("path", "")).strip()
            checkpoint_id = str(result_payload.get("checkpoint_id", "")).strip()
            if path and checkpoint_id:
                return self._truncate_text(f"File written: {path} with checkpoint {checkpoint_id}.", 400)
            if path:
                return self._truncate_text(f"File written: {path}.", 400)
        if result_payload.get("message"):
            return self._truncate_text(str(result_payload["message"]), 400)
        if result_payload.get("path"):
            return self._truncate_text(f"{tool_name} path: {result_payload['path']}", 400)
        return f"{tool_name} succeeded."

    def _update_current_task_after_step(
        self,
        task_id: str,
        original_user_task: str,
        latest_user_message: str,
        tool_payload: dict[str, Any] | None = None,
        tool_result: dict[str, Any] | None = None,
        status: str | None = None,
        final_answer: str | None = None,
        question: str | None = None,
        last_error: str | None = None,
        retry_suggestion: str | None = None,
    ) -> None:
        if self.memory_store is None:
            return
        loaded_task = self.memory_store.load_current_task() or {}
        same_task = str(loaded_task.get("active_task_id") or "") == task_id
        current_task = loaded_task if same_task else {}
        recent_step_summaries = list(current_task.get("recent_step_summaries") or [])
        recent_tool_calls = list(current_task.get("recent_tool_calls") or [])
        recent_tool_result_summaries = list(current_task.get("recent_tool_result_summaries") or [])
        recent_messages = list(current_task.get("recent_messages") or [])
        self._append_recent_message(recent_messages, role="user", content=latest_user_message)
        last_tool_result_summary = current_task.get("last_tool_result_summary")
        if tool_payload is not None:
            summary = f"{tool_payload.get('tool', 'tool')} requested"
            if tool_result is not None:
                outcome = "ok" if tool_result.get("ok") else f"error: {tool_result.get('error', '')}"
                summary = f"{tool_payload.get('tool', 'tool')} -> {outcome}"
                if tool_result.get("approval"):
                    summary += f" | approval: {tool_result['approval'].get('summary', '')}"
                last_tool_result_summary = self._summarize_tool_result_for_memory(tool_payload, tool_result)
                if last_tool_result_summary:
                    recent_tool_result_summaries.append(self._truncate_text(last_tool_result_summary, 400))
            recent_step_summaries.append(self._truncate_text(summary, 240))
            recent_tool_calls.append({"tool": tool_payload.get("tool"), "args": tool_payload.get("args", {})})
        if final_answer is not None:
            self._append_recent_message(recent_messages, role="assistant", content=final_answer)
        if question is not None:
            self._append_recent_message(recent_messages, role="assistant", content=question)
        suggestion_updates: dict[str, Any] = {}
        if tool_payload is not None and tool_result is not None:
            result_payload = tool_result.get("result") or {}
            if not isinstance(result_payload, dict):
                result_payload = {}
            if tool_payload.get("tool") == "desktop_suggest_action" and tool_result.get("ok"):
                suggestion_updates = {
                    "last_desktop_suggestion_id": str(result_payload.get("suggestion_id", "")).strip(),
                    "last_desktop_suggestion_action": str(result_payload.get("action", "")).strip(),
                    "last_desktop_suggestion_target": str(result_payload.get("target", "")).strip(),
                    "last_desktop_suggestion_x": result_payload.get("x"),
                    "last_desktop_suggestion_y": result_payload.get("y"),
                    "last_desktop_suggestion_confidence": result_payload.get("confidence"),
                    "last_desktop_suggestion_expires_at": str(result_payload.get("expires_at", "")).strip(),
                    "last_desktop_suggestion_executed": False,
                    "last_desktop_suggestion_warning": str(result_payload.get("warning", "")).strip(),
                }
            elif tool_payload.get("tool") == "desktop_execute_suggestion":
                suggestion_id = str(result_payload.get("suggestion_id", "")).strip() or str(tool_payload.get("args", {}).get("suggestion_id", "")).strip()
                if suggestion_id:
                    suggestion_updates = {
                        "last_desktop_suggestion_id": suggestion_id,
                        "last_desktop_suggestion_executed": bool(tool_result.get("ok") and result_payload.get("executed")),
                    }
        updates: dict[str, Any] = {
            "active_task_id": task_id,
            "original_user_task": self._truncate_text(original_user_task, 800),
            "latest_user_message": self._truncate_text(latest_user_message, 300),
            "mode": "agent",
            "status": status or current_task.get("status", "active"),
            "last_tool_call": tool_payload or current_task.get("last_tool_call"),
            "last_approval_request": (tool_result or {}).get("approval") if tool_result else current_task.get("last_approval_request"),
            "last_final_answer": self._truncate_text(final_answer, 600) if final_answer is not None else current_task.get("last_final_answer"),
            "recent_step_summaries": recent_step_summaries[-6:],
            "recent_tool_calls": recent_tool_calls[-5:],
            "last_tool_result_summary": self._truncate_text(str(last_tool_result_summary or ""), 800) if last_tool_result_summary else "",
            "recent_tool_result_summaries": recent_tool_result_summaries[-5:],
            "recent_messages": [
                {
                    "role": str(message.get("role", "message")),
                    "content": self._truncate_text(str(message.get("content", "")), 300),
                }
                for message in recent_messages[-8:]
                if isinstance(message, dict)
            ],
            "last_error": self._truncate_text(last_error, 600) if last_error is not None else current_task.get("last_error", ""),
            "retry_suggestion": self._truncate_text(retry_suggestion, 200) if retry_suggestion is not None else current_task.get("retry_suggestion", ""),
        }
        updates.update({key: value for key, value in suggestion_updates.items() if value is not None})
        if question is not None:
            updates["last_question"] = self._truncate_text(question, 300)
        self.memory_store.update_current_task(**updates)

    def _planner_response_preview(self, response_text: str) -> str:
        return self._truncate_text(response_text, 240)

    def _log_planner_call(
        self,
        *,
        task_id: str,
        prompt_build: PromptBuild,
        start_time: float,
        success: bool,
        response_text: str = "",
        error_text: str = "",
    ) -> None:
        if self.planner_log_path is None:
            return
        duration = max(time.perf_counter() - start_time, 0.0)
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "task_id": task_id,
            "model": self.planner_model,
            "tool_count": prompt_build.tool_count,
            "tool_names": prompt_build.tool_names,
            "prompt_char_count": len(prompt_build.system_prompt),
            "working_memory_char_count": len(prompt_build.working_memory),
            "timeout_seconds": self.planner_timeout_seconds,
            "duration_seconds": round(duration, 3),
            "success": success,
            "error": error_text or None,
            "response_preview": self._planner_response_preview(response_text) if response_text else "",
            "task_category": prompt_build.task_category,
            "prompt_mode": prompt_build.prompt_mode,
        }
        with self.planner_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    def _is_context_overflow_error(self, error_text: str) -> bool:
        lowered = error_text.lower()
        return any(
            token in lowered
            for token in (
                "n_keep",
                "n_ctx",
                "context length",
                "context.",
                "prompt too long",
                "exceeded the loaded lm studio context",
            )
        )

    def _context_overflow_message(self) -> str:
        return (
            "The agent memory is available, but the planner prompt exceeded the loaded LM Studio context. "
            f"Increase the planner model context to {CONTEXT_MINIMUM_USABLE} or {CONTEXT_RECOMMENDED}, or retry with compact mode."
        )

    def _call_planner(self, task_id: str, prompt_build: PromptBuild, user_message: str) -> str:
        messages = [
            {"role": "system", "content": prompt_build.system_prompt},
            {"role": "user", "content": user_message},
        ]
        started = time.perf_counter()
        try:
            response_text = self.llm_client.chat_text(
                messages=messages,
                model=self.planner_model,
                max_tokens=1024,
            )
        except Exception as exc:
            self._log_planner_call(
                task_id=task_id,
                prompt_build=prompt_build,
                start_time=started,
                success=False,
                error_text=str(exc),
            )
            raise
        self._log_planner_call(
            task_id=task_id,
            prompt_build=prompt_build,
            start_time=started,
            success=True,
            response_text=response_text,
        )
        return response_text

    def _planner_instruction_for_tool_result(self, tool_payload: dict[str, Any], tool_result: dict[str, Any]) -> str:
        summary = self._summarize_tool_result_for_memory(tool_payload, tool_result)
        compact_result = json.dumps(self.prompt_builder.compact_json(tool_result), ensure_ascii=True)
        return (
            "Continue the active task.\n"
            f"Last tool: {tool_payload.get('tool', 'tool')}\n"
            f"Reason: {self._truncate_text(str(tool_payload.get('reason', '')), 200)}\n"
            f"Tool result summary: {self._truncate_text(summary, 500)}\n"
            f"Structured result: {self._truncate_text(compact_result, 700)}\n"
            "Decide the next single best step. Reply with JSON only."
        )

    def _planner_call_with_recovery(
        self,
        *,
        task_id: str,
        user_message: str,
        current_task: dict[str, Any] | None,
    ) -> str:
        prompt_build = self._build_prompt(user_message, current_task, prompt_mode="standard")
        try:
            return self._call_planner(task_id, prompt_build, user_message)
        except Exception as exc:
            if not self._is_context_overflow_error(str(exc)):
                raise
            ultra_message = self._truncate_text(user_message, 600)
            ultra_prompt = self._build_prompt(ultra_message, current_task, prompt_mode="ultra_compact")
            return self._call_planner(task_id, ultra_prompt, ultra_message)

    def run_task(self, user_task: str, continue_task_id: str | None = None) -> dict[str, Any]:
        current_task = self.memory_store.load_current_task() if self.memory_store is not None else None
        effective_user_task, suggested_task_id = self._prepare_user_task(user_task, current_task)
        task_id = continue_task_id or suggested_task_id or uuid.uuid4().hex[:12]
        started_at = datetime.now().isoformat(timespec="seconds")
        original_user_task = (
            str(current_task.get("original_user_task"))
            if current_task is not None and str(current_task.get("active_task_id") or "") == task_id
            else user_task
        )
        self._update_current_task_after_step(
            task_id=task_id,
            original_user_task=original_user_task,
            latest_user_message=user_task,
            status="active",
            last_error="",
            retry_suggestion="",
        )
        transcript: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        planner_instruction = effective_user_task

        for step_number in range(1, self.max_steps + 1):
            current_task = self.memory_store.load_current_task() if self.memory_store is not None else current_task
            try:
                response_text = self._planner_call_with_recovery(
                    task_id=task_id,
                    user_message=planner_instruction,
                    current_task=current_task,
                )
            except Exception as exc:
                retryable = self._is_context_overflow_error(str(exc))
                error_message = self._context_overflow_message() if retryable else str(exc)
                status = "failed_retryable" if retryable else "failed"
                suggestion = "Increase the planner context to 8192 or 16384, or retry with compact mode." if retryable else ""
                self._update_current_task_after_step(
                    task_id=task_id,
                    original_user_task=original_user_task,
                    latest_user_message=user_task,
                    status=status,
                    last_error=error_message,
                    retry_suggestion=suggestion,
                )
                result = {
                    "ok": False,
                    "status": status,
                    "transcript": transcript,
                    "steps": steps,
                    "task_id": task_id,
                    "error": error_message,
                }
                self._persist_session(result, task_id, user_task, started_at)
                return result

            parsed = self.parse_agent_response(response_text)
            step_record: dict[str, Any] = {
                "step": step_number,
                "ai_message": response_text,
                "parsed_action_type": parsed["type"],
            }

            if parsed["type"] == "tool_call":
                tool_payload = {
                    "tool": parsed.get("tool", ""),
                    "args": parsed.get("args", {}),
                    "reason": parsed.get("reason", ""),
                    "task_id": task_id,
                    "tool_call_id": f"{task_id}_step_{step_number}",
                }
                if isinstance(parsed.get("approval_plan"), dict):
                    tool_payload["approval_plan"] = parsed["approval_plan"]
                transcript.append({"type": "tool_call", "payload": tool_payload})
                tool_result = self.tool_registry.execute_tool_call(tool_payload)
                transcript.append({"type": "tool_result", "payload": tool_result})
                step_record["tool_call"] = tool_payload
                step_record["tool_result"] = tool_result
                if tool_result.get("approval"):
                    step_record["approval"] = tool_result["approval"]
                steps.append(step_record)
                self._update_current_task_after_step(
                    task_id=task_id,
                    original_user_task=original_user_task,
                    latest_user_message=user_task,
                    tool_payload=tool_payload,
                    tool_result=tool_result,
                    status="active",
                    last_error="",
                    retry_suggestion="",
                )
                planner_instruction = self._planner_instruction_for_tool_result(tool_payload, tool_result)
                continue

            if parsed["type"] == "question":
                transcript.append({"type": "question", "payload": parsed})
                step_record["question"] = parsed["message"]
                steps.append(step_record)
                question_status = self._current_task_status_for_question(parsed["message"])
                self._update_current_task_after_step(
                    task_id=task_id,
                    original_user_task=original_user_task,
                    latest_user_message=user_task,
                    status=question_status,
                    question=parsed["message"],
                    last_error="",
                    retry_suggestion="",
                )
                result = {
                    "ok": True,
                    "status": "question",
                    "transcript": transcript,
                    "steps": steps,
                    "message": parsed["message"],
                    "task_id": task_id,
                }
                self._persist_session(result, task_id, user_task, started_at)
                return result

            transcript.append({"type": "final", "payload": parsed})
            step_record["final_answer"] = parsed["message"]
            steps.append(step_record)
            self._update_current_task_after_step(
                task_id=task_id,
                original_user_task=original_user_task,
                latest_user_message=user_task,
                status="completed",
                final_answer=parsed["message"],
                last_error="",
                retry_suggestion="",
            )
            result = {
                "ok": True,
                "status": "final",
                "transcript": transcript,
                "steps": steps,
                "message": parsed["message"],
                "task_id": task_id,
            }
            self._persist_session(result, task_id, user_task, started_at)
            return result

        result = {
            "ok": False,
            "status": "failed",
            "transcript": transcript,
            "steps": steps,
            "task_id": task_id,
            "error": f"Agent exceeded max_steps={self.max_steps} without reaching a final answer.",
        }
        self._update_current_task_after_step(
            task_id=task_id,
            original_user_task=original_user_task,
            latest_user_message=user_task,
            status="failed",
            final_answer=result["error"],
            last_error=result["error"],
            retry_suggestion="Retry the task with a narrower scope.",
        )
        self._persist_session(result, task_id, user_task, started_at)
        return result

    def _persist_session(self, result: dict[str, Any], task_id: str, user_task: str, started_at: str) -> None:
        if self.memory_store is None:
            return
        steps = result.get("steps", []) or []
        tool_calls = [step.get("tool_call") for step in steps if step.get("tool_call")]
        approvals = [step.get("approval") for step in steps if step.get("approval")]
        files_changed = [
            step["tool_result"]["result"].get("path")
            for step in steps
            if step.get("tool_call", {}).get("tool") == "write_file" and step.get("tool_result", {}).get("ok")
        ]
        browser_actions = [
            {"tool": step["tool_call"]["tool"], "args": step["tool_call"].get("args", {})}
            for step in steps
            if str(step.get("tool_call", {}).get("tool", "")).startswith("browser_")
        ]
        tool_names = [str(tool_call.get("tool")) for tool_call in tool_calls if isinstance(tool_call, dict)]
        unresolved_next_steps = ""
        if result.get("status") == "question":
            unresolved_next_steps = result.get("message", "")
        elif result.get("status") in {"error", "failed", "failed_retryable"}:
            unresolved_next_steps = result.get("error", "")
        record = {
            "task_id": task_id,
            "user_task": user_task,
            "mode": "agent",
            "start_time": started_at,
            "end_time": datetime.now().isoformat(timespec="seconds"),
            "model_used": self.planner_model,
            "tool_calls": tool_calls,
            "approvals": approvals,
            "final_answer": result.get("message", ""),
            "errors": [result["error"]] if result.get("error") else [],
            "files_changed": [path for path in files_changed if path],
            "browser_actions": browser_actions,
            "steps": steps,
            "status": result.get("status", "error"),
            "summary": (
                f"User asked: {user_task}\n"
                f"Tools: {', '.join(tool_names) if tool_names else 'none'}\n"
                f"Outcome: {result.get('status', 'unknown')} | {result.get('message') or result.get('error', '')}\n"
                f"Files changed: {len([path for path in files_changed if path])}\n"
                f"Browser actions: {len(browser_actions)}\n"
                f"Unresolved next steps: {unresolved_next_steps or 'none'}"
            ),
            "unresolved_next_steps": unresolved_next_steps,
        }
        result["session_path"] = self.memory_store.save_session(record)
