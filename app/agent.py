from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.lmstudio_client import LMStudioClient
from app.memory import MemoryStore
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
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.planner_model = planner_model
        self.max_steps = max_steps
        self.memory_store = memory_store
        self.root_dir = Path(root_dir).resolve() if root_dir is not None else None

    def parse_agent_response(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Agent response was not valid JSON: {exc}") from exc
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

    def _build_system_prompt(self, working_memory: str = "") -> str:
        tools_json = json.dumps(self.tool_registry.list_tools(), indent=2)
        prompt = (
            "You are LocalPilot, an AI agent. Python is only your tool harness.\n"
            "The user does not want to chat with Python. The user chats with the AI agent.\n"
            "You must decide the plan, choose tools, inspect tool results, and decide when the task is done.\n"
            "Never pretend a tool succeeded. Never claim to have seen output you were not given.\n"
            "For websites, prefer Puppeteer-controlled browser tools before desktop mouse tools.\n"
            "Use browser DOM/text tools for website interaction and screenshot vision tools for visual understanding.\n"
            "Use desktop_suggest_action only for visible Windows desktop or non-browser UI tasks.\n"
            "For desktop tasks in this milestone, observe first, suggest second, and never execute desktop click, type, key press, or hotkey actions.\n"
            "If you need a visual desktop target, capture a screenshot first and then ask for a dry-run suggestion.\n"
            "desktop_move_mouse_preview is preview-only and still requires approval.\n"
            "For timers or reminders on this PC, use set_timer, list_timers, or cancel_timer.\n"
            "Never use run_command with sleep or timeout as a fake timer.\n"
            "When a browser task obviously needs multiple medium-risk steps, you may include an `approval_plan` object on the first tool call.\n"
            "Reply with JSON only.\n"
            "Allowed response formats:\n"
            '{"type":"tool_call","tool":"tool_name","args":{},"reason":"why this tool is needed"}\n'
            '{"type":"final","message":"final answer for the user"}\n'
            '{"type":"question","message":"one clarification question"}\n'
            "Available tools:\n"
            f"{tools_json}"
        )
        rules_text = self._load_pilot_rules()
        if rules_text:
            prompt += f"\n\nPilot rules:\n{rules_text}"
        if working_memory:
            prompt += f"\n\nWorking memory:\n{working_memory}"
        return prompt

    def _load_pilot_rules(self) -> str:
        if self.root_dir is None:
            return ""
        rules_path = self.root_dir / ".pilotrules"
        if not rules_path.exists():
            return ""
        return rules_path.read_text(encoding="utf-8").strip()

    def _build_working_memory(self, current_task: dict[str, Any] | None) -> str:
        lines: list[str] = []
        if current_task:
            lines.extend(
                [
                    f"Current active task id: {current_task.get('active_task_id', '')}",
                    f"Original task: {current_task.get('original_user_task', '')}",
                    f"Latest user message: {current_task.get('latest_user_message', '')}",
                    f"Status: {current_task.get('status', '')}",
                ]
            )
            if current_task.get("status") in {"completed", "final"}:
                lines.append("The last active task finished recently. If the user is clarifying or continuing it, keep this context.")
            last_tool_call = current_task.get("last_tool_call") or {}
            if isinstance(last_tool_call, dict) and last_tool_call.get("tool"):
                lines.append(f"Last tool call: {last_tool_call.get('tool')} {json.dumps(last_tool_call.get('args', {}), ensure_ascii=True)}")
            if current_task.get("last_tool_result_summary"):
                lines.append(f"Last tool result: {current_task.get('last_tool_result_summary')}")
            for recent_tool_call in (current_task.get("recent_tool_calls") or [])[-3:]:
                if isinstance(recent_tool_call, dict) and recent_tool_call.get("tool"):
                    lines.append(
                        f"Recent tool call: {recent_tool_call.get('tool')} {json.dumps(recent_tool_call.get('args', {}), ensure_ascii=True)}"
                    )
            for recent_result_summary in (current_task.get("recent_tool_result_summaries") or [])[-3:]:
                lines.append(f"Recent tool result: {recent_result_summary}")
            last_approval = current_task.get("last_approval_request") or {}
            if isinstance(last_approval, dict) and last_approval.get("summary"):
                lines.append(f"Last approval request: {last_approval.get('summary')}")
            if current_task.get("last_final_answer"):
                lines.append(f"Last final answer: {current_task.get('last_final_answer')}")
            for summary in (current_task.get("recent_step_summaries") or [])[-4:]:
                lines.append(f"Recent step: {summary}")
            recent_messages = current_task.get("recent_messages") or []
            if recent_messages:
                lines.append("Recent conversation turns:")
                for message in recent_messages[-4:]:
                    if not isinstance(message, dict):
                        continue
                    role = str(message.get("role", "message"))
                    content = str(message.get("content", "")).strip()
                    if content:
                        lines.append(f"- {role}: {content}")
        if self.memory_store is not None:
            recent_sessions = self.memory_store.summarize_recent_sessions(limit=2)
            if recent_sessions and recent_sessions != "No recent sessions.":
                lines.append("Recent session summaries:")
                lines.extend(recent_sessions.splitlines())
        return "\n".join(line for line in lines if line)

    def _prepare_user_task(self, user_task: str, current_task: dict[str, Any] | None) -> tuple[str, str | None]:
        if self.memory_store is None or current_task is None:
            return user_task, None
        followup_kind = self.memory_store.followup_kind(user_task)
        if followup_kind is None:
            return user_task, None
        active_task_id = str(current_task.get("active_task_id") or "") or None
        original_task = str(current_task.get("original_user_task") or current_task.get("latest_user_message") or "")
        last_approval = current_task.get("last_approval_request") or {}
        approval_summary = last_approval.get("summary") if isinstance(last_approval, dict) else ""
        last_tool = current_task.get("last_tool_call") or {}
        last_tool_name = last_tool.get("tool") if isinstance(last_tool, dict) else None
        if followup_kind == "approve":
            return (
                "The user approved the pending request or continuation for the active task.\n"
                f"Original task: {original_task}\n"
                f"Pending approval: {approval_summary or 'No explicit approval summary recorded.'}\n"
                f"Last tool call: {last_tool_name or 'none'}\n"
                "Continue the task and avoid restarting from scratch.",
                active_task_id,
            )
        if followup_kind == "deny":
            return (
                "The user denied the pending request for the active task.\n"
                f"Original task: {original_task}\n"
                f"Pending approval: {approval_summary or 'No explicit approval summary recorded.'}\n"
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

    def _summarize_tool_result_for_memory(
        self,
        tool_payload: dict[str, Any],
        tool_result: dict[str, Any],
    ) -> str:
        tool_name = str(tool_payload.get("tool", "tool"))
        if not tool_result.get("ok"):
            return f"{tool_name} failed: {tool_result.get('error', 'unknown error')}"
        result_payload = tool_result.get("result") or {}
        if not isinstance(result_payload, dict):
            return f"{tool_name} succeeded."
        if tool_name == "analyze_screenshot":
            description = str(result_payload.get("description", "")).strip()
            return f"Screenshot analysis: {description}" if description else "Screenshot analysis completed."
        if tool_name == "desktop_suggest_action":
            action = str(result_payload.get("action", "unknown"))
            target = str(result_payload.get("target", "unknown"))
            confidence = result_payload.get("confidence")
            risk = str(result_payload.get("risk", "unknown"))
            confidence_text = ""
            if isinstance(confidence, (int, float)):
                confidence_text = f" at {confidence:.0%} confidence"
            return f"Suggested desktop action: {action} {target}{confidence_text}, risk={risk}."
        if tool_name == "take_screenshot":
            path = str(result_payload.get("path", "")).strip()
            return f"Screenshot saved: {path}" if path else "Screenshot captured."
        if tool_name.startswith("browser_"):
            title = str(result_payload.get("title", "")).strip()
            url = str(result_payload.get("url", "")).strip()
            text_preview = str(result_payload.get("text_preview", "")).strip()
            parts = [part for part in [title, url, text_preview] if part]
            return f"{tool_name} -> {' | '.join(parts)}" if parts else f"{tool_name} succeeded."
        if tool_name == "set_timer":
            label = str(result_payload.get("label", "Timer")).strip()
            fires_at = str(result_payload.get("fires_at", "")).strip()
            return f"Timer set: {label} for {fires_at}."
        if tool_name == "write_file":
            path = str(result_payload.get("path", "")).strip()
            checkpoint_id = str(result_payload.get("checkpoint_id", "")).strip()
            if path and checkpoint_id:
                return f"File written: {path} with checkpoint {checkpoint_id}."
            if path:
                return f"File written: {path}."
        if result_payload.get("message"):
            return str(result_payload["message"])
        if result_payload.get("path"):
            return f"{tool_name} path: {result_payload['path']}"
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
    ) -> None:
        if self.memory_store is None:
            return
        current_task = self.memory_store.load_current_task() or {}
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
                recent_tool_result_summaries.append(last_tool_result_summary)
            recent_step_summaries.append(summary)
            recent_tool_calls.append({"tool": tool_payload.get("tool"), "args": tool_payload.get("args", {})})
        if final_answer is not None:
            self._append_recent_message(recent_messages, role="assistant", content=final_answer)
        if question is not None:
            self._append_recent_message(recent_messages, role="assistant", content=question)
        updates: dict[str, Any] = {
            "active_task_id": task_id,
            "original_user_task": original_user_task,
            "latest_user_message": latest_user_message,
            "mode": "agent",
            "status": status or current_task.get("status", "active"),
            "last_tool_call": tool_payload or current_task.get("last_tool_call"),
            "last_approval_request": (tool_result or {}).get("approval") if tool_result else current_task.get("last_approval_request"),
            "last_final_answer": final_answer if final_answer is not None else current_task.get("last_final_answer"),
            "recent_step_summaries": recent_step_summaries[-6:],
            "recent_tool_calls": recent_tool_calls[-5:],
            "last_tool_result_summary": last_tool_result_summary,
            "recent_tool_result_summaries": recent_tool_result_summaries[-5:],
            "recent_messages": recent_messages[-8:],
        }
        if question is not None:
            updates["last_question"] = question
        self.memory_store.update_current_task(**updates)

    def run_task(self, user_task: str, continue_task_id: str | None = None) -> dict[str, Any]:
        current_task = self.memory_store.load_current_task() if self.memory_store is not None else None
        effective_user_task, suggested_task_id = self._prepare_user_task(user_task, current_task)
        task_id = continue_task_id or suggested_task_id or uuid.uuid4().hex[:12]
        started_at = datetime.now().isoformat(timespec="seconds")
        working_memory = self._build_working_memory(current_task)
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
        )
        messages = [
            {"role": "system", "content": self._build_system_prompt(working_memory)},
            {"role": "user", "content": effective_user_task},
        ]
        transcript: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []

        for step_number in range(1, self.max_steps + 1):
            response_text = self.llm_client.chat_text(
                messages=messages,
                model=self.planner_model,
                max_tokens=1024,
            )
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
                )
                messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=True)})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Tool result:\n"
                            f"{json.dumps(tool_result, ensure_ascii=True)}\n"
                            "Decide the next best step. Reply with JSON only."
                        ),
                    }
                )
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
                )
                result = {"ok": True, "status": "question", "transcript": transcript, "steps": steps, "message": parsed["message"], "task_id": task_id}
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
            )
            result = {"ok": True, "status": "final", "transcript": transcript, "steps": steps, "message": parsed["message"], "task_id": task_id}
            self._persist_session(result, task_id, user_task, started_at)
            return result

        result = {
            "ok": False,
            "status": "error",
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
        elif result.get("status") == "error":
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
