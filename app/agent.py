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

    def _build_system_prompt(self) -> str:
        tools_json = json.dumps(self.tool_registry.list_tools(), indent=2)
        prompt = (
            "You are LocalPilot, an AI agent. Python is only your tool harness.\n"
            "The user does not want to chat with Python. The user chats with the AI agent.\n"
            "You must decide the plan, choose tools, inspect tool results, and decide when the task is done.\n"
            "Never pretend a tool succeeded. Never claim to have seen output you were not given.\n"
            "For websites, prefer Puppeteer-controlled browser tools before desktop mouse tools.\n"
            "Use browser DOM/text tools for website interaction and screenshot vision tools for visual understanding.\n"
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
        return prompt

    def _load_pilot_rules(self) -> str:
        if self.root_dir is None:
            return ""
        rules_path = self.root_dir / ".pilotrules"
        if not rules_path.exists():
            return ""
        return rules_path.read_text(encoding="utf-8").strip()

    def run_task(self, user_task: str) -> dict[str, Any]:
        task_id = uuid.uuid4().hex[:12]
        started_at = datetime.now().isoformat(timespec="seconds")
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_task},
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
                result = {"ok": True, "status": "question", "transcript": transcript, "steps": steps, "message": parsed["message"], "task_id": task_id}
                self._persist_session(result, task_id, user_task, started_at)
                return result

            transcript.append({"type": "final", "payload": parsed})
            step_record["final_answer"] = parsed["message"]
            steps.append(step_record)
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
        }
        result["session_path"] = self.memory_store.save_session(record)
