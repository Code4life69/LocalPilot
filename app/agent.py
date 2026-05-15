from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.lmstudio_client import LMStudioClient
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
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.planner_model = planner_model
        self.max_steps = max_steps

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
        return (
            "You are LocalPilot, an AI agent. Python is only your tool harness.\n"
            "You must decide the plan, choose tools, inspect tool results, and decide when the task is done.\n"
            "Never pretend a tool succeeded. Never claim to have seen output you were not given.\n"
            "For websites, prefer Puppeteer-controlled browser tools before desktop mouse tools.\n"
            "Use browser DOM/text tools for website interaction and screenshot vision tools for visual understanding.\n"
            "Reply with JSON only.\n"
            "Allowed response formats:\n"
            '{"type":"tool_call","tool":"tool_name","args":{},"reason":"why this tool is needed"}\n'
            '{"type":"final","message":"final answer for the user"}\n'
            '{"type":"question","message":"one clarification question"}\n'
            "Available tools:\n"
            f"{tools_json}"
        )

    def run_task(self, user_task: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_task},
        ]
        transcript: list[dict[str, Any]] = []

        for _step in range(self.max_steps):
            response_text = self.llm_client.chat_text(
                messages=messages,
                model=self.planner_model,
                max_tokens=1024,
            )
            parsed = self.parse_agent_response(response_text)

            if parsed["type"] == "tool_call":
                tool_payload = {
                    "tool": parsed.get("tool", ""),
                    "args": parsed.get("args", {}),
                    "reason": parsed.get("reason", ""),
                }
                transcript.append({"type": "tool_call", "payload": tool_payload})
                tool_result = self.tool_registry.execute_tool_call(tool_payload)
                transcript.append({"type": "tool_result", "payload": tool_result})
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
                return {"ok": True, "status": "question", "transcript": transcript, "message": parsed["message"]}

            transcript.append({"type": "final", "payload": parsed})
            return {"ok": True, "status": "final", "transcript": transcript, "message": parsed["message"]}

        return {
            "ok": False,
            "status": "error",
            "transcript": transcript,
            "error": f"Agent exceeded max_steps={self.max_steps} without reaching a final answer.",
        }
