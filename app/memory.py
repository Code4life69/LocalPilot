from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


FOLLOWUP_APPROVE = {"yes", "y", "approve", "approved", "go ahead", "do it"}
FOLLOWUP_DENY = {"no", "n", "deny", "denied", "cancel", "stop"}
FOLLOWUP_CONTINUE = {"continue", "try again"}
FOLLOWUP_STATUS = {"what happened", "status"}


class MemoryStore:
    def __init__(self, memory_dir: str | Path, capability_manifest_path: str | Path) -> None:
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.notes_path = self.memory_dir / "notes.md"
        self.facts_path = self.memory_dir / "learned_facts.json"
        self.current_task_path = self.memory_dir / "current_task.json"
        self.sessions_dir = self.memory_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.capability_manifest_path = Path(capability_manifest_path)
        if not self.notes_path.exists():
            self.notes_path.write_text("# LocalPilot Notes\n\n", encoding="utf-8")
        if not self.facts_path.exists():
            self.facts_path.write_text("{}\n", encoding="utf-8")
        if not self.current_task_path.exists():
            self.current_task_path.write_text("{}\n", encoding="utf-8")

    def load_capabilities(self) -> dict[str, Any]:
        with self.capability_manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_note(self, text: str) -> str:
        note = text.strip()
        if not note:
            return "No note text provided."
        with self.notes_path.open("a", encoding="utf-8") as handle:
            handle.write(f"- {note}\n")
        return "Note saved."

    def search_notes(self, keyword: str) -> list[str]:
        needle = keyword.strip().lower()
        if not needle:
            return []
        lines = self.notes_path.read_text(encoding="utf-8").splitlines()
        matches: list[str] = []
        seen: set[str] = set()
        for line in lines:
            normalized = line.strip()
            if not normalized or normalized.startswith("#"):
                continue
            if needle not in normalized.lower():
                continue
            cleaned = normalized.lstrip("-* ").strip()
            lowered_cleaned = cleaned.lower()
            if cleaned and lowered_cleaned not in seen:
                seen.add(lowered_cleaned)
                matches.append(cleaned)
        return matches

    def show_notes(self) -> str:
        return self.notes_path.read_text(encoding="utf-8")

    def load_facts(self) -> dict[str, Any]:
        with self.facts_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_fact(self, key: str, value: Any) -> str:
        facts = self.load_facts()
        facts[key] = value
        with self.facts_path.open("w", encoding="utf-8") as handle:
            json.dump(facts, handle, indent=2)
            handle.write("\n")
        return f"Saved fact: {key}"

    def save_session(self, record: dict[str, Any]) -> str:
        task_id = str(record.get("task_id") or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        timestamp = str(record.get("start_time") or datetime.now().isoformat(timespec="seconds")).replace(":", "-")
        session_path = self.sessions_dir / f"{timestamp}_{task_id}.json"
        with session_path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        return str(session_path)

    def recent_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for session_path in sorted(self.sessions_dir.glob("*.json"), reverse=True)[: max(limit, 1)]:
            try:
                data = json.loads(session_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                sessions.append(data)
        return sessions

    def list_session_summaries(self, limit: int = 10) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for session_path in sorted(self.sessions_dir.glob("*.json"), reverse=True)[: max(limit, 1)]:
            try:
                data = json.loads(session_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            summaries.append(
                {
                    "session_id": session_path.stem,
                    "task_id": data.get("task_id"),
                    "start_time": data.get("start_time"),
                    "end_time": data.get("end_time"),
                    "status": data.get("status"),
                    "user_task": data.get("user_task"),
                    "final_answer": data.get("final_answer"),
                    "files_changed": data.get("files_changed", []),
                    "browser_actions": data.get("browser_actions", []),
                    "errors": data.get("errors", []),
                    "summary": data.get("summary", ""),
                    "session_path": str(session_path),
                }
            )
        return summaries

    def read_session(self, session_id: str) -> dict[str, Any] | None:
        needle = session_id.strip()
        if not needle:
            return None
        direct_path = self.sessions_dir / f"{needle}.json"
        if direct_path.exists():
            try:
                data = json.loads(direct_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
            if isinstance(data, dict):
                data.setdefault("session_id", direct_path.stem)
                data.setdefault("session_path", str(direct_path))
                return data
        for session_path in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(session_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            if data.get("task_id") == needle or session_path.stem == needle:
                data.setdefault("session_id", session_path.stem)
                data.setdefault("session_path", str(session_path))
                return data
        return None

    def load_current_task(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.current_task_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict) or not data:
            return None
        if not data.get("active_task_id"):
            return None
        return data

    def save_current_task(self, record: dict[str, Any]) -> dict[str, Any]:
        payload = dict(record)
        payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.current_task_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return payload

    def update_current_task(self, **updates: Any) -> dict[str, Any]:
        current = self.load_current_task() or {}
        payload = dict(current)
        payload.update({key: value for key, value in updates.items() if value is not None})
        recent_steps = payload.get("recent_step_summaries") or []
        if isinstance(recent_steps, list):
            payload["recent_step_summaries"] = recent_steps[-6:]
        recent_tool_calls = payload.get("recent_tool_calls") or []
        if isinstance(recent_tool_calls, list):
            payload["recent_tool_calls"] = recent_tool_calls[-5:]
        recent_tool_results = payload.get("recent_tool_result_summaries") or []
        if isinstance(recent_tool_results, list):
            payload["recent_tool_result_summaries"] = recent_tool_results[-5:]
        recent_messages = payload.get("recent_messages") or []
        if isinstance(recent_messages, list):
            payload["recent_messages"] = recent_messages[-8:]
        return self.save_current_task(payload)

    def clear_current_task(self) -> str:
        self.current_task_path.write_text("{}\n", encoding="utf-8")
        return "Current task cleared."

    def summarize_recent_sessions(self, limit: int = 3) -> str:
        sessions = self.list_session_summaries(limit=limit)
        if not sessions:
            return "No recent sessions."
        lines: list[str] = []
        for session in sessions:
            summary = str(session.get("summary", "")).strip()
            if summary:
                lines.append(f"- {summary}")
            else:
                lines.append(
                    f"- {session.get('user_task', '')} | status={session.get('status', 'unknown')} | final={session.get('final_answer', '') or '(none)'}"
                )
        return "\n".join(lines)

    def followup_kind(self, text: str) -> str | None:
        lowered = text.strip().lower()
        if lowered in FOLLOWUP_APPROVE or lowered.startswith(("approve", "approved", "go ahead", "do it", "yes")):
            return "approve"
        if lowered in FOLLOWUP_DENY or lowered.startswith(("deny", "cancel", "stop", "no")):
            return "deny"
        if lowered in FOLLOWUP_CONTINUE or lowered.startswith("i meant "):
            return "continue"
        if lowered in FOLLOWUP_STATUS:
            return "status"
        return None
