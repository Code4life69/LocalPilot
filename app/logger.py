from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


EventCallback = Callable[[dict[str, Any]], None]


class AppLogger:
    def __init__(self, logs_dir: str | Path) -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._text_path = self.logs_dir / "session.log"
        self._jsonl_path = self.logs_dir / "events.jsonl"
        self._callbacks: list[EventCallback] = []
        self._lock = threading.Lock()

    def register_callback(self, callback: EventCallback) -> None:
        self._callbacks.append(callback)

    def event(self, role: str, message: str, persist: bool = True, **extra: Any) -> dict[str, Any]:
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "role": role,
            "message": message,
            "extra": extra,
        }
        line = f"[{entry['timestamp']}] {role}: {message}"
        if extra:
            line += f" | {json.dumps(extra, ensure_ascii=True)}"

        if persist:
            with self._lock:
                with self._text_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                with self._jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

        for callback in list(self._callbacks):
            try:
                callback(entry)
            except Exception:
                continue
        return entry

    def tail_events(self, limit: int = 80) -> list[dict[str, Any]]:
        if not self._jsonl_path.exists():
            return []
        lines = self._jsonl_path.read_text(encoding="utf-8").splitlines()
        events: list[dict[str, Any]] = []
        for line in lines[-max(limit, 1):]:
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    events.append(data)
            except Exception:
                continue
        return events

    def format_event_tail(self, limit: int = 80) -> str:
        events = self.tail_events(limit=limit)
        if not events:
            return "No log events recorded yet."
        lines: list[str] = []
        for entry in events:
            line = f"[{entry.get('timestamp', '')}] {entry.get('role', '')}: {entry.get('message', '')}"
            extra = entry.get("extra") or {}
            if extra:
                line += f" | {json.dumps(extra, ensure_ascii=True)}"
            lines.append(line)
        return "\n".join(lines)
