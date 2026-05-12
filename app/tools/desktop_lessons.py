from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class DesktopLessonStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, lesson_type: str, task: str, reason: str, **extra: Any) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "type": lesson_type,
            "task": task,
            "reason": reason,
            "extra": extra,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        entries: list[dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries[-limit:]

    def render_recent(self, limit: int = 20) -> str:
        lessons = self.recent(limit=limit)
        if not lessons:
            return "No desktop lessons recorded yet."
        lines = ["Desktop lessons:"]
        for lesson in reversed(lessons):
            task = lesson.get("task", "")
            reason = lesson.get("reason", "")
            lesson_type = lesson.get("type", "")
            timestamp = lesson.get("timestamp", "")
            lines.append(f"- [{timestamp}] {lesson_type}: {task}")
            lines.append(f"  {reason}")
        return "\n".join(lines)
