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
