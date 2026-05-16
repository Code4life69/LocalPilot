from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable


NotificationCallback = Callable[[str, str], dict[str, Any]]

SUSPICIOUS_TIMER_LABEL_PATTERNS = (
    ";",
    "&&",
    "|",
    "powershell",
    "cmd.exe",
    "shutdown",
    "restart",
    "remove-item",
    "del ",
    "rm ",
)


def default_notify(title: str, message: str) -> dict[str, Any]:
    try:
        from win10toast import ToastNotifier  # type: ignore

        toaster = ToastNotifier()
        toaster.show_toast(title, message, duration=10, threaded=True)
        return {"ok": True, "method": "win10toast"}
    except Exception:
        pass

    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(title, message)
        root.destroy()
        return {"ok": True, "method": "tkinter_messagebox"}
    except Exception:
        pass

    try:
        import winsound

        winsound.MessageBeep()
        print(f"{title}: {message}")
        return {"ok": True, "method": "winsound_beep"}
    except Exception:
        pass

    print(f"{title}: {message}")
    return {"ok": True, "method": "console_print"}


@dataclass
class ManagedTimer:
    timer_id: str
    duration_seconds: int
    label: str
    notify: bool
    created_at: str
    fires_at: str
    status: str
    notification: str
    fired_at: str | None = None
    cancelled_at: str | None = None


class TimerManager:
    def __init__(
        self,
        timers_path: str | Path,
        notification_callback: NotificationCallback | None = None,
        timer_factory: Callable[[float, Callable[[], None]], threading.Timer] | None = None,
    ) -> None:
        self.timers_path = Path(timers_path)
        self.timers_path.parent.mkdir(parents=True, exist_ok=True)
        self.notification_callback = notification_callback or default_notify
        self.timer_factory = timer_factory or self._default_timer_factory
        self._lock = threading.Lock()
        self._timers: dict[str, dict[str, Any]] = {}
        self._scheduled: dict[str, threading.Timer] = {}
        self._load()
        self._schedule_active_timers()

    def set_timer(self, duration_seconds: int, label: str = "Timer", notify: bool = True) -> dict[str, Any]:
        if duration_seconds <= 0:
            return {"ok": False, "error": "duration_seconds must be greater than zero."}
        if self._label_is_suspicious(label):
            return {"ok": False, "error": "Timer label contains suspicious command-like text."}

        timer_id = f"timer_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        created_at = datetime.now()
        fires_at = created_at + timedelta(seconds=duration_seconds)
        record = ManagedTimer(
            timer_id=timer_id,
            duration_seconds=duration_seconds,
            label=label.strip() or "Timer",
            notify=bool(notify),
            created_at=created_at.isoformat(timespec="seconds"),
            fires_at=fires_at.isoformat(timespec="seconds"),
            status="active",
            notification="scheduled" if notify else "disabled",
        )
        with self._lock:
            self._timers[timer_id] = record.__dict__.copy()
            self._persist_locked()
            self._schedule_timer_locked(timer_id)
        return {
            "ok": True,
            "timer_id": timer_id,
            "duration_seconds": duration_seconds,
            "label": record.label,
            "fires_at": record.fires_at,
            "notification": record.notification,
        }

    def list_timers(self, include_inactive: bool = False) -> dict[str, Any]:
        with self._lock:
            timers = list(self._timers.values())
        if not include_inactive:
            timers = [timer for timer in timers if timer.get("status") == "active"]
        timers.sort(key=lambda timer: str(timer.get("fires_at", "")))
        return {"ok": True, "timers": timers}

    def cancel_timer(self, timer_id: str) -> dict[str, Any]:
        with self._lock:
            timer = self._timers.get(timer_id)
            if timer is None:
                return {"ok": False, "error": f"Timer not found: {timer_id}"}
            scheduled = self._scheduled.pop(timer_id, None)
            if scheduled is not None:
                scheduled.cancel()
            timer["status"] = "cancelled"
            timer["cancelled_at"] = datetime.now().isoformat(timespec="seconds")
            self._persist_locked()
            return {"ok": True, "timer_id": timer_id, "status": "cancelled"}

    def _load(self) -> None:
        if not self.timers_path.exists():
            self.timers_path.write_text(json.dumps({"timers": []}, indent=2) + "\n", encoding="utf-8")
        try:
            payload = json.loads(self.timers_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {"timers": []}
        timers = payload.get("timers", []) if isinstance(payload, dict) else []
        for timer in timers:
            if isinstance(timer, dict) and timer.get("timer_id"):
                self._timers[str(timer["timer_id"])] = timer

    def _schedule_active_timers(self) -> None:
        with self._lock:
            for timer_id, timer in list(self._timers.items()):
                if timer.get("status") != "active":
                    continue
                self._schedule_timer_locked(timer_id)

    def _schedule_timer_locked(self, timer_id: str) -> None:
        if timer_id in self._scheduled:
            return
        timer = self._timers.get(timer_id)
        if timer is None or timer.get("status") != "active":
            return
        try:
            fires_at = datetime.fromisoformat(str(timer["fires_at"]))
        except Exception:
            timer["status"] = "failed"
            self._persist_locked()
            return
        delay = max((fires_at - datetime.now()).total_seconds(), 0.0)
        scheduled = self.timer_factory(delay, lambda: self._fire_timer(timer_id))
        scheduled.daemon = True
        self._scheduled[timer_id] = scheduled
        scheduled.start()

    def _fire_timer(self, timer_id: str) -> None:
        with self._lock:
            timer = self._timers.get(timer_id)
            if timer is None or timer.get("status") != "active":
                return
            timer["status"] = "fired"
            timer["fired_at"] = datetime.now().isoformat(timespec="seconds")
            self._persist_locked()
            self._scheduled.pop(timer_id, None)
            notify_enabled = bool(timer.get("notify", True))
            label = str(timer.get("label", "Timer"))
        if not notify_enabled:
            return
        message = f"{label} finished at {datetime.now().strftime('%I:%M %p').lstrip('0')}."
        try:
            result = self.notification_callback("LocalPilot Timer", message)
        except Exception:
            print(f"LocalPilot Timer: {message}")
            result = {"ok": True, "method": "fallback_console"}
        with self._lock:
            timer = self._timers.get(timer_id)
            if timer is None:
                return
            timer["notification"] = str(result.get("method", "sent"))
            self._persist_locked()

    def _persist_locked(self) -> None:
        payload = {"timers": list(self._timers.values())}
        self.timers_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    def _label_is_suspicious(self, label: str) -> bool:
        lowered = label.lower()
        return any(pattern in lowered for pattern in SUSPICIOUS_TIMER_LABEL_PATTERNS)

    def _default_timer_factory(self, delay: float, callback: Callable[[], None]) -> threading.Timer:
        return threading.Timer(delay, callback)
