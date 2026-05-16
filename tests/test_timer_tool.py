from __future__ import annotations

import time
from pathlib import Path

from app.timer_tool import TimerManager


class FakeScheduledTimer:
    def __init__(self, delay, callback):
        self.delay = delay
        self.callback = callback
        self.daemon = False
        self.started = False
        self.cancelled = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True


def build_timer_manager(tmp_path, notification_callback=None):
    scheduled = []

    def factory(delay, callback):
        timer = FakeScheduledTimer(delay, callback)
        scheduled.append(timer)
        return timer

    manager = TimerManager(
        tmp_path / "memory" / "timers.json",
        notification_callback=notification_callback or (lambda title, message: {"ok": True, "method": "fake", "message": message}),
        timer_factory=factory,
    )
    return manager, scheduled


def test_set_timer_creates_timer_record(tmp_path):
    manager, scheduled = build_timer_manager(tmp_path)

    result = manager.set_timer(duration_seconds=300, label="Check food", notify=True)

    assert result["ok"] is True
    assert result["timer_id"].startswith("timer_")
    assert result["duration_seconds"] == 300
    assert result["notification"] == "scheduled"
    assert scheduled[0].started is True


def test_list_timers_returns_active_timers(tmp_path):
    manager, _scheduled = build_timer_manager(tmp_path)
    created = manager.set_timer(duration_seconds=300, label="Check food", notify=True)

    result = manager.list_timers()

    assert result["ok"] is True
    assert result["timers"][0]["timer_id"] == created["timer_id"]


def test_cancel_timer_marks_timer_cancelled(tmp_path):
    manager, scheduled = build_timer_manager(tmp_path)
    created = manager.set_timer(duration_seconds=300, label="Check food", notify=True)

    result = manager.cancel_timer(created["timer_id"])

    assert result["ok"] is True
    assert result["status"] == "cancelled"
    assert scheduled[0].cancelled is True


def test_set_timer_does_not_block(tmp_path):
    manager, _scheduled = build_timer_manager(tmp_path)

    started = time.monotonic()
    result = manager.set_timer(duration_seconds=120, label="Timer", notify=True)
    elapsed = time.monotonic() - started

    assert result["ok"] is True
    assert elapsed < 0.5


def test_set_timer_label_cannot_execute_commands(tmp_path):
    manager, _scheduled = build_timer_manager(tmp_path)

    result = manager.set_timer(duration_seconds=60, label="powershell -Command Remove-Item", notify=True)

    assert result["ok"] is False
    assert "suspicious" in result["error"].lower()


def test_timer_notification_fallback_does_not_crash(tmp_path):
    manager, scheduled = build_timer_manager(tmp_path, notification_callback=lambda _title, _message: (_ for _ in ()).throw(RuntimeError("boom")))
    created = manager.set_timer(duration_seconds=1, label="Timer", notify=True)

    scheduled[0].callback()

    result = manager.list_timers(include_inactive=True)
    fired = next(timer for timer in result["timers"] if timer["timer_id"] == created["timer_id"])

    assert fired["status"] == "fired"
    assert fired["notification"] == "fallback_console"


def test_set_timer_source_does_not_use_run_command():
    source = Path("app/timer_tool.py").read_text(encoding="utf-8")

    assert "run_command" not in source
