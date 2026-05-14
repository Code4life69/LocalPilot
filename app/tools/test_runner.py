from __future__ import annotations

import queue
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SAFE_TEST_COMMANDS = {
    "run pytest",
    "run tests",
    "pytest",
    "test project",
    "verify project",
    "test localpilot",
    "verify localpilot",
}


class TestRunner:
    __test__ = False

    def __init__(self, app) -> None:
        self.app = app
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._cancel_requested = False
        self._current_command: list[str] | None = None

    def can_handle(self, text: str) -> bool:
        lowered = text.strip().lower()
        return lowered in SAFE_TEST_COMMANDS

    def build_pytest_command(self) -> list[str]:
        venv_python = self.app.root_dir / ".venv" / "Scripts" / "python.exe"
        return [str(venv_python), "-m", "pytest"]

    def is_running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def current_command_display(self) -> str:
        command = self._current_command or self.build_pytest_command()
        return self._display_command(command)

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return {
                    "ok": False,
                    "error": "Tests are already running.",
                    "running": True,
                    "command": self.current_command_display(),
                }
            self._cancel_requested = False
            command = self.build_pytest_command()
            self._current_command = command

        thread = threading.Thread(target=self._run_pytest, args=(command,), daemon=True)
        self._thread = thread
        thread.start()
        return {
            "ok": True,
            "status": "started",
            "message": f"Started tests.\nRunning: {self._display_command(command)}",
            "command": self._display_command(command),
        }

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return {"ok": False, "error": "No test run is active."}
            self._cancel_requested = True
            process.terminate()
        return {"ok": True, "message": "Test cancellation requested."}

    def run_blocking(self) -> dict[str, Any]:
        command = self.build_pytest_command()
        return self._execute_command(command)

    def _run_pytest(self, command: list[str]) -> None:
        result = self._execute_command(command)
        self.app.logger.event(
            "Tests",
            result.get("log_message", "finished"),
            command=self._display_command(command),
            exit_code=result.get("exit_code"),
            duration_seconds=result.get("duration_seconds"),
            summary=result.get("summary", ""),
        )

    def _execute_command(self, command: list[str]) -> dict[str, Any]:
        started_at = time.monotonic()
        started_iso = datetime.now().isoformat(timespec="seconds")
        command_display = self._display_command(command)
        self.app.logger.event("Tests", "started", command=command_display)
        if hasattr(self.app, "task_state"):
            self.app.task_state.update(
                last_test_command=command_display,
                last_test_status="running",
                tests_run=[],
                tests_running=True,
                tests_run_at=started_iso,
                next_recommended_action="Wait for the test run to finish and inspect the summary.",
            )

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        process = subprocess.Popen(
            command,
            cwd=str(self.app.root_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        with self._lock:
            self._process = process

        line_queue: "queue.Queue[tuple[str, str | None]]" = queue.Queue()

        def reader(stream, label: str) -> None:
            try:
                for line in iter(stream.readline, ""):
                    line_queue.put((label, line))
            finally:
                line_queue.put((label, None))

        threads = [
            threading.Thread(target=reader, args=(process.stdout, "stdout"), daemon=True),
            threading.Thread(target=reader, args=(process.stderr, "stderr"), daemon=True),
        ]
        for thread in threads:
            thread.start()

        closed_streams = 0
        while closed_streams < 2:
            label, line = line_queue.get()
            if line is None:
                closed_streams += 1
                continue
            clean = line.rstrip()
            if label == "stdout":
                stdout_lines.append(line)
            else:
                stderr_lines.append(line)
            if clean:
                self.app.logger.event("Tests", clean, stream=label)

        for thread in threads:
            thread.join(timeout=1)
        exit_code = process.wait()
        duration_seconds = round(time.monotonic() - started_at, 2)
        stdout_text = "".join(stdout_lines)
        stderr_text = "".join(stderr_lines)
        summary = self._summarize_output(stdout_text, stderr_text, exit_code)
        cancelled = self._cancel_requested

        with self._lock:
            self._process = None
            self._current_command = None
            self._cancel_requested = False

        status = "cancelled" if cancelled else ("passed" if exit_code == 0 else "failed")
        if hasattr(self.app, "task_state"):
            self.app.task_state.update(
                last_test_command=command_display,
                last_test_exit_code=exit_code,
                last_test_status=status,
                last_test_summary=summary,
                tests_running=False,
                tests_run=[command_display],
                tests_run_at=started_iso,
                next_recommended_action=(
                    "Inspect failing tests and apply a fix."
                    if status == "failed"
                    else "Review the test summary and continue."
                ),
            )

        log_message = "cancelled" if cancelled else ("passed" if exit_code == 0 else "failed")
        return {
            "ok": exit_code == 0 and not cancelled,
            "status": status,
            "log_message": log_message,
            "command": command_display,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "exit_code": exit_code,
            "duration_seconds": duration_seconds,
            "summary": summary,
            "tests_run": [command_display],
        }

    def _summarize_output(self, stdout: str, stderr: str, exit_code: int) -> str:
        combined = "\n".join(part for part in (stdout, stderr) if part)
        for line in reversed([item.strip() for item in combined.splitlines() if item.strip()]):
            if re.search(r"\b(passed|failed|error|errors|skipped)\b", line, flags=re.IGNORECASE):
                return line
        summary_match = re.search(r"=+\s*(.+?)\s*=+\s*$", combined, flags=re.MULTILINE)
        if summary_match:
            return summary_match.group(1).strip()
        if exit_code == 0:
            return "Tests passed."
        return "Tests failed."

    def _display_command(self, command: list[str]) -> str:
        root = self.app.root_dir.resolve()
        venv_python = str((root / ".venv" / "Scripts" / "python.exe").resolve())
        display_parts: list[str] = []
        for part in command:
            if str(Path(part).resolve()) == venv_python:
                display_parts.append(r".\.venv\Scripts\python.exe")
            else:
                display_parts.append(part)
        return " ".join(display_parts)
