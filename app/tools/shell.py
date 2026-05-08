from __future__ import annotations

import subprocess


def run_command(command: str, cwd: str | None = None) -> dict:
    completed = subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        capture_output=True,
        text=True,
    )
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "command": command,
        "cwd": cwd,
    }

