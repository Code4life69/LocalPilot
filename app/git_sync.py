from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path


class GitSyncManager:
    def __init__(self, root_dir: str | Path, settings: dict, logger) -> None:
        self.root_dir = Path(root_dir)
        self.settings = settings
        self.logger = logger

    def sync(self, trigger: str) -> tuple[bool, str]:
        git_settings = self.settings.get("git_sync", {})
        if not git_settings.get("enabled", False):
            return False, "Git sync disabled."

        should_push = (
            trigger == "startup" and git_settings.get("push_on_startup", False)
        ) or (
            trigger == "shutdown" and git_settings.get("push_on_shutdown", False)
        )
        if not should_push:
            return False, f"Git sync skipped for trigger: {trigger}"

        if not self._is_git_repo():
            return False, "Git sync skipped: repository is not initialized."

        remote_check = self._run_git(["remote", "get-url", "origin"])
        if remote_check.returncode != 0:
            return False, "Git sync skipped: git remote `origin` is not configured."

        stage = self._run_git(["add", "-A"])
        if stage.returncode != 0:
            return False, f"Git add failed: {stage.stderr.strip()}"

        status = self._run_git(["status", "--porcelain"])
        if status.returncode != 0:
            return False, f"Git status failed: {status.stderr.strip()}"

        if not status.stdout.strip():
            push = self._run_git(["push", "-u", "origin", "main"])
            if push.returncode == 0:
                return True, "Git sync found no new changes and confirmed push to origin/main."
            return False, f"Git push failed: {push.stderr.strip() or push.stdout.strip()}"

        prefix = git_settings.get("commit_message_prefix", "LocalPilot autosync")
        message = f"{prefix} {trigger} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        commit = self._run_git(["commit", "-m", message])
        if commit.returncode != 0:
            return False, f"Git commit failed: {commit.stderr.strip() or commit.stdout.strip()}"

        push = self._run_git(["push", "-u", "origin", "main"])
        if push.returncode != 0:
            return False, f"Git push failed: {push.stderr.strip() or push.stdout.strip()}"
        return True, f"Git sync completed on {trigger}."

    def _is_git_repo(self) -> bool:
        result = self._run_git(["rev-parse", "--is-inside-work-tree"])
        return result.returncode == 0 and result.stdout.strip() == "true"

    def _run_git(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self.root_dir,
            capture_output=True,
            text=True,
        )

