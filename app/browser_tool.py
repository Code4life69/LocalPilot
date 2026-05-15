from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


class BrowserToolBridge:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.browser_dir = self.root_dir / "browser"
        self.script_path = self.browser_dir / "browser_server.js"

    def _build_payload(self, action: str, **kwargs: Any) -> dict[str, Any]:
        payload = {"action": action}
        payload.update(kwargs)
        return payload

    def run(self, action: str, **kwargs: Any) -> dict[str, Any]:
        if not self.script_path.exists():
            return {
                "ok": False,
                "action": action,
                "error": f"Browser bridge script not found: {self.script_path}",
            }

        payload = self._build_payload(action, **kwargs)
        completed = subprocess.run(
            ["node", str(self.script_path)],
            cwd=str(self.browser_dir),
            input=json.dumps(payload),
            capture_output=True,
            text=True,
        )

        stdout = completed.stdout.strip()
        if not stdout:
            return {
                "ok": False,
                "action": action,
                "error": completed.stderr.strip() or "Browser bridge returned no output.",
            }
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            return {
                "ok": False,
                "action": action,
                "error": f"Browser bridge returned invalid JSON: {stdout}",
            }
        if "action" not in parsed:
            parsed["action"] = action
        return parsed
