from __future__ import annotations

import re
from pathlib import Path

from app.tools import files as file_tools
from app.tools import shell as shell_tools


class CodeMode:
    def __init__(self, app) -> None:
        self.app = app

    def handle(self, request: dict) -> dict:
        text = request["user_text"].strip()
        lowered = text.lower()
        self.app.logger.event("Mode:code", f"Handling code request: {text}")

        if lowered.startswith("list ") or "list folder" in lowered or "list files" in lowered:
            path = self._extract_path(text) or "."
            return file_tools.list_folder(path)

        if lowered.startswith("read ") or "read file" in lowered:
            path = self._extract_path(text)
            if not path:
                return {"ok": False, "error": "No file path provided."}
            return file_tools.read_file(path)

        if lowered.startswith("write ") or "write file" in lowered:
            path, content = self._extract_write_args(text)
            if not path:
                return {"ok": False, "error": "No file path provided for write."}
            if self.app.safety.requires_write_confirmation(path):
                approved = self.app.ask_approval(f"Overwrite existing file?\n{path}")
                if not approved:
                    return {"ok": False, "error": "Write cancelled by user."}
            return file_tools.write_file(path, content)

        if lowered.startswith("append ") or "append file" in lowered:
            path, content = self._extract_write_args(text)
            if not path:
                return {"ok": False, "error": "No file path provided for append."}
            return file_tools.append_file(path, content)

        if lowered.startswith("mkdir ") or "make folder" in lowered or "create folder" in lowered:
            path = self._extract_path(text)
            if not path:
                return {"ok": False, "error": "No folder path provided."}
            return file_tools.make_folder(path)

        if lowered.startswith("copy ") or "copy file" in lowered:
            src, dst = self._extract_two_paths(text)
            if not src or not dst:
                return {"ok": False, "error": "Copy requires source and destination."}
            if self.app.safety.requires_move_confirmation(dst):
                approved = self.app.ask_approval(f"Destination exists. Copy and overwrite?\n{dst}")
                if not approved:
                    return {"ok": False, "error": "Copy cancelled by user."}
            return file_tools.copy_file(src, dst)

        if lowered.startswith("move ") or "move file" in lowered:
            src, dst = self._extract_two_paths(text)
            if not src or not dst:
                return {"ok": False, "error": "Move requires source and destination."}
            approved = self.app.ask_approval(f"Approve file move?\n{src}\n->\n{dst}")
            if not approved:
                return {"ok": False, "error": "Move cancelled by user."}
            return file_tools.move_file(src, dst)

        if lowered.startswith("run ") or lowered.startswith("shell ") or "run command" in lowered:
            command = self._extract_command(text)
            if not command:
                return {"ok": False, "error": "No command provided."}
            if self.app.safety.is_command_blocked(command):
                return {"ok": False, "error": f"Blocked dangerous command: {command}"}
            approved = self.app.ask_approval(f"Command wants to run:\n{command}")
            if not approved:
                return {"ok": False, "error": "Command cancelled by user."}
            return shell_tools.run_command(command, cwd=str(Path.cwd()))

        response = self.app.ollama.chat(self.app.system_prompt, text)
        return {"ok": True, "message": response}

    def _extract_path(self, text: str) -> str | None:
        match = re.search(r'"([^"]+)"', text)
        if match:
            return match.group(1)
        parts = text.split(maxsplit=1)
        if len(parts) == 2:
            return parts[1].replace("folder", "").replace("file", "").strip()
        return None

    def _extract_write_args(self, text: str) -> tuple[str | None, str]:
        quoted = re.findall(r'"([^"]+)"', text)
        if len(quoted) >= 2:
            return quoted[0], quoted[1]
        parts = text.split(maxsplit=2)
        if len(parts) >= 3:
            return parts[1], parts[2]
        if len(parts) == 2:
            return parts[1], ""
        return None, ""

    def _extract_two_paths(self, text: str) -> tuple[str | None, str | None]:
        quoted = re.findall(r'"([^"]+)"', text)
        if len(quoted) >= 2:
            return quoted[0], quoted[1]
        parts = text.split(maxsplit=2)
        if len(parts) >= 3:
            return parts[1], parts[2]
        return None, None

    def _extract_command(self, text: str) -> str:
        lowered = text.lower()
        for prefix in ("run command", "run", "shell"):
            if lowered.startswith(prefix):
                return text[len(prefix):].strip()
        return text.strip()

