from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class CheckpointManager:
    def __init__(self, checkpoints_dir: str | Path) -> None:
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def create_file_checkpoint(
        self,
        target_path: str | Path,
        *,
        task_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> dict[str, Any]:
        path = Path(target_path).resolve()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"{timestamp}_{(task_id or 'manual')}_{uuid.uuid4().hex[:6]}"
        checkpoint_dir = self.checkpoints_dir / checkpoint_id
        files_dir = checkpoint_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        existed_before = path.exists()
        backup_path: Path | None = None
        sha256_before: str | None = None

        if existed_before:
            backup_path = files_dir / self._backup_name_for_path(path)
            shutil.copy2(path, backup_path)
            sha256_before = self._sha256(path)

        manifest = {
            "checkpoint_id": checkpoint_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "task_id": task_id,
            "tool_call_id": tool_call_id,
            "files": [
                {
                    "original_path": str(path),
                    "backup_path": str(backup_path) if backup_path else None,
                    "file_existed_before": existed_before,
                    "sha256_before": sha256_before,
                }
            ],
        }
        manifest_path = checkpoint_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return {
            "ok": True,
            "checkpoint_id": checkpoint_id,
            "checkpoint_dir": str(checkpoint_dir),
            "manifest_path": str(manifest_path),
            "file_existed_before": existed_before,
            "backup_path": str(backup_path) if backup_path else None,
        }

    def list_checkpoints(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for manifest_path in sorted(self.checkpoints_dir.glob("*/manifest.json"), reverse=True):
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                items.append(data)
        return items

    def get_checkpoint(self, checkpoint_id: str) -> dict[str, Any] | None:
        manifest_path = self.checkpoints_dir / checkpoint_id / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def restore_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        manifest_path = self.checkpoints_dir / checkpoint_id / "manifest.json"
        if not manifest_path.exists():
            return {"ok": False, "error": f"Checkpoint not found: {checkpoint_id}"}

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        restored_paths: list[str] = []
        for entry in manifest.get("files", []):
            original_path = Path(str(entry["original_path"]))
            existed_before = bool(entry.get("file_existed_before"))
            backup_path_value = entry.get("backup_path")
            if existed_before:
                if not backup_path_value:
                    return {"ok": False, "error": f"Checkpoint backup missing for {original_path}"}
                backup_path = Path(str(backup_path_value))
                if not backup_path.exists():
                    return {"ok": False, "error": f"Checkpoint backup not found: {backup_path}"}
                original_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_path, original_path)
            elif original_path.exists():
                original_path.unlink()
            restored_paths.append(str(original_path))
        return {"ok": True, "checkpoint_id": checkpoint_id, "restored_paths": restored_paths}

    def _backup_name_for_path(self, path: Path) -> str:
        sanitized = str(path).replace(":", "").replace("\\", "_").replace("/", "_")
        return f"{sanitized}.bak"

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()
