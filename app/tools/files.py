from __future__ import annotations

import shutil
from pathlib import Path


def list_folder(path: str) -> dict:
    folder = Path(path)
    if not folder.exists():
        return {"ok": False, "error": f"Path not found: {folder}"}
    if not folder.is_dir():
        return {"ok": False, "error": f"Not a folder: {folder}"}
    items = [
        {
            "name": item.name,
            "type": "dir" if item.is_dir() else "file",
        }
        for item in sorted(folder.iterdir(), key=lambda value: (not value.is_dir(), value.name.lower()))
    ]
    return {"ok": True, "path": str(folder), "items": items}


def read_file(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {"ok": False, "error": f"File not found: {file_path}"}
    return {"ok": True, "path": str(file_path), "content": file_path.read_text(encoding="utf-8")}


def write_file(path: str, content: str) -> dict:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return {"ok": True, "path": str(file_path), "bytes_written": len(content.encode('utf-8'))}


def append_file(path: str, content: str) -> dict:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(content)
    return {"ok": True, "path": str(file_path), "bytes_appended": len(content.encode('utf-8'))}


def make_folder(path: str) -> dict:
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "path": str(folder)}


def copy_file(src: str, dst: str) -> dict:
    source = Path(src)
    destination = Path(dst)
    if not source.exists():
        return {"ok": False, "error": f"Source file not found: {source}"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {"ok": True, "src": str(source), "dst": str(destination)}


def move_file(src: str, dst: str) -> dict:
    source = Path(src)
    destination = Path(dst)
    if not source.exists():
        return {"ok": False, "error": f"Source file not found: {source}"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return {"ok": True, "src": str(source), "dst": str(destination)}

