from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MemoryStore:
    def __init__(self, memory_dir: str | Path, capability_manifest_path: str | Path) -> None:
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.notes_path = self.memory_dir / "notes.md"
        self.facts_path = self.memory_dir / "learned_facts.json"
        self.capability_manifest_path = Path(capability_manifest_path)
        if not self.notes_path.exists():
            self.notes_path.write_text("# LocalPilot Notes\n\n", encoding="utf-8")
        if not self.facts_path.exists():
            self.facts_path.write_text("{}\n", encoding="utf-8")

    def load_capabilities(self) -> dict[str, Any]:
        with self.capability_manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_note(self, text: str) -> str:
        note = text.strip()
        if not note:
            return "No note text provided."
        with self.notes_path.open("a", encoding="utf-8") as handle:
            handle.write(f"- {note}\n")
        return "Note saved."

    def search_notes(self, keyword: str) -> list[str]:
        needle = keyword.strip().lower()
        if not needle:
            return []
        lines = self.notes_path.read_text(encoding="utf-8").splitlines()
        return [line for line in lines if needle in line.lower()]

    def show_notes(self) -> str:
        return self.notes_path.read_text(encoding="utf-8")

    def load_facts(self) -> dict[str, Any]:
        with self.facts_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_fact(self, key: str, value: Any) -> str:
        facts = self.load_facts()
        facts[key] = value
        with self.facts_path.open("w", encoding="utf-8") as handle:
            json.dump(facts, handle, indent=2)
            handle.write("\n")
        return f"Saved fact: {key}"

