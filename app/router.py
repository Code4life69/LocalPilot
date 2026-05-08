from __future__ import annotations


class KeywordRouter:
    ROUTE_KEYWORDS = {
        "research": ["search", "look up", "research", "find on web", "duckduckgo"],
        "desktop": ["screenshot", "screen", "mouse", "click", "type", "hotkey", "window", "control", "ui"],
        "memory": ["note", "notes", "remember", "memory", "fact"],
        "code": ["file", "folder", "read", "write", "append", "copy", "move", "command", "shell", "run ", "mkdir"],
    }

    def classify(self, text: str) -> str:
        lowered = text.lower()
        for mode, keywords in self.ROUTE_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                return mode
        return "chat"

