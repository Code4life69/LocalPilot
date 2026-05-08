from __future__ import annotations


class KeywordRouter:
    ROUTE_KEYWORDS = {
        "research": ["search", "look up", "research", "find on web", "duckduckgo"],
        "desktop": ["screenshot", "screen", "mouse", "click", "type", "hotkey", "window", "control", "ui"],
        "memory": ["note", "notes", "remember", "memory", "fact"],
        "code": ["file", "folder", "read", "write", "append", "copy", "move", "command", "shell", "run ", "mkdir"],
    }

    RESEARCH_FACT_HINTS = [
        "current",
        "today",
        "latest",
        "as of",
        "president",
        "ceo",
        "governor",
        "mayor",
        "prime minister",
        "price",
        "stock",
        "news",
    ]

    RESEARCH_QUESTION_WORDS = [
        "who",
        "what",
        "when",
        "where",
        "is",
        "are",
        "can",
        "did",
        "does",
    ]

    def classify(self, text: str) -> str:
        lowered = text.lower()
        if self._looks_like_code_project_request(lowered):
            return "code"
        if self._looks_like_current_fact_request(lowered):
            return "research"
        for mode, keywords in self.ROUTE_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                return mode
        return "chat"

    def _looks_like_current_fact_request(self, lowered: str) -> bool:
        starts_like_question = any(lowered.startswith(word + " ") for word in self.RESEARCH_QUESTION_WORDS)
        has_fact_hint = any(hint in lowered for hint in self.RESEARCH_FACT_HINTS)
        has_explicit_date = any(char.isdigit() for char in lowered) and any(sep in lowered for sep in ("/", "-"))
        return (starts_like_question and has_fact_hint) or has_explicit_date and has_fact_hint

    def _looks_like_code_project_request(self, lowered: str) -> bool:
        build_words = ("create", "build", "make")
        project_words = ("app", "program", "script", "calculator", "project")
        path_hint = ":\\" in lowered or " in c:\\" in lowered or "folder" in lowered
        return any(word in lowered for word in build_words) and any(word in lowered for word in project_words) and path_hint
