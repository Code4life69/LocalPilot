from __future__ import annotations

import re


class KeywordRouter:
    ROUTE_KEYWORDS = {
        "research": ["search", "look up", "research", "find on web", "duckduckgo"],
        "desktop": ["screenshot", "screen", "mouse", "click", "type", "hotkey", "window", "control", "ui automation"],
        "memory": ["note", "notes", "remember", "memory", "fact"],
        "code": ["file", "folder", "read", "write", "append", "copy", "move", "command", "shell", "run ", "mkdir"],
    }

    RESEARCH_FACT_HINTS = [
        "current",
        "today",
        "latest",
        "as of",
        "president",
        "vice president",
        "ceo",
        "governor",
        "mayor",
        "prime minister",
        "price",
        "stock",
        "news",
    ]

    DESKTOP_ACTION_HINTS = [
        "on my pc",
        "on google",
        "in the browser",
        "use my mouse",
        "use my keyboard",
        "click through",
        "download that image",
        "download this image",
        "look at the images",
        "open google",
        "open chrome",
        "open browser",
        "visualize desktop",
        "desktop understanding",
        "what you see",
    ]

    DESKTOP_OBSERVATION_PHRASES = [
        "inspect desktop",
        "page inspect",
        "page confidence",
        "show page understanding",
        "show desktop lessons",
        "ocr screenshot",
        "read screen text",
        "page ocr",
        "what window am i on",
        "what window am i in",
        "what is under my mouse",
        "what is under my cursor",
        "get focused control",
        "focused control",
        "list visible controls",
        "show visible controls",
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
        lowered = text.lower().strip()
        if lowered.startswith("professional build ") or lowered.startswith("build this professionally"):
            return "code"
        if self._looks_like_desktop_task_request(lowered):
            return "desktop"
        if self._looks_like_code_verification_request(lowered):
            return "code"
        if self._looks_like_website_project_request(lowered):
            return "code"
        if self._looks_like_code_project_request(lowered):
            return "code"
        if self._looks_like_natural_file_create_request(lowered):
            return "code"
        if self._looks_like_code_tool_request(lowered):
            return "code"
        if self._looks_like_memory_request(lowered):
            return "memory"
        if self._looks_like_current_fact_request(lowered):
            return "research"
        for mode, keywords in self.ROUTE_KEYWORDS.items():
            if any(self._contains_phrase(lowered, keyword) for keyword in keywords):
                return mode
        return "chat"

    def _looks_like_memory_request(self, lowered: str) -> bool:
        return lowered.startswith(("show notes", "search notes", "save note", "remember", "save fact")) or lowered == "notes"

    def _looks_like_current_fact_request(self, lowered: str) -> bool:
        normalized = lowered.strip()
        for prefix in ("no ", "well ", "so ", "okay ", "ok "):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        starts_like_question = any(normalized.startswith(word + " ") for word in self.RESEARCH_QUESTION_WORDS)
        has_fact_hint = any(hint in lowered for hint in self.RESEARCH_FACT_HINTS)
        has_explicit_date = any(char.isdigit() for char in lowered) and any(sep in lowered for sep in ("/", "-"))
        return (starts_like_question and has_fact_hint) or has_explicit_date and has_fact_hint

    def _looks_like_code_project_request(self, lowered: str) -> bool:
        build_words = ("create", "build", "make", "generate", "scaffold")
        scaffold_hints = (
            "app",
            "program",
            "script",
            "gui",
            "double click",
            "double-click",
            "starter",
            "folder",
        )
        project_words = ("calculator", "notepad", "todo", "timer", "project")
        return (
            any(self._contains_phrase(lowered, word) for word in build_words)
            and (
                any(self._contains_phrase(lowered, hint) for hint in scaffold_hints)
                or any(self._contains_phrase(lowered, word) for word in project_words)
            )
        )

    def _looks_like_code_verification_request(self, lowered: str) -> bool:
        return (
            lowered.startswith("verify")
            and any(self._contains_phrase(lowered, phrase) for phrase in ("app", "website", "project", "notepad", "timer", "todo", "calculator"))
            and any(self._contains_phrase(lowered, phrase) for phrase in ("files", "run", "generated"))
        )

    def _looks_like_website_project_request(self, lowered: str) -> bool:
        build_words = ("create", "build", "make", "generate", "scaffold")
        website_words = (
            "website",
            "web site",
            "web page",
            "webpage",
            "landing page",
            "local website",
            "local site",
            "html css js",
            "html css",
            "html css and javascript",
            "html css javascript",
            "javascript",
        )
        disqualifiers = (
            "search the website",
            "open this website",
            "look up websites",
            "websites about",
        )
        if any(phrase in lowered for phrase in disqualifiers):
            return False
        return any(self._contains_phrase(lowered, word) for word in build_words) and any(
            self._contains_phrase(lowered, word) for word in website_words
        )

    def _looks_like_natural_file_create_request(self, lowered: str) -> bool:
        build_words = ("create", "make", "write")
        file_words = ("text file", "file")
        name_words = ("named", "called")
        content_words = ("that says", "with", "containing")
        return (
            any(self._contains_phrase(lowered, word) for word in build_words)
            and any(self._contains_phrase(lowered, word) for word in file_words)
            and any(self._contains_phrase(lowered, word) for word in name_words)
            and any(self._contains_phrase(lowered, word) for word in content_words)
        )

    def _looks_like_code_tool_request(self, lowered: str) -> bool:
        return lowered.startswith(
            (
                "list ",
                "read ",
                "write ",
                "append ",
                "copy ",
                "move ",
                "mkdir ",
                "run ",
                "shell ",
                "read file",
                "write file",
                "append file",
                "copy file",
                "move file",
                "make folder",
                "create folder",
                "run command",
                "list folder",
                "list files",
            )
        )

    def _looks_like_desktop_task_request(self, lowered: str) -> bool:
        if lowered in {"visualize desktop", "visualize desktop understanding", "show me what you see"}:
            return True
        if any(self._contains_phrase(lowered, phrase) for phrase in self.DESKTOP_OBSERVATION_PHRASES):
            return True
        browser_or_pc_action = any(hint in lowered for hint in self.DESKTOP_ACTION_HINTS)
        active_verb = any(word in lowered for word in ("open", "click", "type", "download", "search", "look at", "use"))
        return browser_or_pc_action and active_verb

    def _contains_phrase(self, lowered: str, phrase: str) -> bool:
        if phrase.endswith(" "):
            return lowered.startswith(phrase) or f" {phrase}" in lowered
        parts = [re.escape(part) for part in phrase.split()]
        pattern = r"(?<![a-z0-9])" + r"\s+".join(parts) + r"(?![a-z0-9])"
        return re.search(pattern, lowered) is not None
