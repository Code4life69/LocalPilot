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
        if self._looks_like_website_project_request(lowered):
            return "code"
        if self._looks_like_code_project_request(lowered):
            return "code"
        if self._looks_like_code_tool_request(lowered):
            return "code"
        if self._looks_like_memory_request(lowered):
            return "memory"
        if self._looks_like_desktop_task_request(lowered):
            return "desktop"
        if self._looks_like_current_fact_request(lowered):
            return "research"
        for mode, keywords in self.ROUTE_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
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
        build_words = ("create", "build", "make")
        project_words = ("app", "program", "script", "calculator", "project")
        path_hint = ":\\" in lowered or " in c:\\" in lowered or "folder" in lowered
        return any(word in lowered for word in build_words) and any(word in lowered for word in project_words) and path_hint

    def _looks_like_website_project_request(self, lowered: str) -> bool:
        build_words = ("create", "build", "make", "generate", "scaffold")
        website_words = (
            "website",
            "web site",
            "web page",
            "webpage",
            "local website",
            "local site",
            "html css js",
            "html css and javascript",
            "html css javascript",
        )
        disqualifiers = (
            "search the website",
            "open this website",
            "look up websites",
            "websites about",
        )
        if any(phrase in lowered for phrase in disqualifiers):
            return False
        return any(word in lowered for word in build_words) and any(word in lowered for word in website_words)

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
        browser_or_pc_action = any(hint in lowered for hint in self.DESKTOP_ACTION_HINTS)
        active_verb = any(word in lowered for word in ("open", "click", "type", "download", "search", "look at", "use"))
        return browser_or_pc_action and active_verb
