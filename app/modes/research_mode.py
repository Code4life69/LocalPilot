from __future__ import annotations

from app.tools.web import search_web


class ResearchMode:
    def __init__(self, app) -> None:
        self.app = app

    def handle(self, request: dict) -> dict:
        text = request["user_text"].strip()
        query = self._extract_query(text)
        if not query:
            return {"ok": False, "error": "No research query provided."}
        result = search_web(query, max_results=5)
        if "save" in text.lower() and result.get("ok"):
            summary_lines = [f"## Research: {query}"]
            for item in result["results"]:
                summary_lines.append(f"- {item['title']} | {item['url']}")
                if item["snippet"]:
                    summary_lines.append(f"  {item['snippet']}")
            self.app.memory.save_note("\n".join(summary_lines))
            result["note_saved"] = True
        return result

    def _extract_query(self, text: str) -> str:
        lowered = text.lower()
        for prefix in ("search", "research", "look up", "find on web"):
            if lowered.startswith(prefix):
                return text[len(prefix):].strip(' :')
        return text

