from __future__ import annotations

import re

from app.tools.web import search_web


class ResearchMode:
    def __init__(self, app) -> None:
        self.app = app

    def handle(self, request: dict) -> dict:
        text = request["user_text"].strip()
        query = self._normalize_query_for_search(self._extract_query(text))
        if not query:
            return {"ok": False, "error": "No research query provided."}
        result = search_web(query, max_results=5)
        if not result.get("ok"):
            return result

        if self._looks_like_direct_fact_question(text):
            result["message"] = self._summarize_results(query, result["results"])

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
        cleaned = text.strip()
        lowered = cleaned.lower()
        for prefix in ("search", "research", "look up", "find on web"):
            if lowered.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip(' :')
                lowered = cleaned.lower()
                break

        conversational_prefixes = (
            "can you figure out",
            "can you tell me",
            "could you tell me",
            "please tell me",
            "please find",
            "figure out",
            "tell me",
        )
        for prefix in conversational_prefixes:
            if lowered.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip(" ?:")
                lowered = cleaned.lower()
                break
        return cleaned

    def _looks_like_direct_fact_question(self, text: str) -> bool:
        lowered = text.lower().strip()
        return lowered.startswith(("who ", "what ", "when ", "where ", "is ", "are ", "can ", "does ", "did "))

    def _summarize_results(self, query: str, results: list[dict]) -> str:
        if not results:
            return f"No research results found for: {query}"

        top = results[0]
        lines = [f"Research answer for: {query}"]
        if top.get("snippet"):
            lines.append(top["snippet"])
        else:
            lines.append(top.get("title", "Top result found."))

        lines.append("")
        lines.append("Sources:")
        for item in results[:3]:
            title = item.get("title", "Untitled result")
            url = item.get("url", "")
            lines.append(f"- {title} | {url}")
        return "\n".join(lines)

    def _normalize_query_for_search(self, query: str) -> str:
        normalized = " ".join(query.replace("?", " ").split())
        lowered = normalized.lower()
        date_match = re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", normalized)

        if "current president" in lowered:
            if "united states" not in lowered and "u.s." not in lowered and "usa" not in lowered:
                normalized = normalized + " United States"
            if date_match and "as of" not in lowered:
                normalized = f"current president of the United States as of {date_match.group(0)}"

        if "current vice president" in lowered:
            normalized = "current vice president of the United States"
            if date_match:
                normalized = f"{normalized} {date_match.group(0)}"

        return normalized
