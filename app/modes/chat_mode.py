from __future__ import annotations

from pathlib import Path
import re


class ChatMode:
    def __init__(self, app) -> None:
        self.app = app

    def handle(self, request: dict) -> dict:
        text = request["user_text"].strip()
        lowered = text.lower()
        if lowered in {"help", "/help"}:
            caps = self.app.capabilities
            return {
                "ok": True,
                "message": (
                    f"{caps['name']} modes: {', '.join(caps['modes'])}\n"
                    "Ask me to read/write files, run commands, search the web, save notes, or inspect the desktop."
                ),
            }
        if "what can you do" in lowered or "what do you do" in lowered:
            return {
                "ok": True,
                "message": self.app.describe_capabilities(),
            }
        if lowered == "trust checklist":
            return {
                "ok": True,
                "message": self._load_trust_checklist(),
            }
        if lowered == "model status":
            return {
                "ok": True,
                "message": self.app.describe_model_status(),
            }
        response = self.app.ollama.chat_with_role("main", self._build_chat_prompt(lowered), text)
        if not self._user_used_emoji(text):
            response = self._strip_simple_emoji(response)
        return {"ok": True, "message": response}

    def _load_trust_checklist(self) -> str:
        path = Path(self.app.root_dir) / "docs" / "TRUST_GAUNTLET.md"
        if not path.exists():
            return "Trust checklist is missing. Expected docs/TRUST_GAUNTLET.md."
        return path.read_text(encoding="utf-8")

    def _build_chat_prompt(self, lowered: str) -> str:
        lines = [
            self.app.system_prompt,
            "",
            "Chat mode style rules:",
            "- Answer the user's actual question directly.",
            "- For ordinary conversation, reply naturally in 1 to 3 sentences.",
            "- Do not introduce yourself unless the user asks who you are.",
            "- Do not say you are just a virtual assistant unless the user directly asks about your nature.",
            "- Do not claim personal experiences, memories, or preferences you do not actually have.",
            "- Do not list modes, tools, safety rules, or capabilities unless the user asks about them.",
            "- Do not mention knowledge cutoff dates unless directly relevant.",
            "- Avoid bullet lists for casual chat.",
            "- Avoid emojis unless the user uses them first.",
        ]
        if self._looks_like_small_talk(lowered):
            lines.extend(
                [
                    "- This is ordinary human conversation.",
                    "- Sound warm and normal, not like a support bot or product demo.",
                    "- Do not pivot into a capabilities overview.",
                ]
            )
        return "\n".join(lines)

    def _looks_like_small_talk(self, lowered: str) -> bool:
        normalized = lowered.strip(" .!?")
        exact_matches = {
            "how are you doing",
            "how are you",
            "how's it going",
            "hows it going",
            "what's up",
            "whats up",
            "how have you been",
        }
        if normalized in exact_matches:
            return True
        small_talk_fragments = (
            "rough day",
            "been working on lately",
            "prefer short tasks",
            "bigger projects",
            "if i told you i was stuck",
            "what would you say if i told you",
        )
        return any(fragment in normalized for fragment in small_talk_fragments)

    def _user_used_emoji(self, text: str) -> bool:
        return bool(re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", text))

    def _strip_simple_emoji(self, text: str) -> str:
        cleaned = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", text)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()
