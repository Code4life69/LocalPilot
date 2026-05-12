from __future__ import annotations

import random
import re
import time
import webbrowser
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus

from app.tools.mouse_keyboard import hotkey, press_key, type_text
from app.tools.screen import get_active_window_basic, take_screenshot
from app.tools.windows_ui import get_active_window_title, get_focused_control, list_visible_controls


@dataclass
class PlannedStep:
    name: str
    description: str
    kind: str
    value: str | None = None
    expected_terms: list[str] | None = None
    vision_prompt: str | None = None
    attempts: int = 2


class DesktopExecutionFlow:
    RANDOM_TOPICS = [
        "dolphins",
        "mountains",
        "city skyline",
        "retro keyboards",
        "space nebula",
        "wolves",
    ]

    def __init__(self, app) -> None:
        self.app = app

    def can_handle(self, text: str) -> bool:
        lowered = text.lower()
        return bool(
            re.search(r"https?://\S+", lowered)
            or "open google" in lowered
            or "open browser" in lowered
            or self._looks_like_browser_search_request(lowered)
        )

    def execute(self, text: str) -> dict[str, Any] | None:
        plan = self._build_plan(text)
        if not plan:
            self.app.logger.event("DesktopFlow", "unable to build plan for desktop request")
            return None

        summary = "\n".join(f"- {step.description}" for step in plan)
        if not self.app.ask_approval(f"Approve desktop execution plan?\n{summary}"):
            return {"ok": False, "error": "Desktop execution cancelled by user."}

        self.app.logger.event("DesktopFlow", f"planned {len(plan)} steps")
        step_results: list[dict[str, Any]] = []
        last_snapshot: dict[str, Any] | None = None

        for step in plan:
            self.app.logger.event("DesktopFlow", f"running step: {step.name}")
            ok, detail, snapshot = self._run_step(step)
            step_results.append({"step": step.description, "ok": ok, "detail": detail})
            last_snapshot = snapshot or last_snapshot
            if not ok:
                verification = self._result_verification(last_snapshot, default_verified=False, default_reason=detail)
                return {
                    "ok": False,
                    "content": self._format_summary(step_results, last_snapshot, success=False),
                    "steps": step_results,
                    **verification,
                }

        content = self._format_summary(step_results, last_snapshot, success=True)
        if self._needs_image_download_followup(text):
            content += (
                "\n\nNote: I opened and verified the Google Images search, but selecting, downloading, "
                "and copying a chosen browser image into a folder is not implemented yet. No image file was saved."
            )

        verification = self._result_verification(last_snapshot, default_verified=True, default_reason="Desktop execution completed.")
        return {
            "ok": True,
            "content": content,
            "steps": step_results,
            **verification,
        }

    def _build_plan(self, text: str) -> list[PlannedStep]:
        lowered = text.lower()
        url_match = re.search(r"https?://\S+", text)
        if url_match:
            url = url_match.group(0)
            expected = self._title_terms_for_url(url)
            return [
                PlannedStep(
                    name="open_url",
                    description=f"Open {url} in the default browser",
                    kind="open_url",
                    value=url,
                    expected_terms=expected,
                    vision_prompt=f"Check whether the browser is showing {url}.",
                )
            ]

        if "open google" in lowered or "open browser" in lowered:
            return [
                PlannedStep(
                    name="open_google",
                    description="Open Google in the default browser",
                    kind="open_url",
                    value="https://www.google.com",
                    expected_terms=["google"],
                    vision_prompt="Check whether the browser is showing the Google homepage.",
                )
            ]

        query = self._extract_search_query(text)
        if query:
            search_url = self._build_google_search_url(query, images="image" in lowered)
            expected_terms = [term for term in self._split_terms(query)[:3] if term]
            description = f"Search Google for {query!r}"
            if "image" in lowered:
                description += " in Images"
            return [
                PlannedStep(
                    name="open_google",
                    description="Open Google in the default browser",
                    kind="open_url",
                    value="https://www.google.com",
                    expected_terms=["google"],
                    vision_prompt="Check whether the browser is showing Google.",
                ),
                PlannedStep(
                    name="focus_address_bar",
                    description="Focus the browser address bar",
                    kind="hotkey",
                    value="ctrl+l",
                ),
                PlannedStep(
                    name="type_search_url",
                    description="Type the Google search URL into the address bar",
                    kind="type_text",
                    value=search_url,
                ),
                PlannedStep(
                    name="submit_search",
                    description="Submit the search",
                    kind="press_key",
                    value="enter",
                ),
                PlannedStep(
                    name="verify_search",
                    description="Verify that Google search results are visible",
                    kind="verify",
                    expected_terms=expected_terms or ["google"],
                    vision_prompt=f"Check whether this is a Google results page for {query}.",
                ),
            ]

        return []

    def _run_step(self, step: PlannedStep) -> tuple[bool, str, dict[str, Any] | None]:
        snapshot: dict[str, Any] | None = None
        for attempt in range(1, step.attempts + 1):
            if step.kind == "open_url":
                webbrowser.open(step.value or "", new=0)
                time.sleep(2.5)
            elif step.kind == "hotkey":
                keys = [key.strip() for key in (step.value or "").split("+") if key.strip()]
                self.app.run_guarded_desktop_action(step.description, lambda: hotkey(*keys))
                time.sleep(0.5)
            elif step.kind == "type_text":
                self.app.run_guarded_desktop_action(step.description, lambda: type_text(step.value or ""))
                time.sleep(0.5)
            elif step.kind == "press_key":
                self.app.run_guarded_desktop_action(step.description, lambda: press_key(step.value or "enter"))
                time.sleep(1.0)
            elif step.kind == "verify":
                pass

            if step.expected_terms or step.kind == "verify":
                ok, detail, snapshot = self._verify_step(step)
                if ok:
                    return True, detail, snapshot
            else:
                return True, f"Completed on attempt {attempt}.", None

        if snapshot is None and (step.expected_terms or step.kind == "verify"):
            _, detail, snapshot = self._verify_step(step)
            return False, detail, snapshot
        return False, f"Step failed after {step.attempts} attempts.", snapshot

    def _verify_step(self, step: PlannedStep) -> tuple[bool, str, dict[str, Any]]:
        snapshot = self.inspect(include_vision=False)
        title = snapshot.get("active_window", {}).get("title", "").lower()
        expected_terms = [term.lower() for term in (step.expected_terms or [])]
        title_ok, title_reason, title_conflict = self._verify_active_window_title(step, title, expected_terms)
        if title_ok:
            detail = f"Verified via active window title: {snapshot['active_window'].get('title', '')}"
            return True, detail, self._with_verification(
                snapshot,
                verified=True,
                source="active_window_title",
                reason=detail,
            )

        vision_ok = False
        vision_reason = ""
        if step.vision_prompt:
            snapshot = self.inspect(include_vision=True, vision_prompt=step.vision_prompt)
            analysis = snapshot.get("vision_analysis", "")
            vision_ok, vision_reason = self._verify_vision_analysis(step, analysis, expected_terms)
            if vision_ok and not title_conflict:
                detail = "Verified via screenshot analysis fallback."
                return True, detail, self._with_verification(
                    snapshot,
                    verified=True,
                    source="vision",
                    reason=detail,
                )

        detail = f"Could not verify step via UIA title"
        if step.vision_prompt:
            detail += " or screenshot analysis"
        detail += "."

        failure_reason = vision_reason or title_reason or detail
        if title_conflict and title_reason and vision_reason:
            failure_reason = f"{title_reason} {vision_reason}".strip()

        if title_conflict:
            failure_source = "active_window_title"
        elif vision_reason:
            failure_source = "vision"
        else:
            failure_source = "active_window_title" if title_reason else "vision"
        return False, detail, self._with_verification(
            snapshot,
            verified=False,
            source=failure_source,
            reason=failure_reason,
        )

    def inspect(self, include_vision: bool = False, vision_prompt: str | None = None) -> dict[str, Any]:
        active_window = get_active_window_title()
        if not active_window.get("ok") or not active_window.get("title"):
            active_window = get_active_window_basic()
        snapshot: dict[str, Any] = {
            "active_window": active_window,
            "focused_control": get_focused_control(),
            "visible_controls": list_visible_controls(max_depth=1),
        }

        if include_vision:
            screenshot = take_screenshot(self.app.settings["screenshots_dir"])
            snapshot["screenshot"] = screenshot
            if screenshot.get("ok"):
                self.app.logger.event("Vision", "UIA verification insufficient, using screenshot fallback")
                snapshot["vision_analysis"] = self.app.ollama.analyze_screenshot(
                    vision_prompt or "Describe this screenshot.",
                    screenshot["path"],
                )
        return snapshot

    def _verify_active_window_title(
        self,
        step: PlannedStep,
        title: str,
        expected_terms: list[str],
    ) -> tuple[bool, str, bool]:
        if not title:
            return False, "Active window title was unavailable.", False

        if self._is_browser_verification_step(step):
            if "discord" in title:
                return False, "Active window stayed on Discord instead of the expected browser page.", True
            browser_markers = ("google", "chrome", "edge", "firefox", "brave", "browser", "github")
            if not any(marker in title for marker in browser_markers):
                return False, "Active window title did not indicate the expected browser page.", True
            if expected_terms and all(term in title for term in expected_terms):
                return True, "Active window title confirmed the expected browser page.", False
            return False, "Active window title did not confirm the requested Google query.", False

        if expected_terms and all(term in title for term in expected_terms):
            return True, "Active window title confirmed the expected page.", False
        return False, "Active window title did not confirm the expected page.", False

    def _verify_vision_analysis(self, step: PlannedStep, analysis: str, expected_terms: list[str]) -> tuple[bool, str]:
        lowered = analysis.lower().strip()
        if not lowered:
            return False, "Vision returned no diagnostic text."
        if self._is_negative_vision_response(lowered):
            return False, f"Vision reported a mismatch: {analysis}"

        if self._is_browser_verification_step(step):
            positive_browser_markers = (
                "google results",
                "results page",
                "search results",
                "google search",
                "google homepage",
                "browser is showing",
            )
            if not any(marker in lowered for marker in positive_browser_markers):
                return False, "Vision did not positively confirm the expected Google page."
            if expected_terms and not all(term in lowered for term in expected_terms):
                return False, "Vision did not positively confirm the requested Google query."
            return True, "Vision positively confirmed the expected Google page."

        if expected_terms and all(term in lowered for term in expected_terms):
            return True, "Vision positively confirmed the expected page."
        return False, "Vision did not positively confirm the expected page."

    def _is_negative_vision_response(self, lowered: str) -> bool:
        negative_patterns = (
            r"^\s*no[,\s]",
            r"\bnot a google results page\b",
            r"\bnot the expected\b",
            r"\bdoes not show\b",
            r"\bnot showing\b",
            r"\bis not\b",
            r"\binstead\b",
            r"\brather than\b",
            r"\bmismatch\b",
        )
        return any(re.search(pattern, lowered) for pattern in negative_patterns)

    def _is_browser_verification_step(self, step: PlannedStep) -> bool:
        if step.name == "verify_search":
            return True
        text = f"{step.description} {step.vision_prompt or ''}".lower()
        return "google" in text or "browser" in text

    def _with_verification(
        self,
        snapshot: dict[str, Any],
        *,
        verified: bool,
        source: str,
        reason: str,
    ) -> dict[str, Any]:
        snapshot["verification"] = {
            "verified": verified,
            "verification_source": source,
            "reason": reason,
            "active_window_title": snapshot.get("active_window", {}).get("title", ""),
            "vision_summary": snapshot.get("vision_analysis", ""),
        }
        return snapshot

    def _result_verification(
        self,
        snapshot: dict[str, Any] | None,
        *,
        default_verified: bool,
        default_reason: str,
    ) -> dict[str, Any]:
        if snapshot is None:
            return {
                "verified": default_verified,
                "verification_source": "none",
                "reason": default_reason,
                "active_window_title": "",
                "vision_summary": "",
            }
        verification = snapshot.get("verification", {})
        return {
            "verified": verification.get("verified", default_verified),
            "verification_source": verification.get("verification_source", "none"),
            "reason": verification.get("reason", default_reason),
            "active_window_title": verification.get("active_window_title", snapshot.get("active_window", {}).get("title", "")),
            "vision_summary": verification.get("vision_summary", snapshot.get("vision_analysis", "")),
        }

    def _format_summary(
        self,
        step_results: list[dict[str, Any]],
        snapshot: dict[str, Any] | None,
        success: bool,
    ) -> str:
        lines = ["Desktop execution completed." if success else "Desktop execution stopped."]
        lines.append("")
        lines.append("Steps:")
        for step in step_results:
            marker = "ok" if step["ok"] else "failed"
            lines.append(f"- [{marker}] {step['step']}")
            lines.append(f"  {step['detail']}")
        if snapshot:
            title = snapshot.get("active_window", {}).get("title")
            if title:
                lines.append("")
                lines.append(f"Active window: {title}")
            vision = snapshot.get("vision_analysis")
            if vision:
                lines.append(f"Vision: {vision}")
            screenshot = snapshot.get("screenshot", {}).get("path")
            if screenshot:
                lines.append(f"Screenshot: {screenshot}")
        return "\n".join(lines)

    def _extract_search_query(self, text: str) -> str:
        lowered = text.lower()
        if "random thing" in lowered or "something random" in lowered:
            return random.choice(self.RANDOM_TOPICS)

        patterns = [
            r"search for (.+)",
            r"search (.+?) up on google",
            r"search (.+?) on google",
            r"search (.+?) in the browser",
            r"search (.+)",
        ]
        query = ""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                break
        if not query:
            return ""
        query = re.split(
            r"\s+(?:then|and)\s+(?:go to images|open images|save|download|copy|tell me)\b",
            query,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        query = re.split(
            r"\s+(?:up )?on google\b|\s+in the browser\b|\s+in browser\b|\s+using my pc\b|\s+on my pc\b",
            query,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        query = query.rstrip(" .")
        return query

    def _build_google_search_url(self, query: str, images: bool = False) -> str:
        base = "https://www.google.com/search"
        if images:
            return f"{base}?tbm=isch&q={quote_plus(query)}"
        return f"{base}?q={quote_plus(query)}"

    def _split_terms(self, text: str) -> list[str]:
        return [part for part in re.split(r"[^a-z0-9]+", text.lower()) if len(part) > 2]

    def _title_terms_for_url(self, url: str) -> list[str]:
        terms = self._split_terms(url)
        return terms[:3] or ["browser"]

    def _looks_like_browser_search_request(self, lowered: str) -> bool:
        return (
            "google" in lowered
            and any(word in lowered for word in ("search", "images", "image"))
        ) or ("browser" in lowered and "search" in lowered)

    def _needs_image_download_followup(self, text: str) -> bool:
        lowered = text.lower()
        return (
            any(word in lowered for word in ("save", "download", "copy"))
            and "image" in lowered
            and "folder" in lowered
        )
