from __future__ import annotations

import random
import re
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from app.tools.mouse_keyboard import hotkey, press_key, type_text
from app.tools.page_understanding import PageUnderstandingEngine


@dataclass
class PlannedStep:
    name: str
    description: str
    kind: str
    value: str | None = None
    expected_terms: list[str] | None = None
    vision_prompt: str | None = None
    attempts: int = 2
    metadata: dict[str, Any] | None = None


class DesktopExecutionFlow:
    DEFAULT_OWNER = "Code4life69"
    DEFAULT_REPO = "LocalPilot"
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
        self.page_understanding = PageUnderstandingEngine(app)

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
                self._record_failure_lesson(text, verification)
                status = verification.get("result", "failed")
                return {
                    "ok": status == "completed",
                    "content": self._format_summary(step_results, last_snapshot, status=status),
                    "steps": step_results,
                    **verification,
                }

        verification = self._result_verification(last_snapshot, default_verified=True, default_reason="Desktop execution completed.")
        status = verification.get("result", "completed")
        content = self._format_summary(step_results, last_snapshot, status=status)
        if self._needs_image_download_followup(text):
            content += (
                "\n\nNote: I opened and verified the Google Images search, but selecting, downloading, "
                "and copying a chosen browser image into a folder is not implemented yet. No image file was saved."
            )

        return {
            "ok": status == "completed",
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
            metadata = self._metadata_for_explicit_url(url)
            return [
                PlannedStep(
                    name="open_url",
                    description=f"Open {url} in the default browser",
                    kind="open_url",
                    value=url,
                    expected_terms=expected,
                    vision_prompt=f"Check whether the browser is showing {url}.",
                    metadata=metadata,
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
                    metadata={"page_type": "browser_home", "objective_kind": "page", "require_objective_match": False},
                )
            ]

        query = self._extract_search_query(text)
        github_target = self._extract_github_issue_target(text) or (self._extract_github_issue_target(query) if query else None)
        if github_target is not None:
            owner, repo, issue_number = github_target
            github_url = self._build_github_issue_url(owner, repo, issue_number)
            expected_terms = [owner.lower(), repo.lower(), f"issue {issue_number}"]
            return [
                PlannedStep(
                    name="open_github_issue",
                    description=f"Open GitHub issue #{issue_number} for {owner}/{repo}",
                    kind="open_url",
                    value=github_url,
                    expected_terms=expected_terms,
                    vision_prompt=f"Check whether this is GitHub issue #{issue_number} for {owner}/{repo}.",
                    metadata={
                        "page_type": "github_issue",
                        "objective_kind": "github_issue",
                        "require_objective_match": True,
                        "owner": owner,
                        "repo": repo,
                        "issue_number": issue_number,
                        "target_url": github_url,
                    },
                )
            ]

        if query:
            search_url = self._build_google_search_url(query, images="image" in lowered)
            expected_terms = [term for term in self._split_terms(query)[:3] if term]
            description = f"Open Google search results for {query!r}"
            if "image" in lowered:
                description += " in Images"
            return [
                PlannedStep(
                    name="open_search_results",
                    description=description,
                    kind="open_url",
                    value=search_url,
                    expected_terms=expected_terms or ["google"],
                    vision_prompt=f"Check whether this is a Google results page for {query}.",
                    metadata={
                        "page_type": "google_results",
                        "objective_kind": "generic_search",
                        "require_objective_match": False,
                        "query": query,
                    },
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
        evaluation = self._evaluate_step(step, snapshot)
        if evaluation["result"] == "completed":
            detail = evaluation["reason"]
            return True, detail, self._with_verification(snapshot, evaluation)

        if step.vision_prompt:
            snapshot = self.inspect(include_vision=True, vision_prompt=step.vision_prompt)
            evaluation = self._evaluate_step(step, snapshot)
            if evaluation["result"] == "completed":
                detail = evaluation["reason"]
                return True, detail, self._with_verification(snapshot, evaluation)

        detail = evaluation["reason"]
        return False, detail, self._with_verification(snapshot, evaluation)

    def inspect(self, include_vision: bool = False, vision_prompt: str | None = None) -> dict[str, Any]:
        if include_vision:
            self.app.logger.event("Vision", "UIA verification insufficient, using screenshot fallback")
        snapshot = self.page_understanding.snapshot(
            capture_screenshot=True,
            include_vision=include_vision,
            vision_prompt=vision_prompt,
        )
        if snapshot.get("vision_summary"):
            snapshot["vision_analysis"] = snapshot["vision_summary"]
        return snapshot

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
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        snapshot["verification"] = {
            "verified": evaluation["result"] == "completed",
            "verification_source": evaluation["verification_source"],
            "reason": evaluation["reason"],
            "active_window_title": snapshot.get("active_window", {}).get("title", ""),
            "vision_summary": snapshot.get("vision_analysis", ""),
            "page_state_confidence": evaluation["page_state_confidence"],
            "objective_match_confidence": evaluation["objective_match_confidence"],
            "page_verified": evaluation["page_verified"],
            "objective_verified": evaluation["objective_verified"],
            "result": evaluation["result"],
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
                "page_state_confidence": 0.0,
                "objective_match_confidence": 0.0,
                "page_verified": default_verified,
                "objective_verified": default_verified,
                "result": "completed" if default_verified else "failed",
            }
        verification = snapshot.get("verification", {})
        return {
            "verified": verification.get("verified", default_verified),
            "verification_source": verification.get("verification_source", "none"),
            "reason": verification.get("reason", default_reason),
            "active_window_title": verification.get("active_window_title", snapshot.get("active_window", {}).get("title", "")),
            "vision_summary": verification.get("vision_summary", snapshot.get("vision_analysis", "")),
            "page_state_confidence": verification.get("page_state_confidence", 0.0),
            "objective_match_confidence": verification.get("objective_match_confidence", 0.0),
            "page_verified": verification.get("page_verified", default_verified),
            "objective_verified": verification.get("objective_verified", default_verified),
            "result": verification.get("result", "completed" if default_verified else "failed"),
        }

    def _record_failure_lesson(self, task: str, verification: dict[str, Any]) -> None:
        lessons = getattr(self.app, "desktop_lessons", None)
        if lessons is None:
            return
        lessons.record(
            "verification_failure",
            task,
            verification.get("reason", "Desktop execution verification failed."),
            verification_source=verification.get("verification_source", "unknown"),
            active_window_title=verification.get("active_window_title", ""),
            vision_summary=verification.get("vision_summary", ""),
        )

    def _evaluate_step(self, step: PlannedStep, snapshot: dict[str, Any]) -> dict[str, Any]:
        metadata = step.metadata or {}
        page_type = metadata.get("page_type", "google_results" if self._is_browser_verification_step(step) else "page")
        objective_kind = metadata.get("objective_kind", "generic_search" if self._is_browser_verification_step(step) else "page")
        require_objective = bool(metadata.get("require_objective_match", False))
        title = snapshot.get("active_window", {}).get("title", "")
        title_lower = title.lower()
        ocr_text = snapshot.get("ocr_text", "").lower()
        vision_summary = snapshot.get("vision_analysis", "")
        vision_lower = vision_summary.lower()
        expected_terms = [term.lower() for term in (step.expected_terms or [])]

        if "discord" in title_lower:
            return self._verification_result(
                page_state_confidence=0.0,
                objective_match_confidence=0.0,
                page_verified=False,
                objective_verified=False,
                result="failed",
                source="active_window_title",
                reason="Active window stayed on Discord instead of the expected browser page.",
            )

        if self._is_negative_vision_response(vision_lower):
            return self._verification_result(
                page_state_confidence=0.0,
                objective_match_confidence=0.0,
                page_verified=False,
                objective_verified=False,
                result="failed",
                source="vision",
                reason=f"Vision reported a mismatch: {vision_summary}",
            )

        page_confidence, page_reason = self._page_state_confidence(page_type, metadata, title_lower, ocr_text, vision_lower, expected_terms)
        page_verified = page_confidence >= 0.75

        objective_confidence, objective_reason = self._objective_match_confidence(
            objective_kind,
            page_type,
            metadata,
            title_lower,
            ocr_text,
            vision_lower,
            expected_terms,
            page_verified,
        )
        objective_verified = objective_confidence >= 0.85 if require_objective else page_verified

        if require_objective and page_verified and not objective_verified:
            return self._verification_result(
                page_state_confidence=page_confidence,
                objective_match_confidence=objective_confidence,
                page_verified=True,
                objective_verified=False,
                result="partial",
                source="mixed",
                reason=objective_reason or "Search page opened, but I could not verify the correct target result.",
            )

        if page_verified and objective_verified:
            return self._verification_result(
                page_state_confidence=page_confidence,
                objective_match_confidence=objective_confidence if require_objective else page_confidence,
                page_verified=True,
                objective_verified=True,
                result="completed",
                source="active_window_title" if title else "vision",
                reason=objective_reason or page_reason or f"Verified via active window title: {title}",
            )

        return self._verification_result(
            page_state_confidence=page_confidence,
            objective_match_confidence=objective_confidence,
            page_verified=page_verified,
            objective_verified=False if require_objective else page_verified,
            result="failed",
            source="active_window_title" if title else "vision",
            reason=page_reason or objective_reason or "Could not verify the expected browser page.",
        )

    def _page_state_confidence(
        self,
        page_type: str,
        metadata: dict[str, Any],
        title_lower: str,
        ocr_text: str,
        vision_lower: str,
        expected_terms: list[str],
    ) -> tuple[float, str]:
        confidence = 0.0
        reason = ""
        if page_type == "google_results":
            if "google search" in title_lower and any(marker in title_lower for marker in ("chrome", "edge", "firefox", "brave")):
                confidence += 0.75
                reason = "Active window title confirmed a Google results page."
            elif any(marker in title_lower for marker in ("google", "chrome", "edge", "firefox", "brave")):
                confidence += 0.45
                reason = "Active window title indicated a browser page, but not confirmed Google results."
            if any(marker in ocr_text for marker in ("google", "google.com/search", "images", "search")):
                confidence += 0.10
            if any(marker in vision_lower for marker in ("google results", "search results", "google search", "google homepage")):
                confidence += 0.15
        elif page_type == "github_issue":
            repo = str(metadata.get("repo", "")).lower()
            owner = str(metadata.get("owner", "")).lower()
            issue_number = str(metadata.get("issue_number", "")).lower()
            if repo and repo in title_lower and "issue" in title_lower and issue_number and issue_number in title_lower:
                confidence += 0.75
                reason = "Active window title confirmed the GitHub issue page."
            elif repo and repo in title_lower and any(marker in title_lower for marker in ("github", "chrome", "edge", "firefox", "brave")):
                confidence += 0.50
                reason = "Active window title indicated a GitHub page, but not the exact issue."
            if owner and owner in ocr_text or repo and repo in ocr_text:
                confidence += 0.10
            if issue_number and f"issue {issue_number}" in ocr_text:
                confidence += 0.05
            if any(marker in vision_lower for marker in ("github issue", "issue page", "github page")):
                confidence += 0.15
        else:
            if title_lower:
                confidence += 0.50
                reason = "Active window title was available."
            if ocr_text:
                confidence += 0.10
            if vision_lower:
                confidence += 0.10
        return min(round(confidence, 2), 0.99), reason

    def _objective_match_confidence(
        self,
        objective_kind: str,
        page_type: str,
        metadata: dict[str, Any],
        title_lower: str,
        ocr_text: str,
        vision_lower: str,
        expected_terms: list[str],
        page_verified: bool,
    ) -> tuple[float, str]:
        if objective_kind == "github_issue":
            owner = str(metadata.get("owner", "")).lower()
            repo = str(metadata.get("repo", "")).lower()
            issue_number = str(metadata.get("issue_number", "")).lower()
            if self._contains_missing_marker(ocr_text, owner) or self._contains_missing_marker(vision_lower, owner):
                return 0.0, f"Search page opened, but Google reported Missing: {metadata.get('owner', owner)}."

            if page_type == "google_results" and "unrelated" in vision_lower:
                return 0.0, "Search page opened, but I could not verify the correct target result."

            confidence = 0.0
            if page_type == "github_issue":
                if repo and repo in title_lower and issue_number and issue_number in title_lower and "issue" in title_lower:
                    confidence += 0.65
                if owner and owner in title_lower:
                    confidence += 0.10
                if repo and repo in ocr_text:
                    confidence += 0.10
                if issue_number and (f"issue {issue_number}" in ocr_text or f"#{issue_number}" in ocr_text):
                    confidence += 0.10
                if all(term in vision_lower for term in (repo, issue_number)) and "github" in vision_lower:
                    confidence += 0.15
            else:
                if repo and repo in ocr_text:
                    confidence += 0.20
                if owner and owner in ocr_text:
                    confidence += 0.15
                if issue_number and (f"issue {issue_number}" in ocr_text or f"#{issue_number}" in ocr_text):
                    confidence += 0.20
                if f"github.com/{owner}/{repo}/issues/{issue_number}" in ocr_text:
                    confidence += 0.20
                if all(term in vision_lower for term in (repo, issue_number)) and ("github" in vision_lower or "issue" in vision_lower):
                    confidence += 0.20
                if owner and owner in vision_lower:
                    confidence += 0.10

            if confidence >= 0.85:
                return min(round(confidence, 2), 0.99), f"Verified the requested GitHub issue page for {metadata.get('owner')}/{metadata.get('repo')} issue #{metadata.get('issue_number')}."
            if page_verified:
                return min(round(confidence, 2), 0.99), "Search page opened, but I could not verify the correct target result."
            return min(round(confidence, 2), 0.99), "Could not verify the requested GitHub issue."

        if objective_kind == "generic_search":
            if page_verified:
                return 0.90, "Verified a valid search results page."
            return 0.0, "Could not verify the search results page."

        if page_verified:
            return 0.90, "Verified the requested page."
        return 0.0, "Could not verify the requested page."

    def _contains_missing_marker(self, text: str, term: str) -> bool:
        return bool(term and re.search(rf"missing:\s*{re.escape(term)}", text))

    def _verification_result(
        self,
        *,
        page_state_confidence: float,
        objective_match_confidence: float,
        page_verified: bool,
        objective_verified: bool,
        result: str,
        source: str,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "page_state_confidence": page_state_confidence,
            "objective_match_confidence": objective_match_confidence,
            "page_verified": page_verified,
            "objective_verified": objective_verified,
            "result": result,
            "verification_source": source,
            "reason": reason,
        }

    def _format_summary(
        self,
        step_results: list[dict[str, Any]],
        snapshot: dict[str, Any] | None,
        status: str,
    ) -> str:
        if status == "completed":
            heading = "Desktop execution completed."
        elif status == "partial":
            heading = "Desktop execution partially completed."
        else:
            heading = "Desktop execution stopped."
        lines = [heading]
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

    def _build_github_issue_url(self, owner: str, repo: str, issue_number: str) -> str:
        return f"https://github.com/{owner}/{repo}/issues/{issue_number}"

    def _split_terms(self, text: str) -> list[str]:
        return [part for part in re.split(r"[^a-z0-9]+", text.lower()) if len(part) > 2]

    def _title_terms_for_url(self, url: str) -> list[str]:
        terms = self._split_terms(url)
        return terms[:3] or ["browser"]

    def _metadata_for_explicit_url(self, url: str) -> dict[str, Any]:
        github_issue_match = re.search(
            r"github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/issues/(?P<number>\d+)",
            url,
            re.IGNORECASE,
        )
        if github_issue_match:
            return {
                "page_type": "github_issue",
                "objective_kind": "github_issue",
                "require_objective_match": True,
                "owner": github_issue_match.group("owner"),
                "repo": github_issue_match.group("repo"),
                "issue_number": github_issue_match.group("number"),
                "target_url": url,
            }
        return {"page_type": "browser_page", "objective_kind": "page", "require_objective_match": False}

    def _extract_github_issue_target(self, text: str) -> tuple[str, str, str] | None:
        if not text:
            return None

        slash_match = re.search(
            r"(?P<owner>[A-Za-z0-9_.-]+)\s*/\s*(?P<repo>[A-Za-z0-9_.-]+)\s+issue\s*#?(?P<number>\d+)",
            text,
            re.IGNORECASE,
        )
        if slash_match:
            return slash_match.group("owner"), slash_match.group("repo"), slash_match.group("number")

        spaced_match = re.search(
            r"(?P<owner>[A-Za-z0-9_.-]+)\s+(?P<repo>[A-Za-z0-9_.-]+)\s+issue\s*#?(?P<number>\d+)",
            text,
            re.IGNORECASE,
        )
        if spaced_match:
            return spaced_match.group("owner"), spaced_match.group("repo"), spaced_match.group("number")

        repo_only_match = re.search(
            r"issue\s*#?(?P<number>\d+)\s+(?:for|in)\s+(?P<repo>[A-Za-z0-9_.-]+)",
            text,
            re.IGNORECASE,
        )
        if repo_only_match:
            repo = repo_only_match.group("repo")
            owner, default_repo = self._known_repo_context()
            if repo.lower() == default_repo.lower():
                return owner, default_repo, repo_only_match.group("number")
        return None

    def _known_repo_context(self) -> tuple[str, str]:
        return self.DEFAULT_OWNER, self.DEFAULT_REPO

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
