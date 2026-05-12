from __future__ import annotations

from typing import Any
from pathlib import Path

from app.tools.ocr import read_image
from app.tools.screen import get_active_window_basic, get_mouse_position, take_screenshot
from app.tools.windows_ui import (
    get_active_window_title,
    get_control_at_point,
    get_focused_control,
    list_visible_controls,
)


BROWSER_MARKERS = ("chrome", "google", "edge", "firefox", "brave", "browser")
NEGATIVE_VISION_MARKERS = (
    "not present",
    "not visible",
    "not shown",
    "not on the page",
    "not a",
    "does not show",
    "doesn't show",
    "cannot see",
    "can't see",
    "mismatch",
)


class PageUnderstandingEngine:
    def __init__(self, app) -> None:
        self.app = app
        self.threshold = float(app.settings.get("page_understanding", {}).get("confidence_threshold", 0.85))
        self.debug_views_dir = Path(self.app.root_dir) / "workspace" / "debug_views"

    def snapshot(
        self,
        *,
        capture_screenshot: bool = True,
        include_vision: bool = False,
        vision_prompt: str | None = None,
        target_point: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        active_window = get_active_window_title()
        if not active_window.get("ok") or not active_window.get("title"):
            active_window = get_active_window_basic()
        focused_control = get_focused_control()
        visible_controls = list_visible_controls(max_depth=1)
        mouse_position = get_mouse_position()
        target_control = self._target_control(target_point, mouse_position)
        screenshot = take_screenshot(self.app.settings["screenshots_dir"]) if capture_screenshot else {"ok": False}

        snapshot: dict[str, Any] = {
            "ok": True,
            "active_window": active_window,
            "focused_control": focused_control,
            "visible_controls": visible_controls,
            "mouse_position": mouse_position,
            "target_control": target_control,
            "screenshot": screenshot,
            "screenshot_path": screenshot.get("path", ""),
            "candidate_targets": self._candidate_targets(focused_control, target_control, visible_controls),
            "ocr_available": False,
            "ocr_text": "",
            "ocr_blocks": [],
            "ocr_backend": "",
            "ocr_confidence": 0.0,
        }

        if screenshot.get("ok"):
            ocr_result = read_image(screenshot["path"], output_dir=self.debug_views_dir)
            snapshot["ocr_available"] = bool(ocr_result.get("ok"))
            snapshot["ocr_text"] = ocr_result.get("text", "")
            snapshot["ocr_blocks"] = ocr_result.get("blocks", [])
            snapshot["ocr_backend"] = ocr_result.get("backend", "")
            snapshot["ocr_confidence"] = float(ocr_result.get("confidence", 0.0) or 0.0)
            snapshot["ocr_error"] = ocr_result.get("error", "")
            snapshot["ocr_install_hint"] = ocr_result.get("install_hint", "")
            snapshot["ocr_processed_image"] = ocr_result.get("processed_image", "")

        if include_vision and screenshot.get("ok"):
            prompt = vision_prompt or "Describe this page in one short paragraph."
            snapshot["vision_summary"] = self.app.ollama.analyze_screenshot(prompt, screenshot["path"])
        else:
            snapshot["vision_summary"] = ""

        return snapshot

    def assess(
        self,
        *,
        action_kind: str = "inspect",
        action_text: str = "",
        target_point: tuple[int, int] | None = None,
        include_vision: bool = False,
        vision_prompt: str | None = None,
    ) -> dict[str, Any]:
        snapshot = self.snapshot(
            capture_screenshot=True,
            include_vision=include_vision,
            vision_prompt=vision_prompt,
            target_point=target_point,
        )
        title = snapshot.get("active_window", {}).get("title", "")
        browser_like = any(marker in title.lower() for marker in BROWSER_MARKERS)
        score = 0.0
        reasons: list[str] = []
        blocking_reasons: list[str] = []

        if title:
            score += 0.25
            reasons.append("active window title available")
        else:
            blocking_reasons.append("active window title unavailable")

        if browser_like:
            score += 0.10
            reasons.append("active window looks browser-related")

        focused = snapshot.get("focused_control", {})
        if focused.get("ok"):
            score += 0.20
            reasons.append("focused control available")
            if focused.get("bounds"):
                score += 0.10
                reasons.append("focused control has bounds")
        elif focused.get("reason") == "dependency_missing":
            blocking_reasons.append("UI Automation dependency missing for focused control")
        else:
            blocking_reasons.append("focused control unavailable")

        visible_controls = snapshot.get("visible_controls", {})
        visible_list = visible_controls.get("controls", []) if visible_controls.get("ok") else []
        if visible_list:
            score += 0.15
            reasons.append(f"{len(visible_list)} visible controls detected")
        elif visible_controls.get("reason") == "dependency_missing":
            blocking_reasons.append("UI Automation dependency missing for visible controls")

        if snapshot.get("screenshot_path"):
            score += 0.05
            reasons.append("screenshot captured")

        vision_summary = snapshot.get("vision_summary", "")
        if vision_summary:
            if self._vision_negative(vision_summary):
                blocking_reasons.append("vision says the target or page is not present")
            else:
                reasons.append("vision did not report a mismatch")

        target_control = snapshot.get("target_control", {})
        ocr_text = snapshot.get("ocr_text", "").lower()
        expected_terms = self._expected_terms(action_text)
        if snapshot.get("ocr_available") and ocr_text:
            matched_terms = [term for term in expected_terms if term in ocr_text]
            if matched_terms:
                score += 0.10 if browser_like else 0.05
                reasons.append(f"OCR matched expected text: {', '.join(matched_terms[:3])}")
            elif browser_like and expected_terms:
                score = max(0.0, score - 0.05)
                reasons.append("OCR did not confirm the expected browser text")

        if action_kind == "click":
            if target_control.get("ok") and target_control.get("bounds"):
                score += 0.20
                reasons.append("click target has UIA bounds")
            else:
                blocking_reasons.append("no target bounds exist for the requested click")
        elif action_kind in {"type_text", "hotkey"}:
            if focused.get("ok") and focused.get("bounds"):
                score += 0.15
                reasons.append("text target has focused UIA bounds")
            else:
                blocking_reasons.append("no focused control bounds for typing/hotkey action")

        if action_kind in {"click", "type_text", "hotkey"} and not browser_like and "browser" in action_text.lower():
            blocking_reasons.append("active window is unrelated to the requested browser action")

        score = min(round(score, 2), 0.99)
        allowed = score >= self.threshold and not blocking_reasons
        if allowed:
            primary_reason = "Confidence threshold met."
        elif action_kind == "click" and "no target bounds exist for the requested click" in blocking_reasons:
            primary_reason = "no target bounds exist for the requested click"
        elif action_kind in {"type_text", "hotkey"} and "no focused control bounds for typing/hotkey action" in blocking_reasons:
            primary_reason = "no focused control bounds for typing/hotkey action"
        else:
            primary_reason = blocking_reasons[0] if blocking_reasons else f"confidence {score:.2f} is below threshold {self.threshold:.2f}"

        snapshot.update(
            {
                "confidence_score": score,
                "confidence_threshold": self.threshold,
                "confidence_allowed": allowed,
                "confidence_reason": primary_reason,
                "confidence_evidence": reasons,
                "confidence_blocks": blocking_reasons,
                "browser_like": browser_like,
                "candidate_targets_count": len(snapshot["candidate_targets"]),
            }
        )
        return snapshot

    def post_action_verification(
        self,
        before: dict[str, Any],
        *,
        action_kind: str,
        action_text: str,
        target_point: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        after = self.assess(
            action_kind="inspect",
            action_text=action_text,
            target_point=target_point,
            include_vision=bool(before.get("browser_like")),
            vision_prompt="Describe whether the page still shows the expected target after the action.",
        )
        after_title = after.get("active_window", {}).get("title", "")
        if not after_title:
            verified = False
            reason = "Post-action verification failed because the active window title was unavailable."
            source = "active_window_title"
        elif before.get("browser_like") and not after.get("browser_like"):
            verified = False
            reason = "Post-action verification failed because the active window is no longer browser-related."
            source = "active_window_title"
        elif after.get("vision_summary") and self._vision_negative(after["vision_summary"]):
            verified = False
            reason = "Post-action verification failed because vision reported the expected page or target is not present."
            source = "vision"
        else:
            verified = True
            reason = f"Post-action verification captured a valid page snapshot after {action_kind}."
            source = "page_snapshot"

        after["verification"] = {
            "verified": verified,
            "verification_source": source,
            "reason": reason,
            "active_window_title": after_title,
            "vision_summary": after.get("vision_summary", ""),
        }
        return after

    def render(self, snapshot: dict[str, Any], *, heading: str = "Page understanding") -> str:
        active_title = snapshot.get("active_window", {}).get("title") or "Unknown"
        focused_summary = self._control_summary(snapshot.get("focused_control", {}))
        mouse = snapshot.get("mouse_position", {})
        target_summary = self._control_summary(snapshot.get("target_control", {}))
        visible_controls = snapshot.get("visible_controls", {})
        visible_count = len(visible_controls.get("controls", [])) if visible_controls.get("ok") else 0
        lines = [
            heading,
            f"- Active window: {active_title}",
            f"- Focused control: {focused_summary}",
            f"- Mouse position: ({mouse.get('x', 'unknown')}, {mouse.get('y', 'unknown')})",
            f"- Target under mouse: {target_summary}",
            f"- Visible controls: {visible_count}",
            f"- Screenshot: {snapshot.get('screenshot_path') or 'not captured'}",
            f"- OCR backend: {snapshot.get('ocr_backend') or 'unavailable'}",
            f"- OCR available: {'yes' if snapshot.get('ocr_available') else 'no'}",
            f"- Candidate targets: {snapshot.get('candidate_targets_count', len(snapshot.get('candidate_targets', [])))}",
            f"- Confidence: {snapshot.get('confidence_score', 0.0):.2f} / {snapshot.get('confidence_threshold', self.threshold):.2f}",
            f"- Confidence result: {'allowed' if snapshot.get('confidence_allowed') else 'refused'}",
            f"- Reason: {snapshot.get('confidence_reason', 'n/a')}",
        ]
        if snapshot.get("ocr_text"):
            lines.append(f"- OCR text: {snapshot['ocr_text'][:240]}")
        elif snapshot.get("ocr_error"):
            lines.append(f"- OCR status: {snapshot['ocr_error']}")
        if snapshot.get("vision_summary"):
            lines.append(f"- Vision summary: {snapshot['vision_summary']}")
        candidates = snapshot.get("candidate_targets", [])
        if candidates:
            lines.append("- Candidate targets:")
            for candidate in candidates[:5]:
                lines.append(
                    f"  - {candidate.get('source')}: {candidate.get('control_type') or 'Control'}: {candidate.get('name') or '(unnamed)'}"
                )
        return "\n".join(lines)

    def build_refusal_payload(self, snapshot: dict[str, Any], *, action_name: str) -> dict[str, Any]:
        reason = snapshot.get("confidence_reason", f"confidence {snapshot.get('confidence_score', 0.0):.2f} is below threshold")
        return {
            "ok": False,
            "error": f"Refused {action_name}: {reason}",
            "content": self.render(snapshot, heading="Page understanding gate"),
            "verified": False,
            "verification_source": "confidence_gate",
            "reason": reason,
            "active_window_title": snapshot.get("active_window", {}).get("title", ""),
            "vision_summary": snapshot.get("vision_summary", ""),
            "confidence_score": snapshot.get("confidence_score", 0.0),
            "confidence_threshold": snapshot.get("confidence_threshold", self.threshold),
            "candidate_targets": snapshot.get("candidate_targets", []),
            "screenshot_path": snapshot.get("screenshot_path", ""),
            "ocr_available": snapshot.get("ocr_available", False),
            "ocr_text": snapshot.get("ocr_text", ""),
            "ocr_backend": snapshot.get("ocr_backend", ""),
        }

    def _target_control(self, target_point: tuple[int, int] | None, mouse_position: dict[str, Any]) -> dict[str, Any]:
        if target_point is not None:
            return get_control_at_point(target_point[0], target_point[1])
        if mouse_position.get("ok"):
            return get_control_at_point(mouse_position["x"], mouse_position["y"])
        return {"ok": False, "error": "Target point unavailable."}

    def _candidate_targets(
        self,
        focused_control: dict[str, Any],
        target_control: dict[str, Any],
        visible_controls: dict[str, Any],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        def add_candidate(source: str, payload: dict[str, Any]) -> None:
            if not payload.get("ok"):
                return
            bounds = payload.get("bounds")
            if not bounds:
                return
            key = (
                payload.get("control_type", ""),
                payload.get("name", ""),
                str(bounds),
            )
            if key in seen:
                return
            seen.add(key)
            candidates.append(
                {
                    "source": source,
                    "name": payload.get("name", ""),
                    "control_type": payload.get("control_type", ""),
                    "bounds": bounds,
                }
            )

        add_candidate("focused", focused_control)
        add_candidate("under_mouse", target_control)
        if visible_controls.get("ok"):
            for control in visible_controls.get("controls", [])[:10]:
                add_candidate("visible", {"ok": True, **control})
        return candidates

    def _vision_negative(self, text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in NEGATIVE_VISION_MARKERS)

    def _expected_terms(self, text: str) -> list[str]:
        stopwords = {
            "click",
            "type",
            "text",
            "browser",
            "window",
            "page",
            "screen",
            "google",
            "desktop",
            "mouse",
            "cursor",
            "hotkey",
            "current",
            "position",
            "press",
            "button",
            "field",
            "control",
            "search",
            "open",
            "the",
            "for",
            "and",
            "with",
            "this",
            "that",
        }
        terms: list[str] = []
        for part in text.lower().replace('"', " ").replace("'", " ").split():
            cleaned = "".join(ch for ch in part if ch.isalnum())
            if len(cleaned) <= 2 or cleaned in stopwords:
                continue
            if cleaned not in terms:
                terms.append(cleaned)
        return terms[:6]

    def _control_summary(self, payload: dict[str, Any]) -> str:
        if not payload.get("ok"):
            if payload.get("reason") == "dependency_missing":
                return f"Unavailable (dependency_missing: {payload.get('dependency')})"
            return f"Unavailable ({payload.get('error', 'unknown reason')})"
        return f"{payload.get('control_type') or 'Control'}: {payload.get('name') or '(unnamed)'}"
