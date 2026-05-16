from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.lmstudio_client import LMStudioClient
from app.safety import RISK_DANGEROUS, RISK_MEDIUM


LOW_CONFIDENCE_THRESHOLD = 0.60
WARNING_CONFIDENCE_THRESHOLD = 0.80
SENSITIVE_DESKTOP_PATTERNS = (
    r"\bpassword\b",
    r"\blog in\b",
    r"\blogin\b",
    r"\bsign in\b",
    r"\bcheckout\b",
    r"\bpayment\b",
    r"\bbuy\b",
    r"\bpurchase\b",
    r"\bsend\b",
    r"\bemail\b",
    r"\bmessage\b",
    r"\bdelete\b",
    r"\bconfirm\b",
    r"\bsettings\b",
    r"\badmin\b",
    r"\buac\b",
)


def _import_pyautogui():
    import pyautogui  # type: ignore

    return pyautogui


def get_screen_size(pyautogui_module: Any | None = None) -> dict[str, Any]:
    try:
        pyautogui = pyautogui_module or _import_pyautogui()
    except ImportError as exc:
        return {"ok": False, "error": f"pyautogui not installed: {exc}"}
    width, height = pyautogui.size()
    return {"ok": True, "width": int(width), "height": int(height)}


def get_mouse_position(pyautogui_module: Any | None = None) -> dict[str, Any]:
    try:
        pyautogui = pyautogui_module or _import_pyautogui()
    except ImportError as exc:
        return {"ok": False, "error": f"pyautogui not installed: {exc}"}
    x, y = pyautogui.position()
    return {"ok": True, "x": int(x), "y": int(y)}


def move_mouse_preview(
    x: int,
    y: int,
    *,
    target: str = "",
    confidence: float | None = None,
    duration_seconds: float = 0.15,
    pyautogui_module: Any | None = None,
) -> dict[str, Any]:
    try:
        pyautogui = pyautogui_module or _import_pyautogui()
    except ImportError as exc:
        return {"ok": False, "error": f"pyautogui not installed: {exc}"}
    width, height = pyautogui.size()
    target_x = int(x)
    target_y = int(y)
    if target_x < 0 or target_y < 0 or target_x >= int(width) or target_y >= int(height):
        return {
            "ok": False,
            "error": f"Preview coordinates are outside the screen bounds: ({target_x}, {target_y}) not within {width}x{height}.",
        }
    pyautogui.moveTo(target_x, target_y, duration=max(float(duration_seconds), 0.0))
    return {
        "ok": True,
        "x": target_x,
        "y": target_y,
        "target": target,
        "confidence": confidence,
        "preview_only": True,
        "message": "Mouse moved for preview only. No click was performed.",
    }


def suggest_action_from_screenshot(
    screenshot_path: str | Path,
    instruction: str,
    lmstudio_client: LMStudioClient,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    image_path = Path(screenshot_path)
    if not image_path.exists():
        return {"ok": False, "error": f"Screenshot not found: {image_path}"}
    if not instruction.strip():
        return {"ok": False, "error": "instruction is required for desktop_suggest_action."}

    prompt = (
        "You are analyzing a Windows desktop screenshot for a local AI agent.\n"
        "Return strict JSON only. Do not wrap it in markdown.\n"
        "You may suggest a next desktop action, but do not assume it will be executed.\n"
        "The JSON must include: action, target, x, y, confidence, risk, reason.\n"
        "Use confidence from 0.00 to 1.00.\n"
        "If the visible target appears sensitive, set risk to dangerous.\n"
        "Sensitive examples include: password fields, login submission, payment or checkout, email send buttons, delete confirmations, system settings, or admin prompts.\n"
        "Task instruction:\n"
        f"{instruction.strip()}\n"
        "Return only a single JSON object."
    )

    try:
        raw_response = lmstudio_client.chat_vision(
            prompt=prompt,
            image_path=image_path,
            model=model or lmstudio_client.default_vision_model,
            max_tokens=512,
        )
    except Exception as exc:
        return {"ok": False, "error": f"Vision model failed during desktop_suggest_action: {exc}"}

    try:
        suggestion = _parse_suggestion_response(raw_response)
    except ValueError as exc:
        return {"ok": False, "error": f"desktop_suggest_action could not parse model JSON: {exc}"}

    confidence = suggestion["confidence"]
    sensitive_context = _is_sensitive_desktop_context(f"{suggestion['target']} {suggestion['reason']}")
    risk = suggestion["risk"]
    warning = ""
    next_step = "wait_for_explicit_approval"
    can_preview = True
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        warning = "Confidence is below 0.60. Ask the user for clarification before any desktop execution."
        next_step = "ask_for_clarification"
        can_preview = False
    elif confidence < WARNING_CONFIDENCE_THRESHOLD:
        warning = "Confidence is moderate. Any later desktop execution should include an extra warning and explicit approval."
    if sensitive_context:
        risk = RISK_DANGEROUS
        warning = "Sensitive desktop context detected. Do not execute this action in this milestone."
        next_step = "do_not_execute"
        can_preview = False

    desktop_action = {
        "type": "desktop_action",
        "mode": "dry_run",
        "action": suggestion["action"],
        "x": suggestion["x"],
        "y": suggestion["y"],
        "target": suggestion["target"],
        "confidence": confidence,
        "approved": False,
    }
    return {
        "ok": True,
        "action": suggestion["action"],
        "target": suggestion["target"],
        "x": suggestion["x"],
        "y": suggestion["y"],
        "confidence": confidence,
        "risk": risk,
        "reason": suggestion["reason"],
        "requires_approval_to_execute": True,
        "executed": False,
        "can_execute": False,
        "can_preview_move": can_preview,
        "next_step": next_step,
        "warning": warning,
        "sensitive_context": sensitive_context,
        "desktop_action": desktop_action,
    }


def _parse_suggestion_response(raw_response: str) -> dict[str, Any]:
    payload = _extract_json_object(raw_response)
    required_fields = {"action", "target", "x", "y", "confidence", "risk", "reason"}
    missing = required_fields.difference(payload)
    if missing:
        raise ValueError(f"missing fields: {', '.join(sorted(missing))}")
    try:
        x = int(round(float(payload["x"])))
        y = int(round(float(payload["y"])))
        confidence = float(payload["confidence"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid coordinate or confidence values: {exc}") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    action = str(payload["action"]).strip()
    target = str(payload["target"]).strip()
    reason = str(payload["reason"]).strip()
    if not action or not target or not reason:
        raise ValueError("action, target, and reason must be non-empty strings")
    risk = _normalize_risk(str(payload["risk"]).strip().lower())
    return {
        "action": action,
        "target": target,
        "x": x,
        "y": y,
        "confidence": confidence,
        "risk": risk,
        "reason": reason,
    }


def _extract_json_object(raw_response: str) -> dict[str, Any]:
    stripped = raw_response.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in model response")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("model response JSON must be an object")
    return payload


def _normalize_risk(risk: str) -> str:
    if risk in {"safe", RISK_MEDIUM, RISK_DANGEROUS, "blocked"}:
        return risk
    return RISK_MEDIUM


def _is_sensitive_desktop_context(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in SENSITIVE_DESKTOP_PATTERNS)
