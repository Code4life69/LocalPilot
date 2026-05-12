from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path
from typing import Any

from app.tools.ocr import get_ocr_backend_status


DEPENDENCIES = [
    {
        "label": "UI Automation",
        "package": "uiautomation",
        "module": "uiautomation",
        "fix": r".\.venv\Scripts\python.exe -m pip install uiautomation",
    },
    {
        "label": "PyAutoGUI",
        "package": "pyautogui",
        "module": "pyautogui",
        "fix": r".\.venv\Scripts\python.exe -m pip install pyautogui",
    },
    {
        "label": "MSS",
        "package": "mss",
        "module": "mss",
        "fix": r".\.venv\Scripts\python.exe -m pip install mss",
    },
    {
        "label": "Pillow",
        "package": "Pillow",
        "module": "PIL",
        "fix": r".\.venv\Scripts\python.exe -m pip install Pillow",
    },
    {
        "label": "Requests",
        "package": "requests",
        "module": "requests",
        "fix": r".\.venv\Scripts\python.exe -m pip install requests",
    },
    {
        "label": "DuckDuckGo Search",
        "package": "duckduckgo-search",
        "module": "duckduckgo_search",
        "fix": r".\.venv\Scripts\python.exe -m pip install duckduckgo-search",
    },
    {
        "label": "pytesseract",
        "package": "pytesseract",
        "module": "pytesseract",
        "fix": r".\.venv\Scripts\python.exe -m pip install pytesseract",
    },
]


def build_system_doctor_report(root_dir: str | Path, ollama_reachable: bool) -> str:
    root = Path(root_dir)
    expected_python = root / ".venv" / "Scripts" / "python.exe"
    current_python = Path(sys.executable)
    launcher_uses_venv = _launcher_uses_venv_python(root)
    dependencies = diagnose_dependencies()
    ocr_status = diagnose_ocr_backend()

    lines = [
        "System doctor",
        f"- Ollama reachable: {'yes' if ollama_reachable else 'no'}",
        f"- Python executable: {current_python}",
        f"- Expected LocalPilot venv: {expected_python}",
        f"- Using LocalPilot venv: {'yes' if _same_path(current_python, expected_python) else 'no'}",
        f"- Run LocalPilot.bat uses .venv python: {'yes' if launcher_uses_venv else 'no'}",
        "- Dependency checks:",
    ]

    for item in dependencies:
        lines.append(f"  - {item['package']}: {item['status']}")
        if item.get("detail"):
            lines.append(f"    detail: {item['detail']}")
        if item.get("fix"):
            lines.append(f"    fix: {item['fix']}")

    lines.append("- OCR backend:")
    lines.append(f"  - backend: {ocr_status['backend']}")
    lines.append(f"  - available: {'yes' if ocr_status['available'] else 'no'}")
    if ocr_status.get("tesseract_path"):
        lines.append(f"  - tesseract_path: {ocr_status['tesseract_path']}")
    if ocr_status.get("error"):
        lines.append(f"  - detail: {ocr_status['error']}")
    if ocr_status.get("install_hint"):
        lines.append(f"  - fix: {ocr_status['install_hint']}")

    if any(item["status"] == "dependency_missing" for item in dependencies):
        lines.extend(
            [
                "- Recommended next step:",
                r"  1. .\.venv\Scripts\python.exe -m pip install -r requirements.txt",
                r"  2. Restart LocalPilot using Run LocalPilot.bat",
            ]
        )

    return "\n".join(lines)


def diagnose_dependencies() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for dependency in DEPENDENCIES:
        package = dependency["package"]
        module_name = dependency["module"]
        result: dict[str, Any] = {"package": package, "module": module_name}
        try:
            importlib.import_module(module_name)
            result["status"] = "ok"
        except ModuleNotFoundError as exc:
            missing_target = exc.name or module_name
            if missing_target == module_name:
                result["status"] = "dependency_missing"
                result["detail"] = f"Missing import: {module_name}"
                result["fix"] = dependency["fix"]
            else:
                result["status"] = "runtime_import_error"
                result["detail"] = f"{module_name} failed because dependency {missing_target} is missing: {exc}"
        except Exception as exc:
            result["status"] = "runtime_import_error"
            result["detail"] = str(exc)
        results.append(result)
    return results


def dependency_missing_payload(module_name: str, package_name: str | None = None) -> dict[str, Any]:
    package = package_name or module_name
    fix = rf".\.venv\Scripts\python.exe -m pip install {package}"
    return {
        "ok": False,
        "reason": "dependency_missing",
        "dependency": package,
        "error": f"{module_name} is not installed in the active Python environment.",
        "fix": fix,
    }


def diagnose_ocr_backend() -> dict[str, Any]:
    result = dict(get_ocr_backend_status())
    if not result.get("tesseract_path"):
        discovered = shutil.which("tesseract")
        if discovered:
            result["tesseract_path"] = discovered
    return result


def _launcher_uses_venv_python(root_dir: Path) -> bool:
    launcher = root_dir / "Run LocalPilot.bat"
    if not launcher.exists():
        return False
    try:
        content = launcher.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return r".venv\Scripts\python.exe" in content


def _same_path(first: Path, second: Path) -> bool:
    try:
        return first.resolve() == second.resolve()
    except OSError:
        return str(first).lower() == str(second).lower()
