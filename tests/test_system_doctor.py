import builtins
from pathlib import Path

from app.system_doctor import build_system_doctor_report, diagnose_dependencies
from app.tools import windows_ui


def test_system_doctor_reports_missing_uiautomation(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "uiautomation":
            raise ModuleNotFoundError("No module named 'uiautomation'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    report = build_system_doctor_report(tmp_path, ollama_reachable=False)

    assert "System doctor" in report
    assert "uiautomation: dependency_missing" in report
    assert r".\.venv\Scripts\python.exe -m pip install uiautomation" in report


def test_diagnose_dependencies_returns_missing_status(monkeypatch):
    import importlib

    def fake_import_module(name):
        if name == "uiautomation":
            raise ModuleNotFoundError("No module named 'uiautomation'")
        return object()

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    results = diagnose_dependencies()
    uia = next(item for item in results if item["module"] == "uiautomation")

    assert uia["status"] == "dependency_missing"
    assert "fix" in uia


def test_windows_ui_dependency_missing_payload(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "uiautomation":
            raise ModuleNotFoundError("No module named 'uiautomation'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = windows_ui.get_focused_control()

    assert result["ok"] is False
    assert result["reason"] == "dependency_missing"
    assert result["dependency"] == "uiautomation"
