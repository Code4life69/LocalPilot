import json
from pathlib import Path
import subprocess

from app.browser_tool import BrowserToolBridge


def test_browser_bridge_builds_command_payload(tmp_path):
    root = tmp_path / "repo"
    browser = root / "browser"
    browser.mkdir(parents=True)
    (browser / "browser_server.js").write_text("// stub", encoding="utf-8")
    bridge = BrowserToolBridge(root)

    payload = bridge._build_payload("goto_url", url="https://example.com")

    assert payload == {"action": "goto_url", "url": "https://example.com"}


def test_browser_bridge_handles_successful_json(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    browser = root / "browser"
    browser.mkdir(parents=True)
    (browser / "browser_server.js").write_text("// stub", encoding="utf-8")
    bridge = BrowserToolBridge(root)

    class FakeCompleted:
        stdout = json.dumps({"ok": True, "action": "get_page_info", "title": "Example"})
        stderr = ""

    monkeypatch.setattr("app.browser_tool.subprocess.run", lambda *args, **kwargs: FakeCompleted())

    result = bridge.run("get_page_info")

    assert result["ok"] is True
    assert result["title"] == "Example"


def test_browser_bridge_handles_invalid_json_output(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    browser = root / "browser"
    browser.mkdir(parents=True)
    (browser / "browser_server.js").write_text("// stub", encoding="utf-8")
    bridge = BrowserToolBridge(root)

    class FakeCompleted:
        stdout = "not-json"
        stderr = ""

    monkeypatch.setattr("app.browser_tool.subprocess.run", lambda *args, **kwargs: FakeCompleted())

    result = bridge.run("get_page_info")

    assert result["ok"] is False
    assert "invalid JSON" in result["error"]


def test_browser_bridge_reports_missing_script(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    bridge = BrowserToolBridge(root)

    result = bridge.run("launch_browser")

    assert result["ok"] is False
    assert "Browser bridge script not found" in result["error"]


def test_browser_server_discovery_prefers_env_var(tmp_path):
    script = Path("browser/browser_server.js").resolve()
    fake_browser = tmp_path / "chrome.exe"
    fake_browser.write_text("", encoding="utf-8")
    command = (
        f"const mod=require({json.dumps(str(script))});"
        f"const result=mod.discoverBrowserExecutable({{LOCALPILOT_BROWSER_EXECUTABLE:{json.dumps(str(fake_browser))}}}, []);"
        "console.log(JSON.stringify(result));"
    )
    completed = subprocess.run(["node", "-e", command], capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout)

    assert result["ok"] is True
    assert result["path"] == str(fake_browser)
    assert result["source"] == "LOCALPILOT_BROWSER_EXECUTABLE"


def test_browser_server_reports_clean_error_when_no_executable_found():
    script = Path("browser/browser_server.js").resolve()
    command = (
        f"const mod=require({json.dumps(str(script))});"
        "const result=mod.discoverBrowserExecutable({}, ['Z:/missing/chrome.exe']);"
        "console.log(JSON.stringify(mod.buildBrowserExecutableError('launch_browser', result)));"
    )
    completed = subprocess.run(["node", "-e", command], capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout)

    assert result["ok"] is False
    assert result["action"] == "launch_browser"
    assert "LOCALPILOT_BROWSER_EXECUTABLE" in result["error"]
    assert result["checked_paths"] == ["Z:/missing/chrome.exe"]
