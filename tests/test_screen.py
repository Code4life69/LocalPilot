import sys
from pathlib import Path
from types import SimpleNamespace

from app.tools.screen import latest_screenshot, take_screenshot


def test_take_screenshot_supports_lowercase_mss_factory(monkeypatch, tmp_path):
    output_dir = tmp_path / "screenshots"

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def shot(self, output):
            Path(output).write_text("fake screenshot", encoding="utf-8")

    fake_module = SimpleNamespace(mss=FakeSession)
    monkeypatch.setitem(sys.modules, "mss", fake_module)

    result = take_screenshot(str(output_dir))

    assert result["ok"] is True
    assert Path(result["path"]).exists()


def test_latest_screenshot_returns_most_recent_file(tmp_path):
    output_dir = tmp_path / "screenshots"
    output_dir.mkdir()
    older = output_dir / "screenshot_20260515_120000.png"
    newer = output_dir / "screenshot_20260515_120500.png"
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")

    result = latest_screenshot(output_dir)

    assert result == str(newer)


def test_latest_screenshot_returns_none_when_directory_missing(tmp_path):
    assert latest_screenshot(tmp_path / "missing") is None
