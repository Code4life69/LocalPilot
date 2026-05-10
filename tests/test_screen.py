import sys
from pathlib import Path
from types import SimpleNamespace

from app.tools.screen import take_screenshot


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
