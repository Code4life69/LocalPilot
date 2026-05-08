from pathlib import Path
from types import SimpleNamespace

from app.modes.chat_mode import ChatMode


def test_trust_checklist_command_reads_gauntlet_doc(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "TRUST_GAUNTLET.md").write_text("# LocalPilot Trust Gauntlet\n\n- Test 1\n", encoding="utf-8")

    app = SimpleNamespace(root_dir=tmp_path, capabilities={"name": "LocalPilot", "modes": ["chat"]})
    mode = ChatMode(app)
    result = mode.handle({"user_text": "trust checklist"})

    assert result["ok"]
    assert "# LocalPilot Trust Gauntlet" in result["message"]
