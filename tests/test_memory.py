from app.memory import MemoryStore


def test_notes_and_capabilities(tmp_path):
    memory_dir = tmp_path / "memory"
    manifest = tmp_path / "capabilities.json"
    manifest.write_text('{"name": "LocalPilot", "modes": ["chat"]}', encoding="utf-8")

    store = MemoryStore(memory_dir, manifest)
    assert store.load_capabilities()["name"] == "LocalPilot"
    assert store.save_note("remember this") == "Note saved."
    assert "remember this" in store.show_notes()
    assert store.search_notes("remember")
