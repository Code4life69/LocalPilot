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


def test_search_notes_skips_headers_and_normalizes_bullets(tmp_path):
    memory_dir = tmp_path / "memory"
    manifest = tmp_path / "capabilities.json"
    manifest.write_text('{"name": "LocalPilot", "modes": ["chat"]}', encoding="utf-8")

    store = MemoryStore(memory_dir, manifest)
    store.notes_path.write_text(
        "# LocalPilot Notes\n\n- smoke test note\n* LocalPilot helper note\nplain LocalPilot line\n",
        encoding="utf-8",
    )

    assert store.search_notes("LocalPilot") == ["LocalPilot helper note", "plain LocalPilot line"]
    assert store.search_notes("smoke") == ["smoke test note"]
