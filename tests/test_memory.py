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


def test_search_notes_deduplicates_matches(tmp_path):
    memory_dir = tmp_path / "memory"
    manifest = tmp_path / "capabilities.json"
    manifest.write_text('{"name": "LocalPilot", "modes": ["chat"]}', encoding="utf-8")

    store = MemoryStore(memory_dir, manifest)
    store.notes_path.write_text(
        "# LocalPilot Notes\n\n- duplicate note\n- duplicate note\n- Duplicate Note\n",
        encoding="utf-8",
    )

    assert store.search_notes("duplicate") == ["duplicate note"]


def test_current_task_save_load_and_clear(tmp_path):
    memory_dir = tmp_path / "memory"
    manifest = tmp_path / "capabilities.json"
    manifest.write_text('{"name": "LocalPilot", "modes": ["chat"]}', encoding="utf-8")

    store = MemoryStore(memory_dir, manifest)
    store.save_current_task(
        {
            "active_task_id": "task123",
            "original_user_task": "build a website",
            "latest_user_message": "continue",
            "mode": "agent",
            "status": "active",
        }
    )

    current = store.load_current_task()

    assert current is not None
    assert current["active_task_id"] == "task123"
    assert store.clear_current_task() == "Current task cleared."
    assert store.load_current_task() is None


def test_update_current_task_clamps_recent_memory_lists(tmp_path):
    memory_dir = tmp_path / "memory"
    manifest = tmp_path / "capabilities.json"
    manifest.write_text('{"name": "LocalPilot", "modes": ["chat"]}', encoding="utf-8")

    store = MemoryStore(memory_dir, manifest)
    store.save_current_task({"active_task_id": "task123", "original_user_task": "build a website", "mode": "agent", "status": "active"})
    store.update_current_task(
        recent_step_summaries=[f"step {index}" for index in range(10)],
        recent_tool_calls=[{"tool": f"tool_{index}", "args": {}} for index in range(10)],
        recent_tool_result_summaries=[f"result {index}" for index in range(10)],
        recent_messages=[{"role": "user", "content": f"message {index}"} for index in range(12)],
    )

    current = store.load_current_task()

    assert current["recent_step_summaries"] == [f"step {index}" for index in range(4, 10)]
    assert len(current["recent_tool_calls"]) == 5
    assert current["recent_tool_result_summaries"] == [f"result {index}" for index in range(5, 10)]
    assert len(current["recent_messages"]) == 8
