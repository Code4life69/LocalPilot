import json
from pathlib import Path

from app.task_state import TaskStateStore


def test_task_state_store_initializes_runtime_file(tmp_path):
    state_path = tmp_path / "workspace" / "runtime" / "task_state.json"
    store = TaskStateStore(state_path, safety_constraints={"desktop_requires_confirmation": True})

    assert state_path.exists()
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["active_mode"] == "chat"
    assert data["operating_profile"] == "reliable_stack"
    assert data["safety_constraints"]["desktop_requires_confirmation"] is True


def test_task_state_store_updates_fields(tmp_path):
    store = TaskStateStore(tmp_path / "workspace" / "runtime" / "task_state.json")

    store.update(current_goal="build app", active_mode="code", files_changed=["main.py"])
    snapshot = store.snapshot()

    assert snapshot["current_goal"] == "build app"
    assert snapshot["active_mode"] == "code"
    assert snapshot["files_changed"] == ["main.py"]
