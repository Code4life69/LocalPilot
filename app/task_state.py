from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_TASK_STATE: dict[str, Any] = {
    "current_goal": "",
    "current_plan": [],
    "active_mode": "chat",
    "active_model": "",
    "page_state": {},
    "objective_state": {},
    "build_state": {},
    "research_state": {},
    "last_action": "",
    "last_failure": "",
    "safety_constraints": {},
    "confidence_score": None,
    "files_changed": [],
    "tests_run": [],
    "next_recommended_action": "",
    "operating_profile": "reliable_stack",
}


class TaskStateStore:
    def __init__(
        self,
        path: str | Path,
        safety_constraints: dict[str, Any] | None = None,
        event_callback=None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state = deepcopy(DEFAULT_TASK_STATE)
        self._state["safety_constraints"] = dict(safety_constraints or {})
        self._event_callback = event_callback
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._state.update(loaded)
            except Exception:
                pass
        self._persist()

    def _persist(self) -> None:
        self.path.write_text(json.dumps(self._state, indent=2, ensure_ascii=True), encoding="utf-8")

    def _emit(self, action: str, **extra: Any) -> None:
        if self._event_callback is None:
            return
        try:
            self._event_callback("TaskState", action, **extra)
        except Exception:
            return

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def update(self, **fields: Any) -> dict[str, Any]:
        for key, value in fields.items():
            if value is not None:
                self._state[key] = value
        self._persist()
        self._emit("update", fields=list(fields.keys()))
        return self.snapshot()

    def reset_for_new_goal(self, goal: str, active_mode: str, active_model: str) -> dict[str, Any]:
        safety_constraints = deepcopy(self._state.get("safety_constraints", {}))
        operating_profile = self._state.get("operating_profile", DEFAULT_TASK_STATE["operating_profile"])
        self._state = deepcopy(DEFAULT_TASK_STATE)
        self._state["safety_constraints"] = safety_constraints
        self._state["operating_profile"] = operating_profile
        self._state["current_goal"] = goal
        self._state["active_mode"] = active_mode
        self._state["active_model"] = active_model
        self._persist()
        self._emit("reset", current_goal=goal, active_mode=active_mode, active_model=active_model)
        return self.snapshot()

    def merge_nested(self, key: str, values: dict[str, Any]) -> dict[str, Any]:
        current = self._state.get(key)
        if not isinstance(current, dict):
            current = {}
        current.update(values)
        self._state[key] = current
        self._persist()
        self._emit("merge", key=key, fields=list(values.keys()))
        return self.snapshot()
