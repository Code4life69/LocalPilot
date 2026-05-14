# LocalPilot Cline Agent Skill

Purpose:
Help Cline behave like a careful local Codex-style coding agent for LocalPilot.

Cline role:
- Plan carefully.
- Inspect files before editing.
- Make small focused changes.
- Use tools to read files and run tests.
- Do not guess about repo structure.
- Do not claim done until verification passes.

LocalPilot project:
- Root folder: C:\LocalPilot
- Use project Python only:
  .\.venv\Scripts\python.exe
- Do not use global Python.
- Main GUI/app lifecycle is usually in app/main.py.
- Chat commands usually live in app/modes/chat_mode.py.
- Code/app generation usually lives in app/modes/code_mode.py.
- Desktop behavior usually lives in app/modes/desktop_mode.py.
- Shared task state lives in app/task_state.py.
- Tests live in tests/.

Safe command habits:
Use these commands when appropriate:
- git status
- git diff
- .\.venv\Scripts\python.exe -m pytest
- .\.venv\Scripts\python.exe localpilot.py --model-status
- .\.venv\Scripts\python.exe localpilot.py --task-state

Do not run:
- del /s
- rmdir /s
- format
- powershell destructive commands
- pip install unless explicitly approved
- git reset --hard unless explicitly approved
- git clean unless explicitly approved

Patch/edit discipline:
- Prefer small edits.
- Do not rewrite whole files unless absolutely necessary.
- If search/replace fails, do not keep trying blindly.
- Re-read the exact file section.
- Make a smaller patch.
- If still failing, explain the exact intended change before editing.

Testing discipline:
- After code changes, run:
  .\.venv\Scripts\python.exe -m pytest
- If tests fail, inspect only the related failure.
- Fix the smallest related issue.
- Rerun pytest.
- Do not expand scope.

Final report must include:
- files changed
- what changed
- pytest result
- git status
- known limitations
- whether it is safe to commit

Never change these unless explicitly asked:
- model defaults
- operating profiles
- safety/approval behavior
- desktop automation behavior
- ignored runtime files