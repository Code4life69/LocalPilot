# LocalPilot AI Agent Lessons

This file records lessons learned from using local AI coding agents with Cline on LocalPilot.

The goal is to help the agent avoid repeating mistakes.

## Core verification lessons

1. Do not claim tests passed unless the terminal output clearly shows pytest passed.

Correct proof looks like:
- "156 passed"
- "0 failed"
- pytest exits successfully

If pytest fails, say it failed and show the failure summary.

2. Do not say the working tree is clean unless git status actually says it is clean.

Always verify with:
    git status

3. Do not claim a task is complete just because files were edited.

A task is complete only when:
- requested files changed correctly
- tests passed
- git diff is reviewed
- final report matches reality

4. Do not run setup scripts unless the task explicitly asks.

Avoid these unless directly requested:
- scripts/install_recommended_models.ps1
- scripts/configure_ollama_rtx3060.ps1
- scripts/check_models.ps1
- Run LocalPilot.bat

5. Do not install MCP servers unless the user explicitly asks.

Do not work on Playwright MCP, Git MCP, browser MCP setup, or npm global installs unless the task is specifically about that.

## Patch and editing lessons

6. Prefer small edits over rewriting whole files.

If a search/replace patch fails:
- re-read the exact file section
- make a smaller patch
- do not keep retrying blindly
- do not replace the whole file unless explicitly approved

7. When editing Python tests, indentation matters.

Bad:
    def test_example():
        assert "one" in result
    assert "two" in result

Good:
    def test_example():
        assert "one" in result
        assert "two" in result

8. Do not leave old assertions behind after changing expected behavior.

If a model policy changes from Qwen/Granite to Gemma, remove old assertions that still expect:
- qwen2.5-coder
- qwen3-vl
- granite3.3
- qwen2.5vl
- gemma4:31b

unless the test is specifically checking old or optional behavior.

9. Do not weaken tests just to make them pass.

Fix the real mismatch.
Do not delete important tests unless the user explicitly approves.

10. If tests fail after two fix attempts, stop and explain.

Report:
- failing test name
- exact failure
- file and line
- what was already tried
- likely root cause

Do not keep guessing.

## LocalPilot-specific lessons

11. LocalPilot uses the project venv.

Use:
    .\.venv\Scripts\python.exe -m pytest

Do not use global Python unless explicitly asked.

12. To start a batch file from PowerShell, use the correct syntax.

Correct:
    & ".\Run LocalPilot.bat"

or:
    cmd /c "Run LocalPilot.bat"

Incorrect:
    Run LocalPilot.bat
    powershell -File Run LocalPilot.bat

13. Model role changes usually touch these files:
- config/model_profiles.json
- config/operating_profiles.json
- config/capabilities.json
- tests/test_ollama_client.py

Do not guess other files unless tests or code inspection prove they are needed.

14. Current temporary model policy:

All LLM roles use:
    gemma4:e4b

Embedding remains:
    nomic-embed-text

Reason:
Gemma is not an embedding model.

15. Safety and desktop automation are protected areas.

Do not touch these unless the task specifically asks:
- desktop automation
- approval dialogs
- guarded actions
- destructive command safety
- browser control safety
- safety refusal paths

## Debugging process

16. When fixing an error, follow this order:

1. Read the exact error.
2. Identify the failing file and line.
3. Read the relevant code.
4. Make one small fix.
5. Run the smallest relevant test if possible.
6. Run full pytest before final completion.
7. Review git diff.
8. Report honestly.

17. Do not hide errors with broad try/except.

Bad:
    try:
        do_work()
    except Exception:
        pass

Use clear error handling only when it is part of the correct behavior.

18. For config changes, verify both tests and runtime status.

Use:
    .\.venv\Scripts\python.exe -m pytest
    .\.venv\Scripts\python.exe localpilot.py --model-status

19. Always report exact changed files.

Use:
    git diff --stat
    git status

20. If the user asks for exact changes, do not improvise.

Apply exactly what was requested.
If it seems wrong, stop and ask before changing it.
