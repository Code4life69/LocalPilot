# LocalPilot Small Fix Workflow

Use this workflow for small focused LocalPilot code changes.

Steps:
1. Confirm current directory is C:\LocalPilot.
2. Run git status.
3. Inspect the relevant files before editing.
4. Identify the smallest safe change.
5. Edit only the files needed for the task.
6. Do not touch model defaults, operating profiles, desktop automation, or safety behavior unless the task specifically asks for it.
7. Add or update tests for the change.
8. Run:
   .\.venv\Scripts\python.exe -m pytest
9. If tests fail, fix only the related issue and rerun pytest.
10. Do not claim done until tests pass.
11. Do not commit unless explicitly told to commit.

Final report must include:
- files changed
- what changed
- pytest result
- git status