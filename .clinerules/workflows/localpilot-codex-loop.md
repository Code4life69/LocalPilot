# LocalPilot Codex Loop Workflow

Use this workflow for coding tasks in LocalPilot.

Goal:
Act like a local Codex-style coding agent:
inspect -> plan -> edit -> test -> fix -> report.

Steps:

1. Confirm location
- Confirm current directory is C:\LocalPilot.
- Run:
  git status

2. Understand task
- Restate the task in one short paragraph.
- Identify the likely files to inspect.
- Do not edit yet.

3. Inspect first
- Read the relevant files.
- Find the smallest safe place to make the change.
- Do not guess file structure.

4. Make a small plan
Report:
- files likely to edit
- what each edit will do
- tests to add/update
- risks

5. Edit
- Make the smallest focused code change.
- Do not rewrite whole files.
- Do not touch unrelated areas.
- Do not change model defaults, operating profiles, desktop automation, or safety unless specifically asked.

6. Test
Run:
.\.venv\Scripts\python.exe -m pytest

7. If tests fail
- Read the failure.
- Fix only the related issue.
- Rerun pytest.
- Stop if the failure is unrelated or risky.

8. Review diff
Run:
git diff

Check:
- no unrelated files changed
- no accidental rewrites
- no safety/model/desktop changes unless requested
- tests cover the new behavior

9. Final report
Include:
- files changed
- what changed
- pytest result
- git status
- known limitations
- whether the change is safe to commit

10. Commit
Do not commit unless the user explicitly says to commit.