# LocalPilot Run Tests Workflow

Use this workflow when asked to test or verify LocalPilot.

Steps:
1. Confirm current directory is C:\LocalPilot.
2. Run:
   .\.venv\Scripts\python.exe -m pytest
3. Capture:
   - pass/fail result
   - failed test names if any
   - error summary
   - warnings
4. If tests fail, inspect only the relevant files.
5. Fix only the issue directly related to the failure.
6. Rerun:
   .\.venv\Scripts\python.exe -m pytest
7. Stop when tests pass or a real blocker is found.

Final report:
- pytest result
- failures fixed
- files changed
- git status