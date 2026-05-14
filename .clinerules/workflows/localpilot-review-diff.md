# LocalPilot Review Diff Workflow

Use this workflow before committing or after making code changes.

Steps:
1. Run:
   git status
2. Run:
   git diff
3. Review the diff for:
   - unrelated changes
   - model default changes
   - operating profile changes
   - desktop automation changes
   - safety/approval behavior changes
   - large accidental rewrites
4. If unrelated changes exist, stop and ask before continuing.
5. If the diff is focused and tests pass, summarize it.

Final report:
- changed files
- whether changes are focused
- whether any risky areas were touched
- pytest result if already run
- whether it is safe to commit