# LocalPilot Codex-Plus Architecture Plan

## 1. Recommended Architecture

LocalPilot should be a layered local agent system:

1. `Intent + Safety Layer`
   - router
   - destructive-action refusal
   - approval gates
   - operating profile selection

2. `Shared State Layer`
   - runtime `task_state`
   - request history
   - current plan
   - current verification status
   - last failure
   - files changed
   - tests run

3. `Execution Layer`
   - code/file actions
   - test runner
   - desktop actions
   - page understanding
   - OCR
   - browser verification

4. `Verification Layer`
   - syntax/static checks
   - test results
   - page-state verification
   - objective completion verification
   - confidence gating
   - launch verification

5. `Reflection Layer`
   - self-review
   - failure lessons
   - next recommended action
   - bounded retry/improvement loop

The control rule should stay:

`plan -> act -> verify -> reflect -> continue or stop`

## 2. Modules To Add

Recommended core modules:

- `app/agent_loop.py`
  - orchestrates multi-step task execution
  - reads/writes task state every step

- `app/planner.py`
  - creates bounded plans
  - separates code tasks, desktop tasks, research tasks, and mixed tasks

- `app/verifier.py`
  - central verification hub
  - unifies tests, file checks, page verification, and objective verification

- `app/context_manager.py`
  - summarizes large task histories
  - preserves current goal, active plan, blockers, and verified facts

- `app/tools/test_runner.py`
  - safe project verification runner

- `app/tools/browser_inspector.py`
  - future Playwright inspection wrapper

- `app/tools/objective_verifier.py`
  - explicit goal-completion verification

- `app/tools/action_recovery.py`
  - retry logic for safe recoverable failures

## 3. Implementation Order

1. Shared task state
2. Test runner
3. Central verifier
4. Unified agent loop
5. Better plan representation
6. Playwright inspection
7. Approved browser form filling
8. Job helper on top of the stable browser/verification stack

## 4. Safety Gates

Keep these hard gates:

- destructive local actions blocked by safety unless a future explicit safe workflow exists
- no auto-submit of forms
- no clicking/typing below confidence threshold
- no arbitrary shell commands without approval
- no writing outside workspace unless explicitly approved
- no package installs without approval
- no running downloaded code without approval

Additional safety gates to add:

- `action_scope`
  - what app/window/site is allowed right now

- `goal_scope`
  - what exact objective is being verified

- `stop_conditions`
  - stop on unrelated active window
  - stop on negative verification
  - stop on repeated failed retries

## 5. Model Policy

Use specialized models by default unless benchmarks prove a simpler stack is better.

Recommended policy:

- `reliable_stack`
  - default baseline
  - best safety and predictable behavior

- `quality_max`
  - explicit best-results mode
  - allowed to be slower/heavier

- `one_model`
  - benchmark experiment only
  - not default until it beats the baseline on real tasks

Model roles:

- `main`
  - planning, reasoning, summaries, self-review

- `coder`
  - code generation, file edits, debugging, fixes

- `vision`
  - screenshot analysis, page understanding, verification support

- `router`
  - optional fast intent classification later

- `ocr`
  - support signal only, never sole click authority

## 6. Context Compaction Strategy

LocalPilot should never rely on raw conversation history alone.

Keep:

- current goal
- current plan
- what has been verified
- what has failed
- files changed
- tests run
- current page state
- next recommended action

Compact away:

- long repetitive logs
- repeated stdout/stderr
- stale screenshots
- old intermediate reasoning

Mechanism:

- summarize after each major phase
- keep raw artifacts on disk
- keep compact summaries in `task_state`
- point to file paths for large outputs instead of inlining them

## 7. Test Runner Design

The test runner should:

- use only the project venv by default
- run in a worker thread
- stream stdout/stderr into logs and GUI
- record:
  - command
  - exit code
  - duration
  - summary
  - timestamp
- support cancellation
- write the final result into `task_state`

Future extensions:

- named verification presets
  - `pytest`
  - `python -m py_compile`
  - app launch checks
  - website static checks

## 8. Page / Objective Verification Design

Keep page verification and objective verification separate.

`page_state_confidence`
- answers: am I on the expected page type?

`objective_match_confidence`
- answers: did I reach the exact target?

Signals:

- active window title
- UI Automation
- OCR text
- vision summary
- browser DOM inspection later

Rules:

- page pass is not objective pass
- negative vision or contradictory OCR should reduce confidence
- unrelated window should fail immediately
- OCR text alone cannot authorize clicking

## 9. Professional Build Loop Design

Every serious build should follow:

1. brief
2. acceptance checklist
3. first implementation
4. verification
5. self-review
6. fix pass
7. re-verification
8. stop only when checks pass or bounded stop condition triggers

The build loop should use:

- coder for implementation
- main/reviewer for critique
- test runner for verification
- launch verification where safe

## 10. Playwright Integration Plan

Playwright should be added only as a browser inspection and approved action layer.

Phase 1:

- open page
- inspect URL/title
- read DOM text
- inspect forms/buttons/labels
- take deterministic browser screenshots

Phase 2:

- suggest field mapping
- compare DOM text with OCR/UIA/vision
- support approved field filling

Phase 3:

- verified multi-step browser workflows

Never:

- auto-submit forms
- continue after failed objective verification

## 11. Job Helper Plan

The job helper should remain an optional skill pack, not the app identity.

Version 1:

- local profile
- saved answers
- tracker CSV
- cover letters
- application question drafts

Version 2:

- browser page inspection for job forms
- approved field suggestions
- approved field filling only

Blocked until browser verification is mature:

- auto-filled multi-step applications
- submit flows

## 12. Benchmark Scoring System

Benchmark complete operating modes, not just isolated model prompts.

Benchmark categories:

- coding task quality
- self-correction quality
- professional build quality
- screenshot understanding
- page verification
- objective completion
- OCR/form understanding
- safety refusal correctness
- tool-following
- speed as secondary metric

Scoring rule:

- safety regressions are disqualifying
- false success claims are heavily penalized
- quality wins only matter if verification stays reliable

## 13. What Should Be Built Next

Recommended next steps:

1. central verifier module
2. unified bounded agent loop
3. richer plan objects in task state
4. Playwright inspection only
5. browser-field mapping with approval
6. job helper v1 on top of that stable foundation

## 14. What Should Stay Out Of Scope For Now

Keep these out for now:

- auto-submit browser forms
- uncontrolled self-modifying logic
- autonomous long-running background agents
- more random model hunting
- package auto-installation
- destructive local automation
- OCR-only clicking
- blind browser navigation without objective verification

## Recommended Immediate Direction

The next architecture move that gets LocalPilot closest to a real Codex-plus local PC agent is:

1. finish the safe test runner and central verification story
2. unify code/desktop/research around one agent loop
3. add Playwright inspection as a verification source
4. only then build approved browser workflows and job-form assistance
