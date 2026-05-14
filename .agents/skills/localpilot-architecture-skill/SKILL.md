---
name: localpilot-architecture-skill
description: Brief description of what this skill does
---

# localpilot-architecture-skill

Instructions for the AI agent...

## Usage

Describe when and how to use this skill.

## Steps
# LocalPilot Architecture Skill

LocalPilot is a local Codex-plus desktop agent.

Core goals:
- Code like a local Codex-style agent.
- Use the PC safely with approval.
- See the screen with vision/OCR/UI Automation.
- Verify objectives before claiming success.
- Avoid false success.
- Keep runtime/private files out of Git.

Important architecture:
- app/main.py handles GUI and app-level request lifecycle.
- app/modes/chat_mode.py handles chat commands and normal chat.
- app/modes/code_mode.py handles code/app/site generation.
- app/modes/desktop_mode.py handles desktop actions and observation.
- app/modes/research_mode.py handles research.
- app/task_state.py stores shared runtime task state.
- config/operating_profiles.json defines reliable_stack, quality_max, and one-model modes.

Model policy:
- Do not change model defaults unless explicitly asked.
- reliable_stack should remain available.
- quality_max is optional for high-quality/heavy tasks.
- qwen2.5-coder is used for coding.
- qwen2.5vl or qwen3-vl is used for vision/page understanding.
- gemma4:31b is a heavy reviewer/writing model.

Safety policy:
- Do not remove approval gates.
- Do not bypass safety.
- Do not auto-submit forms.
- Do not blindly click.
- Do not claim success unless verification proves it.

Coding policy:
- Keep changes focused.
- Add tests for new behavior.
- Run .\.venv\Scripts\python.exe -m pytest.
- Do not use global Python.
- Do not commit unless explicitly told.
1. First step
2. Second step
3. Third step
