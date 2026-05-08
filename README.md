# LocalPilot

LocalPilot is a Windows-first local AI assistant starter project. It runs on your machine with Ollama, keeps its memory in plain local files, exposes explicit operating modes, and uses Python tools for file work, shell access, web lookup, screenshots, and guarded desktop control.

The first version is intentionally boring: no giant agent framework, no autonomous background loops, no vector database, and no self-modifying core logic. It is built to be easy to inspect, test, and extend.

## Features

- CLI chat entrypoint: `python localpilot.py`
- Optional Tkinter GUI with live activity timeline
- Explicit modes: chat, code, research, desktop, memory
- Local Ollama integration for text and placeholder vision
- Keyword router instead of opaque autonomous planning
- File and shell tools with confirmation gates
- DuckDuckGo research with 5-result cap
- Plain-file notes and learned facts memory
- Windows UI Automation before screenshot vision
- Structured JSONL logging plus readable text logs

## Installation

```powershell
cd C:\LocalPilot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen3:30b
ollama pull qwen2.5-vl:7b
python localpilot.py
```

## Required Ollama Models

- Main reasoning / chat model: `qwen3:30b`
- Vision model: `qwen2.5-vl:7b`

LocalPilot keeps reasoning/chat and visual analysis separate by default:

- `qwen3:30b` handles planning, coding, chat, and tool decisions
- `qwen2.5-vl:7b` is reserved for screenshots and visual inspection

LocalPilot expects a reachable Ollama API, typically at `http://127.0.0.1:11434`.

## How To Run

```powershell
cd C:\LocalPilot
python localpilot.py
```

By default LocalPilot starts the CLI and tries to open the GUI alongside it. If Tkinter is unavailable, the CLI still runs.

For double-click launch on Windows, use [Run LocalPilot.bat](</C:/LocalPilot/Run LocalPilot.bat>). It will:

- create `.venv` automatically on first run if it does not exist
- install `requirements.txt` into that virtual environment
- use `.venv\Scripts\python.exe` for all normal launches
- keep the window open if startup fails so the error stays visible

## Safety Rules

- File overwrite requires approval.
- File move into an existing target requires approval.
- Shell commands require approval.
- Dangerous shell commands are blocked by default.
- Desktop click, type, and hotkey actions require approval.
- Windows UI Automation is preferred over screenshots.
- Vision is only used when desktop inspection needs it.
- The assistant should not rewrite its own core logic without user approval.

## Project Structure

```text
C:\LocalPilot
  README.md
  .gitignore
  requirements.txt
  localpilot.py
  config\
    settings.json
    model_profiles.json
    capabilities.json
  app\
    __init__.py
    main.py
    router.py
    safety.py
    memory.py
    logger.py
  app\llm\
    __init__.py
    ollama_client.py
    prompts.py
  app\tools\
    __init__.py
    files.py
    shell.py
    web.py
    screen.py
    mouse_keyboard.py
    windows_ui.py
  app\modes\
    __init__.py
    chat_mode.py
    code_mode.py
    research_mode.py
    desktop_mode.py
  memory\
    notes.md
    learned_facts.json
  logs\
    .gitkeep
  workspace\
    .gitkeep
  tests\
    test_safety.py
    test_files.py
    test_router.py
    test_memory.py
```

## What Works In v1

- Terminal chat loop
- GUI status window with role and mode activity
- Keyword routing
- File listing, read, write, append, mkdir, copy, move
- Shell execution with approval and dangerous-command blocking
- Web search with capped results
- Notes save/search/show
- Screenshot capture and mouse position
- Basic active window and UI Automation inspection
- Vision entrypoint placeholder with graceful failure path

## Current TODO

- More reliable natural-language parsing for code and desktop commands
- Richer GUI controls for approvals and history filtering
- Stronger Windows UI Automation control actions
- Better structured learned facts updates
- Full multimodal Ollama verification for `qwen2.5-vl:7b`

## First Roadmap

1. Stabilize the existing tool interfaces and tests.
2. Improve command parsing without turning the app into one giant loop.
3. Add better memory indexing while staying file-based.
4. Expand Windows UI Automation read-only inspection.
5. Make vision analysis production-ready once the local Ollama multimodal path is verified.

## GitHub

Target repository: `https://github.com/Code4life69/LocalPilot`

If `gh` is not installed, create the repository manually on GitHub as a public repo named `LocalPilot`, then run:

```powershell
cd C:\LocalPilot
git init
git add .
git commit -m "Initial LocalPilot starter"
git branch -M main
git remote add origin https://github.com/Code4life69/LocalPilot.git
git push -u origin main
```
