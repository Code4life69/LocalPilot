from pathlib import Path

from app.modes.code_mode import APP_TEMPLATES, CodeMode


class DummyLogger:
    def event(self, *args, **kwargs):
        return None


class DummySafety:
    def requires_write_confirmation(self, path):
        return Path(path).exists()

    def requires_move_confirmation(self, destination):
        return Path(destination).exists()

    def is_command_blocked(self, command):
        return False


class DummyApp:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.logger = DummyLogger()
        self.safety = DummySafety()
        self.task_state = type(
            "TaskStateStub",
            (),
            {
                "snapshot": lambda self: {},
                "update": lambda self, **kwargs: kwargs,
            },
        )()
        self.settings = {
            "professional_build": {
                "enabled": True,
                "max_passes": 3,
                "allow_web_research": True,
                "require_acceptance_checklist": True,
                "stop_on_failed_verification": True,
                "launch_verification_enabled": True,
                "launch_timeout_seconds": 8,
            }
        }
        self.system_prompt = "system prompt"
        self.ollama = type(
            "StubOllama",
            (),
            {
                "chat_with_role": lambda self, role, system_prompt, user_text: f"{role}|{user_text}",
            },
        )()
        self.start_project_tests = lambda: {"ok": True, "status": "started", "message": "Started tests."}
        self.cancel_project_tests = lambda: {"ok": True, "message": "Test cancellation requested."}

    def ask_approval(self, prompt):
        return True

    def resolve_runtime_model_for_role(self, role):
        return {"main": "qwen3:8b", "coder": "qwen2.5-coder:14b-instruct-q3_K_M"}.get(role, "")


def test_notepad_request_without_explicit_folder_creates_default_project(tmp_path):
    app = DummyApp(tmp_path)
    mode = CodeMode(app)

    result = mode.handle(
        {
            "user_text": "make me a notepad app with a gui and double click starter"
        }
    )

    assert result["ok"]
    project_path = Path(result["project_path"])
    assert project_path.exists()
    assert project_path.parent == tmp_path / "workspace" / "generated_apps"
    assert (project_path / "main.py").exists()
    assert (project_path / "Run Notepad.bat").exists()
    assert (project_path / "README.txt").exists()
    assert result["verification"]["syntax_verified"] is True
    assert "Double-click Run Notepad.bat" in result["message"]


def test_website_request_creates_expected_static_files(tmp_path):
    app = DummyApp(tmp_path)
    mode = CodeMode(app)

    result = mode.handle(
        {
            "user_text": "I want you to make me a website locally on my pc in a new folder make it basic"
        }
    )

    assert result["ok"]
    project_path = Path(result["project_path"])
    assert project_path.name.startswith("Website_")
    assert (project_path / "index.html").exists()
    assert (project_path / "style.css").exists()
    assert (project_path / "script.js").exists()
    assert (project_path / "Run Website.bat").exists()
    assert (project_path / "README.txt").exists()
    assert result["verification"]["static_files_verified"] is True
    assert "Double-click Run Website.bat" in result["message"]
    assert "Website type:" in result["message"]
    assert "Open it by double-clicking Run Website.bat." in result["message"]


def test_prompt_aware_website_generator_creates_different_site_content(tmp_path):
    app = DummyApp(tmp_path)
    mode = CodeMode(app)

    lawn_result = mode.handle({"user_text": "make me a local website for a lawn care business"})
    portfolio_result = mode.handle({"user_text": "make me a portfolio website for my coding projects"})
    ai_result = mode.handle({"user_text": "make me a dark futuristic website for an AI assistant"})

    assert lawn_result["ok"]
    assert portfolio_result["ok"]
    assert ai_result["ok"]

    lawn_path = Path(lawn_result["project_path"])
    portfolio_path = Path(portfolio_result["project_path"])
    ai_path = Path(ai_result["project_path"])

    lawn_html = (lawn_path / "index.html").read_text(encoding="utf-8")
    portfolio_html = (portfolio_path / "index.html").read_text(encoding="utf-8")
    ai_html = (ai_path / "index.html").read_text(encoding="utf-8")

    lawn_css = (lawn_path / "style.css").read_text(encoding="utf-8")
    portfolio_css = (portfolio_path / "style.css").read_text(encoding="utf-8")
    ai_css = (ai_path / "style.css").read_text(encoding="utf-8")

    assert "FreshCut Lawn Care" in lawn_html
    assert "Request A Quote" in lawn_html
    assert "Local Business Site" in lawn_result["message"]
    assert "#2f9e44" in lawn_css

    assert "Project Portfolio" in portfolio_html
    assert "View My Projects" in portfolio_html
    assert "Portfolio" in portfolio_result["message"]
    assert "#2563eb" in portfolio_css

    assert "NeonPilot AI" in ai_html
    assert "Enter The Workflow" in ai_html
    assert "Dark" in ai_result["message"]
    assert "#61dafb" in ai_css

    assert lawn_html != portfolio_html
    assert portfolio_html != ai_html
    assert lawn_css != ai_css


def test_generated_app_verification_uses_latest_matching_folder(tmp_path):
    app = DummyApp(tmp_path)
    mode = CodeMode(app)

    create_result = mode.handle({"user_text": "make me a timer app with a gui and double click starter"})
    assert create_result["ok"]

    verify_result = mode.handle({"user_text": "verify the generated timer app files and tell me how to run it"})
    assert verify_result["ok"]
    assert "Run Timer.bat" in verify_result["message"]
    assert verify_result["verification"]["syntax_verified"] is True


def test_supported_app_kind_detection():
    mode = CodeMode(DummyApp(Path(".")))
    assert mode._detect_supported_app_kind("make me a basic website") == "website"
    assert mode._detect_supported_app_kind("make me a todo list app") == "todo"
    assert mode._detect_supported_app_kind("make me a notepad app") == "notepad"
    assert mode._detect_supported_app_kind("make me a timer app") == "timer"
    assert mode._detect_supported_app_kind("make me a calculator app") == "calculator"
    assert mode._detect_supported_app_kind("make me a helper script tool") == "script"


def test_app_scaffold_request_requires_real_app_language():
    mode = CodeMode(DummyApp(Path(".")))
    assert not mode._is_app_scaffold_request("write a note in memory")
    assert mode._is_app_scaffold_request("make me a calculator app with gui and double click starter")


def test_all_templates_have_required_files():
    for template in APP_TEMPLATES.values():
        assert template["main_filename"] in {"main.py", "index.html"}
        assert template["launcher_name"].endswith(".bat")
        assert template["readme_name"] == "README.txt"


def test_code_mode_uses_coder_role_for_llm_fallback(tmp_path):
    app = DummyApp(tmp_path)
    mode = CodeMode(app)

    result = mode.handle({"user_text": "explain this code bug"})

    assert result["ok"]
    assert result["message"] == "coder|explain this code bug"


def test_website_verification_checks_asset_links(tmp_path):
    app = DummyApp(tmp_path)
    mode = CodeMode(app)

    result = mode.handle({"user_text": "make me a product landing page for LocalPilot"})
    project_path = Path(result["project_path"])
    index_path = project_path / "index.html"
    broken_html = index_path.read_text(encoding="utf-8").replace('src="script.js"', "")
    index_path.write_text(broken_html, encoding="utf-8")

    verification = mode._verify_app_outputs(project_path, "website")

    assert verification["ok"] is False
    assert "script.js link missing" in verification["error"]


def test_natural_language_file_creation_creates_verified_workspace_file(tmp_path):
    app = DummyApp(tmp_path)
    mode = CodeMode(app)

    result = mode.handle(
        {
            "user_text": "create a text file in workspace named trust_test.txt that says LocalPilot file test"
        }
    )

    target = tmp_path / "workspace" / "trust_test.txt"
    assert result["ok"]
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "LocalPilot file test"
    assert result["path"] == str(target)
    assert "Created file:" in result["message"]


def test_natural_language_file_creation_asks_before_overwrite(tmp_path):
    app = DummyApp(tmp_path)
    app.ask_approval = lambda prompt: False
    mode = CodeMode(app)
    target = tmp_path / "workspace" / "notes.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("existing", encoding="utf-8")

    result = mode.handle(
        {
            "user_text": "make a text file called notes.txt in workspace with hello world"
        }
    )

    assert result["ok"] is False
    assert result["error"] == "Write cancelled by user."
    assert target.read_text(encoding="utf-8") == "existing"


def test_professional_build_creates_acceptance_checklist_and_verifies(tmp_path, monkeypatch):
    app = DummyApp(tmp_path)
    mode = CodeMode(app)
    monkeypatch.setattr(
        mode,
        "_verify_launch_readiness",
        lambda project_path, app_kind, settings: {
            "status": "passed",
            "passed": True,
            "reason": "Launch verification passed in test.",
            "stdout": "",
            "stderr": "",
            "window_title": APP_TEMPLATES[app_kind]["display_name"],
            "cleanup_performed": True,
        },
    )

    result = mode.handle({"user_text": "professional build make me a notepad app with a gui and double click starter"})

    assert result["ok"] is True
    assert result["status"] == "completed"
    assert result["acceptance_checklist"]
    assert any(item["name"] == "Acceptance checklist created" for item in result["acceptance_checklist"])
    assert result["verification_history"]
    assert result["verification_history"][0]["checks_performed"]
    assert result["project_path"].startswith(str(tmp_path / "workspace" / "generated_apps"))
    assert "Professional build completed." in result["message"]
    assert "Known limitations:" in result["message"]
    assert "Launch verification passed or skipped with reason" in result["message"]


def test_professional_build_does_not_claim_done_if_verification_fails(tmp_path, monkeypatch):
    app = DummyApp(tmp_path)
    mode = CodeMode(app)
    monkeypatch.setattr(
        mode,
        "_verify_launch_readiness",
        lambda project_path, app_kind, settings: {
            "status": "passed",
            "passed": True,
            "reason": "Launch verification passed in test.",
            "stdout": "",
            "stderr": "",
            "window_title": APP_TEMPLATES[app_kind]["display_name"],
            "cleanup_performed": True,
        },
    )

    original_verify = mode._run_professional_verification

    def failing_verify(*args, **kwargs):
        result = original_verify(*args, **kwargs)
        result["ok"] = False
        result["tests_or_checks_passed"] = False
        result["error"] = "Syntax verification failed for generated project."
        return result

    mode._run_professional_verification = failing_verify
    result = mode.handle({"user_text": "professional build make me a timer app with a gui and double click starter"})

    assert result["ok"] is False
    assert result["status"] == "verification_failed"
    assert "Professional build stopped after a failed verification step." in result["message"]
    assert "Syntax verification failed for generated project." in result["message"]


def test_professional_build_stops_after_max_passes(tmp_path, monkeypatch):
    app = DummyApp(tmp_path)
    app.settings["professional_build"]["max_passes"] = 2
    mode = CodeMode(app)
    monkeypatch.setattr(
        mode,
        "_verify_launch_readiness",
        lambda project_path, app_kind, settings: {
            "status": "passed",
            "passed": True,
            "reason": "Launch verification passed in test.",
            "stdout": "",
            "stderr": "",
            "window_title": APP_TEMPLATES[app_kind]["display_name"],
            "cleanup_performed": True,
        },
    )

    original_verify = mode._run_professional_verification
    mode._run_professional_verification = original_verify

    def failing_checklist(*args, **kwargs):
        checklist = mode._build_acceptance_checklist_original(*args, **kwargs)
        checklist[-1] = {
            "name": "Artificial blocker",
            "passed": False,
            "detail": "Keep iterating for the test.",
        }
        return checklist

    mode._build_acceptance_checklist_original = mode._build_acceptance_checklist
    mode._build_acceptance_checklist = failing_checklist

    result = mode.handle({"user_text": "professional build make me a calculator app with a gui and double click starter"})

    assert result["ok"] is False
    assert result["status"] == "stopped_at_max_passes"
    assert result["passes_completed"] == 2


def test_launch_verifier_handles_successful_subprocess(tmp_path, monkeypatch):
    app = DummyApp(tmp_path)
    mode = CodeMode(app)
    project_path = tmp_path / "workspace" / "generated_apps" / "NotepadApp_Test"
    project_path.mkdir(parents=True, exist_ok=True)
    main_path = project_path / "main.py"
    main_path.write_text("print('ok')\n", encoding="utf-8")

    class FakeProcess:
        def __init__(self):
            self.terminated = False

        def poll(self):
            return None

        def communicate(self, timeout=None):
            return ("", "")

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    fake_process = FakeProcess()
    monkeypatch.setattr(mode, "_spawn_launch_process", lambda command, cwd: fake_process)
    monkeypatch.setattr(mode, "_detect_window_title", lambda expected: "Notepad")

    result = mode._verify_launch_readiness(project_path, "notepad", app.settings["professional_build"])

    assert result["passed"] is True
    assert result["status"] == "passed"
    assert result["window_title"] == "Notepad"
    assert result["cleanup_performed"] is True
    assert fake_process.terminated is True


def test_launch_verifier_handles_timeout(tmp_path, monkeypatch):
    app = DummyApp(tmp_path)
    app.settings["professional_build"]["launch_timeout_seconds"] = 1
    mode = CodeMode(app)
    project_path = tmp_path / "workspace" / "generated_apps" / "TimerApp_Test"
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "main.py").write_text("print('ok')\n", encoding="utf-8")

    class FakeProcess:
        def poll(self):
            return None

        def communicate(self, timeout=None):
            return ("", "")

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    fake_process = FakeProcess()
    times = iter([0.0, 2.0])
    monkeypatch.setattr(mode, "_spawn_launch_process", lambda command, cwd: fake_process)
    monkeypatch.setattr(mode, "_detect_window_title", lambda expected: "")
    monkeypatch.setattr("app.modes.code_mode.time.monotonic", lambda: next(times))
    monkeypatch.setattr("app.modes.code_mode.time.sleep", lambda seconds: None)

    result = mode._verify_launch_readiness(project_path, "timer", app.settings["professional_build"])

    assert result["passed"] is False
    assert result["status"] == "timeout"
    assert "timed out" in result["reason"].lower()


def test_launch_verifier_handles_crash_and_stderr(tmp_path, monkeypatch):
    app = DummyApp(tmp_path)
    mode = CodeMode(app)
    project_path = tmp_path / "workspace" / "generated_apps" / "CalculatorApp_Test"
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "main.py").write_text("print('ok')\n", encoding="utf-8")

    class FakeProcess:
        def poll(self):
            return 1

        def communicate(self, timeout=None):
            return ("", "Traceback: boom")

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(mode, "_spawn_launch_process", lambda command, cwd: FakeProcess())

    result = mode._verify_launch_readiness(project_path, "calculator", app.settings["professional_build"])

    assert result["passed"] is False
    assert result["status"] == "failed"
    assert "traceback: boom" in result["reason"].lower()
    assert result["stderr"] == "Traceback: boom"


def test_professional_build_does_not_mark_completed_when_launch_verification_fails(tmp_path, monkeypatch):
    app = DummyApp(tmp_path)
    app.settings["professional_build"]["max_passes"] = 2
    mode = CodeMode(app)
    monkeypatch.setattr(
        mode,
        "_verify_launch_readiness",
        lambda project_path, app_kind, settings: {
            "status": "failed",
            "passed": False,
            "reason": "Launch verification failed: window title never appeared.",
            "stdout": "",
            "stderr": "",
            "window_title": "",
            "cleanup_performed": True,
        },
    )
    result = mode.handle({"user_text": "professional build make me a notepad app with a gui and double click starter"})

    assert result["ok"] is False
    assert result["status"] == "verification_failed"
    assert "Launch verification failed" in result["message"]


def test_launch_verifier_calls_cleanup(tmp_path, monkeypatch):
    app = DummyApp(tmp_path)
    mode = CodeMode(app)
    project_path = tmp_path / "workspace" / "generated_apps" / "TodoApp_Test"
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "main.py").write_text("print('ok')\n", encoding="utf-8")

    class FakeProcess:
        def poll(self):
            return None

        def communicate(self, timeout=None):
            return ("", "")

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    cleanup_calls = []
    monkeypatch.setattr(mode, "_spawn_launch_process", lambda command, cwd: FakeProcess())
    monkeypatch.setattr(mode, "_detect_window_title", lambda expected: "Todo List")
    monkeypatch.setattr(mode, "_cleanup_launch_process", lambda process: cleanup_calls.append(True) or True)

    result = mode._verify_launch_readiness(project_path, "todo", app.settings["professional_build"])

    assert result["passed"] is True
    assert cleanup_calls == [True]


def test_professional_build_respects_overwrite_approval(tmp_path):
    app = DummyApp(tmp_path)
    app.ask_approval = lambda prompt: False
    mode = CodeMode(app)
    existing_dir = tmp_path / "workspace" / "generated_apps" / "ExistingApp"
    existing_dir.mkdir(parents=True, exist_ok=True)
    (existing_dir / "main.py").write_text("print('existing')\n", encoding="utf-8")
    (existing_dir / "Run Notepad.bat").write_text("@echo off\n", encoding="utf-8")
    (existing_dir / "README.txt").write_text("existing\n", encoding="utf-8")

    result = mode.handle(
        {
            "user_text": f'professional build make me a notepad app with a gui and double click starter in "{existing_dir}"'
        }
    )

    assert result["ok"] is False
    assert result["error"] == "Professional build cancelled by user."


def test_professional_build_can_request_research(tmp_path, monkeypatch):
    app = DummyApp(tmp_path)
    mode = CodeMode(app)
    calls = []
    monkeypatch.setattr(
        mode,
        "_verify_launch_readiness",
        lambda project_path, app_kind, settings: {
            "status": "skipped",
            "passed": True,
            "reason": "Launch verification is only required for generated Python GUI apps.",
            "stdout": "",
            "stderr": "",
            "window_title": "",
            "cleanup_performed": False,
        },
    )

    def fake_search(query, max_results=3):
        calls.append(query)
        return {
            "ok": True,
            "results": [
                {"title": "sqlite3 — Python documentation", "url": "https://docs.python.org/3/library/sqlite3.html", "snippet": "Official sqlite3 docs."}
            ],
        }

    monkeypatch.setattr("app.modes.code_mode.search_web", fake_search)

    result = mode.handle({"user_text": "professional build make me a script tool that uses sqlite to store notes"})

    assert result["status"] == "completed"
    assert calls
    assert "sqlite3" in result["research"]["summary"].lower()


def test_code_mode_starts_test_runner_for_run_pytest(tmp_path):
    app = DummyApp(tmp_path)
    mode = CodeMode(app)

    result = mode.handle({"user_text": "run pytest"})

    assert result["ok"] is True
    assert result["status"] == "started"


def test_code_mode_can_cancel_running_tests(tmp_path):
    app = DummyApp(tmp_path)
    mode = CodeMode(app)

    result = mode.handle({"user_text": "cancel tests"})

    assert result["ok"] is True
    assert result["message"] == "Test cancellation requested."
