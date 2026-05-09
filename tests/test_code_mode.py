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
        self.system_prompt = "system prompt"
        self.ollama = type(
            "StubOllama",
            (),
            {
                "chat_with_role": lambda self, role, system_prompt, user_text: f"{role}|{user_text}",
            },
        )()

    def ask_approval(self, prompt):
        return True


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
