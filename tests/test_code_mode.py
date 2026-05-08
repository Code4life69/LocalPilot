from pathlib import Path

from app.modes.code_mode import CodeMode


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

    def ask_approval(self, prompt):
        return True


def test_calculator_request_without_explicit_folder_creates_default_project(tmp_path):
    app = DummyApp(tmp_path)
    mode = CodeMode(app)

    result = mode.handle(
        {
            "user_text": "I want you to make me a new calculator program with a gui in a new folder and tell me where the folder is and make it a double click to start program so its easy to start for me"
        }
    )

    assert result["ok"]
    project_path = Path(result["project_path"])
    assert project_path.exists()
    assert project_path.parent == tmp_path / "workspace" / "generated_apps"
    assert (project_path / "calculator.py").exists()
    assert (project_path / "Run Calculator.bat").exists()
    assert "Double-click Run Calculator.bat" in result["message"]


def test_calculator_scaffold_detector_is_not_triggered_by_generic_in_phrase(tmp_path):
    app = DummyApp(tmp_path)
    mode = CodeMode(app)

    assert not mode._is_calculator_scaffold_request("write a note in memory")
    assert mode._is_calculator_scaffold_request(
        "make a calculator with gui in a new folder and make it double click to start"
    )
