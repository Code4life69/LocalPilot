from pathlib import Path

from app.tools.test_runner import TestRunner


class DummyLogger:
    def __init__(self):
        self.events = []

    def event(self, role, message, **extra):
        self.events.append((role, message, extra))


class DummyTaskState:
    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)
        return kwargs


class DummyApp:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.logger = DummyLogger()
        self.task_state = DummyTaskState()


def test_test_runner_builds_venv_pytest_command(tmp_path):
    app = DummyApp(tmp_path)
    runner = TestRunner(app)

    command = runner.build_pytest_command()

    assert command == [str(tmp_path / ".venv" / "Scripts" / "python.exe"), "-m", "pytest"]


def test_test_runner_captures_successful_command_output(tmp_path):
    app = DummyApp(tmp_path)
    runner = TestRunner(app)
    script = tmp_path / "ok.py"
    script.write_text("print('2 passed in 0.01s')\n", encoding="utf-8")

    result = runner._execute_command(["python", str(script)])

    assert result["ok"] is True
    assert result["exit_code"] == 0
    assert "2 passed in 0.01s" in result["stdout"]
    assert result["summary"] == "2 passed in 0.01s"
    assert app.task_state.updates[-1]["last_test_status"] == "passed"


def test_test_runner_captures_failed_command_output(tmp_path):
    app = DummyApp(tmp_path)
    runner = TestRunner(app)
    script = tmp_path / "fail.py"
    script.write_text("import sys\nprint('1 failed in 0.02s', file=sys.stderr)\nsys.exit(2)\n", encoding="utf-8")

    result = runner._execute_command(["python", str(script)])

    assert result["ok"] is False
    assert result["exit_code"] == 2
    assert "1 failed in 0.02s" in result["stderr"]
    assert result["summary"] == "1 failed in 0.02s"
    assert app.task_state.updates[-1]["last_test_exit_code"] == 2


def test_test_runner_does_not_use_global_python_in_default_command(tmp_path):
    app = DummyApp(tmp_path)
    runner = TestRunner(app)

    command = runner.build_pytest_command()

    assert command[0].endswith(r".venv\Scripts\python.exe")
    assert command[0] != "python"
