from app.safety import SafetyManager


def test_blocks_dangerous_shell_commands():
    safety = SafetyManager()
    assert safety.is_command_blocked("del important.txt")
    assert safety.is_command_blocked("Remove-Item foo")
    assert safety.is_command_blocked("git push --force origin main")
    assert not safety.is_command_blocked("git status")


def test_write_confirmation_depends_on_existing_target(tmp_path):
    safety = SafetyManager()
    target = tmp_path / "data.txt"
    assert not safety.requires_write_confirmation(target)
    target.write_text("hello", encoding="utf-8")
    assert safety.requires_write_confirmation(target)


def test_move_confirmation_depends_on_existing_destination(tmp_path):
    safety = SafetyManager()
    target = tmp_path / "dst.txt"
    assert not safety.requires_move_confirmation(target)
    target.write_text("hello", encoding="utf-8")
    assert safety.requires_move_confirmation(target)


def test_detects_broad_destructive_requests():
    safety = SafetyManager()
    assert safety.is_broad_destructive_request("delete everything in C:\\LocalPilot\\workspace")
    assert safety.is_broad_destructive_request("wipe folder C:\\Temp")
    assert safety.is_broad_destructive_request("remove all files from workspace")
    assert safety.is_broad_destructive_request("erase workspace")
    assert safety.is_broad_destructive_request("rm -rf C:\\LocalPilot\\workspace")
    assert not safety.is_broad_destructive_request("list everything in C:\\LocalPilot\\workspace")
