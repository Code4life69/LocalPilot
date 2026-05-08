from app.tools.files import append_file, copy_file, list_folder, make_folder, move_file, read_file, write_file


def test_write_read_and_append_file(tmp_path):
    file_path = tmp_path / "note.txt"
    write_result = write_file(str(file_path), "hello")
    assert write_result["ok"]

    append_result = append_file(str(file_path), " world")
    assert append_result["ok"]

    read_result = read_file(str(file_path))
    assert read_result["ok"]
    assert read_result["content"] == "hello world"


def test_make_folder_and_list_folder(tmp_path):
    folder = tmp_path / "demo"
    make_result = make_folder(str(folder))
    assert make_result["ok"]
    write_file(str(folder / "a.txt"), "x")

    list_result = list_folder(str(folder))
    assert list_result["ok"]
    assert list_result["items"][0]["name"] == "a.txt"


def test_copy_and_move_file(tmp_path):
    source = tmp_path / "src.txt"
    copied = tmp_path / "copied.txt"
    moved = tmp_path / "moved.txt"
    write_file(str(source), "payload")

    copy_result = copy_file(str(source), str(copied))
    assert copy_result["ok"]
    assert copied.read_text(encoding="utf-8") == "payload"

    move_result = move_file(str(copied), str(moved))
    assert move_result["ok"]
    assert moved.read_text(encoding="utf-8") == "payload"

