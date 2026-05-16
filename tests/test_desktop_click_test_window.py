import importlib


def test_safe_click_test_window_module_imports_without_side_effects():
    module = importlib.import_module("app.desktop_click_test_window")

    assert hasattr(module, "SafeClickTestWindow")
    assert hasattr(module, "main")
