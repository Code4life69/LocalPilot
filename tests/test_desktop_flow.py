from unittest.mock import patch

from app.tools.desktop_flow import DesktopExecutionFlow


class DummyApp:
    settings = {"screenshots_dir": "workspace/screenshots"}


def test_desktop_flow_detects_google_search_requests():
    flow = DesktopExecutionFlow(DummyApp())
    assert flow.can_handle("search for dolphins on google in the browser")
    plan = flow._build_plan("search for dolphins on google in the browser")
    assert [step.name for step in plan] == [
        "open_google",
        "focus_address_bar",
        "type_search_url",
        "submit_search",
        "verify_search",
    ]
    assert "q=dolphins" in plan[2].value


def test_desktop_flow_uses_image_search_when_requested():
    flow = DesktopExecutionFlow(DummyApp())
    plan = flow._build_plan("search for dolphins images on google")
    assert "tbm=isch" in plan[2].value


def test_desktop_flow_handles_explicit_urls():
    flow = DesktopExecutionFlow(DummyApp())
    plan = flow._build_plan("open https://github.com/Code4life69/LocalPilot/issues/4 in the browser")
    assert len(plan) == 1
    assert plan[0].kind == "open_url"
    assert "github" in plan[0].value


def test_desktop_flow_handles_conversational_google_images_request():
    flow = DesktopExecutionFlow(DummyApp())
    with patch("app.tools.desktop_flow.random.choice", return_value="dolphins"):
        plan = flow._build_plan(
            "can you search a random thing up on google then go to images then save and copy your favourite image in a folder"
        )
    assert [step.name for step in plan] == [
        "open_google",
        "focus_address_bar",
        "type_search_url",
        "submit_search",
        "verify_search",
    ]
    assert "tbm=isch" in plan[2].value
    assert "q=dolphins" in plan[2].value


def test_desktop_flow_detects_image_download_followup_request():
    flow = DesktopExecutionFlow(DummyApp())
    assert flow._needs_image_download_followup(
        "search a random thing up on google then go to images then save and copy your favourite image in a folder"
    )
