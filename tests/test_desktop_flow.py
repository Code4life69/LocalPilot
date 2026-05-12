from unittest.mock import patch

from app.tools.desktop_flow import DesktopExecutionFlow, PlannedStep


class DummyApp:
    settings = {"screenshots_dir": "workspace/screenshots"}

    def __init__(self):
        self.logger = type("Logger", (), {"event": staticmethod(lambda *args, **kwargs: None)})()

    def ask_approval(self, _prompt):
        return True

    def run_guarded_desktop_action(self, _action_name, action):
        return action()


def test_desktop_flow_detects_google_search_requests():
    flow = DesktopExecutionFlow(DummyApp())
    assert flow.can_handle("search for dolphins on google in the browser")
    plan = flow._build_plan("search for dolphins on google in the browser")
    assert [step.name for step in plan] == ["open_search_results"]
    assert "q=dolphins" in plan[0].value


def test_desktop_flow_uses_image_search_when_requested():
    flow = DesktopExecutionFlow(DummyApp())
    plan = flow._build_plan("search for dolphins images on google")
    assert len(plan) == 1
    assert "tbm=isch" in plan[0].value


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
    assert [step.name for step in plan] == ["open_search_results"]
    assert "tbm=isch" in plan[0].value
    assert "q=dolphins" in plan[0].value


def test_desktop_flow_detects_image_download_followup_request():
    flow = DesktopExecutionFlow(DummyApp())
    assert flow._needs_image_download_followup(
        "search a random thing up on google then go to images then save and copy your favourite image in a folder"
    )


def test_negative_vision_response_with_expected_terms_still_fails():
    flow = DesktopExecutionFlow(DummyApp())
    step = PlannedStep(
        name="verify_search",
        description="Verify that Google search results are visible",
        kind="verify",
        expected_terms=["code4life69", "localpilot", "issue"],
        vision_prompt="Check whether this is a Google results page for Code4life69 LocalPilot issue 4.",
    )

    def fake_inspect(include_vision=False, vision_prompt=None):
        if include_vision:
            return {
                "active_window": {"title": "Google Chrome"},
                "vision_analysis": "No, this is not a Google results page for Code4life69 LocalPilot issue 4.",
            }
        return {"active_window": {"title": "Google Chrome"}}

    flow.inspect = fake_inspect

    ok, _detail, snapshot = flow._verify_step(step)

    assert ok is False
    assert snapshot["verification"]["verified"] is False
    assert snapshot["verification"]["verification_source"] == "vision"
    assert "mismatch" in snapshot["verification"]["reason"].lower()


def test_discord_active_window_fails_google_verification_even_if_vision_mentions_query():
    flow = DesktopExecutionFlow(DummyApp())
    step = PlannedStep(
        name="verify_search",
        description="Verify that Google search results are visible",
        kind="verify",
        expected_terms=["code4life69", "localpilot", "issue"],
        vision_prompt="Check whether this is a Google results page for Code4life69 LocalPilot issue 4.",
    )

    def fake_inspect(include_vision=False, vision_prompt=None):
        if include_vision:
            return {
                "active_window": {"title": "Chatting - Discord"},
                "vision_analysis": "This is a Google search results page for Code4life69 LocalPilot issue 4.",
            }
        return {"active_window": {"title": "Chatting - Discord"}}

    flow.inspect = fake_inspect

    ok, _detail, snapshot = flow._verify_step(step)

    assert ok is False
    assert snapshot["verification"]["verified"] is False
    assert snapshot["verification"]["verification_source"] == "active_window_title"
    assert "discord" in snapshot["verification"]["reason"].lower()


def test_successful_google_title_passes_verification():
    flow = DesktopExecutionFlow(DummyApp())
    step = PlannedStep(
        name="verify_search",
        description="Verify that Google search results are visible",
        kind="verify",
        expected_terms=["code4life69", "localpilot", "issue"],
        vision_prompt="Check whether this is a Google results page for Code4life69 LocalPilot issue 4.",
    )

    flow.inspect = lambda include_vision=False, vision_prompt=None: {
        "active_window": {"title": "Code4life69 LocalPilot issue 4 - Google Search - Google Chrome"}
    }

    ok, _detail, snapshot = flow._verify_step(step)

    assert ok is True
    assert snapshot["verification"]["verified"] is True
    assert snapshot["verification"]["verification_source"] == "active_window_title"


def test_failed_verification_returns_ok_false():
    flow = DesktopExecutionFlow(DummyApp())
    step = PlannedStep(
        name="verify_search",
        description="Verify that Google search results are visible",
        kind="verify",
        expected_terms=["code4life69", "localpilot", "issue"],
        vision_prompt="Check whether this is a Google results page for Code4life69 LocalPilot issue 4.",
    )
    failing_snapshot = {
        "active_window": {"title": "Chatting - Discord"},
        "vision_analysis": "No, this is not a Google results page for Code4life69 LocalPilot issue 4.",
        "verification": {
            "verified": False,
            "verification_source": "active_window_title",
            "reason": "Active window stayed on Discord instead of the expected browser page.",
            "active_window_title": "Chatting - Discord",
            "vision_summary": "No, this is not a Google results page for Code4life69 LocalPilot issue 4.",
        },
    }
    flow._build_plan = lambda _text: [step]
    flow._run_step = lambda _step: (False, "Could not verify step via UIA title or screenshot analysis.", failing_snapshot)

    result = flow.execute("search for Code4life69 LocalPilot issue 4 on google in the browser")

    assert result["ok"] is False
    assert result["verified"] is False
    assert result["verification_source"] == "active_window_title"
    assert result["active_window_title"] == "Chatting - Discord"
    assert result["vision_summary"] == "No, this is not a Google results page for Code4life69 LocalPilot issue 4."
    assert result["content"].startswith("Desktop execution stopped.")
