from unittest.mock import patch

from app.tools.desktop_flow import DesktopExecutionFlow, PlannedStep


class DummyApp:
    settings = {
        "screenshots_dir": "workspace/screenshots",
        "page_understanding": {"confidence_threshold": 0.85},
    }

    def __init__(self):
        self.logger = type("Logger", (), {"event": staticmethod(lambda *args, **kwargs: None)})()
        self.desktop_lessons = type(
            "Lessons",
            (),
            {
                "__init__": lambda self: setattr(self, "entries", []),
                "record": lambda self, lesson_type, task, reason, **extra: self.entries.append(
                    {"type": lesson_type, "task": task, "reason": reason, "extra": extra}
                ),
            },
        )()

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


def test_github_issue_phrase_resolves_to_direct_github_url():
    flow = DesktopExecutionFlow(DummyApp())
    plan = flow._build_plan("search for Code4life69 LocalPilot issue 4 on google in the browser")

    assert [step.name for step in plan] == ["open_github_issue"]
    assert plan[0].value == "https://github.com/Code4life69/LocalPilot/issues/4"
    assert plan[0].metadata["objective_kind"] == "github_issue"
    assert plan[0].metadata["require_objective_match"] is True


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


def test_google_results_with_missing_owner_fail_objective_verification():
    flow = DesktopExecutionFlow(DummyApp())
    step = PlannedStep(
        name="verify_search",
        description="Verify that Google search results are visible",
        kind="verify",
        expected_terms=["code4life69", "localpilot", "issue"],
        vision_prompt="Check whether this is a Google results page for Code4life69 LocalPilot issue 4.",
        metadata={
            "page_type": "google_results",
            "objective_kind": "github_issue",
            "require_objective_match": True,
            "owner": "Code4life69",
            "repo": "LocalPilot",
            "issue_number": "4",
        },
    )

    flow.inspect = lambda include_vision=False, vision_prompt=None: {
        "active_window": {"title": "Code4life69 LocalPilot issue 4 - Google Search - Google Chrome"},
        "ocr_text": "Missing: Code4life69\nSearch results",
        "vision_analysis": "",
    }

    ok, detail, snapshot = flow._verify_step(step)

    assert ok is False
    assert snapshot["verification"]["page_verified"] is True
    assert snapshot["verification"]["objective_verified"] is False
    assert snapshot["verification"]["result"] == "partial"
    assert "Missing: Code4life69" in snapshot["verification"]["reason"]


def test_google_results_with_unrelated_target_are_partial_not_success():
    flow = DesktopExecutionFlow(DummyApp())
    step = PlannedStep(
        name="verify_search",
        description="Verify that Google search results are visible",
        kind="verify",
        expected_terms=["code4life69", "localpilot", "issue"],
        vision_prompt="Check whether this is a Google results page for Code4life69 LocalPilot issue 4.",
        metadata={
            "page_type": "google_results",
            "objective_kind": "github_issue",
            "require_objective_match": True,
            "owner": "Code4life69",
            "repo": "LocalPilot",
            "issue_number": "4",
        },
    )

    flow.inspect = lambda include_vision=False, vision_prompt=None: {
        "active_window": {"title": "Code4life69 LocalPilot issue 4 - Google Search - Google Chrome"},
        "ocr_text": "danielgross/localpilot unrelated result",
        "vision_analysis": "The page is a Google search results page. The first result appears unrelated to Code4life69.",
    }

    ok, detail, snapshot = flow._verify_step(step)

    assert ok is False
    assert snapshot["verification"]["page_verified"] is True
    assert snapshot["verification"]["objective_verified"] is False
    assert snapshot["verification"]["result"] == "partial"
    assert "could not verify the correct target result" in snapshot["verification"]["reason"].lower()


def test_direct_github_issue_url_verification_passes():
    flow = DesktopExecutionFlow(DummyApp())
    step = PlannedStep(
        name="open_github_issue",
        description="Open GitHub issue #4 for Code4life69/LocalPilot",
        kind="open_url",
        value="https://github.com/Code4life69/LocalPilot/issues/4",
        expected_terms=["code4life69", "localpilot", "issue 4"],
        vision_prompt="Check whether this is GitHub issue #4 for Code4life69/LocalPilot.",
        metadata={
            "page_type": "github_issue",
            "objective_kind": "github_issue",
            "require_objective_match": True,
            "owner": "Code4life69",
            "repo": "LocalPilot",
            "issue_number": "4",
        },
    )

    flow.inspect = lambda include_vision=False, vision_prompt=None: {
        "active_window": {"title": "Polish GUI formatting and professional desktop UI · Issue #4 · Code4life69/LocalPilot - Google Chrome"},
        "ocr_text": "Code4life69/LocalPilot Issue 4",
        "vision_analysis": "",
    }

    ok, detail, snapshot = flow._verify_step(step)

    assert ok is True
    assert snapshot["verification"]["page_verified"] is True
    assert snapshot["verification"]["objective_verified"] is True
    assert snapshot["verification"]["result"] == "completed"


def test_generic_google_search_can_complete_with_page_verification_only():
    flow = DesktopExecutionFlow(DummyApp())
    step = PlannedStep(
        name="open_search_results",
        description="Open Google search results for 'dolphins'",
        kind="open_url",
        value="https://www.google.com/search?q=dolphins",
        expected_terms=["dolphins"],
        vision_prompt="Check whether this is a Google results page for dolphins.",
        metadata={
            "page_type": "google_results",
            "objective_kind": "generic_search",
            "require_objective_match": False,
            "query": "dolphins",
        },
    )

    flow.inspect = lambda include_vision=False, vision_prompt=None: {
        "active_window": {"title": "dolphins - Google Search - Google Chrome"},
        "ocr_text": "Google Search dolphins",
        "vision_analysis": "",
    }

    ok, detail, snapshot = flow._verify_step(step)

    assert ok is True
    assert snapshot["verification"]["page_verified"] is True
    assert snapshot["verification"]["objective_verified"] is True
    assert snapshot["verification"]["result"] == "completed"


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
    assert flow.app.desktop_lessons.entries[0]["type"] == "verification_failure"


def test_partial_verification_summary_exposes_page_and_objective_state():
    flow = DesktopExecutionFlow(DummyApp())
    step = PlannedStep(
        name="verify_search",
        description="Verify that Google search results are visible",
        kind="verify",
        expected_terms=["code4life69", "localpilot", "issue"],
        vision_prompt="Check whether this is a Google results page for Code4life69 LocalPilot issue 4.",
    )
    partial_snapshot = {
        "active_window": {"title": "Code4life69 LocalPilot issue 4 - Google Search - Google Chrome"},
        "vision_analysis": "The page is a Google search results page, but the first result appears unrelated.",
        "verification": {
            "verified": False,
            "verification_source": "mixed",
            "reason": "Search page opened, but I could not verify the correct target result.",
            "active_window_title": "Code4life69 LocalPilot issue 4 - Google Search - Google Chrome",
            "vision_summary": "The page is a Google search results page, but the first result appears unrelated.",
            "page_state_confidence": 0.9,
            "objective_match_confidence": 0.2,
            "page_verified": True,
            "objective_verified": False,
            "result": "partial",
        },
    }
    flow._build_plan = lambda _text: [step]
    flow._run_step = lambda _step: (False, "Search page opened, but I could not verify the correct target result.", partial_snapshot)

    result = flow.execute("search for Code4life69 LocalPilot issue 4 on google in the browser")

    assert result["ok"] is False
    assert result["result"] == "partial"
    assert "Desktop execution partially completed." in result["content"]
    assert "Page verified: True" in result["content"]
    assert "Objective verified: False" in result["content"]
    assert "Page state confidence: 0.90" in result["content"]
    assert "Objective match confidence: 0.20" in result["content"]
