from app.prompt_builder import PromptBuilder


def _sample_tools():
    return [
        {
            "name": "take_screenshot",
            "description": "Capture the current screen.",
            "argument_schema": {"type": "object", "properties": {}},
            "risk_level": "safe",
            "approval_required": False,
        },
        {
            "name": "analyze_screenshot",
            "description": "Describe a screenshot with the vision model.",
            "argument_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            "risk_level": "safe",
            "approval_required": False,
        },
        {
            "name": "desktop_suggest_action",
            "description": "Suggest the next desktop action without executing it.",
            "argument_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "instruction": {"type": "string"}},
                "required": ["path", "instruction"],
            },
            "risk_level": "safe",
            "approval_required": False,
        },
        {
            "name": "browser_launch",
            "description": "Launch the browser.",
            "argument_schema": {"type": "object", "properties": {"headless": {"type": "boolean"}}},
            "risk_level": "medium",
            "approval_required": True,
        },
        {
            "name": "browser_search",
            "description": "Search the web.",
            "argument_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            "risk_level": "medium",
            "approval_required": True,
        },
        {
            "name": "browser_get_page_info",
            "description": "Get page info.",
            "argument_schema": {"type": "object", "properties": {}},
            "risk_level": "safe",
            "approval_required": False,
        },
        {
            "name": "set_timer",
            "description": "Set a timer.",
            "argument_schema": {"type": "object", "properties": {"duration_seconds": {"type": "integer"}}, "required": ["duration_seconds"]},
            "risk_level": "safe",
            "approval_required": False,
        },
        {
            "name": "list_timers",
            "description": "List timers.",
            "argument_schema": {"type": "object", "properties": {}},
            "risk_level": "safe",
            "approval_required": False,
        },
        {
            "name": "cancel_timer",
            "description": "Cancel a timer.",
            "argument_schema": {"type": "object", "properties": {"timer_id": {"type": "string"}}, "required": ["timer_id"]},
            "risk_level": "safe",
            "approval_required": False,
        },
        {
            "name": "list_files",
            "description": "List files.",
            "argument_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            "risk_level": "safe",
            "approval_required": False,
        },
        {
            "name": "write_file",
            "description": "Write a file.",
            "argument_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
            "risk_level": "medium",
            "approval_required": True,
        },
        {
            "name": "get_current_task",
            "description": "Read the current task.",
            "argument_schema": {"type": "object", "properties": {}},
            "risk_level": "safe",
            "approval_required": False,
        },
        {
            "name": "summarize_recent_sessions",
            "description": "Summarize recent sessions.",
            "argument_schema": {"type": "object", "properties": {}},
            "risk_level": "safe",
            "approval_required": False,
        },
        {
            "name": "list_sessions",
            "description": "List sessions.",
            "argument_schema": {"type": "object", "properties": {}},
            "risk_level": "safe",
            "approval_required": False,
        },
        {
            "name": "read_session",
            "description": "Read a session.",
            "argument_schema": {"type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]},
            "risk_level": "safe",
            "approval_required": False,
        },
        {
            "name": "run_command",
            "description": "Run a shell command.",
            "argument_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
            "risk_level": "medium",
            "approval_required": True,
        },
    ]


def test_prompt_builder_does_not_include_full_session_json():
    builder = PromptBuilder(planner_context_length=16384)
    current_task = {
        "active_task_id": "task123",
        "original_user_task": "Open Google and search cats.",
        "latest_user_message": "what did you look up?",
        "status": "completed",
        "last_tool_result_summary": "browser_search -> cats - Google Search",
        "recent_messages": [{"role": "user", "content": "Open Google and search cats."}],
        "session_dump": {"tool_calls": [{"tool": "browser_search"}], "steps": [{"step": 1}]},
    }

    prompt = builder.build(
        user_message="what did you look up?",
        current_task=current_task,
        available_tools=_sample_tools(),
        rules_text="- The AI must make the plan and choose tools.",
    ).system_prompt

    assert '"tool_calls"' not in prompt
    assert '"steps"' not in prompt


def test_prompt_builder_limits_recent_messages_and_tool_summaries():
    builder = PromptBuilder(planner_context_length=16384)
    current_task = {
        "active_task_id": "task123",
        "original_user_task": "Describe my screen briefly.",
        "latest_user_message": "What about the sidebar?",
        "status": "completed",
        "last_tool_result_summary": "X" * 1200,
        "recent_tool_result_summaries": ["A" * 500, "B" * 500, "C" * 500, "D" * 500],
        "recent_messages": [
            {"role": "user", "content": "u1" * 200},
            {"role": "assistant", "content": "a1" * 200},
            {"role": "user", "content": "u2" * 200},
            {"role": "assistant", "content": "a2" * 200},
        ],
    }

    working_memory = builder.build(
        user_message="What about the sidebar?",
        current_task=current_task,
        available_tools=_sample_tools(),
        rules_text="- The AI must make the plan and choose tools.",
    ).working_memory

    assert len(working_memory) < 4000
    assert working_memory.count("- user:") + working_memory.count("- assistant:") <= 3
    assert "..." in working_memory


def test_desktop_followup_prompt_includes_desktop_tools_only():
    builder = PromptBuilder(planner_context_length=16384)
    prompt = builder.build(
        user_message="What about the sidebar?",
        current_task={
            "active_task_id": "task123",
            "original_user_task": "Describe my screen briefly.",
            "recent_tool_calls": [{"tool": "analyze_screenshot", "args": {"path": "demo.png"}}],
        },
        available_tools=_sample_tools(),
        rules_text="- The AI must make the plan and choose tools.",
    ).system_prompt

    assert "take_screenshot" in prompt
    assert "desktop_suggest_action" in prompt
    assert "browser_launch" not in prompt
    assert "set_timer" not in prompt


def test_browser_followup_prompt_includes_browser_tools_only():
    builder = PromptBuilder(planner_context_length=16384)
    prompt = builder.build(
        user_message="what did you look up?",
        current_task={
            "active_task_id": "task123",
            "original_user_task": "Open Google and search cats.",
            "recent_tool_calls": [{"tool": "browser_search", "args": {"query": "cats"}}],
        },
        available_tools=_sample_tools(),
        rules_text="- The AI must make the plan and choose tools.",
    ).system_prompt

    assert "browser_search" in prompt
    assert "browser_get_page_info" in prompt
    assert "desktop_suggest_action" not in prompt
    assert "set_timer" not in prompt


def test_timer_prompt_includes_timer_tools_only():
    builder = PromptBuilder(planner_context_length=16384)
    prompt = builder.build(
        user_message="set a timer for 1 minute on my pc",
        current_task=None,
        available_tools=_sample_tools(),
        rules_text="- The AI must make the plan and choose tools.",
    ).system_prompt

    assert "set_timer" in prompt
    assert "list_timers" in prompt
    assert "cancel_timer" in prompt
    assert "run_command" not in prompt
    assert "browser_search" not in prompt


def test_prompt_builder_warns_when_context_is_too_low():
    builder = PromptBuilder(planner_context_length=4096)

    warning = builder.planner_context_warning()

    assert warning is not None
    assert "too small" in warning.lower()
    assert "16384" in warning
