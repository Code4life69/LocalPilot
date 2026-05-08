from app.modes.research_mode import ResearchMode


class DummyApp:
    memory = None


def test_normalize_query_for_current_vice_president():
    mode = ResearchMode(DummyApp())
    query = mode._normalize_query_for_search("who is the current vice president of the united states as of 05/08/2026")
    assert query == "current vice president of the United States 05/08/2026"
