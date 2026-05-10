from app.modes.research_mode import ResearchMode


class DummyMemory:
    def __init__(self):
        self.saved = []

    def save_note(self, text):
        self.saved.append(text)
        return "Note saved."


class DummyApp:
    def __init__(self):
        self.memory = DummyMemory()


def test_normalize_query_for_current_vice_president():
    mode = ResearchMode(DummyApp())
    query = mode._normalize_query_for_search("who is the current vice president of the united states as of 05/08/2026")
    assert query == "current vice president of the United States 05/08/2026"


def test_extract_query_strips_save_suffix():
    mode = ResearchMode(DummyApp())
    query = mode._extract_query("search web for the current Ollama model name for Qwen vision and save the useful result to notes")
    assert query == "the current Ollama model name for Qwen vision"


def test_zero_result_research_does_not_save_note(monkeypatch):
    app = DummyApp()
    mode = ResearchMode(app)
    monkeypatch.setattr("app.modes.research_mode.search_web", lambda query, max_results=5: {"ok": True, "query": query, "results": []})

    result = mode.handle(
        {
            "user_text": "search web for the current Ollama model name for Qwen vision and save the useful result to notes"
        }
    )

    assert result["ok"]
    assert result["message"] == "No useful results found, nothing saved."
    assert app.memory.saved == []


def test_useful_research_result_can_save_note(monkeypatch):
    app = DummyApp()
    mode = ResearchMode(app)
    monkeypatch.setattr(
        "app.modes.research_mode.search_web",
        lambda query, max_results=5: {
            "ok": True,
            "query": query,
            "results": [
                {
                    "title": "Qwen2.5-VL model",
                    "url": "https://ollama.com/library/qwen2.5vl",
                    "snippet": "Qwen2.5-VL is the current Qwen vision model in Ollama.",
                }
            ],
        },
    )

    result = mode.handle(
        {
            "user_text": "search web for the current Ollama model name for Qwen vision and save the useful result to notes"
        }
    )

    assert result["ok"]
    assert result["note_saved"] is True
    assert app.memory.saved
    assert "Research: Ollama Qwen vision model" in app.memory.saved[0]
