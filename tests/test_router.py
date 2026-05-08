from app.router import KeywordRouter


def test_keyword_routing():
    router = KeywordRouter()
    assert router.classify("search python logging best practices") == "research"
    assert router.classify("read file C:\\temp\\x.txt") == "code"
    assert router.classify("take screenshot now") == "desktop"
    assert router.classify("save note remember this") == "memory"
    assert router.classify("hello there") == "chat"
    assert router.classify("who is the current president as of 05/08/2026") == "research"
