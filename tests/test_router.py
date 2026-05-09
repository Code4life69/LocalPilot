from app.router import KeywordRouter


def test_keyword_routing():
    router = KeywordRouter()
    assert router.classify("search python logging best practices") == "research"
    assert router.classify("read file C:\\temp\\x.txt") == "code"
    assert router.classify("take screenshot now") == "desktop"
    assert router.classify("save note remember this") == "memory"
    assert router.classify("hello there") == "chat"
    assert router.classify("who is the current president as of 05/08/2026") == "research"
    assert router.classify("no who is the vice president of america?") == "research"
    assert (
        router.classify(
            "search for dolphins on my pc on google and look at the images and download that image"
        )
        == "desktop"
    )
    assert router.classify("visualize desktop") == "desktop"
    assert router.classify("visualize desktop understanding") == "desktop"
    assert router.classify("show me what you see") == "desktop"
    assert router.classify('write "workspace/final_pass/test_note.txt" "hello"') == "code"
    assert router.classify('read "workspace/final_pass/test_note.txt"') == "code"
    assert router.classify("make me a new website locally on my pc") == "code"
    assert router.classify("build me a website") == "code"
    assert router.classify("make a basic local website") == "code"
    assert router.classify("create a simple website") == "code"
    assert router.classify("make me a website with html css and javascript") == "code"
    assert router.classify("make a local website in a new folder") == "code"
    assert router.classify("search the website for prices") != "code"
    assert router.classify("open this website") != "code"
    assert router.classify("look up websites about local ai") != "code"
