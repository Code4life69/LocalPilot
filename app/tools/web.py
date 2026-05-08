from __future__ import annotations

def search_web(query: str, max_results: int = 5) -> dict:
    try:
        from duckduckgo_search import DDGS
    except ImportError as exc:
        return {"ok": False, "error": f"duckduckgo-search not installed: {exc}"}

    results = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                }
            )
            if len(results) >= max_results:
                break
    return {"ok": True, "query": query, "results": results}
