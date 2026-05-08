from __future__ import annotations

import warnings
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup


def search_web(query: str, max_results: int = 5) -> dict:
    try:
        results = _search_with_ddgs(query, max_results)
        if results:
            return {"ok": True, "query": query, "results": results}
    except Exception:
        pass

    try:
        fallback_results = _search_with_duckduckgo_html(query, max_results)
        return {"ok": True, "query": query, "results": fallback_results}
    except Exception as exc:
        return {"ok": False, "error": f"Web search failed: {exc}", "query": query, "results": []}


def _search_with_ddgs(query: str, max_results: int) -> list[dict]:
    try:
        from duckduckgo_search import DDGS
    except ImportError as exc:
        raise RuntimeError(f"duckduckgo-search not installed: {exc}") from exc

    results = []
    original_warn = warnings.warn

    def _filtered_warn(message, category=None, *args, **kwargs):
        text = str(message)
        if category is RuntimeWarning and "has been renamed to `ddgs`" in text:
            return
        return original_warn(message, category=category, *args, **kwargs)

    warnings.warn = _filtered_warn
    try:
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
    finally:
        warnings.warn = original_warn
    return results


def _search_with_duckduckgo_html(query: str, max_results: int) -> list[dict]:
    response = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    results: list[dict] = []

    for block in soup.select(".result")[:max_results]:
        link = block.select_one(".result__title a")
        snippet = block.select_one(".result__snippet")
        if link is None:
            continue
        title = link.get_text(" ", strip=True)
        href = _unwrap_duckduckgo_link(link.get("href", ""))
        body = snippet.get_text(" ", strip=True) if snippet else ""
        results.append({"title": title, "url": href, "snippet": body})
    return results


def _unwrap_duckduckgo_link(url: str) -> str:
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path == "/l/":
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target) if target else url
    return url
