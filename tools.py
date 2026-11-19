"""Web tools. Plain functions, no LLM involved, so they run fast and can run in parallel."""

import os

import requests
from bs4 import BeautifulSoup
from langchain_tavily import TavilySearch

MAX_PAGE_CHARS = 3000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _clean(text: str) -> str:
    lines = (line.strip() for line in (text or "").splitlines())
    text = "\n".join(line for line in lines if line)
    return text[:MAX_PAGE_CHARS] + "\n...[truncated]" if len(text) > MAX_PAGE_CHARS else text


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """One Tavily call that also returns each page's text, which saves scraping most pages.
    Returns [{title, url, snippet, text}]. Raises on failure so the pipeline can report it."""
    tavily = TavilySearch(
        max_results=max_results,
        search_depth="basic",
        include_raw_content="text",
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
    )
    response = tavily.invoke({"query": query})
    if isinstance(response, str) or (isinstance(response, dict) and response.get("error")):
        err = response if isinstance(response, str) else response["error"]
        raise RuntimeError(f"Web search failed: {err}")

    results = []
    for r in response.get("results", []):
        if not r.get("url"):
            continue
        results.append({
            "title": (r.get("title") or r["url"]).strip(),
            "url": r["url"],
            "snippet": " ".join((r.get("content") or "").split())[:300],
            "text": _clean(r.get("raw_content") or ""),
        })
    return results


def scrape_url(url: str) -> str:
    """Fallback for pages Tavily returned no text for. Returns "" on any failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()
        return _clean(soup.get_text(separator="\n"))
    except Exception:
        return ""
