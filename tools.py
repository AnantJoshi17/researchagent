import os

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

MAX_PAGE_CHARS = 3000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _get_tavily() -> TavilySearch:
    # Built lazily so importing this module doesn't require the key to be set.
    return TavilySearch(max_results=5, tavily_api_key=os.getenv("TAVILY_API_KEY"))


@tool
def web_search(query: str) -> str:
    """Search the web for a query and return the top results as numbered text."""
    try:
        response = _get_tavily().invoke({"query": query})
        if isinstance(response, dict) and response.get("error"):
            return f"Search failed: {response['error']}"

        lines = []
        for i, r in enumerate(response.get("results", []), start=1):
            snippet = " ".join(r.get("content", "").split())[:300]
            lines.append(f"{i}. {r.get('title', '')}\nURL: {r.get('url', '')}\n{snippet}")
        return "\n\n".join(lines) or "Search failed: no results"
    except Exception as e:
        return f"Search failed: {e}"


@tool
def scrape_url(url: str) -> str:
    """Fetch a web page and return its main readable text."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()

        lines = (line.strip() for line in soup.get_text(separator="\n").splitlines())
        text = "\n".join(line for line in lines if line)

        if len(text) > MAX_PAGE_CHARS:
            text = text[:MAX_PAGE_CHARS] + "\n...[truncated]"
        return text
    except Exception as e:
        return f"Could not fetch {url}: {e}"
