"""The agents. Only the ones that need judgement call the LLM:

    router  -> decides chat vs research (keyword rule, no LLM call, instant)
    chat    -> LLM, streams a conversational answer
    search  -> Tavily (no LLM)
    reader  -> reads the top pages in parallel (no LLM)
    writer  -> LLM, streams the cited report      (chains.py)
    critic  -> LLM, scores the report out of 10   (chains.py)

Search and reader used to be LLM tool-calling agents. Doing them directly cuts a report from
about six sequential LLM calls to two, which is most of the speed-up.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

import tools

# How many pages the reader passes to the writer. Each is capped at MAX_PAGE_CHARS in tools.py.
MAX_SOURCES = int(os.getenv("MAX_SOURCES", "3"))
MIN_PAGE_CHARS = 400   # below this, Tavily's text is too thin and we scrape the page ourselves


def get_llm(temperature: float = 0.3) -> ChatGroq:
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    extra = {}
    # gpt-oss and qwen are reasoning models. Low effort keeps answers good and cuts the
    # "thinking" time before the first token by a lot. Other models reject the parameter.
    if model.startswith(("openai/gpt-oss", "qwen")):
        extra["reasoning_effort"] = os.getenv("GROQ_REASONING_EFFORT", "low")
    return ChatGroq(model=model, api_key=os.getenv("GROQ_API_KEY"), temperature=temperature,
                    max_retries=2, **extra)


# ---------------------------------------------------------------- router

RESEARCH_WORD = re.compile(r"\bresearch\b", re.IGNORECASE)
LEADING_FILLER = re.compile(
    r"^(?:(?:please|can|could|would|will|you|kindly|do|make|write|give|me|a|an|some|"
    r"deep|full|detailed|report|on|about|into|for|of|regarding|the topic|topic)\b[\s,:\-]*)+",
    re.IGNORECASE,
)


def route(prompt: str) -> tuple[str, str]:
    """("research", topic) when the prompt contains the word "research", else ("chat", prompt).

    "research cricket"                  -> ("research", "cricket")
    "do a research report on cricket"   -> ("research", "cricket")
    "cricket"                           -> ("chat", "cricket")
    """
    prompt = prompt.strip()
    if not RESEARCH_WORD.search(prompt):
        return "chat", prompt
    topic = RESEARCH_WORD.sub(" ", prompt)
    topic = " ".join(topic.split())
    topic = LEADING_FILLER.sub("", topic).strip(" ,:-?.!")
    topic = re.sub(r"\s+report$", "", topic, flags=re.IGNORECASE).strip()
    return "research", topic


# ---------------------------------------------------------------- chat agent

CHAT_SYSTEM = (
    "You are a friendly, concise research assistant chatting with a user. Answer helpfully in "
    "a few short paragraphs or bullets. If the user would benefit from a full cited report, "
    "mention they can start their message with the word \"research\" (for example "
    "\"research {example}\")."
)


def stream_chat(prompt: str, history: list[dict], memory: str = "") -> Iterator[str]:
    """history: earlier chat turns as [{"role": "user"|"assistant", "content": str}]."""
    system = CHAT_SYSTEM.format(example=prompt[:40] or "cricket")
    if memory:
        system += f"\n\nReports researched earlier in this session:\n{memory}"
    messages = [SystemMessage(system)]
    for turn in history[-6:]:
        cls = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(cls(turn["content"]))
    messages.append(HumanMessage(prompt))
    for chunk in get_llm(temperature=0.5).stream(messages):
        if chunk.content:
            yield chunk.content


# ---------------------------------------------------------------- query rewrite (follow-ups only)

def standalone_query(topic: str, memory: str) -> str:
    """Turn a follow-up like "its economic impact" into a full search query. Only called when
    the session already has reports, so first questions skip this LLM call."""
    if not memory:
        return topic
    msg = get_llm(temperature=0).invoke([
        SystemMessage(
            "Rewrite the user's research request as one standalone web search query. If it refers "
            "to earlier research (it, that, they, compare...), name the subject explicitly. "
            "Reply with the query only, under 15 words."
        ),
        HumanMessage(f"Earlier research:\n{memory}\n\nRequest: {topic}"),
    ])
    query = " ".join(str(msg.content).split()).strip("\"'")
    return query or topic


# ---------------------------------------------------------------- search and reader agents

def search(query: str) -> list[dict]:
    return tools.web_search(query, max_results=max(5, MAX_SOURCES + 2))


def _read_one(result: dict) -> dict:
    text = result["text"]
    if len(text) < MIN_PAGE_CHARS:
        text = tools.scrape_url(result["url"]) or text or result["snippet"]
    return {**result, "text": text}


def _diverse(results: list[dict]) -> list[dict]:
    """Put one result per website first, so the report isn't built from a single site."""
    seen, first, rest = set(), [], []
    for r in results:
        host = (urlparse(r["url"]).hostname or "").removeprefix("www.")
        (rest if host in seen else first).append(r)
        seen.add(host)
    return first + rest


def read(results: list[dict]) -> list[dict]:
    """Read the top MAX_SOURCES pages in parallel, skipping ones that come back empty."""
    results = _diverse(results)
    with ThreadPoolExecutor(max_workers=max(1, len(results))) as pool:
        pages = list(pool.map(_read_one, results))
    good = [p for p in pages if len(p["text"]) >= 200]
    return (good or pages)[:MAX_SOURCES]


def notes_from(pages: list[dict]) -> str:
    return "\n\n".join(f"SOURCE: {p['url']}\nTITLE: {p['title']}\n{p['text']}" for p in pages)
