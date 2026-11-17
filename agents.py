import os
from typing import Iterator

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq

from tools import scrape_url, web_search

# How many pages the reader scrapes per report. Each page is capped at MAX_PAGE_CHARS in
# tools.py, so this is the main lever on tokens per run (and on Groq's per-minute budget).
MAX_SOURCES = int(os.getenv("MAX_SOURCES", "3"))


def get_llm(temperature: float = 0.3) -> ChatGroq:
    # A small model by default: each report makes several sequential LLM calls, and on
    # larger models that exhausts Groq's free tokens-per-minute budget, so the SDK backs
    # off mid-run and the total time roughly triples.
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
    )


def _stream_agent(agent, prompt: str) -> Iterator[tuple[str, object]]:
    """Run an agent graph, yielding ("tool_call", {name, args}) as the model requests
    tools and finally ("final", text). Lets callers show progress mid-run."""
    final = ""
    for update in agent.stream(
        {"messages": [{"role": "user", "content": prompt}]}, stream_mode="updates"
    ):
        for node_state in update.values():
            for msg in (node_state or {}).get("messages", []):
                if not isinstance(msg, AIMessage):
                    continue
                for call in msg.tool_calls or []:
                    yield "tool_call", {"name": call["name"], "args": call["args"]}
                if msg.content and not msg.tool_calls:
                    final = msg.content
    yield "final", final


def _run_agent(agent, prompt: str) -> str:
    return next(v for kind, v in _stream_agent(agent, prompt) if kind == "final")


def _search_agent():
    return create_agent(
        model=get_llm(),
        tools=[web_search],
        system_prompt=(
            "You are a research search specialist. Call web_search once or twice with "
            "well-chosen queries, then list the most useful sources, each on its own line as "
            "'Title - URL - one line on why it matters'. Do not write a report, only list sources. "
            "If you are given earlier research from this session and the new request is a "
            "follow-up (it says 'it', 'that', 'compare', etc.), resolve what it refers to "
            "before choosing search queries."
        ),
    )


def _reader_agent():
    return create_agent(
        model=get_llm(),
        tools=[scrape_url],
        system_prompt=(
            f"You are a research reader. Given a list of sources, pick the {MAX_SOURCES} most "
            "relevant and substantial URLs from different sites and call scrape_url on each "
            "(you may call them in parallel). If a page fails or comes back empty, replace it "
            "with the next best URL once. Then write detailed notes on each page you read: key "
            "facts, figures and quotes. Start each page's notes with the line "
            "'SOURCE: <the url you read>'. Note where sources agree or disagree."
        ),
    )


def search_prompt(topic: str, memory: str = "") -> str:
    prompt = f"Find sources on: {topic}"
    if memory:
        prompt = f"Earlier research in this session:\n{memory}\n\n{prompt}"
    return prompt


def stream_search_agent(topic: str, memory: str = ""):
    return _stream_agent(_search_agent(), search_prompt(topic, memory))


def stream_reader_agent(topic: str, sources: str):
    return _stream_agent(_reader_agent(), f"Topic: {topic}\n\nSources:\n{sources}")


def run_search_agent(topic: str, memory: str = "") -> str:
    return _run_agent(_search_agent(), search_prompt(topic, memory))


def run_reader_agent(topic: str, sources: str) -> str:
    return _run_agent(_reader_agent(), f"Topic: {topic}\n\nSources:\n{sources}")
