import json
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents import get_llm

WRITER_SYSTEM = """You are a research writer. Write a clear, factual report in markdown using ONLY the research provided.
Never invent facts, figures or URLs. Cite sources inline as markdown links like [source](url); never wrap a bare URL in bracket characters.
Where the sources disagree, say so.

Use exactly these three sections:

## Introduction
2 to 3 sentences framing the topic.

## Key Findings
At least 3 bullet points, each a bolded claim followed by a sentence of supporting detail.

## Sources
Markdown links to the URLs that actually appeared in the research."""

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", WRITER_SYSTEM),
    ("human", "{memory}Topic: {topic}\n\nSources:\n{sources}\n\nDetailed reading:\n{content}"),
])


def writer_chain():
    # Built per call, not at import, so the app can start (and say so) without API keys.
    # 0.4 rather than 0.3: this is the only step writing real prose, so it runs slightly warmer.
    return writer_prompt | get_llm(temperature=0.4) | StrOutputParser()


def _writer_inputs(topic: str, sources: str, content: str, memory: str = "") -> dict:
    memory_block = (
        f"Earlier research in this session (context only; do not cite it):\n{memory}\n\n"
        if memory else ""
    )
    return {"topic": topic, "sources": sources, "content": content, "memory": memory_block}


def run_writer_chain(topic: str, sources: str, content: str, memory: str = "") -> str:
    return writer_chain().invoke(_writer_inputs(topic, sources, content, memory))


def stream_writer_chain(topic: str, sources: str, content: str, memory: str = ""):
    """Yield the report as text chunks while the model writes it."""
    yield from writer_chain().stream(_writer_inputs(topic, sources, content, memory))


# --- Critic chain -------------------------------------------------------
# Temperature 0 because a score that changes between runs is not a score.

CRITIC_SYSTEM = """You are a strict research editor. Review the report against the research notes it was written from.
Judge: factual grounding in the notes, coverage of the topic, clarity, and correct citation of sources.
Penalise any claim, number or URL that does not appear in the notes.

Reply with ONLY a JSON object, no prose and no code fences, in exactly this shape:
{{"score": <integer 1-10>, "verdict": "<one sentence>", "strengths": ["...", "..."], "weaknesses": ["...", "..."]}}
Give 2 to 3 strengths and 2 to 3 weaknesses, each under 25 words."""

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", CRITIC_SYSTEM),
    ("human", "Topic: {topic}\n\nResearch notes:\n{content}\n\nReport:\n{report}"),
])


def critic_chain():
    return critic_prompt | get_llm(temperature=0.0) | StrOutputParser()


def parse_critique(raw: str) -> dict:
    """Pull the JSON object out of the critic's reply. Small models sometimes add a fence
    or a sentence around it, so search for the outermost braces rather than trusting it."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}

    if "score" not in data:
        # Last resort: a bare "7/10" or "score: 7" somewhere in the text.
        m = re.search(r"(\d{1,2})\s*/\s*10|score\D{0,5}(\d{1,2})", raw, re.IGNORECASE)
        data["score"] = int(next(g for g in m.groups() if g)) if m else None

    score = data.get("score")
    try:
        score = max(1, min(10, int(score))) if score is not None else None
    except (TypeError, ValueError):
        score = None

    def _list(key):
        value = data.get(key) or []
        return [str(v) for v in value][:3] if isinstance(value, list) else [str(value)]

    return {
        "score": score,
        "verdict": str(data.get("verdict") or ""),
        "strengths": _list("strengths"),
        "weaknesses": _list("weaknesses"),
    }


def run_critic_chain(topic: str, content: str, report: str) -> dict:
    return parse_critique(critic_chain().invoke({"topic": topic, "content": content, "report": report}))
