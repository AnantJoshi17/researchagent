import re
import time
from typing import Iterator

import memory
from agents import stream_reader_agent, stream_search_agent
from chains import run_critic_chain, stream_writer_chain

URL_RE = re.compile(r"https?://[^\s<>\")\]]+")


def parse_sources(text: str) -> list[dict]:
    """Turn the search agent's 'Title - URL - why' lines into structured cards. The model
    doesn't always follow the format exactly, so anything with a URL on it counts."""
    items, seen = [], set()
    for line in text.splitlines():
        m = URL_RE.search(line)
        if not m:
            continue
        url = m.group(0).rstrip(".,;:")
        if url in seen:
            continue
        seen.add(url)
        before, after = line[: m.start()], line[m.end():]
        title = re.sub(r"^[\s\-*\d.)#|]+", "", before)          # list markers / numbering
        title = re.sub(r"[\s\-–—:|(\[]+$", "", title)            # trailing separators
        title = title.replace("**", "").strip(" []")
        why = re.sub(r"^[\s)\]\-–—:|]+", "", after).replace("**", "").strip()
        items.append({"title": title or url, "url": url, "why": why})
    return items


def _read_urls(notes: str, attempted: list[str]) -> list[str]:
    # Prefer the pages the reader actually reported on; fall back to what it tried.
    reported = [u.rstrip(".,;:") for u in re.findall(r"SOURCE:\s*(\S+)", notes)]
    return list(dict.fromkeys(reported or attempted))


def research_events(topic: str, session_id: str | None = None) -> Iterator[dict]:
    """Run the full pipeline, yielding progress events the CLI and web UI both render."""
    started = time.time()
    try:
        past = memory.memory_context(session_id)

        yield {"type": "stage", "stage": "search", "status": "active"}
        sources_text = ""
        for kind, value in stream_search_agent(topic, past):
            if kind == "tool_call" and value["name"] == "web_search":
                yield {"type": "query", "query": value["args"].get("query", "")}
            elif kind == "final":
                sources_text = value
        sources = parse_sources(sources_text)
        if not sources:
            raise RuntimeError("The search step found no usable sources. Try rephrasing the topic.")
        yield {"type": "sources", "items": sources}
        yield {"type": "stage", "stage": "search", "status": "done"}

        yield {"type": "stage", "stage": "read", "status": "active"}
        notes, attempted = "", []
        for kind, value in stream_reader_agent(topic, sources_text):
            if kind == "tool_call" and value["name"] == "scrape_url":
                url = value["args"].get("url", "")
                attempted.append(url)
                yield {"type": "reading", "url": url}
            elif kind == "final":
                notes = value
        read_urls = _read_urls(notes, attempted)
        known = {s["url"] for s in sources}
        extra = [{"title": u, "url": u, "why": ""} for u in read_urls if u not in known]
        if extra:
            sources += extra
            yield {"type": "sources", "items": sources}
        yield {"type": "read_done", "urls": read_urls}
        yield {"type": "stage", "stage": "read", "status": "done"}

        # The writer never sees the raw web, only what the reader chose to pass on.
        yield {"type": "stage", "stage": "write", "status": "active"}
        report = ""
        for chunk in stream_writer_chain(topic, sources_text, notes, past):
            report += chunk
            yield {"type": "token", "text": chunk}
        yield {"type": "stage", "stage": "write", "status": "done"}

        yield {"type": "stage", "stage": "critique", "status": "active"}
        critique = run_critic_chain(topic, notes, report)
        yield {"type": "critique", **critique}
        yield {"type": "stage", "stage": "critique", "status": "done"}

        report_id = None
        if session_id:
            report_id = memory.save_report(session_id, topic, report, sources, read_urls, critique)
        yield {"type": "done", "id": report_id, "elapsed": round(time.time() - started, 1)}
    except Exception as e:  # surfaced to the user rather than killing the stream
        yield {"type": "error", "message": str(e) or e.__class__.__name__}


STAGE_LABELS = {
    "search": "[1/4] Searching the web...",
    "read": "[2/4] Reading the best sources...",
    "write": "[3/4] Writing the report...",
    "critique": "[4/4] Critiquing the report...",
}


def run_research(topic: str) -> str:
    """CLI entry point: prints progress and returns the report with the critique appended."""
    report, critique = "", None
    for ev in research_events(topic):
        if ev["type"] == "stage" and ev["status"] == "active":
            print(STAGE_LABELS[ev["stage"]])
        elif ev["type"] == "reading":
            print(f"      reading {ev['url']}")
        elif ev["type"] == "token":
            report += ev["text"]
        elif ev["type"] == "critique":
            critique = ev
        elif ev["type"] == "error":
            raise RuntimeError(ev["message"])

    if critique:
        score = f"{critique['score']}/10" if critique["score"] else "n/a"
        lines = [f"\n\n---\n\n## Critic score: {score}", critique["verdict"], "", "**Strengths**"]
        lines += [f"- {s}" for s in critique["strengths"]] + ["", "**Weaknesses**"]
        lines += [f"- {w}" for w in critique["weaknesses"]]
        report += "\n".join(lines)
    return report
