import time
from typing import Iterator

import agents
import memory
from chains import run_critic_chain, stream_writer_chain


def research_events(topic: str, session_id: str | None = None) -> Iterator[dict]:
    """Run search -> read -> write -> critique, yielding progress events the CLI and the web
    UI both render. Only the writer and the critic call the LLM (plus one short query rewrite
    for follow-ups), so a report takes a few seconds instead of a minute."""
    started = time.time()
    try:
        past = memory.memory_context(session_id)

        yield {"type": "stage", "stage": "search", "status": "active"}
        query = agents.standalone_query(topic, past)
        yield {"type": "query", "query": query}
        results = agents.search(query)
        if not results:
            raise RuntimeError("The search step found no usable sources. Try rephrasing the topic.")
        sources = [{"title": r["title"], "url": r["url"], "why": r["snippet"]} for r in results]
        yield {"type": "sources", "items": sources}
        yield {"type": "stage", "stage": "search", "status": "done"}

        yield {"type": "stage", "stage": "read", "status": "active"}
        for r in agents._diverse(results)[: agents.MAX_SOURCES]:
            yield {"type": "reading", "url": r["url"]}
        pages = agents.read(results)
        read_urls = [p["url"] for p in pages]
        notes = agents.notes_from(pages)
        yield {"type": "read_done", "urls": read_urls}
        yield {"type": "stage", "stage": "read", "status": "done"}

        # The writer never sees the raw web, only the pages the reader chose.
        yield {"type": "stage", "stage": "write", "status": "active"}
        sources_text = "\n".join(f"- {s['title']} - {s['url']}" for s in sources)
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


def chat_reply(prompt: str, session_id: str | None = None) -> Iterator[str]:
    """Stream a chat answer, then store the turn in session memory."""
    history = memory.chat_history(session_id) if session_id else []
    answer = ""
    for chunk in agents.stream_chat(prompt, history, memory.memory_context(session_id)):
        answer += chunk
        yield chunk
    if session_id and answer:
        memory.save_chat(session_id, prompt, answer)


STAGE_LABELS = {
    "search": "[1/4] Searching the web...",
    "read": "[2/4] Reading the best sources...",
    "write": "[3/4] Writing the report...",
    "critique": "[4/4] Critiquing the report...",
}


def run_research(topic: str, session_id: str | None = None) -> str:
    """CLI: prints progress and returns the report with the critique appended."""
    report, critique = "", None
    for ev in research_events(topic, session_id):
        if ev["type"] == "stage" and ev["status"] == "active":
            print(STAGE_LABELS[ev["stage"]])
        elif ev["type"] == "reading":
            print(f"      reading {ev['url']}")
        elif ev["type"] == "token":
            report += ev["text"]
        elif ev["type"] == "critique":
            critique = ev
        elif ev["type"] == "done":
            print(f"      done in {ev['elapsed']}s")
        elif ev["type"] == "error":
            raise RuntimeError(ev["message"])

    if critique:
        score = f"{critique['score']}/10" if critique["score"] else "n/a"
        lines = [f"\n\n---\n\n## Critic score: {score}", critique["verdict"], "", "**Strengths**"]
        lines += [f"- {s}" for s in critique["strengths"]] + ["", "**Weaknesses**"]
        lines += [f"- {w}" for w in critique["weaknesses"]]
        report += "\n".join(lines)
    return report
