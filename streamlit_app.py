"""Streamlit front end, the one deployed to Streamlit Community Cloud.

Run locally:  streamlit run streamlit_app.py

A message containing the word "research" (e.g. "research cricket") runs the multi-agent
report pipeline. Anything else (e.g. "cricket") gets a fast chatbot answer.
"""

import os
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="researchagent", page_icon="🔮", layout="centered",
                   initial_sidebar_state="auto")

# Keys come from .env locally and from st.secrets on Streamlit Cloud. Both must be in
# os.environ before pipeline is imported, because the chains build their LLM at import.
load_dotenv()
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except FileNotFoundError:  # no secrets.toml locally
    pass
except Exception:
    pass

import memory  # noqa: E402
import st_blocks as ui  # noqa: E402
from agents import route  # noqa: E402
from pipeline import chat_reply, research_events  # noqa: E402

REQUIRED_KEYS = ("GROQ_API_KEY", "TAVILY_API_KEY")
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "20"))
CHAT_LIMIT_PER_HOUR = int(os.getenv("CHAT_LIMIT_PER_HOUR", "120"))
SUGGESTIONS = [
    "cricket",
    "research cricket",
    "research impact of AI on healthcare",
    "How do GLP-1 drugs work?",
]
CSS = (Path(__file__).parent / "streamlit.css").read_text()


@st.cache_resource
def _rate_windows() -> dict:
    # Shared across all sessions in this process: every report spends the deployer's quota.
    return defaultdict(deque)


def rate_limited(client: str, kind: str = "research") -> bool:
    limit = RATE_LIMIT_PER_HOUR if kind == "research" else CHAT_LIMIT_PER_HOUR
    now, window = time.time(), _rate_windows()[(client, kind)]
    while window and now - window[0] > 3600:
        window.popleft()
    if len(window) >= limit:
        return True
    window.append(now)
    return False


def new_run(topic: str, kind: str = "research") -> dict:
    return {
        "kind": kind, "topic": topic, "answer": "", "status": "running", "started": time.time(), "elapsed": 0.0,
        "stages": {k: "pending" for k, _, _ in ui.STAGES}, "log": [], "sources": [],
        "reading": [], "read_urls": [], "report": "", "critique": None, "error": None,
    }


def history_run(item: dict) -> dict:
    if item["kind"] == "chat":
        run = new_run(item["topic"], "chat")
        run.update(status="history", answer=item["answer"])
        return run
    run = new_run(item["topic"])
    run.update(status="history", stages={k: "done" for k, _, _ in ui.STAGES},
               sources=item["sources"], read_urls=item["read_urls"], report=item["report"],
               critique=item["critique"])
    return run


def apply_event(run: dict, ev: dict) -> None:
    t = ev["type"]
    run["elapsed"] = time.time() - run["started"]
    if t == "stage":
        run["stages"][ev["stage"]] = ev["status"]
        if ev["status"] == "active":
            hint = next(h for k, _, h in ui.STAGES if k == ev["stage"])
            run["log"].append(f"{hint.lower()}…")
    elif t == "query":
        run["log"].append(f'web_search("{ev["query"]}")')
    elif t == "sources":
        run["sources"] = ev["items"]
        run["log"].append(f"found {len(ev['items'])} sources")
    elif t == "reading":
        run["reading"].append(ev["url"])
        run["log"].append(f"scrape {ui.domain_of(ev['url'])}")
    elif t == "read_done":
        run["read_urls"] = ev["urls"]
        run["log"].append(f"read {len(ev['urls'])} pages")
    elif t == "token":
        run["report"] += ev["text"]
    elif t == "critique":
        run["critique"] = {k: ev[k] for k in ("score", "verdict", "strengths", "weaknesses")}
        run["log"].append(f"scored {ev['score'] or '?'}/10")
    elif t == "done":
        run["status"], run["elapsed"] = "done", ev["elapsed"]
        run["log"].append(f"done in {ev['elapsed']}s")
    elif t == "error":
        run["status"], run["error"] = "error", ev["message"]
        run["stages"] = {k: ("error" if v == "active" else v) for k, v in run["stages"].items()}
        run["log"].append("stopped")


def render_chat(run: dict) -> None:
    with st.chat_message("user"):
        st.markdown(run["topic"])
    with st.chat_message("assistant"):
        if run["error"]:
            st.html(ui.error(run["error"]))
        else:
            st.markdown(run["answer"])


def stream_chat(prompt: str, session_id: str) -> dict:
    run = new_run(prompt, "chat")
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            run["answer"] = st.write_stream(chat_reply(prompt, session_id))
            run["status"] = "done"
        except Exception as e:
            run["status"], run["error"] = "error", str(e) or e.__class__.__name__
            st.html(ui.error(run["error"]))
    return run


def render_run(run: dict, idx: int) -> None:
    if run["kind"] == "chat":
        render_chat(run)
        return
    st.html(ui.topic(run["topic"]))
    if run["status"] != "history":
        st.html(ui.pipeline(run))
    if run["error"]:
        st.html(ui.error(run["error"]))
    if run["sources"]:
        st.html(ui.sources(run))
    if run["report"]:
        st.html(ui.eyebrow("Report"))
        with st.container(key=f"report_{idx}"):
            st.markdown(run["report"])
        st.download_button("Download .md", run["report"], file_name="report.md",
                           mime="text/markdown", key=f"dl_{idx}")
    if run["critique"]:
        st.html(ui.critique(run["critique"]))


def stream_run(topic: str, session_id: str, idx: int) -> dict:
    run = new_run(topic)
    st.html(ui.topic(topic))
    pipe = st.empty()
    err = st.empty()
    srcs = st.empty()
    report_slot = st.empty()
    crit = st.empty()
    report_box = None
    last_paint = 0.0

    for ev in research_events(topic, session_id):
        apply_event(run, ev)
        t = ev["type"]
        now = time.time()
        if t != "token" or now - last_paint > 0.5:   # keep the timer moving while writing
            pipe.html(ui.pipeline(run))
        if t in ("sources", "reading", "read_done"):
            srcs.html(ui.sources(run))
        if t == "token":
            if report_box is None:
                with report_slot.container():
                    st.html(ui.eyebrow("Report"))
                    report_box = st.container(key=f"report_{idx}").empty()
            if now - last_paint > 0.05:
                report_box.markdown(run["report"] + " ▍")
                last_paint = now
        if t == "stage" and ev["stage"] == "write" and ev["status"] == "done" and report_box:
            report_box.markdown(run["report"])
        if t == "stage" and ev["stage"] == "critique" and ev["status"] == "active":
            crit.html(ui.critique_pending())
        if t == "critique":
            crit.html(ui.critique(run["critique"]))
        if t == "error":
            err.html(ui.error(ev["message"]))
    return run


# ---------------------------------------------------------------- page

st.html(f"<style>{CSS}</style>")

# The session id lives in the URL so a refresh (or a shared link) keeps the memory.
if "session_id" not in st.session_state:
    sid = st.query_params.get("s", "")
    st.session_state.session_id = sid if 8 <= len(sid) <= 64 and sid.replace("-", "").isalnum() else str(uuid.uuid4())
    st.session_state.runs = [history_run(i) for i in memory.get_history(st.session_state.session_id)]
st.query_params["s"] = st.session_state.session_id
session_id = st.session_state.session_id
runs = st.session_state.runs
missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]

with st.sidebar:
    st.html(ui.brand())
    if st.button("＋  New session", key="new_session", width="stretch"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.runs = []
        st.query_params["s"] = st.session_state.session_id
        st.rerun()
    reports = [r for r in runs if r["kind"] == "research"]
    if reports:
        st.html(ui.sidebar_runs(reports))
    st.caption('Tip: type **cricket** to chat, or **research cricket** for a cited report.')
    st.html(ui.status(not missing, os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")))

topic = st.chat_input(
    "Ask a follow-up, or start with “research” for a cited report…" if runs
    else "Chat about anything, or type “research <topic>” for a report…",
    max_chars=300,
    disabled=bool(missing),
    key="prompt",  # stable identity even though the placeholder text changes
)
topic = topic or st.session_state.pop("suggested", None)

# Always emit this slot (even empty) so the hero doesn't linger, greyed out, mid-run.
hero_slot = st.empty()
if not runs and not topic:
    with hero_slot.container():
        st.html(ui.hero(missing))
        with st.container(key="chips", horizontal=True, horizontal_alignment="center"):
            for s in SUGGESTIONS:
                if st.button(s, key=f"chip_{s}", disabled=bool(missing)):
                    st.session_state.suggested = s
                    st.rerun()

for i, r in enumerate(runs):
    with st.container(key=f"run_{i}"):
        render_run(r, i)

if topic and (topic := topic.strip()):
    client = st.context.ip_address or session_id
    mode, text = route(topic)
    if mode == "research" and len(text) < 2:
        st.toast("Tell me what to research, e.g. “research cricket”.")
    elif len(text) < 1:
        st.toast("Give the agent a little more to go on.")
    elif rate_limited(client, mode):
        limit = RATE_LIMIT_PER_HOUR if mode == "research" else CHAT_LIMIT_PER_HOUR
        st.html(ui.error(f"Limit of {limit} {mode} requests per hour reached. Try again later."))
    elif mode == "chat":
        with st.container(key=f"run_{len(runs)}"):
            runs.append(stream_chat(text, session_id))
    else:
        with st.container(key=f"run_{len(runs)}"):
            finished = stream_run(text, session_id, len(runs))
        runs.append(finished)
        if finished["status"] == "done":
            st.rerun()  # refresh the sidebar list and swap live placeholders for static blocks
