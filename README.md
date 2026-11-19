# researchagent

A multi-agent research assistant with a Streamlit web app.

- Type **`cricket`** and it chats like a normal chatbot (about 2 seconds).
- Type **`research cricket`** and a team of agents searches the web, reads the best
  pages, writes a cited markdown report and has a critic score it out of 10 (about 4–8 seconds).

Any message containing the word "research" becomes a report. Everything else is chat.

```
              ┌─► chat agent (LLM, streamed)
user ─► router┤
              └─► search agent ─► reader agent ─► writer agent ─► critic agent
                  (Tavily)        (3 pages,       (LLM, streams   (LLM, score /10)
                                   in parallel)    cited report)
                         ▲
        session memory (SQLite): earlier chats and reports feed follow-ups
```

**Why it's fast:** only the agents that need judgement call the LLM. The router is a keyword
rule, search is a single Tavily call that also returns page text, and the reader fetches pages
in parallel. So a report makes 2 LLM calls instead of about 6. Follow-ups add one short
call that rewrites "its impact" into a full search query. gpt-oss models run with
`reasoning_effort=low`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env   # then fill in your Groq and Tavily keys
```

Keys: Groq at <https://console.groq.com/keys>, Tavily at <https://app.tavily.com> (both have free tiers).

## Run

```bash
streamlit run streamlit_app.py          # web app -> http://localhost:8501
python3 main.py "research cricket"      # CLI report
python3 main.py "cricket"               # CLI chat
python3 main.py                         # CLI interactive chat loop
```

Preview the web app without API keys (fake LLM, canned results): `streamlit run dev_demo.py`

Tests (offline, no keys needed): `python3 -m unittest discover tests`

## Files

| File | What it does |
|---|---|
| `agents.py` | Router, chat agent, search agent, reader agent, query rewrite for follow-ups |
| `chains.py` | Writer agent and critic agent (prompt + LLM chains) |
| `tools.py` | Tavily web search and a BeautifulSoup page-scraper fallback |
| `pipeline.py` | Runs the research steps and yields progress events. Also stores chat replies |
| `memory.py` | Session memory in SQLite (chats and reports) |
| `streamlit_app.py`, `st_blocks.py`, `streamlit.css` | Web app |
| `main.py` | CLI |

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub: `git push origin <branch>`.
2. Go to [share.streamlit.io](https://share.streamlit.io), click **Create app**, pick the
   repo and branch, and set the main file to `streamlit_app.py`.
3. Under **Advanced settings**, choose Python 3.12 or newer, and paste into **Secrets**:
   ```toml
   GROQ_API_KEY = "your_groq_key"
   TAVILY_API_KEY = "your_tavily_key"
   ```
4. Click **Deploy**.

Every request spends your Groq and Tavily quota, so each visitor is limited to
`RATE_LIMIT_PER_HOUR` reports (default 20) and `CHAT_LIMIT_PER_HOUR` chats (default 120).
Session history lives in a local SQLite file, so on Streamlit Cloud it resets whenever the
app restarts.

## Settings (`.env` or Streamlit secrets)

| Name | Default | Meaning |
|---|---|---|
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Any Groq chat model |
| `GROQ_REASONING_EFFORT` | `low` | Only used for gpt-oss / qwen models |
| `MAX_SOURCES` | `3` | Pages the reader passes to the writer |
| `RATE_LIMIT_PER_HOUR` | `20` | Reports per visitor per hour |
| `CHAT_LIMIT_PER_HOUR` | `120` | Chat messages per visitor per hour |
