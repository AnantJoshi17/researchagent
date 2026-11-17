# researchagent

A research agent that searches the web, reads the strongest sources, writes a cited
markdown report, and has a critic score it out of 10. Use it from the CLI or the
Streamlit web app, which shows every step live.

```
search agent ──► reader agent ──► writer chain ──► critic chain
 (Tavily)        (scrapes 3 pages)  (streams report)  (score /10, temp 0)
                         ▲
               session memory (SQLite): earlier reports feed follow-ups
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env   # then fill in your Groq and Tavily keys
```

## Run

```bash
streamlit run streamlit_app.py                 # web app -> http://localhost:8501
python3 main.py "impact of AI on healthcare"   # CLI
```

Preview the web app without API keys (fake LLM, canned results): `streamlit run dev_demo.py`

Tests (offline, no keys needed): `python3 -m unittest discover tests`

## Files

| File | What it does |
|---|---|
| `tools.py` | `web_search` (Tavily) and `scrape_url` (BeautifulSoup) tools |
| `agents.py` | Search agent and reader agent (reads up to `MAX_SOURCES` pages) |
| `chains.py` | Writer chain and critic chain |
| `pipeline.py` | Runs the four steps and yields progress events |
| `memory.py` | Session memory in SQLite |
| `streamlit_app.py`, `st_blocks.py`, `streamlit.css` | Web app |
| `main.py` | CLI |

## Deploy on Streamlit Community Cloud

1. Push this folder to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), click **Create app**, pick the
   repo and branch, and set the main file to `streamlit_app.py`.
3. Under **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your_groq_key"
   TAVILY_API_KEY = "your_tavily_key"
   ```
4. Click **Deploy**.

Every report spends your Groq and Tavily quota, so the app limits each visitor to
`RATE_LIMIT_PER_HOUR` reports (default 20). Session history lives in a local SQLite file,
so on Streamlit Cloud it resets whenever the app restarts.

## Progress

- [x] Web search tool (Tavily)
- [x] Page scraper tool (BeautifulSoup)
- [x] Search agent
- [x] Reader agent
- [x] Writer chain
- [x] Critic chain - scores the report out of 10
- [x] Read multiple sources instead of one
- [x] Web interface (Streamlit)
- [x] Session memory
- [x] Deployment (Streamlit Community Cloud)
