"""HTML fragments for the Streamlit UI. Everything that came from the model or the web is
escaped before it goes into markup."""

from html import escape
from urllib.parse import urlparse

STAGES = [
    ("search", "Search", "Querying the web"),
    ("read", "Read", "Scraping top sources"),
    ("write", "Write", "Drafting the report"),
    ("critique", "Critique", "Scoring the draft"),
]

CHECK = (
    '<svg viewBox="0 0 16 16" width="14" height="14"><path d="M3.5 8.5l3 3 6-7" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


def domain_of(url: str) -> str:
    host = urlparse(url).hostname or url
    return host.removeprefix("www.")


def _hue(text: str) -> int:
    h = 0
    for c in text:
        h = (h * 31 + ord(c)) % 360
    return h


def hero(missing: list[str]) -> str:
    banner = (
        f'<div class="ra-banner">Missing {escape(" and ".join(missing))}. Add them to the app\'s '
        "secrets (Streamlit Cloud: ⋮ → Settings → Secrets) or to <code>.env</code> locally.</div>"
        if missing else ""
    )
    return f"""
<div class="ra-hero">
  {banner}
  <span class="ra-hero-pill"><span class="ra-dot ok"></span>chat · search · read · write · critique</span>
  <h1>Ask anything. <span class="ra-accent">Research it properly.</span></h1>
  <p>Type <b>cricket</b> and it chats. Type <b>research cricket</b> and a team of agents
  searches the web, reads the strongest sources, writes a cited report, then a strict editor
  scores it. Follow-ups remember the session.</p>
</div>"""


def topic(text: str) -> str:
    return f'<h2 class="ra-topic">{escape(text)}</h2>'


def pipeline(run: dict) -> str:
    nodes = []
    for i, (key, label, hint) in enumerate(STAGES, start=1):
        state = run["stages"][key]
        inner = CHECK if state == "done" else f'<span class="ra-num">{i}</span>'
        nodes.append(
            f'<div class="ra-stage ra-stage-{state}"><span class="ra-node">{inner}</span>'
            f'<span class="ra-stage-text"><span class="ra-stage-label">{label}</span>'
            f'<span class="ra-stage-hint">{hint}</span></span></div>'
        )
    running = run["status"] == "running"
    activity = escape(run["log"][-1] if run["log"] else "waiting")
    caret = '<span class="ra-caret"></span>' if running else ""
    return f"""
<div class="ra-glass ra-pipeline">
  <div class="ra-track">{''.join(nodes)}</div>
  <div class="ra-pipe-foot"><span class="ra-activity {'live' if running else ''}">
    <span class="ra-prompt">›</span> {activity}{caret}</span>
    <span class="ra-elapsed">{run['elapsed']:.1f}s</span></div>
</div>"""


def sources(run: dict) -> str:
    items = run["sources"]
    if not items:
        return ""
    read, reading = set(run["read_urls"]), set(run["reading"])
    cards = []
    for i, s in enumerate(items):
        domain = domain_of(s["url"])
        state = "read" if s["url"] in read else "reading" if s["url"] in reading else ""
        badge = f'<span class="ra-badge ra-badge-{state}">{state}</span>' if state else ""
        title = domain if s["title"] == s["url"] else s["title"]
        why = f'<span class="ra-why">{escape(s["why"])}</span>' if s.get("why") else ""
        cards.append(
            f'<a class="ra-glass ra-source {state}" href="{escape(s["url"], quote=True)}" target="_blank" '
            f'rel="noreferrer" style="--hue:{_hue(domain)};--i:{i}">'
            f'<span class="ra-source-top"><span class="ra-fav">{escape(domain[:1].upper())}</span>'
            f'<span class="ra-domain">{escape(domain)}</span>{badge}</span>'
            f'<span class="ra-source-title">{escape(title)}</span>{why}</a>'
        )
    read_count = f'<span class="ra-count read">{len(read)} read</span>' if read else ""
    return (
        f'<h3 class="ra-eyebrow">Sources <span class="ra-count">{len(items)}</span>{read_count}</h3>'
        f'<div class="ra-sources">{"".join(cards)}</div>'
    )


def eyebrow(text: str) -> str:
    return f'<h3 class="ra-eyebrow">{escape(text)}</h3>'


def critique_pending() -> str:
    lines = "".join(f'<span class="ra-shimmer" style="width:{w}%"></span>' for w in (60, 85, 40))
    return f'{eyebrow("Critic review")}<div class="ra-glass ra-critique-pending">{lines}</div>'


def critique(c: dict) -> str:
    score = c.get("score")
    tone = "muted" if not score else "good" if score >= 8 else "mid" if score >= 5 else "low"
    strengths = "".join(f"<li>{escape(s)}</li>" for s in c.get("strengths", []))
    weaknesses = "".join(f"<li>{escape(s)}</li>" for s in c.get("weaknesses", []))
    return f"""
{eyebrow("Critic review")}
<div class="ra-glass ra-critique">
  <div class="ra-score">
    <div class="ra-ring-wrap">
      <div class="ra-ring tone-{tone}" style="--pct:{(score or 0) * 10}%"></div>
      <span class="ra-ring-value"><b>{score if score else '–'}</b><i>/10</i></span>
    </div>
    <p class="ra-verdict">{escape(c.get('verdict') or 'No verdict returned.')}</p>
  </div>
  <div class="ra-lists">
    <div><h4 class="ra-list-title good">Strengths</h4><ul>{strengths}</ul></div>
    <div><h4 class="ra-list-title low">Weaknesses</h4><ul>{weaknesses}</ul></div>
  </div>
</div>"""


def error(message: str) -> str:
    return f'<div class="ra-error"><strong>Something went wrong.</strong> {escape(message)}</div>'


def sidebar_runs(runs: list[dict]) -> str:
    rows = []
    for r in runs:
        score = (r.get("critique") or {}).get("score")
        tone = "" if not score else "good" if score >= 8 else "mid" if score >= 5 else "low"
        pill = f'<span class="ra-pill {tone}">{score}</span>' if score else ""
        rows.append(f'<div class="ra-nav-item"><span>{escape(r["topic"])}</span>{pill}</div>')
    return f'<h4 class="ra-eyebrow">This session</h4>{"".join(rows)}' if rows else ""


def brand() -> str:
    return '<div class="ra-brand">researchagent</div>'


def status(ok: bool, model: str) -> str:
    text = escape(model) if ok else "API keys missing"
    return f'<div class="ra-status"><span class="ra-dot {"ok" if ok else "bad"}"></span>{text}</div>'
