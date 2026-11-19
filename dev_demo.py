"""Preview the Streamlit UI without API keys.

Runs the real app and pipeline, but swaps the LLM and web steps for canned, timed fakes
so every stage of the UI animates. Never use this for real research.

    streamlit run dev_demo.py
"""

import os
import time
from pathlib import Path

os.environ.setdefault("GROQ_API_KEY", "demo")
os.environ.setdefault("TAVILY_API_KEY", "demo")
os.environ.setdefault("GROQ_MODEL", "demo-mode (fake LLM)")
os.environ.setdefault("DB_PATH", "data/demo.db")

import pipeline  # noqa: E402

RESULTS = [
    {"title": "Artificial intelligence in healthcare: past, present and future",
     "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5829945/", "snippet": "Widely cited review of clinical AI"},
    {"title": "WHO guidance on ethics & governance of AI for health",
     "url": "https://www.who.int/publications/i/item/9789240029200", "snippet": "Global policy framework"},
    {"title": "AI in radiology: opportunities and challenges",
     "url": "https://www.nature.com/articles/s41568-018-0016-5", "snippet": "Diagnostic imaging deep dive"},
    {"title": "The potential for artificial intelligence in healthcare",
     "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6616181/", "snippet": "Adoption barriers and outlook"},
]

REPORT = """## Introduction
Artificial intelligence is moving from research labs into everyday clinical work, from reading scans to triaging patients. The sources agree on its promise but differ on how fast safe adoption can happen.

## Key Findings
- **Diagnostic imaging is the most mature use case.** Deep-learning models match specialist accuracy on several radiology and pathology tasks [source](https://www.nature.com/articles/s41568-018-0016-5).
- **Governance lags behind capability.** The WHO sets out six principles, including transparency and accountability, that most deployments do not yet meet [source](https://www.who.int/publications/i/item/9789240029200).
- **Data quality is the main bottleneck.** Reviews point to fragmented, biased records as the biggest barrier to models that generalise [source](https://pmc.ncbi.nlm.nih.gov/articles/PMC5829945/).
- **Operational gains arrive first.** Scheduling and documentation tools show measurable time savings before clinical tools reach scale.

## Sources
- [AI in healthcare: past, present and future](https://pmc.ncbi.nlm.nih.gov/articles/PMC5829945/)
- [WHO guidance on AI for health](https://www.who.int/publications/i/item/9789240029200)
- [AI in radiology](https://www.nature.com/articles/s41568-018-0016-5)
"""


def fake_search(query):
    time.sleep(0.8)
    return [{**r, "text": "page text " * 50} for r in RESULTS]


def fake_read(results):
    time.sleep(0.8)
    return results[:3]


def fake_chat(prompt, history, memory=""):
    answer = (f"Here's a quick take on **{prompt}**. This is demo mode, so the answer is canned. "
              f"Type **research {prompt}** to see the full report pipeline animate.")
    for word in answer.split(" "):
        time.sleep(0.03)
        yield word + " "


def fake_writer(*args):
    for word in REPORT.split(" "):
        time.sleep(0.018)
        yield word + " "


def fake_critic(*args):
    time.sleep(1.6)
    return {
        "score": 8,
        "verdict": "Well grounded and clearly cited, but light on numbers.",
        "strengths": ["Every claim maps to a cited source.", "Notes where the sources disagree."],
        "weaknesses": ["Few concrete figures or effect sizes.", "Operational claim has no citation."],
    }


pipeline.agents.search = fake_search
pipeline.agents.read = fake_read
pipeline.agents.stream_chat = fake_chat
pipeline.agents.standalone_query = lambda topic, memory="": topic
pipeline.stream_writer_chain = fake_writer
pipeline.run_critic_chain = fake_critic

exec((Path(__file__).parent / "streamlit_app.py").read_text())
