"""Offline tests: the LLM and the web are replaced by fakes, so no API keys are needed.

Run:  python -m unittest discover tests
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.db")
os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("TAVILY_API_KEY", "test")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

import agents  # noqa: E402
import pipeline  # noqa: E402
from chains import parse_critique  # noqa: E402


class ToolFakeModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


SEARCH_SCRIPT = [
    AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "ai healthcare"}, "id": "1"}]),
    AIMessage(content=(
        "1. **AI in Diagnostics** - https://a.example.com/diag - peer-reviewed overview\n"
        "2. [WHO report](https://who.example.org/ai) - policy view\n"
        "Some chatter with no link\n"
        "3. Duplicate - https://a.example.com/diag - again"
    )),
]
READER_SCRIPT = [
    AIMessage(content="", tool_calls=[
        {"name": "scrape_url", "args": {"url": "https://a.example.com/diag"}, "id": "2"},
        {"name": "scrape_url", "args": {"url": "https://who.example.org/ai"}, "id": "3"},
    ]),
    AIMessage(content="SOURCE: https://a.example.com/diag\nFact one.\n\nSOURCE: https://who.example.org/ai\nFact two."),
]


@tool
def web_search(query: str) -> str:
    """fake search"""
    return "results"


@tool
def scrape_url(url: str) -> str:
    """fake scrape"""
    return f"page text of {url}"


def _fake_llms():
    scripts = iter([SEARCH_SCRIPT, READER_SCRIPT])
    return lambda temperature=0.3: ToolFakeModel(messages=iter(next(scripts)))


def _patched():
    return [
        mock.patch.object(agents, "get_llm", _fake_llms()),
        mock.patch.object(agents, "web_search", web_search),
        mock.patch.object(agents, "scrape_url", scrape_url),
        mock.patch.object(pipeline, "stream_writer_chain",
                          lambda *a: iter(["## Introduction\n", "Body [s](https://a.example.com/diag)"])),
        mock.patch.object(pipeline, "run_critic_chain",
                          lambda *a: {"score": 8, "verdict": "Solid.", "strengths": ["a"], "weaknesses": ["b"]}),
    ]


class PipelineTest(unittest.TestCase):
    def setUp(self):
        for p in _patched():
            p.start()
            self.addCleanup(p.stop)

    def test_events_end_to_end_and_memory(self):
        events = list(pipeline.research_events("AI in healthcare", "session-1234"))
        types = [e["type"] for e in events]
        self.assertNotIn("error", types, events)
        self.assertEqual(types[-1], "done")

        sources = next(e for e in events if e["type"] == "sources")["items"]
        self.assertEqual([s["url"] for s in sources],
                         ["https://a.example.com/diag", "https://who.example.org/ai"])
        self.assertEqual(sources[0]["title"], "AI in Diagnostics")
        self.assertEqual(sources[1]["title"], "WHO report")

        reading = [e["url"] for e in events if e["type"] == "reading"]
        self.assertEqual(len(reading), 2)
        self.assertEqual(next(e for e in events if e["type"] == "query")["query"], "ai healthcare")
        self.assertEqual(next(e for e in events if e["type"] == "critique")["score"], 8)

        stages = [(e["stage"], e["status"]) for e in events if e["type"] == "stage"]
        self.assertEqual([s for s, st in stages if st == "done"], ["search", "read", "write", "critique"])

        import memory
        history = memory.get_history("session-1234")
        self.assertEqual(history[-1]["topic"], "AI in healthcare")
        self.assertIn("AI in healthcare", memory.memory_context("session-1234"))

    def test_cli_appends_critique(self):
        with mock.patch("builtins.print"):
            report = pipeline.run_research("AI in healthcare")
        self.assertIn("Critic score: 8/10", report)


class ParseCritiqueTest(unittest.TestCase):
    def test_fenced_json(self):
        raw = '```json\n{"score": 7, "verdict": "ok", "strengths": ["x"], "weaknesses": ["y", "z"]}\n```'
        self.assertEqual(parse_critique(raw)["score"], 7)

    def test_clamps_and_falls_back(self):
        self.assertEqual(parse_critique('{"score": 14}')["score"], 10)
        self.assertEqual(parse_critique("I would give this 6/10 overall.")["score"], 6)
        self.assertIsNone(parse_critique("no idea")["score"])



if __name__ == "__main__":
    unittest.main()
