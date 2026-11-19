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

import agents  # noqa: E402
import memory  # noqa: E402
import pipeline  # noqa: E402
from chains import parse_critique  # noqa: E402

RESULTS = [
    {"title": "AI in Diagnostics", "url": "https://a.example.com/diag",
     "snippet": "peer-reviewed overview", "text": "Fact one. " * 60},
    {"title": "WHO report", "url": "https://who.example.org/ai",
     "snippet": "policy view", "text": ""},          # too thin: reader must scrape it
    {"title": "Blog", "url": "https://blog.example.net/x", "snippet": "opinion", "text": "Fact three. " * 60},
    {"title": "Forum", "url": "https://forum.example.net/y", "snippet": "chatter", "text": "Fact four. " * 60},
]


class RouterTest(unittest.TestCase):
    def test_routes(self):
        cases = {
            "cricket": ("chat", "cricket"),
            "research cricket": ("research", "cricket"),
            "Research: cricket": ("research", "cricket"),
            "cricket research": ("research", "cricket"),
            "can you do a research report on the history of cricket?":
                ("research", "the history of cricket"),
            "write a research report about AI in healthcare": ("research", "AI in healthcare"),
            "who are the best researchers?": ("chat", "who are the best researchers?"),
            "research": ("research", ""),
        }
        for prompt, expected in cases.items():
            self.assertEqual(agents.route(prompt), expected, prompt)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        patches = [
            mock.patch("tools.web_search", lambda q, max_results=5: [dict(r) for r in RESULTS]),
            mock.patch("tools.scrape_url", lambda url: "Scraped WHO text. " * 30),
            mock.patch.object(agents, "standalone_query", lambda topic, memory="": f"q:{topic}"),
            mock.patch.object(pipeline, "stream_writer_chain",
                              lambda *a: iter(["## Introduction\n", "Body [s](https://a.example.com/diag)"])),
            mock.patch.object(pipeline, "run_critic_chain",
                              lambda *a: {"score": 8, "verdict": "Solid.", "strengths": ["a"], "weaknesses": ["b"]}),
            mock.patch.object(agents, "stream_chat", lambda p, h, m="": iter(["Cricket is ", "a bat-and-ball game."])),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_research_events_end_to_end_and_memory(self):
        events = list(pipeline.research_events("AI in healthcare", "session-1234"))
        types = [e["type"] for e in events]
        self.assertNotIn("error", types, events)
        self.assertEqual(types[-1], "done")
        self.assertEqual(next(e for e in events if e["type"] == "query")["query"], "q:AI in healthcare")

        sources = next(e for e in events if e["type"] == "sources")["items"]
        self.assertEqual(len(sources), 4)
        self.assertEqual(sources[0]["title"], "AI in Diagnostics")

        read_urls = next(e for e in events if e["type"] == "read_done")["urls"]
        self.assertEqual(len(read_urls), agents.MAX_SOURCES)
        self.assertIn("https://who.example.org/ai", read_urls)   # scraped fallback worked

        stages = [(e["stage"], e["status"]) for e in events if e["type"] == "stage"]
        self.assertEqual([s for s, st in stages if st == "done"], ["search", "read", "write", "critique"])
        self.assertEqual(next(e for e in events if e["type"] == "critique")["score"], 8)

        history = memory.get_history("session-1234")
        self.assertEqual(history[-1]["kind"], "research")
        self.assertIn("AI in healthcare", memory.memory_context("session-1234"))

    def test_chat_reply_is_stored(self):
        answer = "".join(pipeline.chat_reply("cricket", "session-chat"))
        self.assertEqual(answer, "Cricket is a bat-and-ball game.")
        self.assertEqual(memory.chat_history("session-chat")[0], {"role": "user", "content": "cricket"})
        self.assertEqual(memory.get_history("session-chat")[0]["kind"], "chat")

    def test_search_failure_is_reported(self):
        with mock.patch("tools.web_search", side_effect=RuntimeError("Web search failed: bad key")):
            events = list(pipeline.research_events("x", None))
        self.assertEqual(events[-1], {"type": "error", "message": "Web search failed: bad key"})

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
