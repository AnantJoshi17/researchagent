"""Session memory: every finished report is stored in SQLite under its session id, and the
most recent ones are fed back into the next run so follow-ups like "what are its risks?"
resolve against what was researched before."""

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", Path(__file__).parent / "data" / "research.db"))
MEMORY_RUNS = 3          # how many earlier reports the next run sees
MEMORY_EXCERPT_CHARS = 600

_lock = threading.Lock()


@contextmanager
def _db():
    """Open, commit on success, and always close (sqlite3's own `with` only commits)."""
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            report TEXT NOT NULL,
            sources TEXT NOT NULL,
            read_urls TEXT NOT NULL,
            critique TEXT NOT NULL,
            created_at REAL NOT NULL
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON reports(session_id, id)")
    return conn


def save_report(session_id: str, topic: str, report: str, sources: list, read_urls: list,
                critique: dict) -> int:
    with _lock, _db() as conn:
        cur = conn.execute(
            "INSERT INTO reports (session_id, topic, report, sources, read_urls, critique, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, topic, report, json.dumps(sources), json.dumps(read_urls),
             json.dumps(critique), time.time()),
        )
        return cur.lastrowid


def get_history(session_id: str) -> list[dict]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM reports WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
    return [
        {
            "id": r["id"],
            "topic": r["topic"],
            "report": r["report"],
            "sources": json.loads(r["sources"]),
            "read_urls": json.loads(r["read_urls"]),
            "critique": json.loads(r["critique"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def clear_session(session_id: str) -> None:
    with _lock, _db() as conn:
        conn.execute("DELETE FROM reports WHERE session_id = ?", (session_id,))


def memory_context(session_id: str | None) -> str:
    """A compact digest of the session's last few reports, oldest first."""
    if not session_id:
        return ""
    with _db() as conn:
        rows = conn.execute(
            "SELECT topic, report FROM reports WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, MEMORY_RUNS),
        ).fetchall()
    parts = []
    for i, r in enumerate(reversed(rows), start=1):
        excerpt = " ".join(r["report"].split())[:MEMORY_EXCERPT_CHARS]
        parts.append(f"{i}. Topic: {r['topic']}\n   Summary: {excerpt}")
    return "\n".join(parts)
