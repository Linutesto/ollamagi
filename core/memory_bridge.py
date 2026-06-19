"""Read from and write to Hermes cognitive_memory.sqlite."""
import sqlite3
import hashlib
import time
from pathlib import Path
from typing import Optional
from core.config import HERMES_DB


def _connect():
    db = sqlite3.connect(HERMES_DB, timeout=10)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=30000")
    db.row_factory = sqlite3.Row
    return db


def get_relevant_context(query: str, limit: int = 8) -> list[dict]:
    """Pull beliefs + memories relevant to query (keyword match fallback — no embeddings needed)."""
    terms = [t.lower() for t in query.split() if len(t) > 3]
    if not terms:
        return []
    like_clause = " OR ".join(f"LOWER(statement) LIKE '%{t}%'" for t in terms[:5])
    mem_clause = " OR ".join(f"LOWER(content) LIKE '%{t}%'" for t in terms[:5])
    results = []
    try:
        with _connect() as db:
            rows = db.execute(
                f"SELECT statement, confidence, source FROM beliefs WHERE {like_clause} "
                f"ORDER BY confidence DESC, evidence_count DESC LIMIT {limit}"
            ).fetchall()
            for r in rows:
                results.append({"type": "belief", "content": r["statement"],
                                 "confidence": r["confidence"], "source": r["source"]})
            mem_rows = db.execute(
                f"SELECT content, created_at FROM memories WHERE {mem_clause} "
                f"ORDER BY created_at DESC LIMIT {limit}"
            ).fetchall()
            for r in mem_rows:
                results.append({"type": "memory", "content": r["content"], "source": "memory"})
    except Exception:
        pass
    return results[:limit]


def store_memory(content: str, source: str, tags: list[str] | None = None) -> bool:
    """Store a new memory in Hermes."""
    try:
        with _connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO memories (content, source, created_at, tags) "
                "VALUES (?, ?, ?, ?)",
                (content, source, int(time.time()), ",".join(tags or []))
            )
        return True
    except Exception:
        return False


def store_belief(content: str, confidence: float, source: str) -> bool:
    """Store or reinforce a belief in Hermes."""
    try:
        with _connect() as db:
            existing = db.execute(
                "SELECT id, evidence_count FROM beliefs WHERE statement = ?", (content,)
            ).fetchone()
            if existing:
                db.execute(
                    "UPDATE beliefs SET evidence_count = evidence_count + 1, "
                    "confidence = MIN(confidence + 0.05, 1.0), updated_at = ? WHERE id = ?",
                    (int(time.time()), existing["id"])
                )
            else:
                db.execute(
                    "INSERT INTO beliefs (statement, confidence, source, created_at, updated_at, "
                    "evidence_count, status) VALUES (?, ?, ?, ?, ?, 1, 'active')",
                    (content, confidence, source, int(time.time()), int(time.time()))
                )
        return True
    except Exception:
        return False


def get_goals(status: str = "active", limit: int = 10) -> list[dict]:
    """Get current Hermes goals."""
    try:
        with _connect() as db:
            rows = db.execute(
                "SELECT title, description, priority FROM goals "
                "ORDER BY priority DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        return []


def context_for_task(task_description: str) -> str:
    """Build a context string from Hermes for injection into task prompts."""
    items = get_relevant_context(task_description, limit=6)
    if not items:
        return ""
    lines = ["[HERMES CONTEXT — from prior sessions]"]
    for item in items:
        prefix = "BELIEF" if item["type"] == "belief" else "MEMORY"
        conf = f" (conf={item.get('confidence', '?'):.2f})" if item["type"] == "belief" else ""
        lines.append(f"• [{prefix}{conf}] {item['content'][:200]}")
    return "\n".join(lines)
