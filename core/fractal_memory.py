"""
Fractal Memory — hierarchical semantic SQLite memory.

Levels:
  0 (leaves)  : raw memories — facts, code, observations
  1 (concepts): clusters of related memories
  2 (domains) : clusters of concepts
  3+ (meta)   : higher-order abstractions

Every node has the same shape: content + embedding + parent + children.
That self-similarity at every scale is the fractal property.

Insert: embed → find nearest cluster → join or create → update centroid → propagate
Query:  embed → beam-search from top level downward → collect leaves + lineage
"""
import json
import re
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
FRACTAL_DB    = Path(__file__).parent.parent / "fractal_memory.db"
EMBED_MODEL   = "mxbai-embed-large"
EMBED_DIM     = 1024
EMBED_URL     = "http://localhost:11434/api/embed"
MAX_LEVEL     = 4          # 0..3 (3 = root meta-concepts)
JOIN_THRESH   = 0.52       # empirically validated for mxbai-embed-large (within-topic μ=0.56, cross-topic μ=0.44)
SPLIT_AT      = 12         # children before splitting a cluster
BEAM_WIDTH    = 6          # candidates kept per level during query
DIRECT_SCAN_LIMIT = 2000  # below this leaf count, skip beam search and scan leaves directly


# ── Low-level helpers ─────────────────────────────────────────────────────────
def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)

def _unpack(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))

def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))

def _centroid(vecs: list[list[float]]) -> list[float]:
    arr = np.array(vecs, dtype=np.float32)
    return arr.mean(axis=0).tolist()


# ── DB connection (per-thread) ────────────────────────────────────────────────
_local = threading.local()
_write_lock = threading.RLock()


def _conn(path: Path = FRACTAL_DB) -> sqlite3.Connection:
    c = getattr(_local, "conn", None)
    if c is None:
        c = sqlite3.connect(str(path), check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA busy_timeout=15000")
        _local.conn = c
    return c


# ── Schema ────────────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    level        INTEGER NOT NULL DEFAULT 0,
    content      TEXT    NOT NULL,
    label        TEXT,
    embedding    BLOB,
    parent_id    INTEGER REFERENCES nodes(id),
    child_count  INTEGER DEFAULT 0,
    access_count INTEGER DEFAULT 0,
    created_at   REAL    DEFAULT (unixepoch('now','subsec')),
    last_accessed REAL   DEFAULT (unixepoch('now','subsec')),
    flow_id      TEXT    DEFAULT '',
    tags         TEXT    DEFAULT '[]',
    metadata     TEXT    DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_nodes_level  ON nodes(level);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_flow   ON nodes(flow_id);
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts
    USING fts5(content, tags, content=nodes, content_rowid=id);
CREATE TRIGGER IF NOT EXISTS nodes_fts_insert
    AFTER INSERT ON nodes BEGIN
        INSERT INTO nodes_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
    END;
CREATE TRIGGER IF NOT EXISTS nodes_fts_delete
    AFTER DELETE ON nodes BEGIN
        INSERT INTO nodes_fts(nodes_fts, rowid, content, tags)
            VALUES ('delete', old.id, old.content, old.tags);
    END;
"""


def _init_db():
    with _write_lock:
        db = _conn()
        db.executescript(_SCHEMA)
        db.commit()


# ── Embedding ─────────────────────────────────────────────────────────────────
def _embed(text: str) -> list[float]:
    try:
        r = httpx.post(EMBED_URL,
                       json={"model": EMBED_MODEL, "input": text[:4096]},
                       timeout=30.0)
        r.raise_for_status()
        return r.json()["embeddings"][0]
    except Exception:
        return []   # empty = no embedding; FTS fallback still works


# ── Core tree logic ───────────────────────────────────────────────────────────
def _children_embeddings(parent_id: int) -> list[tuple[int, list[float]]]:
    """Return (child_id, embedding) for all direct children with embeddings."""
    db = _conn()
    rows = db.execute(
        "SELECT id, embedding FROM nodes WHERE parent_id=? AND embedding IS NOT NULL",
        (parent_id,)
    ).fetchall()
    return [(r["id"], _unpack(r["embedding"])) for r in rows]


def _update_centroid(cluster_id: int):
    """Recompute a cluster node's embedding as the centroid of its children."""
    children = _children_embeddings(cluster_id)
    if not children:
        return
    centroid = _centroid([e for _, e in children])
    with _write_lock:
        db = _conn()
        db.execute("UPDATE nodes SET embedding=? WHERE id=?",
                   (_pack(centroid), cluster_id))
        db.commit()


def _kmeans2(vecs: list[list[float]]) -> tuple[list[int], list[int]]:
    """Split vectors into 2 groups by k-means (k=2, 10 iters)."""
    arr = np.array(vecs, dtype=np.float32)
    # Init: pick two most distant points
    c0 = arr[0]
    dists = np.linalg.norm(arr - c0, axis=1)
    c1 = arr[np.argmax(dists)]
    for _ in range(10):
        d0 = np.linalg.norm(arr - c0, axis=1)
        d1 = np.linalg.norm(arr - c1, axis=1)
        labels = (d1 < d0).astype(int)
        if labels.sum() == 0 or labels.sum() == len(labels):
            break
        c0 = arr[labels == 0].mean(axis=0)
        c1 = arr[labels == 1].mean(axis=0)
    g0 = [i for i, l in enumerate(labels) if l == 0]
    g1 = [i for i, l in enumerate(labels) if l == 1]
    return g0 or [0], g1 or [1]


def _split_cluster(cluster_id: int, level: int):
    """Split an overgrown cluster into two sibling clusters."""
    db = _conn()
    children = db.execute(
        "SELECT id, content, embedding FROM nodes WHERE parent_id=?", (cluster_id,)
    ).fetchall()
    if len(children) < 2:
        return

    ids   = [r["id"]   for r in children]
    vecs  = [_unpack(r["embedding"]) if r["embedding"] else [] for r in children]
    texts = [r["content"] for r in children]

    # Only split children that have embeddings
    valid = [(i, v, t) for i, v, t in zip(ids, vecs, texts) if v]
    if len(valid) < 2:
        return

    v_ids, v_vecs, v_texts = zip(*valid)
    g0, g1 = _kmeans2(list(v_vecs))

    def _make_cluster(group_indices):
        gvecs = [v_vecs[i] for i in group_indices]
        gids  = [v_ids[i]  for i in group_indices]
        centroid = _centroid(gvecs)
        label = f"[Cluster L{level}] {v_texts[group_indices[0]][:60]}…"
        with _write_lock:
            db2 = _conn()
            cur = db2.execute(
                "INSERT INTO nodes(level,content,label,embedding,parent_id,child_count) "
                "VALUES (?,?,?,?,?,?)",
                (level, label, label, _pack(centroid), cluster_id, len(gids))
            )
            new_id = cur.lastrowid
            db2.execute(f"UPDATE nodes SET parent_id=? WHERE id IN ({','.join('?'*len(gids))})",
                        [new_id] + list(gids))
            db2.execute("UPDATE nodes SET child_count=child_count+? WHERE id=?",
                        (len(gids), cluster_id))
            db2.commit()
        return new_id, centroid

    c0_id, c0_vec = _make_cluster(g0)
    c1_id, c1_vec = _make_cluster(g1)

    # Remove direct children from old cluster (now attached to new sub-clusters)
    with _write_lock:
        db2 = _conn()
        valid_ids = list(v_ids)
        db2.execute(
            f"UPDATE nodes SET child_count=child_count-? WHERE id=?",
            (len(valid_ids), cluster_id)
        )
        db2.commit()

    # Recurse: place the two new clusters into hierarchy above
    _place_in_hierarchy(c0_id, c0_vec, level)
    _place_in_hierarchy(c1_id, c1_vec, level)


def _place_in_hierarchy(node_id: int, embedding: list[float], node_level: int):
    """Find or create a parent cluster at node_level+1 for node_id."""
    parent_level = node_level + 1
    if parent_level > MAX_LEVEL or not embedding:
        return

    db = _conn()
    clusters = db.execute(
        "SELECT id, embedding, child_count FROM nodes WHERE level=? AND embedding IS NOT NULL",
        (parent_level,)
    ).fetchall()

    best_id, best_sim = None, -1.0
    for c in clusters:
        sim = _cosine(embedding, _unpack(c["embedding"]))
        if sim > best_sim:
            best_sim, best_id = sim, c["id"]

    with _write_lock:
        db2 = _conn()
        if best_id and best_sim >= JOIN_THRESH:
            # Join existing cluster
            db2.execute("UPDATE nodes SET parent_id=?, last_accessed=? WHERE id=?",
                        (best_id, time.time(), node_id))
            db2.execute("UPDATE nodes SET child_count=child_count+1 WHERE id=?", (best_id,))
            db2.commit()
            _update_centroid(best_id)
            child_count = db2.execute(
                "SELECT child_count FROM nodes WHERE id=?", (best_id,)
            ).fetchone()["child_count"]
            if child_count >= SPLIT_AT:
                _split_cluster(best_id, parent_level)
            else:
                new_centroid = _unpack(db2.execute(
                    "SELECT embedding FROM nodes WHERE id=?", (best_id,)
                ).fetchone()["embedding"] or _pack(embedding))
                _place_in_hierarchy(best_id, new_centroid, parent_level)
        else:
            # Create new cluster
            db2 = _conn()
            node_content = db2.execute(
                "SELECT content FROM nodes WHERE id=?", (node_id,)
            ).fetchone()["content"]
            label = f"[L{parent_level}] {node_content[:80]}"
            cur = db2.execute(
                "INSERT INTO nodes(level,content,label,embedding,child_count) VALUES (?,?,?,?,1)",
                (parent_level, label, label, _pack(embedding))
            )
            cluster_id = cur.lastrowid
            db2.execute("UPDATE nodes SET parent_id=? WHERE id=?", (cluster_id, node_id))
            db2.commit()
            _place_in_hierarchy(cluster_id, embedding, parent_level)


# ── Public API ────────────────────────────────────────────────────────────────
def insert(content: str,
           flow_id: str = "",
           tags: list[str] | None = None,
           metadata: dict | None = None) -> int:
    """Insert a new memory leaf and self-organize into the fractal hierarchy."""
    _init_db()
    embedding = _embed(content)

    with _write_lock:
        db = _conn()
        cur = db.execute(
            "INSERT INTO nodes(level,content,embedding,flow_id,tags,metadata) "
            "VALUES (0,?,?,?,?,?)",
            (content,
             _pack(embedding) if embedding else None,
             flow_id,
             json.dumps(tags or []),
             json.dumps(metadata or {}))
        )
        node_id = cur.lastrowid
        db.commit()

    if embedding:
        _place_in_hierarchy(node_id, embedding, node_level=0)

    return node_id


def query(text: str,
          limit: int = 8,
          flow_id: str | None = None) -> list[dict]:
    """
    Semantic beam-search through the fractal hierarchy.
    Returns leaf memories (level 0) ranked by similarity, each with lineage.
    Falls back to FTS when embeddings are unavailable.
    """
    _init_db()
    q_emb = _embed(text)
    db = _conn()

    if not q_emb:
        # FTS fallback — strip FTS5 special chars (hyphens, quotes, parens) to avoid parse errors
        clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        terms = " OR ".join(w for w in clean.split()[:8] if len(w) > 1)
        rows = db.execute(
            "SELECT n.id, n.content, n.flow_id, n.tags, n.metadata, n.created_at "
            "FROM nodes_fts f JOIN nodes n ON n.id=f.rowid "
            "WHERE nodes_fts MATCH ? AND n.level=0 "
            + ("AND n.flow_id=? " if flow_id else "")
            + "ORDER BY rank LIMIT ?",
            ([terms, flow_id, limit] if flow_id else [terms, limit])
        ).fetchall()
        results = []
        for r in rows:
            results.append({
                "id": r["id"], "content": r["content"],
                "score": 0.5, "lineage": get_lineage(r["id"]),
                "flow_id": r["flow_id"],
                "tags": json.loads(r["tags"] or "[]"),
                "created_at": r["created_at"],
            })
        return results

    # Choose search strategy based on leaf count.
    # Direct scan is exact and fast for small collections; beam search scales
    # to millions but sacrifices recall when many root clusters exist.
    leaf_count = db.execute(
        "SELECT COUNT(*) FROM nodes WHERE level=0 AND embedding IS NOT NULL"
        + (" AND flow_id=?" if flow_id else ""),
        ([flow_id] if flow_id else [])
    ).fetchone()[0]

    candidates: list[tuple[float, int]] = []

    if leaf_count <= DIRECT_SCAN_LIMIT:
        # Fast exhaustive scan — exact top-K by cosine similarity
        rows = db.execute(
            "SELECT id, embedding FROM nodes WHERE level=0 AND embedding IS NOT NULL"
            + (" AND flow_id=?" if flow_id else ""),
            ([flow_id] if flow_id else [])
        ).fetchall()
        for r in rows:
            sim = _cosine(q_emb, _unpack(r["embedding"]))
            candidates.append((sim, r["id"]))
        candidates.sort(reverse=True)
        candidates = candidates[:limit]
    else:
        # Beam search from top level for large collections
        top_level = db.execute(
            "SELECT MAX(level) as ml FROM nodes WHERE embedding IS NOT NULL"
        ).fetchone()["ml"] or 0

        roots = db.execute(
            "SELECT id, embedding FROM nodes WHERE level=? AND parent_id IS NULL "
            "AND embedding IS NOT NULL", (top_level,)
        ).fetchall()
        candidates = sorted(
            [(_cosine(q_emb, _unpack(r["embedding"])), r["id"]) for r in roots],
            reverse=True
        )[:BEAM_WIDTH]

        # Drill down until leaves (generous cap handles splits at same level)
        for _ in range((MAX_LEVEL + 1) * 2):
            next_candidates = []
            for _, parent_id in candidates:
                children = db.execute(
                    "SELECT id, embedding FROM nodes WHERE parent_id=? AND embedding IS NOT NULL",
                    (parent_id,)
                ).fetchall()
                for c in children:
                    sim = _cosine(q_emb, _unpack(c["embedding"]))
                    next_candidates.append((sim, c["id"]))
            if not next_candidates:
                break
            next_candidates.sort(reverse=True)
            candidates = next_candidates[:BEAM_WIDTH]

    # Record access
    ids = [nid for _, nid in candidates]
    if ids:
        with _write_lock:
            db2 = _conn()
            db2.execute(
                f"UPDATE nodes SET access_count=access_count+1, last_accessed=? "
                f"WHERE id IN ({','.join('?'*len(ids))})",
                [time.time()] + ids
            )
            db2.commit()

    results = []
    for score, nid in candidates[:limit]:
        row = db.execute(
            "SELECT id, content, flow_id, tags, created_at FROM nodes WHERE id=?", (nid,)
        ).fetchone()
        if row and (flow_id is None or row["flow_id"] == flow_id):
            results.append({
                "id": row["id"],
                "content": row["content"],
                "score": round(score, 4),
                "lineage": get_lineage(row["id"]),
                "flow_id": row["flow_id"],
                "tags": json.loads(row["tags"] or "[]"),
                "created_at": row["created_at"],
            })

    return results


def get_lineage(node_id: int) -> list[dict]:
    """Walk up the parent chain, return path from root to this node."""
    db = _conn()
    path = []
    current = node_id
    seen = set()
    while current and current not in seen:
        seen.add(current)
        row = db.execute(
            "SELECT id, level, label, content, parent_id FROM nodes WHERE id=?",
            (current,)
        ).fetchone()
        if not row:
            break
        path.append({
            "id": row["id"],
            "level": row["level"],
            "label": row["label"] or row["content"][:60],
        })
        current = row["parent_id"]
    path.reverse()  # root → leaf
    return path


def _clean_label(label: str) -> str:
    """Strip auto-generated [Cluster Lx] / [Lx] prefixes and trim to ~50 chars."""
    clean = re.sub(r"^\[(?:Cluster )?L\d+\]\s*", "", label).strip()
    # Remove repeated prefix patterns like "[L4] [L3] [L2] [L1] content" → "content"
    clean = re.sub(r"(\[(?:Cluster )?L\d+\]\s*)+", "", clean).strip()
    return clean[:50] if clean else label[:50]


def context_for_task(description: str, limit: int = 6) -> str:
    """
    Drop-in replacement for memory_bridge.context_for_task.
    Returns a compact context string to inject into agent prompts.
    """
    results = query(description, limit=limit)
    if not results:
        return ""
    lines = ["[MEMORY — relevant prior knowledge]"]
    for r in results:
        # Build a short breadcrumb using only the nearest named ancestor
        ancs = [x for x in r["lineage"][:-1] if x["level"] > 0]
        if ancs:
            # Use the lowest-level (closest) ancestor with a meaningful label
            parent_label = _clean_label(ancs[-1]["label"])
            prefix = f"[{parent_label}] " if parent_label else ""
        else:
            prefix = ""
        score_str = f"(relevance {r['score']:.2f}) " if r["score"] < 0.9 else ""
        lines.append(f"• {score_str}{prefix}{r['content'][:300]}")
    return "\n".join(lines)


def store_from_result(content: str, flow_id: str = "",
                      confidence: float = 0.7) -> bool:
    """Store a distilled fact/belief from a task result."""
    try:
        insert(content, flow_id=flow_id,
               tags=["belief"],
               metadata={"confidence": confidence})
        return True
    except Exception:
        return False


def stats() -> dict:
    """Return DB statistics."""
    _init_db()
    db = _conn()
    total = db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    by_level = {}
    for row in db.execute("SELECT level, COUNT(*) as c FROM nodes GROUP BY level"):
        by_level[row["level"]] = row["c"]
    try:
        db_size_kb = FRACTAL_DB.stat().st_size // 1024
    except Exception:
        db_size_kb = 0
    return {
        "total_nodes": total,
        "by_level": by_level,
        "db_path": str(FRACTAL_DB),
        "embed_model": EMBED_MODEL,
        "db_size_kb": db_size_kb,
    }
