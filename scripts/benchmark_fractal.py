"""
Fractal Memory Benchmark Suite
Measures: insert throughput, query latency, hierarchy formation,
cluster splitting, FTS fallback, lineage depth, cross-topic retrieval precision.
"""
import json
import os
import sys
import time
import sqlite3
import threading
import statistics
import random
import struct
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# ─── Corpus ──────────────────────────────────────────────────────────────────
CORPUS = {
    "python": [
        "Python async/await enables non-blocking I/O with coroutines and event loops.",
        "List comprehensions in Python are faster than equivalent for loops.",
        "Python generators yield values lazily, saving memory on large datasets.",
        "Type hints in Python 3.10+ use the | operator for union types.",
        "The GIL prevents true parallel CPU execution in CPython threads.",
        "Python dataclasses auto-generate __init__, __repr__, and __eq__.",
        "f-strings are faster than .format() and % string formatting.",
        "Context managers using __enter__ and __exit__ guarantee cleanup.",
        "Python decorators are higher-order functions wrapping callables.",
        "The walrus operator := assigns inside expressions.",
    ],
    "machine_learning": [
        "Transformer attention scales quadratically with sequence length.",
        "Batch normalization reduces internal covariate shift during training.",
        "Dropout randomly zeroes activations to prevent overfitting.",
        "Adam optimizer combines momentum and RMSProp adaptive learning.",
        "Cross-entropy loss is standard for multi-class classification.",
        "Gradient clipping prevents exploding gradients in deep networks.",
        "Transfer learning fine-tunes pretrained models on new tasks.",
        "The vanishing gradient problem plagues deep sigmoid networks.",
        "Quantization reduces model size by using lower precision weights.",
        "Contrastive learning trains models to distinguish similar from dissimilar.",
    ],
    "databases": [
        "SQLite WAL mode allows concurrent reads during writes.",
        "B-tree indexes make range queries O(log n) instead of O(n).",
        "VACUUM in SQLite reclaims fragmented space and rewrites the DB.",
        "FTS5 in SQLite provides full-text search with BM25 ranking.",
        "Covering indexes include all queried columns to avoid table lookups.",
        "PostgreSQL MVCC allows readers and writers without locking each other.",
        "Redis Sorted Sets store members with scores for ranked queries.",
        "Columnar storage compresses and scans better than row storage.",
        "Sharding splits data horizontally across multiple nodes.",
        "Write-ahead logging ensures durability without fsync on every write.",
    ],
    "security": [
        "SQL injection exploits unsanitized user input in database queries.",
        "CSRF tokens prevent cross-site request forgery attacks.",
        "Content Security Policy headers restrict resource loading origins.",
        "JWT tokens are stateless authentication with HMAC or RSA signatures.",
        "OWASP Top 10 covers the most critical web application vulnerabilities.",
        "TLS 1.3 eliminates weak cipher suites and reduces handshake latency.",
        "Bcrypt password hashing includes a salt and is computationally expensive.",
        "Certificate pinning prevents MITM by hardcoding expected TLS certificates.",
        "Rate limiting protects APIs from abuse and credential stuffing.",
        "Input validation should happen at every trust boundary in the system.",
    ],
    "docker": [
        "Docker multi-stage builds reduce final image size by discarding build tools.",
        "Bind mounts map host directories into containers for development.",
        "Named volumes persist data beyond container lifecycle.",
        "Docker networking: bridge for single host, overlay for swarm clusters.",
        "The COPY instruction adds files to image layers; each layer is cached.",
        "Resource limits via --cpus and --memory prevent noisy neighbor issues.",
        "Health checks allow orchestrators to restart unhealthy containers.",
        "Docker BuildKit enables parallel build stages and better caching.",
        "USER instruction reduces attack surface by dropping root privileges.",
        "Compose V2 uses BuildKit by default and supports profiles.",
    ],
}

ALL_MEMORIES = [(topic, mem) for topic, mems in CORPUS.items() for mem in mems]
random.shuffle(ALL_MEMORIES)

# ─── Utilities ────────────────────────────────────────────────────────────────
def sep(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

def row(label, value, unit=""):
    print(f"  {label:<40} {value} {unit}")

# ─── Import under test ───────────────────────────────────────────────────────
import core.fractal_memory as fm

# Use a temp DB so benchmarks don't pollute production
BENCH_DB = Path(tempfile.mkdtemp()) / "bench.db"
fm.FRACTAL_DB = BENCH_DB
# Reset thread-local connection
import threading
fm._local = threading.local()


# ══════════════════════════════════════════════════════════════════════════════
# B1: Embedding latency
# ══════════════════════════════════════════════════════════════════════════════
sep("B1 — Embedding latency (mxbai-embed-large)")
embed_times = []
for text in ALL_MEMORIES[:10]:
    t0 = time.perf_counter()
    vec = fm._embed(text[1])
    elapsed = time.perf_counter() - t0
    if vec:
        embed_times.append(elapsed)
        print(f"  [{elapsed*1000:6.1f}ms]  dim={len(vec)}  '{text[1][:50]}…'")
    else:
        print(f"  [FAILED]  '{text[1][:50]}…'")

if embed_times:
    row("Mean embed latency", f"{statistics.mean(embed_times)*1000:.1f}", "ms")
    row("Median embed latency", f"{statistics.median(embed_times)*1000:.1f}", "ms")
    row("Min / Max", f"{min(embed_times)*1000:.1f} / {max(embed_times)*1000:.1f}", "ms")
    row("Throughput", f"{1/statistics.mean(embed_times):.1f}", "embeds/sec")
    EMBEDDING_AVAILABLE = True
else:
    print("  !! Ollama unavailable — embedding benchmarks will use FTS fallback")
    EMBEDDING_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# B2: Insert throughput
# ══════════════════════════════════════════════════════════════════════════════
sep("B2 — Insert throughput (50 memories, 5 topics)")
fm._init_db()

insert_times = []
batch_t0 = time.perf_counter()

for i, (topic, content) in enumerate(ALL_MEMORIES[:50]):
    t0 = time.perf_counter()
    node_id = fm.insert(content, flow_id="bench", tags=[topic],
                        metadata={"topic": topic, "bench": True})
    elapsed = time.perf_counter() - t0
    insert_times.append(elapsed)
    print(f"  [{i+1:2d}] id={node_id:4d}  {elapsed*1000:6.1f}ms  [{topic}] '{content[:45]}…'")

batch_elapsed = time.perf_counter() - batch_t0
print()
row("Total inserts", 50)
row("Total time", f"{batch_elapsed:.2f}", "s")
row("Mean insert time", f"{statistics.mean(insert_times)*1000:.1f}", "ms")
row("Median insert time", f"{statistics.median(insert_times)*1000:.1f}", "ms")
row("Throughput", f"{50/batch_elapsed:.2f}", "inserts/sec")


# ══════════════════════════════════════════════════════════════════════════════
# B3: DB stats & hierarchy formation
# ══════════════════════════════════════════════════════════════════════════════
sep("B3 — Hierarchy formation after 50 inserts")
s = fm.stats()
row("Total nodes", s["total_nodes"])
for lvl in sorted(s["by_level"]):
    label = {0: "leaves", 1: "concepts", 2: "domains", 3: "meta", 4: "root"}.get(lvl, f"L{lvl}")
    row(f"  Level {lvl} ({label})", s["by_level"][lvl], "nodes")

db = fm._conn(BENCH_DB)
clusters_with_children = db.execute(
    "SELECT id, level, child_count, label FROM nodes WHERE level > 0 AND child_count > 0 "
    "ORDER BY level DESC, child_count DESC"
).fetchall()
print()
print("  Top clusters by child count:")
for c in clusters_with_children[:10]:
    print(f"    L{c['level']}  children={c['child_count']:2d}  {str(c['label'] or '')[:60]}")

# Compression ratio: leaves → unique parent clusters
if s["by_level"].get(0) and s["by_level"].get(1):
    ratio = s["by_level"][0] / s["by_level"][1]
    row("\n  Leaves-to-L1-cluster ratio", f"{ratio:.1f}", "leaves/cluster")
if s["by_level"].get(0) and s["by_level"].get(2):
    ratio2 = s["by_level"][0] / s["by_level"].get(2, 1)
    row("  Leaves-to-L2-cluster ratio", f"{ratio2:.1f}", "leaves/cluster")


# ══════════════════════════════════════════════════════════════════════════════
# B4: Query latency & precision
# ══════════════════════════════════════════════════════════════════════════════
sep("B4 — Query latency & precision (10 queries)")

QUERIES = [
    ("python asyncio event loop", "python"),
    ("neural network training optimization", "machine_learning"),
    ("SQLite concurrent access WAL", "databases"),
    ("web security injection attacks", "security"),
    ("Docker container image layers", "docker"),
    ("Python memory efficient iteration", "python"),
    ("deep learning gradient problems", "machine_learning"),
    ("database indexing performance", "databases"),
    ("TLS certificate authentication", "security"),
    ("container resource management", "docker"),
]

precision_scores = []
query_times = []

for query_text, expected_topic in QUERIES:
    t0 = time.perf_counter()
    results = fm.query(query_text, limit=5)
    elapsed = time.perf_counter() - t0
    query_times.append(elapsed)

    # Precision: how many of top-5 results match the expected topic?
    correct = 0
    for r in results[:5]:
        tags = r.get("tags", [])
        if expected_topic in tags:
            correct += 1

    precision = correct / min(5, len(results)) if results else 0
    precision_scores.append(precision)

    top_score = results[0]["score"] if results else 0
    print(f"  [{elapsed*1000:5.1f}ms]  P@5={precision:.0%}  top={top_score:.3f}  Q: '{query_text}'")
    if results:
        print(f"           → '{results[0]['content'][:70]}…'")

print()
row("Mean query latency", f"{statistics.mean(query_times)*1000:.1f}", "ms")
row("Median query latency", f"{statistics.median(query_times)*1000:.1f}", "ms")
row("Mean P@5 precision", f"{statistics.mean(precision_scores):.0%}")
row("Queries with P@5 ≥ 60%", f"{sum(1 for p in precision_scores if p >= 0.6)}/{len(precision_scores)}")


# ══════════════════════════════════════════════════════════════════════════════
# B5: Lineage depth & structure
# ══════════════════════════════════════════════════════════════════════════════
sep("B5 — Lineage depth distribution")

leaf_ids = [r["id"] for r in db.execute(
    "SELECT id FROM nodes WHERE level=0 LIMIT 20"
).fetchall()]

depths = []
for lid in leaf_ids:
    lineage = fm.get_lineage(lid)
    depths.append(len(lineage))

if depths:
    row("Mean lineage depth", f"{statistics.mean(depths):.1f}", "levels")
    row("Min depth", min(depths))
    row("Max depth", max(depths))
    row("Depth distribution", str({d: depths.count(d) for d in sorted(set(depths))}))

# Show 2 example lineages
print()
print("  Example lineages:")
for lid in leaf_ids[:2]:
    lineage = fm.get_lineage(lid)
    chain = " › ".join(f"L{n['level']}:{(n['label'] or '')[:30]}" for n in lineage)
    print(f"    {chain}")


# ══════════════════════════════════════════════════════════════════════════════
# B6: FTS fallback (no embeddings)
# ══════════════════════════════════════════════════════════════════════════════
sep("B6 — FTS fallback (simulated embedding failure)")
original_embed = fm._embed

def _broken_embed(text):
    return []

fm._embed = _broken_embed

fts_queries = [
    ("async coroutines generators", "python"),
    ("batch normalization dropout", "machine_learning"),
    ("B-tree index range query", "databases"),
    ("injection csrf security", "security"),
]

fts_times = []
fts_precisions = []

for query_text, expected_topic in fts_queries:
    t0 = time.perf_counter()
    results = fm.query(query_text, limit=5)
    elapsed = time.perf_counter() - t0
    fts_times.append(elapsed)

    correct = sum(1 for r in results[:5] if expected_topic in r.get("tags", []))
    precision = correct / min(5, len(results)) if results else 0
    fts_precisions.append(precision)

    print(f"  [{elapsed*1000:5.1f}ms]  P@5={precision:.0%}  '{query_text}'")
    if results:
        print(f"           → '{results[0]['content'][:70]}…'")

print()
row("Mean FTS latency", f"{statistics.mean(fts_times)*1000:.1f}", "ms")
row("Mean FTS P@5 precision", f"{statistics.mean(fts_precisions):.0%}")

fm._embed = original_embed  # restore


# ══════════════════════════════════════════════════════════════════════════════
# B7: Concurrent write safety
# ══════════════════════════════════════════════════════════════════════════════
sep("B7 — Concurrent write safety (4 threads × 5 inserts)")
errors = []
thread_times = []

def _worker(tid, memories):
    t0 = time.perf_counter()
    for content in memories:
        try:
            fm.insert(content, flow_id=f"thread-{tid}", tags=["concurrent"])
        except Exception as e:
            errors.append(str(e))
    thread_times.append(time.perf_counter() - t0)

thread_memories = [
    [f"Thread {i} memory {j}: concurrent write test with semantic content about topic {j % 5}"
     for j in range(5)]
    for i in range(4)
]

threads = [threading.Thread(target=_worker, args=(i, thread_memories[i])) for i in range(4)]
t0 = time.perf_counter()
for t in threads:
    t.start()
for t in threads:
    t.join()
total_concurrent = time.perf_counter() - t0

s2 = fm.stats()
row("Write errors", len(errors))
row("Total concurrent inserts", 20)
row("Wall time", f"{total_concurrent:.2f}", "s")
row("Nodes added", s2["total_nodes"] - s["total_nodes"])
if errors:
    for e in errors[:3]:
        print(f"  !! {e}")
else:
    print("  All concurrent writes succeeded without errors.")


# ══════════════════════════════════════════════════════════════════════════════
# B8: Cosine similarity distribution within vs across topics
# ══════════════════════════════════════════════════════════════════════════════
sep("B8 — Semantic coherence: within-topic vs cross-topic similarity")
if EMBEDDING_AVAILABLE:
    topic_embeddings = {}
    for topic, mems in CORPUS.items():
        vecs = [fm._embed(m) for m in mems[:5]]
        vecs = [v for v in vecs if v]
        topic_embeddings[topic] = vecs

    within_sims = []
    cross_sims = []
    topics = list(topic_embeddings.keys())

    for topic in topics:
        vecs = topic_embeddings[topic]
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                within_sims.append(fm._cosine(vecs[i], vecs[j]))

    for i, t1 in enumerate(topics):
        for j, t2 in enumerate(topics):
            if i >= j:
                continue
            v1 = topic_embeddings[t1][0] if topic_embeddings[t1] else None
            v2 = topic_embeddings[t2][0] if topic_embeddings[t2] else None
            if v1 and v2:
                cross_sims.append(fm._cosine(v1, v2))

    if within_sims and cross_sims:
        row("Within-topic mean cosine sim", f"{statistics.mean(within_sims):.4f}")
        row("Within-topic min/max",
            f"{min(within_sims):.4f} / {max(within_sims):.4f}")
        row("Cross-topic mean cosine sim", f"{statistics.mean(cross_sims):.4f}")
        row("Cross-topic min/max",
            f"{min(cross_sims):.4f} / {max(cross_sims):.4f}")
        sep_score = statistics.mean(within_sims) - statistics.mean(cross_sims)
        row("Separability (within − cross)", f"{sep_score:.4f}")
        row("JOIN_THRESH effectiveness",
            f"{'GOOD' if statistics.mean(within_sims) > fm.JOIN_THRESH > statistics.mean(cross_sims) else 'CHECK'}")
else:
    print("  Skipped — embeddings unavailable.")


# ══════════════════════════════════════════════════════════════════════════════
# B9: context_for_task output quality
# ══════════════════════════════════════════════════════════════════════════════
sep("B9 — context_for_task output (agent prompt injection test)")
ctx = fm.context_for_task("I need to build a Python async web scraper with database storage", limit=6)
print(ctx if ctx else "  (no context returned)")
print()
row("Context lines", len(ctx.splitlines()) if ctx else 0)
row("Context chars", len(ctx))


# ══════════════════════════════════════════════════════════════════════════════
# B10: DB file size efficiency
# ══════════════════════════════════════════════════════════════════════════════
sep("B10 — Storage efficiency")
db_size = BENCH_DB.stat().st_size
final_stats = fm.stats()
row("Total nodes", final_stats["total_nodes"])
row("DB file size", f"{db_size / 1024:.1f}", "KB")
row("Bytes per node", f"{db_size / max(final_stats['total_nodes'], 1):.0f}", "bytes")
row("Embed dim", fm.EMBED_DIM)
raw_embed_bytes = fm.EMBED_DIM * 4 * final_stats["total_nodes"]
row("Raw embedding bytes", f"{raw_embed_bytes / 1024:.1f}", "KB")
row("Storage overhead factor", f"{db_size / max(raw_embed_bytes, 1):.2f}x", "(1.0 = perfect packing)")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
sep("SUMMARY")
row("Embedding model", fm.EMBED_MODEL)
row("Embedding dim", fm.EMBED_DIM)
row("JOIN_THRESH", fm.JOIN_THRESH)
row("SPLIT_AT", fm.SPLIT_AT)
row("BEAM_WIDTH", fm.BEAM_WIDTH)
row("MAX_LEVEL", fm.MAX_LEVEL)
print()
if embed_times:
    row("Embed latency (mean)", f"{statistics.mean(embed_times)*1000:.1f} ms")
if insert_times:
    row("Insert latency (mean)", f"{statistics.mean(insert_times)*1000:.1f} ms")
if query_times:
    row("Query latency (mean)", f"{statistics.mean(query_times)*1000:.1f} ms")
if precision_scores:
    row("Query P@5 precision (mean)", f"{statistics.mean(precision_scores):.0%}")
row("Concurrent write errors", len(errors))
row("Total benchmark nodes", final_stats["total_nodes"])
row("DB size", f"{db_size / 1024:.1f} KB")

# Cleanup
import shutil
shutil.rmtree(BENCH_DB.parent, ignore_errors=True)
print()
print("  Benchmark DB cleaned up.")
