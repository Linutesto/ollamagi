"""
Thin compatibility shim — all memory now goes through fractal_memory.
"""
from core.fractal_memory import (
    insert,
    query as _query,
    context_for_task,
    store_from_result as store_belief,
    stats,
)


def get_relevant_context(query_text: str, limit: int = 8) -> list[dict]:
    return _query(query_text, limit=limit)


def store_memory(content: str, source: str = "",
                 tags: list[str] | None = None) -> bool:
    try:
        insert(content, tags=tags or [], metadata={"source": source})
        return True
    except Exception:
        return False


def get_goals(status: str = "active", limit: int = 10) -> list[dict]:
    # Goals are no longer a separate concept — return empty to avoid polluting prompts
    return []
