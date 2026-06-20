"""Fractal memory tools — agents can store and retrieve knowledge."""
from core.tool_registry import tool


@tool("memory_store", "Store a fact or insight in the fractal memory for future use", {
    "content": {"type": "string", "description": "The fact, insight, or knowledge to store"},
    "tags":    {"type": "array",  "description": "Optional list of topic tags", "default": []},
})
def memory_store(content: str, tags: list | None = None) -> dict:
    from core.fractal_memory import insert
    node_id = insert(content, tags=tags or [])
    return {"stored": True, "id": node_id}


@tool("memory_search", "Search the fractal memory for relevant prior knowledge", {
    "query": {"type": "string",  "description": "What to search for"},
    "limit": {"type": "integer", "description": "Max results to return", "default": 5},
})
def memory_search(query: str, limit: int = 5) -> list[dict]:
    from core.fractal_memory import query as fquery
    results = fquery(query, limit=limit)
    return [
        {
            "content": r["content"],
            "score": r["score"],
            "path": " › ".join(x["label"] for x in r["lineage"]),
        }
        for r in results
    ]
