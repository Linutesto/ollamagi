"""Extract knowledge from completed flows and write to cognitive memory."""
import re
from core.memory_bridge import store_memory, store_belief
from core.model_router import chat


def extract_and_store(flow_id: str, objective: str, task_results: list) -> int:
    """Distill findings from all task outputs into cognitive memories and beliefs."""
    combined = f"FLOW: {objective}\n\n"
    successful_results = [r for r in task_results if r.status == "success"]
    for r in successful_results:
        combined += f"TASK [{r.title}] ({r.status}):\n{r.output[:2000]}\n\n"

    # Never turn failed or unverified outputs into durable beliefs.
    if not successful_results:
        store_memory(
            f"OllamAGI flow '{objective}' completed with no verified successful tasks.",
            f"ollamagi:flow:{flow_id}",
            tags=["ollamagi", "flow_summary", "failed"],
        )
        return 0

    # Ask the model to extract key findings from verified successful tasks only.
    system = (
        "You extract factual, reusable knowledge from agent task outputs. "
        "Return a JSON array of objects with keys: "
        "type ('belief'|'memory'), content (str, max 200 chars), confidence (0.0-1.0). "
        "Focus on: discovered facts, working techniques, target info, patterns. "
        "Return ONLY valid JSON, no markdown."
    )
    raw = chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": combined[:6000]}],
        task_type="fast",
        flow_id=flow_id,
        max_tokens=512,
        timeout_s=90,
    )
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    import json
    try:
        items = json.loads(raw)
    except Exception:
        items = []

    stored = 0
    source = f"ollamagi:flow:{flow_id}"
    for item in items:
        content = item.get("content", "").strip()
        if not content or len(content) < 10:
            continue
        if item.get("type") == "belief":
            if store_belief(content, float(item.get("confidence", 0.6)), source):
                stored += 1
        else:
            if store_memory(content, source, tags=["ollamagi", "flow"]):
                stored += 1

    # Always store a summary memory
    summary = f"OllamAGI flow '{objective}': {len(task_results)} tasks, " \
              f"{len(successful_results)} succeeded."
    store_memory(summary, source, tags=["ollamagi", "flow_summary"])
    return stored
