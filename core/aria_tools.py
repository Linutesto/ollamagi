"""Aria tool definitions, executors, and the streaming tool-use loop."""
import json
import queue
import threading
import time
import httpx

from core.config import OLLAMA_URL, MODELS, WORKSPACE_DIR


ARIA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the internet using SearxNG (aggregates Google, Bing, DuckDuckGo). "
                "Use for current info, docs, prices, news, or anything needing real-time data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Number of results (default 8, max 15)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from a flow's workspace. Use to inspect generated code, reports, data, or configs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flow_id": {"type": "string", "description": "Flow ID prefix (first 8 chars is enough)"},
                    "path": {"type": "string", "description": "Path relative to workspace, e.g. 'main.py' or 'reports/analysis.md'"},
                },
                "required": ["flow_id", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files produced by a flow in its workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flow_id": {"type": "string", "description": "Flow ID prefix"},
                },
                "required": ["flow_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_flow",
            "description": (
                "Launch a new OllamAGI flow. Agents run autonomously in Docker containers. "
                "Returns the flow ID immediately; the flow continues in background. "
                "Optionally pass base_flow_ids to give agents access to other projects' files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {"type": "string", "description": "Specific, actionable objective for the agents"},
                    "flow_type": {
                        "type": "string",
                        "enum": [
                            "agent_development", "product_development", "research",
                            "security", "data_engineering", "devops", "automation", "content", "general",
                        ],
                        "description": "Flow type (auto-detected from objective if omitted)",
                    },
                    "base_flow_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Flow IDs whose workspace files should be made available to this flow's agents at /work/_context/{id}/",
                    },
                },
                "required": ["objective"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_flow",
            "description": "Stop a running flow immediately. Cannot be undone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flow_id": {"type": "string", "description": "Flow ID to stop"},
                },
                "required": ["flow_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Semantic search in fractal memory — past knowledge, code patterns, project learnings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for"},
                    "limit": {"type": "integer", "description": "Max results (default 6, max 12)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "store_memory",
            "description": "Permanently store a fact, insight, or code pattern in fractal memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The fact to store"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Category tags, e.g. ['python', 'performance']",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_flow_logs",
            "description": "Get execution logs for a flow — what agents did, errors, and results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flow_id": {"type": "string", "description": "Flow ID"},
                    "lines": {"type": "integer", "description": "Last N log entries (default 60)"},
                },
                "required": ["flow_id"],
            },
        },
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _resolve_flow_dir(flow_id: str):
    """Resolve flow ID (or prefix) to its workspace Path, or None."""
    if not WORKSPACE_DIR.exists():
        return None
    exact = WORKSPACE_DIR / flow_id
    if exact.exists():
        return exact
    for d in sorted(WORKSPACE_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if d.is_dir() and d.name.startswith(flow_id):
            return d
    return None


# ── Tool executors ─────────────────────────────────────────────────────────

def _execute_tool(name: str, args: dict) -> str:
    dispatch = {
        "web_search":    lambda: _tool_web_search(args.get("query", ""), args.get("max_results", 8)),
        "read_file":     lambda: _tool_read_file(args.get("flow_id", ""), args.get("path", "")),
        "list_files":    lambda: _tool_list_files(args.get("flow_id", "")),
        "run_flow":      lambda: _tool_run_flow(args.get("objective", ""), args.get("flow_type"), args.get("base_flow_ids")),
        "stop_flow":     lambda: _tool_stop_flow(args.get("flow_id", "")),
        "search_memory": lambda: _tool_search_memory(args.get("query", ""), args.get("limit", 6)),
        "store_memory":  lambda: _tool_store_memory(args.get("content", ""), args.get("tags", [])),
        "get_flow_logs": lambda: _tool_get_flow_logs(args.get("flow_id", ""), args.get("lines", 60)),
    }
    fn = dispatch.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    try:
        return fn()
    except Exception as exc:
        return f"Tool error ({name}): {exc}"


def _tool_web_search(query: str, max_results: int = 8) -> str:
    if not query.strip():
        return "Error: query required"
    max_results = min(max(1, int(max_results)), 15)
    try:
        resp = httpx.get(
            "http://localhost:4000/search",
            params={"q": query, "format": "json", "categories": "general"},
            timeout=12.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])[:max_results]
    except httpx.ConnectError:
        return "Error: SearxNG unreachable at localhost:4000 — search service may be down."
    except Exception as e:
        return f"Search error: {e}"

    if not results:
        return f"No results for: {query}"
    lines = [f"Web search: **{query}**\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", r.get("href", ""))
        body = (r.get("content") or r.get("body") or "")[:300]
        lines.append(f"{i}. **{title}**\n   {url}\n   {body}\n")
    return "\n".join(lines)


def _tool_read_file(flow_id: str, path: str) -> str:
    if not flow_id or not path:
        return "Error: flow_id and path required"
    work_dir = _resolve_flow_dir(flow_id)
    if not work_dir:
        return f"Error: no workspace for flow '{flow_id}'"
    try:
        target = (work_dir / path).resolve()
        target.relative_to(work_dir.resolve())
    except ValueError:
        return "Error: path traversal blocked"
    if not target.exists():
        return f"File not found: {path} in {work_dir.name}"
    if not target.is_file():
        return f"'{path}' is a directory — use list_files instead"
    content = target.read_text(errors="replace")
    size = target.stat().st_size
    if len(content) > 12000:
        content = content[:12000] + f"\n\n[truncated — {size} bytes total]"
    return f"**{path}** ({size} bytes, flow {work_dir.name})\n{'─'*50}\n{content}"


def _tool_list_files(flow_id: str) -> str:
    if not flow_id:
        return "Error: flow_id required"
    work_dir = _resolve_flow_dir(flow_id)
    if not work_dir:
        return f"Error: no workspace for flow '{flow_id}'"
    skip = {"flow.json", "flow_log.jsonl", "llm_calls.jsonl"}
    files = [
        (str(f.relative_to(work_dir)), f.stat().st_size)
        for f in sorted(work_dir.rglob("*"))
        if f.is_file() and f.name not in skip
    ]
    if not files:
        return f"Workspace {work_dir.name}: no files yet"
    lines = [f"**{work_dir.name}** — {len(files)} file(s):\n"]
    for rel, size in files:
        lines.append(f"  {rel}  ({size:,} B)")
    return "\n".join(lines)


def _tool_run_flow(objective: str, flow_type: str | None = None,
                   base_flow_ids: list | None = None) -> str:
    if not objective.strip():
        return "Error: objective required"
    from core.orchestrator import run_flow as _run_flow, _flows

    # Optional WS broadcast (avoid circular import by checking sys.modules)
    import sys
    _broadcast = None
    if "api.server" in sys.modules:
        try:
            _broadcast = sys.modules["api.server"].broadcast_sync
        except AttributeError:
            pass

    id_box: dict = {}

    def _bg():
        try:
            flow = _run_flow(
                objective,
                flow_type=flow_type,
                broadcast=_broadcast,
                base_flows=base_flow_ids or [],
            )
            id_box["id"] = flow.id
            id_box["status"] = flow.status
        except Exception as exc:
            id_box["error"] = str(exc)

    pre_ids = set(_flows.keys())
    threading.Thread(target=_bg, daemon=True).start()

    # Poll for the flow to register itself (happens in first ~100ms of run_flow)
    deadline = time.time() + 4.0
    while time.time() < deadline:
        new = set(_flows.keys()) - pre_ids
        if new:
            fid = list(new)[0]
            return (
                f"Flow launched!\n"
                f"**ID:** `{fid}`\n"
                f"**Objective:** {objective}\n"
                f"**Type:** {flow_type or 'auto-detected'}\n"
                + (f"**Base projects:** {', '.join(f'`{b}`' for b in (base_flow_ids or []))}\n" if base_flow_ids else "")
                + f"Monitor in the Flows tab or call `get_flow_logs {fid}`."
            )
        time.sleep(0.05)

    if "error" in id_box:
        return f"Flow failed to start: {id_box['error']}"
    return "Flow started — ID not yet available. Check the Flows tab."


def _tool_stop_flow(flow_id: str) -> str:
    if not flow_id:
        return "Error: flow_id required"
    work_dir = _resolve_flow_dir(flow_id)
    actual_id = work_dir.name if work_dir else flow_id
    from core.orchestrator import request_stop
    request_stop(actual_id)
    return f"Stop requested for flow `{actual_id}`. Will halt within ~500ms."


def _tool_search_memory(query: str, limit: int = 6) -> str:
    if not query.strip():
        return "Error: query required"
    limit = min(max(1, int(limit)), 12)
    from core.fractal_memory import query as fquery
    results = fquery(query, limit=limit)
    if not results:
        return f"No memories matching: {query}"
    lines = [f"Memory search: **{query}**\n"]
    for i, r in enumerate(results, 1):
        sim = r.get("similarity", 0)
        content = r.get("content", "")[:200]
        tags = r.get("tags", [])
        lineage = " > ".join(r.get("lineage", []))
        lines.append(f"{i}. [{sim:.2f}] {content}")
        if tags:
            lines.append(f"   Tags: {', '.join(tags)}")
        if lineage:
            lines.append(f"   Path: {lineage}")
        lines.append("")
    return "\n".join(lines)


def _tool_store_memory(content: str, tags: list | None = None) -> str:
    if not content.strip():
        return "Error: content required"
    from core.fractal_memory import insert as finsert
    node_id = finsert(content, tags=tags or [], metadata={"source": "aria_chat"})
    return f"Stored in memory. Node: {node_id}\nContent: {content[:120]}"


def _tool_get_flow_logs(flow_id: str, lines: int = 60) -> str:
    if not flow_id:
        return "Error: flow_id required"
    work_dir = _resolve_flow_dir(flow_id)
    if not work_dir:
        return f"Error: no workspace for flow '{flow_id}'"
    log_file = work_dir / "flow_log.jsonl"
    if not log_file.exists():
        return f"No log file for flow {work_dir.name}"
    entries = []
    with open(log_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                entries.append({"level": "raw", "msg": line})
    entries = entries[-int(lines):]
    if not entries:
        return f"Log empty for {work_dir.name}"
    out = [f"**{work_dir.name}** — last {len(entries)} log entries:\n"]
    for e in entries:
        level = e.get("level", "info").upper()
        msg = e.get("msg", str(e))[:250]
        out.append(f"[{level}] {msg}")
    return "\n".join(out)


# ── Streaming tool-use loop ────────────────────────────────────────────────

def run_aria_loop(messages: list, system_ctx: str, token_q: queue.Queue):
    """
    Streaming Aria tool-use loop. Emits JSON events into token_q:
      {"type": "token", "token": "..."}
      {"type": "tool_start", "name": "...", "args": {...}}
      {"type": "tool_done",  "name": "...", "preview": "..."}
    Terminates with: token_q.put(None)
    """
    model = MODELS.get("orchestrator", "")
    loop_messages = [{"role": "system", "content": system_ctx}] + list(messages[-30:])
    MAX_ROUNDS = 8

    try:
        for _round in range(MAX_ROUNDS):
            accumulated_tool_calls: list = []
            accumulated_content = ""

            with httpx.stream(
                "POST",
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": loop_messages,
                    "tools": ARIA_TOOLS,
                    "stream": True,
                    "options": {"temperature": 0.2, "num_predict": 2048},
                },
                timeout=httpx.Timeout(120.0),
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        d = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    msg = d.get("message", {})
                    chunk = msg.get("content", "")
                    tc_in_chunk = msg.get("tool_calls", [])

                    if tc_in_chunk:
                        accumulated_tool_calls.extend(tc_in_chunk)

                    # Stream content only when no tool calls have appeared yet in this round
                    if chunk and not accumulated_tool_calls:
                        accumulated_content += chunk
                        token_q.put(json.dumps({"type": "token", "token": chunk}))

            if not accumulated_tool_calls:
                # Final answer was already streamed — done
                token_q.put(None)
                return

            # Append assistant's tool-use message to history
            loop_messages.append({
                "role": "assistant",
                "content": accumulated_content,
                "tool_calls": accumulated_tool_calls,
            })

            # Execute each tool call
            for tc in accumulated_tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "unknown")
                raw_args = fn.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        raw_args = {}

                token_q.put(json.dumps({"type": "tool_start", "name": name, "args": raw_args}))

                result = _execute_tool(name, raw_args)

                preview = (result or "(no output)")[:500]
                token_q.put(json.dumps({"type": "tool_done", "name": name, "preview": preview}))

                loop_messages.append({"role": "tool", "content": result})

        # Exhausted rounds
        token_q.put(json.dumps({"type": "token", "token": "\n\n[Max tool rounds reached]"}))
        token_q.put(None)

    except Exception as exc:
        token_q.put(json.dumps({"type": "token", "token": f"\n\n[Error: {exc}]"}))
        token_q.put(None)
