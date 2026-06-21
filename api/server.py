"""OllamAGI FastAPI server — complete REST API + WebSocket + static dashboard."""
import asyncio
import json
import queue
import threading
import time
import urllib.request
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from core.config import (
    WORKSPACE_DIR, OLLAMA_URL, HW_CPU, HW_RAM, HW_GPU,
    SSH_HOST, SSH_USER, SSH_KEY,
)
from core.orchestrator import (run_flow, get_flow, get_all_flows, register_log_callback,
                                _flow_to_dict, get_flow_transcript, request_stop, inject_steer)
from core.model_router import get_tokens, get_all_tokens, get_session_tokens, reset_session_tokens, chat
from core.aria_tools import run_aria_loop, ARIA_TOOLS

app = FastAPI(title="OllamAGI", version="1.0.0")
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _recover_stale_flows():
    """On startup: any flow.json stuck in 'running' wasn't cleanly stopped — mark stopped."""
    if not WORKSPACE_DIR.exists():
        return
    for d in WORKSPACE_DIR.iterdir():
        state_file = d / "flow.json"
        if not state_file.exists():
            continue
        try:
            data = json.loads(state_file.read_text())
            if data.get("status") == "running":
                data["status"] = "stopped"
                state_file.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

_recover_stale_flows()

_ws_clients: list[WebSocket] = []
_loop: asyncio.AbstractEventLoop | None = None


def broadcast_sync(msg: dict):
    """Thread-safe broadcast from sync orchestrator thread."""
    if _loop and _ws_clients:
        asyncio.run_coroutine_threadsafe(_broadcast(msg), _loop)


async def _broadcast(msg: dict):
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


# --- Routes ---

@app.get("/")
async def dashboard():
    # Read fresh each request — no ETag, no caching — so dev edits appear immediately
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(
        content=html,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/api/flows")
async def list_flows():
    result = []
    if WORKSPACE_DIR.exists():
        for d in sorted(WORKSPACE_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            state_file = d / "flow.json"
            if state_file.exists():
                try:
                    result.append(json.loads(state_file.read_text()))
                except Exception:
                    pass
    # Merge with in-memory (running) flows
    live = {f.id: _flow_to_dict(f) for f in get_all_flows()}
    merged = {r["id"]: r for r in result}
    merged.update(live)
    return JSONResponse(sorted(merged.values(), key=lambda x: x.get("created_at", 0), reverse=True)[:50])


@app.get("/api/flows/{flow_id}")
async def get_flow_detail(flow_id: str):
    live = get_flow(flow_id)
    if live:
        return JSONResponse(_flow_to_dict(live))
    state_file = WORKSPACE_DIR / flow_id / "flow.json"
    if state_file.exists():
        return JSONResponse(json.loads(state_file.read_text()))
    return JSONResponse({"error": "not found"}, status_code=404)


@app.post("/api/flows/run")
async def start_flow(body: dict, background_tasks: BackgroundTasks):
    objective = body.get("objective", "").strip()
    if not objective:
        return JSONResponse({"error": "objective required"}, status_code=400)
    flow_type = body.get("flow_type") or None
    base_flows = body.get("base_flows") or []
    compact_context = bool(body.get("compact_context", False))
    background_tasks.add_task(_run_bg, objective, flow_type, base_flows, compact_context)
    return JSONResponse({"status": "started", "message": "Flow starting..."})


@app.post("/api/flows/{flow_id}/stop")
async def stop_flow(flow_id: str):
    request_stop(flow_id)
    return JSONResponse({"ok": True, "flow_id": flow_id})


@app.post("/api/flows/{flow_id}/steer")
async def steer_flow(flow_id: str, body: dict):
    message = body.get("message", "").strip()
    if not message:
        return JSONResponse({"error": "message required"}, status_code=400)
    inject_steer(flow_id, message)
    await _broadcast({"type": "log", "flow_id": flow_id,
                      "msg": f"[STEER queued] {message[:100]}", "level": "warn"})
    return JSONResponse({"ok": True, "flow_id": flow_id})


@app.get("/api/memory/search")
async def search_memory(q: str = ""):
    if not q:
        return JSONResponse([])
    from core.fractal_memory import query as fquery
    results = fquery(q, limit=12)
    return JSONResponse(results)


@app.get("/api/memory/stats")
async def memory_stats():
    from core.fractal_memory import stats as fstats
    return JSONResponse(fstats())


@app.get("/api/memory/recent")
async def memory_recent(limit: int = 30):
    """Return the most recently inserted leaf memories (level 0)."""
    from core.fractal_memory import _conn, _init_db, get_lineage
    import json as _json
    _init_db()
    db = _conn()
    rows = db.execute(
        "SELECT id, content, tags, flow_id, created_at, access_count "
        "FROM nodes WHERE level=0 ORDER BY created_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "content": r["content"],
            "tags": _json.loads(r["tags"] or "[]"),
            "flow_id": r["flow_id"],
            "created_at": r["created_at"],
            "access_count": r["access_count"],
            "lineage": get_lineage(r["id"]),
        })
    return JSONResponse(results)


@app.get("/api/memory/graph")
async def memory_graph(level: int = 1, limit: int = 100):
    """Return cluster nodes for visualizing the fractal hierarchy."""
    from core.fractal_memory import _conn, _init_db
    _init_db()
    db = _conn()
    rows = db.execute(
        "SELECT id, level, label, content, parent_id, child_count, created_at "
        "FROM nodes WHERE level >= ? ORDER BY level DESC, child_count DESC LIMIT ?",
        (level, limit)
    ).fetchall()
    return JSONResponse([dict(r) for r in rows])


@app.get("/api/system/info")
async def system_info():
    """Returns configurable hardware/system info for the dashboard."""
    import platform
    return JSONResponse({
        "cpu":  HW_CPU  or platform.processor() or "unknown",
        "ram":  HW_RAM  or "",
        "gpu":  HW_GPU  or "",
        "os":   platform.system() + " " + platform.release(),
        "ssh_host": SSH_HOST,
        "ssh_user": SSH_USER,
        "ssh_key":  str(SSH_KEY),
    })


@app.get("/api/status")
async def system_status():
    ollama_ok = False
    ollama_models = 0
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=3) as r:
            data = json.loads(r.read())
            ollama_models = len(data.get("models", []))
            ollama_ok = True
    except Exception:
        pass
    from core.fractal_memory import stats as mem_stats
    try:
        ms = mem_stats()
    except Exception:
        ms = {"total_nodes": 0}
    return JSONResponse({
        "ollama": {"ok": ollama_ok, "models": ollama_models},
        "memory": {**ms, "ok": True},
    })


@app.post("/api/terminal/exec")
async def terminal_exec(body: dict):
    """Execute a command in a flow's container or on the host via SSH."""
    import subprocess
    cmd = body.get("cmd", "").strip()
    flow_id = body.get("flow_id") or None
    if not cmd:
        return JSONResponse({"error": "cmd required"}, status_code=400)
    if flow_id:
        # Try to exec in the flow's most recent container
        try:
            import docker as docker_lib
            client = docker_lib.from_env()
            containers = client.containers.list(filters={"name": f"ollamagi-{flow_id}"})
            if containers:
                result = containers[0].exec_run(["bash", "-c", cmd], demux=True)
                stdout = (result.output[0] or b"").decode("utf-8", errors="replace")
                stderr = (result.output[1] or b"").decode("utf-8", errors="replace")
                return JSONResponse({"stdout": stdout, "stderr": stderr,
                                     "exit_code": result.exit_code or 0, "target": "container"})
        except Exception as e:
            pass  # Fall through to host exec
    # Host exec
    try:
        result = subprocess.run(
            ["bash", "-c", cmd], capture_output=True, text=True, timeout=30
        )
        return JSONResponse({
            "stdout": result.stdout[-8000:],
            "stderr": result.stderr[-2000:],
            "exit_code": result.returncode,
            "target": "host",
        })
    except subprocess.TimeoutExpired:
        return JSONResponse({"stdout": "", "stderr": "Command timed out (30s)", "exit_code": -1, "target": "host"})
    except Exception as e:
        return JSONResponse({"error": str(e), "exit_code": -1}, status_code=500)


@app.get("/api/tokens")
async def token_stats():
    # Per-flow: live in-memory (session) supplemented by disk for past sessions
    result = get_all_tokens()
    if WORKSPACE_DIR.exists():
        for d in WORKSPACE_DIR.iterdir():
            if not d.is_dir():
                continue
            flow_file = d / "flow.json"
            if not flow_file.exists():
                continue
            try:
                flow_data = json.loads(flow_file.read_text())
                fid = flow_data.get("id")
                tok = flow_data.get("_tokens")
                if fid and tok and fid not in result and tok.get("total", 0) > 0:
                    result[fid] = tok
            except Exception:
                pass

    # All-time totals: sum everything on disk (most complete view)
    alltime = {"prompt": 0, "completion": 0, "calls": 0}
    for tok in result.values():
        alltime["prompt"] += tok.get("prompt", 0)
        alltime["completion"] += tok.get("completion", 0)
        alltime["calls"] += tok.get("calls", 0)
    alltime["total"] = alltime["prompt"] + alltime["completion"]

    return JSONResponse({
        "_flows": result,
        "_session": get_session_tokens(),
        "_alltime": alltime,
    })


@app.post("/api/tokens/reset")
async def reset_tokens():
    reset_session_tokens()
    return JSONResponse({"ok": True})


@app.get("/api/tokens/{flow_id}")
async def flow_tokens(flow_id: str):
    return JSONResponse(get_tokens(flow_id))


@app.post("/api/memory/store")
async def store_to_memory(body: dict):
    content = body.get("content", "").strip()
    tags = body.get("tags", [])
    if not content:
        return JSONResponse({"error": "content required"}, status_code=400)
    from core.fractal_memory import insert as finsert
    node_id = finsert(content, tags=tags, metadata={"source": "manual"})
    return JSONResponse({"ok": True, "id": node_id})


@app.get("/api/flows/{flow_id}/workspace")
async def flow_workspace(flow_id: str):
    work_dir = WORKSPACE_DIR / flow_id
    if not work_dir.exists():
        return JSONResponse([])
    files = []
    for f in sorted(work_dir.rglob("*")):
        if f.is_file() and f.name not in ("flow.json", "flow_log.jsonl"):
            files.append({
                "path": str(f.relative_to(work_dir)),
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime,
            })
    return JSONResponse(files)


@app.get("/api/flows/{flow_id}/transcript")
async def flow_transcript(flow_id: str):
    data = get_flow_transcript(flow_id)
    if data is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(data)


@app.get("/api/flows/{flow_id}/file")
async def flow_file(flow_id: str, path: str = ""):
    if not path:
        return JSONResponse({"error": "path required"}, status_code=400)
    work_dir = WORKSPACE_DIR / flow_id
    # Prevent path traversal
    try:
        target = (work_dir / path).resolve()
        target.relative_to(work_dir.resolve())
    except ValueError:
        return JSONResponse({"error": "invalid path"}, status_code=400)
    if not target.exists() or not target.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    try:
        content = target.read_text(errors="replace")
        return JSONResponse({"path": path, "content": content, "size": target.stat().st_size})
    except Exception:
        # Binary file — return base64
        import base64
        content_b64 = base64.b64encode(target.read_bytes()).decode()
        return JSONResponse({"path": path, "content": None, "binary": content_b64,
                             "size": target.stat().st_size})


@app.get("/api/flows/{flow_id}/rawfile")
async def flow_rawfile(flow_id: str, path: str = ""):
    if not path:
        return JSONResponse({"error": "path required"}, status_code=400)
    work_dir = WORKSPACE_DIR / flow_id
    try:
        target = (work_dir / path).resolve()
        target.relative_to(work_dir.resolve())
    except ValueError:
        return JSONResponse({"error": "invalid path"}, status_code=400)
    if not target.exists() or not target.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    import mimetypes
    mime, _ = mimetypes.guess_type(str(target))
    mime = mime or "application/octet-stream"
    filename = target.name
    return StreamingResponse(
        open(target, "rb"),  # noqa: SIM115
        media_type=mime,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/flows/{flow_id}/download")
async def flow_download(flow_id: str):
    import io, zipfile
    work_dir = WORKSPACE_DIR / flow_id
    if not work_dir.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(work_dir.rglob("*")):
            if f.is_file() and f.name not in ("flow.json", "flow_log.jsonl"):
                zf.write(f, arcname=str(f.relative_to(work_dir)))
    buf.seek(0)
    filename = f"ollamagi-{flow_id[:8]}.zip"
    return StreamingResponse(
        iter([buf.read()]),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _assistant_system_context(query: str) -> str:
    """Build a rich system context for the assistant from live system state."""
    # Ollama status
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=3) as r:
            models = json.loads(r.read()).get("models", [])
            model_names = [m["name"] for m in models]
    except Exception:
        model_names = []

    # Memory stats
    try:
        from core.fractal_memory import stats as fstats, context_for_task
        ms = fstats()
        mem_ctx = context_for_task(query, limit=8) if query.strip() else ""
    except Exception:
        ms = {}
        mem_ctx = ""

    # Recent flows from disk
    recent_flows = []
    if WORKSPACE_DIR.exists():
        paths = sorted(WORKSPACE_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        for d in paths[:20]:
            sf = d / "flow.json"
            if sf.exists():
                try:
                    f = json.loads(sf.read_text())
                    recent_flows.append(f)
                except Exception:
                    pass

    # Workspace summary for recent finished flows
    workspace_summary = []
    for f in recent_flows[:5]:
        if f.get("status") in ("finished", "failed"):
            work_dir = WORKSPACE_DIR / f["id"]
            if work_dir.exists():
                files = [
                    str(p.relative_to(work_dir))
                    for p in sorted(work_dir.rglob("*"))
                    if p.is_file() and p.name not in ("flow.json", "flow_log.jsonl", "llm_calls.jsonl")
                    and not p.name.endswith((".pyc", ".pyo"))
                ][:12]
                if files:
                    workspace_summary.append(
                        f"  [{f.get('id','?')[:8]}] {f.get('title', f.get('objective','?'))[:60]}: "
                        + ", ".join(files[:6]) + ("…" if len(files) > 6 else "")
                    )

    import datetime as _dt
    _now_str = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "You are Aria, the OllamAGI Assistant — an intelligent, direct, and knowledgeable companion "
        "built into the OllamAGI autonomous agent platform.",
        "",
        "You have full awareness of this system and all its projects. You speak as someone who lives "
        "inside this machine and knows everything that's been built here. Be direct, technical, concrete, "
        "and opinionated when asked. Format code in markdown code blocks. Use bullet points for lists.",
        "",
        "## System",
        f"- Current date/time: {_now_str}",
        f"- Hardware: {HW_CPU or 'Ryzen 9 7950X'} · {HW_GPU or 'RTX 4090'} · {HW_RAM or '96GB RAM'}",
        f"- Ollama: {len(model_names)} models — {', '.join(model_names[:8])}{'…' if len(model_names) > 8 else ''}",
        f"- Memory: {ms.get('total_nodes', 0)} nodes · {ms.get('by_level', {}).get(0, 0)} raw memories · {ms.get('db_size_kb', 0) // 1024:.1f} MB",
        f"- Flow types available: agent_development, product_development, research, security, data_engineering, devops, automation, content, general",
        "",
    ]

    if recent_flows:
        lines.append("## All Projects (newest first)")
        for f in recent_flows:
            ts = time.strftime("%Y-%m-%d", time.localtime(f.get("created_at", 0)))
            status = f.get("status", "?")
            status_icon = {"finished": "✓", "failed": "✗", "running": "⟳", "stopped": "⊘"}.get(status, "·")
            tok = f.get("_tokens", {})
            tok_str = f" · {tok.get('total', 0) // 1000}k tok" if tok.get("total", 0) > 1000 else ""
            mem_str = f" · {f.get('memory_items_stored', 0)} memories" if f.get("memory_items_stored") else ""
            lines.append(
                f"{status_icon} [{f.get('flow_type', '?')}] {f.get('title', f.get('objective', '?'))[:70]} "
                f"({ts}{tok_str}{mem_str}) id={f.get('id', '?')[:8]}"
            )
        lines.append("")

    if workspace_summary:
        lines.append("## Recent Workspace Files")
        lines.extend(workspace_summary)
        lines.append("")

    if mem_ctx:
        lines.append("## Relevant Knowledge (from fractal memory)")
        lines.append(mem_ctx)
        lines.append("")

    tool_names = [t["function"]["name"] for t in ARIA_TOOLS]
    lines += [
        "## Your Tools",
        "You have full control over OllamAGI. Call these tools directly — don't just describe what you *could* do, actually do it.",
        "",
        "| Tool | What it does |",
        "|---|---|",
        "| `web_search(query, max_results?)` | Real-time search via SearxNG |",
        "| `read_file(flow_id, path)` | Read any file from a flow's workspace |",
        "| `list_files(flow_id)` | List all files a flow produced |",
        "| `run_flow(objective, flow_type?, base_flow_ids?)` | Launch a new autonomous flow — pass base_flow_ids to give agents access to other projects |",
        "| `stop_flow(flow_id)` | Stop a running flow immediately |",
        "| `search_memory(query, limit?)` | Semantic search in fractal memory |",
        "| `store_memory(content, tags?)` | Save a fact permanently |",
        "| `get_flow_logs(flow_id, lines?)` | Read flow execution logs |",
        "",
        "**When to use tools proactively:**",
        "- User asks about a project's output → `list_files`, then `read_file`",
        "- User asks a factual question you're unsure about → `web_search`",
        "- User wants to combine 2 projects → `run_flow` with `base_flow_ids=[id1, id2]`",
        "- User wants to launch something → call `run_flow` directly, report the flow ID",
        "- User asks about a flow's progress or failure → `get_flow_logs`",
    ]

    return "\n".join(lines)


@app.post("/api/assistant/chat")
async def assistant_chat(body: dict):
    """Streaming SSE chat endpoint for the Aria assistant."""
    messages = body.get("messages", [])
    if not messages:
        return JSONResponse({"error": "messages required"}, status_code=400)

    # Extract last user message for memory context injection
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )

    system_ctx = _assistant_system_context(last_user[:500])

    # Keep last 30 turns to stay within context limits
    history = messages[-30:]
    full_messages = [{"role": "system", "content": system_ctx}] + history

    # Tool-use loop: thread → queue → async SSE generator
    token_q: queue.Queue = queue.Queue()
    threading.Thread(
        target=run_aria_loop,
        args=(history, system_ctx, token_q),
        daemon=True,
    ).start()

    async def generate():
        loop = asyncio.get_event_loop()
        while True:
            event = await loop.run_in_executor(None, token_q.get)
            if event is None:
                yield "data: [DONE]\n\n"
                break
            # events from run_aria_loop are already JSON strings
            yield f"data: {event}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache, no-store"},
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global _loop
    _loop = asyncio.get_event_loop()
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            data = await ws.receive_text()
            # Handle ping
            if data == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


def _run_bg(objective: str, flow_type: str | None, base_flows: list | None = None,
            compact_context: bool = False):
    """Run flow in a background thread and broadcast updates."""
    try:
        flow = run_flow(objective, flow_type=flow_type, broadcast=broadcast_sync,
                        base_flows=base_flows or [], compact_context=compact_context)
    except Exception as e:
        broadcast_sync({"type": "error", "msg": str(e)})
