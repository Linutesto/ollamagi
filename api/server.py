"""OllamAGI FastAPI server — complete REST API + WebSocket + static dashboard."""
import asyncio
import json
import sqlite3
import threading
import uuid
import urllib.request
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from core.config import WORKSPACE_DIR, HERMES_DB, HW_CPU, HW_RAM, HW_GPU, SSH_HOST, SSH_USER, SSH_KEY
from core.orchestrator import (run_flow, get_flow, get_all_flows, register_log_callback,
                                _flow_to_dict, request_stop, inject_steer)
from core.memory_bridge import get_relevant_context, get_goals, store_memory
from core.model_router import get_tokens, get_all_tokens, get_session_tokens, reset_session_tokens

app = FastAPI(title="OllamAGI", version="1.0.0")
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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
    return FileResponse(str(STATIC_DIR / "index.html"))


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
    background_tasks.add_task(_run_bg, objective, flow_type)
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


@app.get("/api/memory/goals")
async def list_goals():
    return JSONResponse(get_goals(limit=15))


@app.get("/api/memory/search")
async def search_memory(q: str = ""):
    if not q:
        return JSONResponse([])
    return JSONResponse(get_relevant_context(q, limit=12))


@app.get("/api/memory/stats")
async def memory_stats():
    try:
        db = sqlite3.connect(HERMES_DB, timeout=5)
        beliefs = db.execute("SELECT COUNT(*) FROM beliefs").fetchone()[0]
        memories = db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        goals = db.execute("SELECT COUNT(*) FROM goals").fetchone()[0]
        db.close()
        return JSONResponse({"beliefs": beliefs, "memories": memories, "goals": goals})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            data = json.loads(r.read())
            ollama_models = len(data.get("models", []))
            ollama_ok = True
    except Exception:
        pass
    hermes_ok = False
    try:
        db = sqlite3.connect(HERMES_DB, timeout=3)
        db.execute("SELECT 1 FROM beliefs LIMIT 1")
        db.close()
        hermes_ok = True
    except Exception:
        pass
    return JSONResponse({
        "ollama": {"ok": ollama_ok, "models": ollama_models},
        "hermes": {"ok": hermes_ok},
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
async def store_to_hermes(body: dict):
    content = body.get("content", "").strip()
    if not content:
        return JSONResponse({"error": "content required"}, status_code=400)
    ok = store_memory(content, source="ollamagi:manual", tags=["manual"])
    return JSONResponse({"ok": ok})


@app.get("/api/flows/{flow_id}/workspace")
async def flow_workspace(flow_id: str):
    work_dir = WORKSPACE_DIR / flow_id
    if not work_dir.exists():
        return JSONResponse([])
    files = []
    for f in sorted(work_dir.rglob("*")):
        if f.is_file() and f.name != "flow.json":
            files.append({
                "path": str(f.relative_to(work_dir)),
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime,
            })
    return JSONResponse(files)


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


def _run_bg(objective: str, flow_type: str | None):
    """Run flow in a background thread and broadcast updates."""
    try:
        flow = run_flow(objective, flow_type=flow_type, broadcast=broadcast_sync)
    except Exception as e:
        broadcast_sync({"type": "error", "msg": str(e)})
