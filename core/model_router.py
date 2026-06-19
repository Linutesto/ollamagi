"""Route tasks to the right Ollama model — with token counting and per-flow cancellation."""
import httpx
import json
import queue
import threading
import time
from typing import Iterator
from core.config import OLLAMA_URL, MODELS, OLLAMA_CTX

# Global token counter — thread-safe
_lock = threading.Lock()
_token_totals: dict[str, dict] = {}   # flow_id → {prompt, completion, calls}

# Session-level accumulator (reset on demand, tracks since server start / last reset)
_session_tokens = {"prompt": 0, "completion": 0, "calls": 0}
_session_start: float = time.time()

# Stop signals registered by the orchestrator
_stop_events: dict[str, threading.Event] = {}
_stop_lock = threading.Lock()


class FlowStoppedException(BaseException):
    """Raised when a flow is cancelled. BaseException — bypasses bare except Exception."""
    pass


def register_stop_event(flow_id: str, event: threading.Event):
    with _stop_lock:
        _stop_events[flow_id] = event


def cancel_flow(flow_id: str):
    """Mark a flow as stopped so the next poll interval in chat() raises FlowStoppedException."""
    with _stop_lock:
        ev = _stop_events.get(flow_id)
    if ev:
        ev.set()


def _is_stopped(flow_id: str | None) -> bool:
    if not flow_id:
        return False
    with _stop_lock:
        ev = _stop_events.get(flow_id)
    return ev is not None and ev.is_set()


def interruptible_sleep(seconds: float, flow_id: str | None):
    """Sleep in 0.5s chunks, raising FlowStoppedException immediately if stop is requested."""
    elapsed = 0.0
    while elapsed < seconds:
        if _is_stopped(flow_id):
            raise FlowStoppedException(f"Flow {flow_id} stopped during sleep")
        chunk = min(0.5, seconds - elapsed)
        time.sleep(chunk)
        elapsed += chunk


def _interruptible_post(payload: dict, flow_id: str | None) -> dict:
    """
    Run the httpx POST in a daemon thread and poll for stop every 0.5s.
    This makes any LLM call instantly cancellable without waiting for Ollama to respond.
    The daemon thread finishes in the background (Ollama completes the request),
    but the calling flow sees an immediate stop.
    """
    result_q: queue.Queue = queue.Queue()

    def _worker():
        try:
            resp = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload,
                              timeout=httpx.Timeout(600.0))
            resp.raise_for_status()
            result_q.put(("ok", resp.json()))
        except Exception as e:
            result_q.put(("err", e))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    while True:
        if _is_stopped(flow_id):
            raise FlowStoppedException(f"Flow {flow_id} stopped")
        try:
            status, value = result_q.get(timeout=0.5)
            if status == "ok":
                return value
            raise value
        except queue.Empty:
            continue


def record_tokens(flow_id: str | None, prompt_tokens: int, completion_tokens: int):
    with _lock:
        _session_tokens["prompt"] += prompt_tokens
        _session_tokens["completion"] += completion_tokens
        _session_tokens["calls"] += 1
        if not flow_id:
            return
        if flow_id not in _token_totals:
            _token_totals[flow_id] = {"prompt": 0, "completion": 0, "calls": 0}
        _token_totals[flow_id]["prompt"] += prompt_tokens
        _token_totals[flow_id]["completion"] += completion_tokens
        _token_totals[flow_id]["calls"] += 1


def get_session_tokens() -> dict:
    with _lock:
        s = dict(_session_tokens)
        return {**s, "total": s["prompt"] + s["completion"], "since": _session_start}


def reset_session_tokens():
    global _session_start
    with _lock:
        _session_tokens["prompt"] = 0
        _session_tokens["completion"] = 0
        _session_tokens["calls"] = 0
        _session_start = time.time()


def get_tokens(flow_id: str) -> dict:
    with _lock:
        t = _token_totals.get(flow_id, {"prompt": 0, "completion": 0, "calls": 0})
        return {**t, "total": t["prompt"] + t["completion"]}


def get_all_tokens() -> dict:
    with _lock:
        return {fid: {**t, "total": t["prompt"] + t["completion"]}
                for fid, t in _token_totals.items()}


def _model_for_task(task_type: str) -> str:
    return MODELS.get(task_type, MODELS["orchestrator"])


def chat(messages: list[dict], task_type: str = "orchestrator",
         tools: list[dict] | None = None, stream: bool = False,
         flow_id: str | None = None) -> str | Iterator[str]:
    model = _model_for_task(task_type)
    payload = {
        "model": model,
        "messages": messages,
        "options": {"num_ctx": OLLAMA_CTX, "temperature": 0.2},
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools

    if not stream:
        data = _interruptible_post(payload, flow_id)
        record_tokens(flow_id, data.get("prompt_eval_count", 0), data.get("eval_count", 0))
        return data["message"]["content"]

    # Streaming: run in daemon thread, yield tokens via queue, poll stop signal
    def _stream():
        token_q: queue.Queue = queue.Queue()

        def _worker():
            try:
                with httpx.stream("POST", f"{OLLAMA_URL}/api/chat",
                                  json=payload, timeout=httpx.Timeout(600.0)) as r:
                    for line in r.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            token_q.put(("tok", chunk["message"]["content"]))
                        if chunk.get("done"):
                            record_tokens(flow_id,
                                          chunk.get("prompt_eval_count", 0),
                                          chunk.get("eval_count", 0))
                            token_q.put(("done", None))
            except Exception as e:
                token_q.put(("err", e))

        threading.Thread(target=_worker, daemon=True).start()

        while True:
            if _is_stopped(flow_id):
                raise FlowStoppedException(f"Flow {flow_id} stopped")
            try:
                kind, val = token_q.get(timeout=0.5)
                if kind == "tok":
                    yield val
                elif kind == "done":
                    return
                else:
                    raise val
            except queue.Empty:
                continue

    return _stream()


def decompose_flow(objective: str, context: str = "", flow_id: str | None = None) -> list[dict]:
    system = (
        "You are OllamAGI, a task decomposition engine. "
        "Break the objective into 3-7 concrete tasks. "
        "Each task must be actionable in a Linux shell. "
        "Return ONLY valid JSON: a list of objects with keys: "
        "id (int), title (str), type (pentest|research|code|analysis), "
        "description (str), container (pentest|python|generic), timeout (int seconds)."
    )
    user_parts = [f"OBJECTIVE: {objective}"]
    if context:
        user_parts.append(context)
    user_parts.append("Return only the JSON array, no other text.")

    raw = chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": "\n\n".join(user_parts)}],
        task_type="orchestrator",
        flow_id=flow_id,
    )
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return [{"id": 1, "title": objective[:80], "type": "analysis",
                 "description": objective, "container": "python", "timeout": 180}]


def generate_task_script(task: dict, context: str = "", flow_id: str | None = None) -> str:
    from core.config import SSH_HOST, SSH_USER
    system = (
        "You are OllamAGI's code generator. "
        "Generate a complete, executable bash script for the given task. "
        "The script runs inside a Linux container. "
        "/work is bind-mounted to the host workspace directory. "
        "The host home directory is bind-mounted and accessible. "
        f"SSH access to host: ssh {SSH_HOST} (key at /root/.ssh/id_ed25519, user={SSH_USER}). "
        "Save all outputs to /work/. Include proper error handling. "
        "Return ONLY the script, no markdown, no explanation."
    )
    user = f"TASK: {task['title']}\n\nDETAILS: {task['description']}"
    if context:
        user += f"\n\n{context}"
    return chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        task_type="coder",
        flow_id=flow_id,
    )
