"""Route tasks to the right Ollama model — with token counting and per-flow cancellation."""
import httpx
import json
import queue
import threading
import time
from typing import Iterator, Callable
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

# LLM call log callbacks — registered per flow by orchestrator
_llm_callbacks: dict[str, Callable] = {}
_llm_counters: dict[str, int] = {}
_llm_lock = threading.Lock()


def register_llm_callback(flow_id: str, cb: Callable):
    with _llm_lock:
        _llm_callbacks[flow_id] = cb
        _llm_counters[flow_id] = 0


def _fire_llm_log(flow_id: str | None, messages: list, response: str,
                  task_type: str, model: str, prompt_tokens: int, completion_tokens: int,
                  thinking: str = ""):
    if not flow_id:
        return
    with _llm_lock:
        cb = _llm_callbacks.get(flow_id)
        if not cb:
            return
        _llm_counters[flow_id] = _llm_counters.get(flow_id, 0) + 1
        n = _llm_counters[flow_id]
    try:
        cb({
            "type": "llm_call",
            "flow_id": flow_id,
            "ts": time.time(),
            "n": n,
            "task_type": task_type,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "messages": messages,
            "thinking": thinking,
            "response": response,
        })
    except Exception:
        pass
_model_cache_lock = threading.Lock()
_available_models_cache: tuple[float, list[str]] = (0.0, [])


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


def _interruptible_post(payload: dict, flow_id: str | None, timeout_s: float) -> dict:
    """
    Run the httpx POST in a daemon thread and poll for stop every 0.5s.
    This makes any LLM call instantly cancellable without waiting for Ollama to respond.
    The daemon thread finishes in the background (Ollama completes the request),
    but the calling flow sees an immediate stop.
    """
    result_q: queue.Queue = queue.Queue()
    deadline = time.monotonic() + timeout_s

    def _worker():
        try:
            resp = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload,
                              timeout=httpx.Timeout(timeout_s))
            resp.raise_for_status()
            result_q.put(("ok", resp.json()))
        except Exception as e:
            result_q.put(("err", e))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    while True:
        if _is_stopped(flow_id):
            raise FlowStoppedException(f"Flow {flow_id} stopped")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Ollama call timed out after {timeout_s:.0f}s")
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


def _available_models(force: bool = False) -> list[str]:
    global _available_models_cache
    with _model_cache_lock:
        cached_at, cached = _available_models_cache
        if cached and not force and time.time() - cached_at < 30:
            return list(cached)
    try:
        response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
        response.raise_for_status()
        names = [
            item.get("name") or item.get("model")
            for item in response.json().get("models", [])
        ]
        names = [name for name in names if name]
    except Exception:
        names = []
    with _model_cache_lock:
        _available_models_cache = (time.time(), names)
    return names


def _model_for_task(task_type: str) -> str:
    requested = MODELS.get(task_type, MODELS["orchestrator"])
    available = _available_models()
    if not available or requested in available:
        return requested
    raise RuntimeError(
        f"Configured single model '{requested}' is not installed in Ollama. "
        f"Available models: {', '.join(available[:12])}"
    )


_CALL_PROFILES = {
    # Small structured calls must stay small. Without num_predict, a local
    # thinking model can generate until the full context limit is exhausted.
    "fast":         {"num_predict": 384,  "timeout": 90.0,  "think": False},
    "orchestrator": {"num_predict": 2048, "timeout": 600.0, "think": False},
    "analysis":     {"num_predict": 3072, "timeout": 600.0, "think": True},
    "tools":        {"num_predict": 3072, "timeout": 600.0, "think": False},
    "coder":        {"num_predict": 4096, "timeout": 600.0, "think": True},
}


def chat(messages: list[dict], task_type: str = "orchestrator",
         tools: list[dict] | None = None, stream: bool = False,
         flow_id: str | None = None, max_tokens: int | None = None,
         timeout_s: float | None = None,
         think: bool | None = None) -> str | Iterator[str]:
    model = _model_for_task(task_type)
    profile = _CALL_PROFILES.get(task_type, _CALL_PROFILES["orchestrator"])
    call_timeout = timeout_s or profile["timeout"]
    payload = {
        "model": model,
        "messages": messages,
        "options": {
            "num_ctx": OLLAMA_CTX,
            "temperature": 0.2,
            "num_predict": max_tokens or profile["num_predict"],
        },
        "stream": stream,
    }
    if think is not None:
        payload["think"] = think
    elif "think" in profile:
        payload["think"] = profile["think"]
    if tools:
        payload["tools"] = tools

    if not stream:
        data = _interruptible_post(payload, flow_id, call_timeout)
        pt = data.get("prompt_eval_count", 0)
        ct = data.get("eval_count", 0)
        record_tokens(flow_id, pt, ct)
        response_text = data["message"]["content"]
        thinking_text = data["message"].get("thinking", "") or ""
        _fire_llm_log(flow_id, messages, response_text, task_type, model, pt, ct, thinking_text)
        return response_text

    # Streaming: run in daemon thread, yield tokens via queue, poll stop signal
    def _stream():
        token_q: queue.Queue = queue.Queue()
        deadline = time.monotonic() + call_timeout

        def _worker():
            try:
                with httpx.stream("POST", f"{OLLAMA_URL}/api/chat",
                                  json=payload, timeout=httpx.Timeout(call_timeout)) as r:
                    for line in r.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        msg = chunk.get("message", {})
                        if msg.get("thinking"):
                            token_q.put(("think", msg["thinking"]))
                        if msg.get("content"):
                            token_q.put(("tok", msg["content"]))
                        if chunk.get("done"):
                            pt = chunk.get("prompt_eval_count", 0)
                            ct = chunk.get("eval_count", 0)
                            record_tokens(flow_id, pt, ct)
                            token_q.put(("done", (pt, ct)))
            except Exception as e:
                token_q.put(("err", e))

        threading.Thread(target=_worker, daemon=True).start()

        accumulated: list[str] = []
        thinking_acc: list[str] = []
        while True:
            if _is_stopped(flow_id):
                raise FlowStoppedException(f"Flow {flow_id} stopped")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Ollama stream timed out after {call_timeout:.0f}s")
            try:
                kind, val = token_q.get(timeout=0.5)
                if kind == "tok":
                    accumulated.append(val)
                    yield val
                elif kind == "think":
                    thinking_acc.append(val)
                elif kind == "done":
                    pt, ct = val
                    _fire_llm_log(flow_id, messages, "".join(accumulated),
                                  task_type, model, pt, ct, "".join(thinking_acc))
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
