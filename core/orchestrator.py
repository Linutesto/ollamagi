"""OllamAGI orchestrator — flow lifecycle with stop/steer/autofix/replan/memory agents."""
import uuid
import time
import json
import threading
import ctypes
from dataclasses import dataclass, field
from typing import Callable
from pathlib import Path

from core.config import WORKSPACE_DIR
from core.agents import run_agent, ALL_ROLES, FLOW_TYPE_ROLES
from core.memory_bridge import context_for_task, store_memory, store_belief
from core.model_router import chat, get_tokens, cancel_flow, register_stop_event, FlowStoppedException, interruptible_sleep
from executor.docker_manager import create_container, exec_script, exec_python, stop_container, sync_workspace
from memory.extractor import extract_and_store

MAX_RETRIES = 2  # auto-fix attempts per subtask before giving up


@dataclass
class Subtask:
    id: str
    task_id: str
    title: str
    description: str
    agent: str
    status: str = "created"   # created|running|finished|failed|retrying
    result: str = ""
    output: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    container_type: str = "python"
    needs_container: bool = False
    attempts: int = 0


@dataclass
class Task:
    id: str
    flow_id: str
    title: str
    description: str
    agent: str
    subtasks: list[Subtask] = field(default_factory=list)
    status: str = "created"
    result: str = ""
    started_at: float | None = None
    finished_at: float | None = None


@dataclass
class Flow:
    id: str
    title: str
    objective: str
    flow_type: str
    tasks: list[Task] = field(default_factory=list)
    status: str = "created"   # created|running|stopped|finished|failed
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    hermes_items_stored: int = 0
    replan_count: int = 0


# ── Registries ──────────────────────────────────────────────────────────────
_flows: dict[str, Flow] = {}
_log_callbacks: dict[str, list[Callable]] = {}
_stop_signals: dict[str, threading.Event] = {}
_steer_queue: dict[str, list[str]] = {}
_flow_threads: dict[str, int] = {}   # flow_id → thread.ident


# ── Public control API ───────────────────────────────────────────────────────
def get_flow(flow_id: str) -> Flow | None:
    return _flows.get(flow_id)

def get_all_flows() -> list[Flow]:
    return list(_flows.values())

def request_stop(flow_id: str):
    # 1. Set the stop event first so all checks see it immediately
    ev = _stop_signals.get(flow_id)
    if ev:
        ev.set()

    # 2. Close the flow's httpx client — immediately unblocks any pending LLM call
    #    (raises FlowStoppedException in the blocked thread)
    cancel_flow(flow_id)

    # 3. Kill any running Docker containers for this flow
    try:
        import docker as docker_lib
        dclient = docker_lib.from_env()
        for c in dclient.containers.list(filters={"name": f"ollamagi-{flow_id}"}):
            try:
                c.kill()
            except Exception:
                pass
    except Exception:
        pass

def inject_steer(flow_id: str, message: str):
    _steer_queue.setdefault(flow_id, []).append(message)

def register_log_callback(flow_id: str, cb: Callable):
    _log_callbacks.setdefault(flow_id, []).append(cb)

def _is_stopped(flow_id: str) -> bool:
    return _stop_signals.get(flow_id, threading.Event()).is_set()

def _drain_steer(flow_id: str) -> list[str]:
    msgs = _steer_queue.pop(flow_id, [])
    return msgs


# ── Persistence ──────────────────────────────────────────────────────────────
def _log(flow_id: str, msg: str, level: str = "info"):
    for cb in _log_callbacks.get(flow_id, []):
        try:
            cb({"flow_id": flow_id, "msg": msg, "level": level, "ts": time.time()})
        except Exception:
            pass

def _save(flow: Flow):
    work_dir = WORKSPACE_DIR / flow.id
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "flow.json").write_text(json.dumps(_flow_to_dict(flow), indent=2))

def _flow_to_dict(flow: Flow) -> dict:
    tok = get_tokens(flow.id)
    return {
        "id": flow.id,
        "title": flow.title,
        "objective": flow.objective,
        "flow_type": flow.flow_type,
        "status": flow.status,
        "created_at": flow.created_at,
        "finished_at": flow.finished_at,
        "hermes_items_stored": flow.hermes_items_stored,
        "replan_count": flow.replan_count,
        "_tokens": tok,
        "tasks": [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "agent": t.agent,
                "status": t.status,
                "result": t.result[:500] if t.result else "",
                "started_at": t.started_at,
                "finished_at": t.finished_at,
                "subtasks": [
                    {
                        "id": s.id,
                        "title": s.title,
                        "agent": s.agent,
                        "status": s.status,
                        "result": s.result[:300] if s.result else "",
                        "attempts": s.attempts,
                        "started_at": s.started_at,
                        "finished_at": s.finished_at,
                    }
                    for s in t.subtasks
                ],
            }
            for t in flow.tasks
        ],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────
def _detect_flow_type(objective: str) -> str:
    obj = objective.lower()
    if any(w in obj for w in ["agent", "autonomous", "skill", "tool", "ai system", "bot"]):
        return "agent_development"
    if any(w in obj for w in ["product", "saas", "revenue", "monetize", "business", "roi", "sell"]):
        return "product_development"
    if any(w in obj for w in ["research", "analyze", "study", "explore", "discover", "map"]):
        return "research"
    if any(w in obj for w in ["pentest", "hack", "security", "vuln", "exploit", "bug bounty"]):
        return "security"
    return "general"

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return text.strip()


# ── Planning ──────────────────────────────────────────────────────────────────
def _generate_tasks(flow: Flow, hermes_ctx: str) -> list[Task]:
    roles = FLOW_TYPE_ROLES.get(flow.flow_type, FLOW_TYPE_ROLES["general"])
    role_list = ", ".join(roles)
    system = (
        f"You are decomposing a '{flow.flow_type}' flow into tasks.\n"
        f"Available agents: {role_list}\n"
        "Return ONLY valid JSON array with keys: id(int), title, description, agent, "
        "needs_container(bool), container_type('pentest'|'python'|'generic')\n"
        "3-7 tasks. Concrete and actionable. No markdown."
    )
    user_parts = [f"OBJECTIVE: {flow.objective}"]
    if hermes_ctx:
        user_parts.append(hermes_ctx)
    raw = chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": "\n\n".join(user_parts)}],
        task_type="orchestrator", flow_id=flow.id,
    )
    raw = _strip_fences(raw.strip())
    try:
        task_defs = json.loads(raw)
    except Exception:
        task_defs = [{"id": 1, "title": flow.objective[:80], "description": flow.objective,
                      "agent": "primary_agent", "needs_container": False, "container_type": "python"}]
    tasks = []
    for td in task_defs:
        tasks.append(Task(
            id=f"t{td.get('id', len(tasks)+1)}",
            flow_id=flow.id,
            title=td.get("title", "Task"),
            description=td.get("description", ""),
            agent=td.get("agent", "primary_agent"),
        ))
    return tasks


def _generate_subtasks(task: Task, flow: Flow, hermes_ctx: str) -> list[Subtask]:
    system = (
        "You are the Generator agent. Break this task into 2-5 concrete subtasks.\n"
        "Return ONLY JSON array: {id(int), title, description, agent, "
        "needs_container(bool), container_type('pentest'|'python'|'generic')}\n"
        "No markdown."
    )
    user = f"TASK: {task.title}\n\nDESCRIPTION: {task.description}"
    if hermes_ctx:
        user += f"\n\n{hermes_ctx}"
    raw = chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        task_type="orchestrator", flow_id=flow.id,
    )
    raw = _strip_fences(raw.strip())
    try:
        sub_defs = json.loads(raw)
    except Exception:
        return [Subtask(id=f"{task.id}-s1", task_id=task.id, title=task.title,
                        description=task.description, agent=task.agent)]
    subtasks = []
    for sd in sub_defs:
        subtasks.append(Subtask(
            id=f"{task.id}-s{sd.get('id', len(subtasks)+1)}",
            task_id=task.id,
            title=sd.get("title", "Subtask"),
            description=sd.get("description", ""),
            agent=sd.get("agent", task.agent),
            needs_container=sd.get("needs_container", False),
            container_type=sd.get("container_type", "python"),
        ))
    return subtasks


def _replan_remaining(flow: Flow, from_idx: int, completed: list[Task],
                      steer_msgs: list[str]) -> list[Task]:
    """Ask the orchestrator to produce new tasks for remaining work."""
    roles = FLOW_TYPE_ROLES.get(flow.flow_type, FLOW_TYPE_ROLES["general"])
    done_summary = "\n".join(
        f"✓ {t.title}: {(t.result or '')[:120]}" for t in completed if t.status == "finished"
    ) or "nothing yet"
    steer_note = "\n".join(f"[USER STEER]: {m}" for m in steer_msgs)

    system = (
        "You are replanning a flow after failure or user steering.\n"
        f"Available agents: {', '.join(roles)}\n"
        "Return ONLY valid JSON array: {id(int), title, description, agent, "
        "needs_container(bool), container_type}. 2-5 tasks."
    )
    user = (
        f"OBJECTIVE: {flow.objective}\n\n"
        f"COMPLETED:\n{done_summary}\n\n"
        + (f"USER STEERING:\n{steer_note}\n\n" if steer_note else "")
        + "Plan REMAINING tasks only. Skip what's done. Fix failed approaches."
    )
    raw = chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        task_type="orchestrator", flow_id=flow.id,
    )
    raw = _strip_fences(raw.strip())
    try:
        defs = json.loads(raw)
    except Exception:
        return []
    tasks = []
    for i, td in enumerate(defs):
        tasks.append(Task(
            id=f"r{from_idx + i + 1}",
            flow_id=flow.id,
            title=td.get("title", "Task"),
            description=td.get("description", ""),
            agent=td.get("agent", "primary_agent"),
        ))
    return tasks


# ── Auto-fix ──────────────────────────────────────────────────────────────────
def _fix_python(code: str, error_output: str, description: str, flow_id: str | None) -> str:
    prompt = (
        "Fix this Python script that failed. Return ONLY corrected Python code — no markdown.\n"
        "To install packages use: import sys,subprocess; subprocess.run([sys.executable,'-m','pip','install','pkg'],check=False,capture_output=True)\n\n"
        f"TASK: {description}\n\n"
        f"ERROR OUTPUT:\n{error_output[:2000]}\n\n"
        f"FAILING CODE:\n{code[:3000]}"
    )
    fixed = chat([{"role": "user", "content": prompt}], task_type="coder", flow_id=flow_id)
    return _strip_fences(fixed)

def _fix_bash(script: str, error_output: str, description: str, flow_id: str | None) -> str:
    prompt = (
        "Fix this bash script that failed. Do NOT use heredocs. Return ONLY corrected bash — no markdown.\n\n"
        f"TASK: {description}\n\n"
        f"ERROR OUTPUT:\n{error_output[:2000]}\n\n"
        f"FAILING SCRIPT:\n{script[:3000]}"
    )
    fixed = chat([{"role": "user", "content": prompt}], task_type="coder", flow_id=flow_id)
    return _strip_fences(fixed)


# ── Memory agent ──────────────────────────────────────────────────────────────
def _memory_distill(flow_id: str, task_title: str, result: str) -> int:
    """Extract key facts from a task result and store as Hermes beliefs."""
    if not result or len(result) < 60:
        return 0
    prompt = (
        f"Extract 1-3 concise, reusable facts or learnings from this task result.\n"
        f"Task: {task_title}\nResult: {result[:2000]}\n\n"
        "Return ONLY a JSON array of strings (the facts). No markdown. Max 3 items."
    )
    raw = chat([{"role": "user", "content": prompt}], task_type="fast", flow_id=flow_id)
    raw = _strip_fences(raw.strip())
    try:
        facts = json.loads(raw)
        if isinstance(facts, list):
            count = 0
            for f in facts[:3]:
                if isinstance(f, str) and len(f) > 20:
                    store_belief(f, confidence=0.72, source=f"ollamagi:flow:{flow_id}")
                    count += 1
            return count
    except Exception:
        pass
    return 0


# ── Subtask execution with auto-fix retry ─────────────────────────────────────
def _execute_subtask(subtask: Subtask, flow: Flow, task: Task,
                     history: list[dict], log_fn: Callable,
                     flow_id: str | None = None) -> str:
    hermes_ctx = context_for_task(subtask.description)
    extra_ctx = f"FLOW: {flow.objective}\nTASK: {task.title}\n"
    if hermes_ctx:
        extra_ctx += f"\n{hermes_ctx}"

    if not subtask.needs_container:
        messages = history + [{"role": "user", "content": subtask.description}]
        return run_agent(subtask.agent, messages, extra_ctx, flow_id=flow_id)

    use_python = subtask.agent in ("coder", "installer")

    if use_python:
        code_prompt = (
            f"Write a Python 3 script that accomplishes this subtask:\n\n"
            f"{subtask.description}\n\n"
            "Environment: Python 3 container. Pre-installed: requests, httpx, aiohttp, rich, "
            "beautifulsoup4, python-dotenv, pyyaml, psutil.\n\n"
            "Rules:\n"
            "- Save ALL output files to /work/ using open('/work/filename', 'w')\n"
            "- Install missing packages with pip using --prefer-binary to avoid Rust/C compilation:\n"
            "  import sys, subprocess\n"
            "  subprocess.run([sys.executable, '-m', 'pip', 'install', '--prefer-binary', 'pkg'], check=False, capture_output=True)\n"
            "- NEVER use pydantic v2 — use pydantic v1 (pip install 'pydantic<2') or plain dataclasses instead\n"
            "- NEVER use packages that require Rust compilation (polars, cryptography>=42, pydantic-core, etc.)\n"
            "- Prefer packages that have pre-built wheels: requests, httpx, bs4, lxml, playwright, selenium\n"
            "- Print progress to stdout so the user can track execution\n"
            "- Wrap the main logic in try/except and print any errors clearly\n"
            "- Return ONLY raw Python code — NO markdown fences, NO explanation"
        )
        messages = history + [{"role": "user", "content": code_prompt}]
        code = _strip_fences(run_agent(subtask.agent, messages, extra_ctx, flow_id=flow_id))

        for attempt in range(MAX_RETRIES + 1):
            subtask.attempts = attempt + 1
            container = None
            try:
                container = create_container(flow.id, f"{subtask.id}-a{attempt}", subtask.container_type)
                exit_code, output = exec_python(container, code, timeout=300)
                if exit_code == 0:
                    return output[-4000:] if output else "(done)"
                # Non-zero exit — auto-fix if retries remain
                if attempt < MAX_RETRIES:
                    log_fn(f"  ⚠ exit {exit_code} — auto-fixing (attempt {attempt+1}/{MAX_RETRIES})…", "warn")
                    code = _fix_python(code, output, subtask.description, flow_id)
                else:
                    return f"[FAILED after {attempt+1} attempts]\n{output[-2000:]}"
            except Exception as e:
                if attempt < MAX_RETRIES:
                    log_fn(f"  ⚠ container error — retrying: {e}", "warn")
                else:
                    return f"Container error: {e}"
            finally:
                if container:
                    stop_container(container)
        return "[FAILED — max retries exceeded]"

    else:
        script_prompt = (
            f"Write a bash script for this subtask:\n{subtask.description}\n"
            "Environment: Debian Linux container with apt-get. "
            "python3 and pip3 may not be pre-installed — if you need them, add: "
            "apt-get install -y -qq python3 python3-pip 2>/dev/null || true\n"
            "Do NOT use heredocs. Do NOT embed Python code inline in bash.\n"
            "Do NOT assume any tool is available — install what you need via apt-get.\n"
            "Save all outputs to /work/. Return ONLY the bash script, no markdown."
        )
        messages = history + [{"role": "user", "content": script_prompt}]
        script = _strip_fences(run_agent(subtask.agent, messages, extra_ctx, flow_id=flow_id))

        for attempt in range(MAX_RETRIES + 1):
            subtask.attempts = attempt + 1
            container = None
            try:
                container = create_container(flow.id, f"{subtask.id}-a{attempt}", subtask.container_type)
                exit_code, output = exec_script(container, script, timeout=300)
                if exit_code == 0:
                    return output[-4000:] if output else "(done)"
                if attempt < MAX_RETRIES:
                    log_fn(f"  ⚠ exit {exit_code} — auto-fixing bash (attempt {attempt+1}/{MAX_RETRIES})…", "warn")
                    script = _fix_bash(script, output, subtask.description, flow_id)
                else:
                    return f"[FAILED after {attempt+1} attempts]\n{output[-2000:]}"
            except Exception as e:
                if attempt < MAX_RETRIES:
                    log_fn(f"  ⚠ container error — retrying: {e}", "warn")
                else:
                    return f"Container error: {e}"
            finally:
                if container:
                    stop_container(container)
        return "[FAILED — max retries exceeded]"


# ── Main flow runner ──────────────────────────────────────────────────────────
def run_flow(objective: str, flow_type: str | None = None,
             broadcast: Callable | None = None) -> Flow:
    flow_id = uuid.uuid4().hex[:8]
    detected_type = flow_type or _detect_flow_type(objective)
    title = objective[:60] + ("..." if len(objective) > 60 else "")

    flow = Flow(id=flow_id, title=title, objective=objective, flow_type=detected_type)
    _flows[flow_id] = flow
    stop_event = threading.Event()
    _stop_signals[flow_id] = stop_event
    _steer_queue[flow_id] = []
    _flow_threads[flow_id] = threading.current_thread().ident
    register_stop_event(flow_id, stop_event)  # model_router polls this every 0.5s

    def log(msg: str, level: str = "info"):
        _log(flow_id, msg, level)
        if broadcast:
            broadcast({"type": "log", "flow_id": flow_id, "msg": msg, "level": level})

    def _mark_stopped():
        flow.status = "stopped"
        flow.finished_at = time.time()
        _save(flow)
        log("Flow force-stopped", "warn")
        if broadcast:
            broadcast({"type": "flow_done", "flow_id": flow_id,
                       "status": "stopped", "data": _flow_to_dict(flow)})

    if broadcast:
        broadcast({"type": "flow_start", "flow_id": flow_id})

    log(f"Flow {flow_id} started — type: {detected_type}")
    _save(flow)

    hermes_ctx = context_for_task(objective)
    if hermes_ctx:
        log(f"Hermes: {len(hermes_ctx.splitlines())} relevant memories loaded")

    try:
        flow.status = "running"
        log("primary_agent: decomposing objective…")
        for _attempt in range(3):
            try:
                flow.tasks = _generate_tasks(flow, hermes_ctx)
                break
            except (SystemExit, KeyboardInterrupt, FlowStoppedException):
                raise
            except Exception as e:
                if _attempt == 2:
                    raise
                log(f"  ⚠ task decomposition error (attempt {_attempt+1}/3): {e} — retrying in 5s…", "warn")
                interruptible_sleep(5, flow_id)
        log(f"primary_agent: {len(flow.tasks)} tasks planned")
        _save(flow)

        conversation_history: list[dict] = [{"role": "user", "content": f"Objective: {objective}"}]
        all_results: list[dict] = []
        completed_tasks: list[Task] = []
        consecutive_failures = 0
        task_idx = 0

        while task_idx < len(flow.tasks):
            # ── Steer check (before each task) ──
            steers = _drain_steer(flow_id)
            if steers:
                for m in steers:
                    log(f"[STEER] {m[:120]}", "warn")
                log("Replanning remaining tasks based on steering…", "warn")
                new_tasks = _replan_remaining(flow, task_idx, completed_tasks, steers)
                if new_tasks:
                    flow.tasks = flow.tasks[:task_idx] + new_tasks
                    flow.replan_count += 1
                    log(f"Replanned: {len(new_tasks)} new tasks (replan #{flow.replan_count})")
                    if broadcast:
                        broadcast({"type": "replan", "flow_id": flow_id,
                                   "msg": f"Replanned — {len(new_tasks)} new tasks"})
                    _save(flow)

            task = flow.tasks[task_idx]
            task.status = "running"
            task.started_at = time.time()
            log(f"task [{task.id}] {task.title} — agent: {task.agent}")
            _save(flow)

            log(f"  generator: planning subtasks for '{task.title}'…")
            for _attempt in range(3):
                try:
                    task.subtasks = _generate_subtasks(task, flow, hermes_ctx)
                    break
                except (SystemExit, KeyboardInterrupt, FlowStoppedException):
                    raise
                except Exception as e:
                    if _attempt == 2:
                        raise
                    log(f"  ⚠ subtask planning error (attempt {_attempt+1}/3): {e} — retrying in 5s…", "warn")
                    interruptible_sleep(5, flow_id)
            log(f"  generator: {len(task.subtasks)} subtasks")
            _save(flow)

            task_results = []
            task_had_failure = False

            for subtask in task.subtasks:
                subtask.status = "running"
                subtask.started_at = time.time()
                log(f"  [{subtask.agent}] {subtask.title}")
                _save(flow)

                try:
                    result = _execute_subtask(
                        subtask, flow, task, conversation_history, log, flow_id
                    )
                    failed = result.startswith("[FAILED")
                    subtask.result = result
                    subtask.status = "failed" if failed else "finished"
                    if failed:
                        log(f"  ✗ {subtask.title[:60]} — gave up after {MAX_RETRIES+1} attempts", "error")
                        task_had_failure = True
                        reflection = run_agent(
                            "reflector",
                            [{"role": "user", "content":
                              f"Subtask '{subtask.title}' failed.\n"
                              f"Error: {result[:500]}\nTask: {task.description}"}],
                            flow_id=flow_id,
                        )
                        log(f"  reflector: {reflection[:200]}")
                    else:
                        log(f"  ✓ {subtask.title[:60]}")
                except Exception as e:
                    subtask.result = str(e)
                    subtask.status = "failed"
                    task_had_failure = True
                    log(f"  ✗ {subtask.title[:60]}: {e}", "error")

                subtask.finished_at = time.time()
                task_results.append({"subtask": subtask.title, "result": subtask.result[:1000]})
                conversation_history.append({
                    "role": "assistant",
                    "content": f"Subtask '{subtask.title}': {subtask.result[:500]}"
                })
                _save(flow)

            synthesis_prompt = (
                f"Synthesize the results of task '{task.title}':\n"
                + "\n".join(f"- {r['subtask']}: {r['result'][:200]}" for r in task_results)
            )
            task.result = run_agent("primary_agent",
                                    [{"role": "user", "content": synthesis_prompt}],
                                    flow_id=flow_id)
            task.status = "failed" if task_had_failure else "finished"
            task.finished_at = time.time()
            log(f"task [{task.id}] {'failed' if task_had_failure else 'done'}")
            all_results.append({"task": task.title, "result": task.result})

            if task.status == "finished":
                n = _memory_distill(flow_id, task.title, task.result)
                if n:
                    flow.hermes_items_stored += n
                    log(f"  memory agent: {n} belief(s) stored in Hermes")
                completed_tasks.append(task)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 2 and task_idx + 1 < len(flow.tasks):
                    log("2 consecutive task failures — triggering replan…", "warn")
                    new_tasks = _replan_remaining(flow, task_idx + 1, completed_tasks, [])
                    if new_tasks:
                        flow.tasks = flow.tasks[:task_idx + 1] + new_tasks
                        flow.replan_count += 1
                        log(f"Auto-replanned: {len(new_tasks)} new tasks (replan #{flow.replan_count})")
                        if broadcast:
                            broadcast({"type": "replan", "flow_id": flow_id,
                                       "msg": f"Auto-replanned after failures — {len(new_tasks)} tasks"})
                    consecutive_failures = 0

            _save(flow)
            task_idx += 1

        # Final Hermes extraction
        log("Extracting knowledge into Hermes memory…")
        from executor.task_runner import TaskResult
        fake = [
            TaskResult(task_id=i, title=r["task"], status="success",
                       exit_code=0, output=r["result"], duration=0)
            for i, r in enumerate(all_results)
        ]
        n = extract_and_store(flow_id, objective, fake)
        flow.hermes_items_stored += n
        log(f"{n} more items stored in Hermes (total: {flow.hermes_items_stored})")

        flow.status = "finished"
        flow.finished_at = time.time()
        _save(flow)
        log(f"Flow {flow_id} complete — workspace: {WORKSPACE_DIR / flow_id}")
        if broadcast:
            broadcast({"type": "flow_done", "flow_id": flow_id,
                       "status": "finished", "data": _flow_to_dict(flow)})

    except (SystemExit, KeyboardInterrupt, FlowStoppedException):
        _mark_stopped()

    except Exception as e:
        flow.status = "failed"
        flow.finished_at = time.time()
        _save(flow)
        log(f"Flow failed: {e}", "error")
        if broadcast:
            broadcast({"type": "flow_done", "flow_id": flow_id,
                       "status": "failed", "data": _flow_to_dict(flow)})

    return flow
