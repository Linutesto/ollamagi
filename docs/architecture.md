# OllamAGI Architecture

## Overview

OllamAGI is a 5-layer system connecting a web interface to local LLMs via a flow orchestrator that executes code in Docker containers and feeds results into persistent memory.

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 5 — INTERFACE                                    │
│  Mobile-first SPA (single HTML file, no build step)     │
│  WebSocket for live log streaming                       │
│  Tabs: Flows · Run · Terminal · Logs · Memory · System  │
├─────────────────────────────────────────────────────────┤
│  LAYER 4 — ORCHESTRATOR  (core/orchestrator.py)         │
│  Flow → Task decomposition → Subtask execution          │
│  Auto-fix (2 retries) · Replan · Steer · Memory distill │
│  Immediate stop via BaseException + queue polling       │
├─────────────────────────────────────────────────────────┤
│  LAYER 3 — MODEL ROUTER  (core/model_router.py)         │
│  Routes calls to the right Ollama model per role        │
│  Token counting (per-flow + session)                    │
│  Interruptible LLM calls (daemon thread + queue)        │
├─────────────────────────────────────────────────────────┤
│  LAYER 2 — EXECUTOR  (executor/docker_manager.py)       │
│  Spins up Docker containers per subtask                 │
│  Injects SSH key · bootstraps Python packages           │
│  exec_python(): writes .py via tar, runs with python3   │
│  exec_script(): bash scripts                            │
├─────────────────────────────────────────────────────────┤
│  LAYER 1 — MEMORY  (core/memory_bridge.py)              │
│  Cognitive memory SQLite: beliefs, memories, goals, RAG │
│  context_for_task(): semantic search per subtask        │
│  _memory_distill(): extracts 1-3 beliefs per task       │
└─────────────────────────────────────────────────────────┘
```

---

## Flow Lifecycle

```
User submits objective
       ↓
_generate_tasks()       — LLM decomposes into 3-7 tasks
       ↓
for each Task:
  _generate_subtasks()  — LLM plans 3-5 subtasks
  for each Subtask:
    _execute_subtask()  — runs code in Docker container
    if exit_code != 0:
      _fix_python/bash() — LLM auto-fixes, retry up to 2×
    if subtask fails:
      reflector agent   — extracts lesson
  if 2 consecutive task failures:
    _replan_remaining() — LLM replans rest of flow
  task done → _memory_distill() → cognitive memory
flow done → extract_and_store() → cognitive memory bulk write
```

---

## Agent Roles

| Role | Model key | Purpose |
|---|---|---|
| `primary_agent` | orchestrator | Main planning and synthesis |
| `generator` | orchestrator | Subtask plan generation |
| `refiner` | orchestrator | Plan review and improvement |
| `coder` | coder | Python/bash code writing |
| `installer` | coder | System setup, DevOps |
| `researcher` | orchestrator | Information gathering |
| `adviser` | orchestrator | Strategic guidance |
| `reflector` | orchestrator | Failure analysis |
| `architect` | orchestrator | System design |
| `monetizer` | orchestrator | ROI/opportunity analysis |
| `pentester` | tools | Security testing (authorized only) |

---

## Immediate Stop Design

The naive approach (closing the httpx client, injecting exceptions via ctypes) doesn't work because Ollama calls block at the C level. OllamAGI uses a **daemon thread + queue** pattern:

1. LLM call runs in a daemon thread
2. Main thread polls a `queue.Queue` with 0.5s timeout
3. Between polls, checks `_stop_events[flow_id]`
4. On stop: raises `FlowStoppedException(BaseException)`
5. `BaseException` bypasses all `except Exception` handlers and reaches the top-level catch in `run_flow()`

The Ollama daemon thread completes in the background (result discarded).

---

## Cognitive Memory

The cognitive memory system is optional. It is a SQLite database covering:
- `beliefs` — factual assertions extracted from task results
- `memories` — episodic records of past work
- `goals` — current objectives and priorities
- Semantic index — vector embeddings for RAG retrieval

OllamAGI reads memory context before each task and writes new beliefs after completion. If `MEMORY_DB` does not exist, memory features are silently disabled.

---

## Container Strategy

Each subtask gets its own fresh container:

- **python:latest** — Python code tasks (coder/installer agents)
- **debian:latest** — generic bash tasks
- **kali-linux** — security/pentest tasks

Containers have:
- `/work` bind-mounted to `workspace/<flow-id>/` on the host
- Host home directory bind-mounted for file persistence
- Docker socket for DinD (Docker-in-Docker) if needed
- SSH key injected for host access
- Common Python packages pre-installed at startup (non-blocking)

Code is injected via `put_archive` (tar) and run with `python3` directly — never via bash heredocs, which break on Python syntax and special characters.

---

## Token Counting

Every Ollama call records `prompt_eval_count` and `eval_count` to:
- Per-flow totals (persisted in `flow.json`, survives restarts)
- Session totals (in-memory, resettable via dashboard)
- All-time totals (aggregated from all `flow.json` files on disk)
