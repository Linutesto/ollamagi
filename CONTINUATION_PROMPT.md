# OllamAGI — Continuation Prompt

Paste this at the start of a new Claude Code session to resume work instantly.

---

You are continuing development of **OllamAGI** — a local-first autonomous agent platform built in Python/FastAPI. The working directory is `/home/yan/ollamagi`.

## What this is

OllamAGI takes a natural-language objective, decomposes it into tasks/subtasks via Ollama LLMs, executes code in Docker containers, auto-fixes failures, and stores results in Hermes (a 100-table SQLite cognitive memory at `~/.hermes/cognitive_memory.sqlite`).

**Read `CODEX.md` first** — it has the full architecture, all API endpoints, known issues, and session history.

## Stack

- Python 3.11, FastAPI, uvicorn, httpx, docker SDK
- Ollama at `http://localhost:11434` (local models, no cloud)
- Docker for agent container execution
- Single-file SPA at `api/static/index.html` (no build step)
- systemd service: `ollamagi.service` on port 7654
- SSH tunnel from Termux: `ssh -L 7654:localhost:7654 yan@100.75.11.55 -N`

## Current model config (`.env` / actual running values)

```
MODEL_ORCHESTRATOR = vaultbox/qwen3.5-uncensored:27b
MODEL_CODER        = qwen3-coder:30b
MODEL_FAST         = jaahas/qwen3.5-uncensored:2b
MODEL_EMBEDDINGS   = mxbai-embed-large:latest
OLLAMA_CTX         = 65536
```

## Service management

```bash
# Check if running
curl -s http://localhost:7654/api/status

# Restart (kills current server, starts new one)
kill $(pgrep -f "ollamagi.py serve") 2>/dev/null
PYTHONPATH=/home/yan/ollamagi python3 /home/yan/ollamagi/ollamagi.py serve --port 7654 &

# Health check
bash /home/yan/ollamagi/scripts/health_check.sh

# Verify imports after changes
python3 -c "import core.orchestrator; import core.model_router; import api.server; print('OK')"
```

## Key files to know

| File | Why you'll touch it |
|---|---|
| `core/config.py` | All env-var config — never hardcode paths elsewhere |
| `core/orchestrator.py` | Flow lifecycle, retry logic, stop/steer/replan |
| `core/model_router.py` | LLM calls, token counting, `interruptible_sleep`, stop events |
| `core/agents.py` | Agent role definitions — add new roles here |
| `executor/docker_manager.py` | Container creation, `exec_python()`, bootstrap |
| `api/server.py` | All REST endpoints + WebSocket |
| `api/static/index.html` | Complete dashboard SPA — JS, CSS, HTML in one file |

## Active known issues (check CODEX.md for full list)

1. **LLM sometimes generates agent names not in ALL_ROLES** (`planning_agent`, `research_agent`) — they silently fall back to `primary_agent`. To fix properly: either add these as aliases in `ALL_ROLES`, or add a validation step in `_generate_subtasks` that constrains the agent names the LLM can pick.

2. **Workspace files** — agent outputs land in `workspace/<flow-id>/`. The directory is git-ignored. Files from generated projects (Discord bots, scrapers, etc.) live here. They are NOT cleaned up between runs.

3. **Hermes memory** — runs on the host at `~/.hermes/cognitive_memory.sqlite`. The memory bridge silently degrades if the file doesn't exist. If Hermes shows "unreachable" in the dashboard, check the path in `.env`.

## GitHub repo

https://github.com/Linutesto/ollamagi  
Push changes with:
```bash
cd /home/yan/ollamagi
git add <files>
git commit -m "description"
git push
```

## Session context

The last session focused on:
- Fixing flows stuck in "running" state after server restart
- Fixing Ollama 500 errors leaving flows stuck (now marks as `failed`)
- Adding token usage display per flow (detail view + list cards)
- Token dashboard: session / all-time / per-flow + reset button
- Increasing Ollama timeout to 600s with 3-retry interruptible planning
- GitHub release: audit, sanitize, push to Linutesto/ollamagi
- Fixing debian containers missing python3 (bootstrap now apt-gets it)
- Fixing bash script prompt to not assume python3 is available

**Last thing done:** Created CODEX.md and this file, committed and pushed to GitHub.

## How to continue

1. Run `bash scripts/health_check.sh` to confirm system is healthy
2. Check `curl -s http://localhost:7654/api/flows` for any stuck flows
3. Read CODEX.md § "Known issues" for current bugs
4. The user accesses via mobile browser over SSH tunnel — always test UI changes with that in mind (small screen, touch, no hover)
