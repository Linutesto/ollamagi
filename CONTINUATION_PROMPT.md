# OllamAGI — Continuation Prompt

Paste this at the start of a new Claude Code session to resume work instantly.

---

You are continuing development of **OllamAGI** — a local-first autonomous agent platform built in Python/FastAPI. The working directory is `/home/yan/ollamagi`.

## What this is

OllamAGI takes a natural-language objective, decomposes it into tasks/subtasks via Ollama LLMs, executes code in Docker containers, auto-fixes failures, and stores results in a SQLite cognitive memory at `~/.ollamagi/cognitive_memory.sqlite`.

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
MODEL_SINGLE     = vaultbox/qwen3.5-uncensored:27b
MODEL_EMBEDDINGS = mxbai-embed-large:latest
OLLAMA_CTX       = 65536
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

## Current reliability model

1. Generated role aliases are normalized and constrained to the selected workflow type.
2. Every subtask has an explicit deliverable contract (`source`, `documentation`, `dependency`, `configuration`, `report`, `dataset`, `test`, etc.) plus expected `/work` paths when known.
3. Container work is accepted only with observable evidence: valid changed artifacts or substantive test output.
4. Agent-development flows receive project-level syntax, completeness, offline-mode, and bounded runtime smoke validation.
5. Final validation failures receive up to two bounded repair/revalidation attempts against the actual workspace.
6. Workspace files live in `workspace/<flow-id>/` and are intentionally preserved between process restarts.
7. Cognitive memory silently degrades if `~/.ollamagi/cognitive_memory.sqlite` is unavailable.

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

The last session focused on workflow reliability:
- Replaced keyword-based artifact guessing with explicit per-subtask deliverable contracts
- Prevented README and requirements tasks from being falsely treated as source-code tasks
- Added exact expected-artifact path validation and safe path normalization
- Added contract-based task recovery after later validated subtasks
- Added deterministic final repair attempts for broken imports/interfaces/runtime smoke tests
- Added cross-workflow tests for agent, product, research, security, and general flows
- Directly writes single-file documentation/reports instead of generating fragile Python writer scripts
- Preflights generated Python build-script syntax and explicitly repairs nested triple-quote failures
- Treats exact artifact paths as authoritative even when their suffix is unconventional
- Uses final deterministic workspace validation as the flow outcome; failed attempts remain visible as history
- Requires persistent dependency manifests for Python projects with third-party imports
- Runs credentialed bot smoke tests through explicit offline modes with external networking blocked
- Rejects mixed Telegram SDKs and hardcoded token-like credentials
- Requires every agent to provide a network-isolated offline self-test
- Prevents memory context from adding unrequested Redis/databases/cloud services
- Writes configuration/dependency bundles directly without runtime writer dependencies
- Requires all paths in multi-file deliverable contracts
- Requires durable reports for research, product, and security workflows
- Separates final repairs from replans and marks replaced attempts as `superseded`
- Adds objective-specific validation for email, API, chat, data-pipeline, crawler, trading, and web-automation agents
- Excludes bytecode, caches, logs, and temporary files from artifact evidence
- Verifies requested logging and explicit error handling in generated source
- Confirmed the server is healthy and only Qwen 3.5 27B is loaded

**Last thing done:** Rebuilt email flow `74f8d3ea` as a real local `.eml` automation agent and completed a cross-workflow reliability audit.

## How to continue

1. Run `bash scripts/health_check.sh` to confirm system is healthy
2. Check `curl -s http://localhost:7654/api/flows` for any stuck flows
3. Read CODEX.md § "Known issues" for current bugs
4. The user accesses via mobile browser over SSH tunnel — always test UI changes with that in mind (small screen, touch, no hover)
