# OllamAGI

**Local-first autonomous agent platform powered by Ollama.**

OllamAGI turns any machine with a GPU into a self-contained AI agent that can plan, code, research, scrape, and deploy — all without sending your data to a cloud API. Give it an objective; it breaks the work into tasks, spins up Docker containers, executes code, retries failures automatically, and feeds everything it learns back into a persistent fractal memory system.

---

## Why OllamAGI?

Most AI agent frameworks assume cloud LLMs. OllamAGI is built from the ground up for **local Ollama models**:

- Zero API costs — your GPU does the work
- No data leaves your machine
- Works offline
- Mobile-accessible dashboard over Tailscale or SSH tunnel
- Self-organizing fractal memory that improves with every flow

---

## Core Features

| Feature | Description |
|---|---|
| **Flow engine** | Objective → Tasks → Subtasks → Docker execution |
| **Multi-agent** | 10 specialized roles: Architect, Coder, Researcher, Pentester… |
| **Auto-fix** | Failed code is automatically debugged and retried (up to 2×) |
| **Replanner** | 2 consecutive task failures trigger LLM replanning of remaining tasks |
| **Steer** | Inject a prompt mid-flow to redirect agents without stopping |
| **Instant stop** | Stop any flow within ~500ms — doesn't wait for the current LLM call |
| **Fractal memory** | Self-organizing SQLite memory with semantic search, beam retrieval, and live dashboard tab |
| **Mobile dashboard** | Full-featured mobile web UI, works over SSH tunnel from Termux |
| **Live terminal** | Execute commands in agent containers from the dashboard |
| **Token tracking** | Per-flow and session-level token usage with reset button |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Web Dashboard (mobile-first SPA)                       │
│  WebSocket live updates · Stop · Steer · Terminal       │
│  Memory tab: hierarchy · search · live recent feed      │
├─────────────────────────────────────────────────────────┤
│  FastAPI Server  (default port 8000)                    │
├─────────────────────────────────────────────────────────┤
│  Orchestrator                                           │
│  Flow → Tasks → Subtasks                                │
│  Model router · Auto-fix · Replan · Memory distill      │
│  (distills facts after every text subtask + task end)   │
├─────────────────────────────────────────────────────────┤
│  Agent Roles                                            │
│  primary_agent · architect · coder · researcher         │
│  installer · adviser · reflector · monetizer · pentester│
├─────────────────────────────────────────────────────────┤
│  Executor                                               │
│  Docker containers (python / debian / kali)             │
│  SSH host access · /work bind-mount                     │
├─────────────────────────────────────────────────────────┤
│  Fractal Memory  (core/fractal_memory.py)               │
│  L0 leaves → L1 concepts → L2 domains → L3/L4 meta     │
│  mxbai-embed-large · cosine clustering · FTS5 fallback  │
│  context_for_task() injects prior knowledge into prompts│
└─────────────────────────────────────────────────────────┘
         ↕  Ollama  (local, port 11434)
         ↕  SearxNG (local, port 4000)
```

See [docs/FRACTAL_MEMORY_WHITEPAPER.md](docs/FRACTAL_MEMORY_WHITEPAPER.md) for the full memory system deep-dive.

---

## Requirements

- Linux (tested on Fedora; should work on Ubuntu/Debian)
- Python 3.11+
- [Ollama](https://ollama.ai) with at least one pulled model
- Docker (for agent container execution)
- GPU recommended (CPU works but is slow)

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/linutesto/ollamagi.git
cd ollamagi
pip3 install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set MODEL_SINGLE to the model used by every agent role
```

### 3. Pull an Ollama model

```bash
ollama pull vaultbox/qwen3.5-uncensored:27b
```

### 4. Start

```bash
python3 ollamagi.py serve --port 8000
# Dashboard: http://localhost:8000
```

### 5. Health check

```bash
bash scripts/health_check.sh
```

---

## Fedora Setup (fresh machine)

```bash
bash scripts/setup_fedora.sh
```

This installs: Python, Docker, Ollama, creates the agent SSH key, and prints next steps.

---

## Mobile Access (Termux / Android)

OllamAGI is designed to be used from a phone.

**Via Tailscale** (recommended — no tunnel needed):
```bash
# Just open the Tailscale IP in your browser, e.g.:
http://100.x.x.x:8000
```

**Via SSH tunnel** from Termux:
```bash
ssh -L 8000:localhost:8000 youruser@yourserver -N
# Then open http://localhost:8000
```

See [docs/mobile-first.md](docs/mobile-first.md) for the full workflow.

---

## Agent SSH Access (optional)

Agents running in Docker containers can SSH back to the host to run commands with full access. This requires a dedicated key:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/ollamagi_agent -N ""
cat ~/.ssh/ollamagi_agent.pub >> ~/.ssh/authorized_keys
```

Then set in `.env`:
```
SSH_KEY=~/.ssh/ollamagi_agent
SSH_USER=youruser
```

---

## Fractal Memory

OllamAGI ships with a built-in self-organizing memory system (`core/fractal_memory.py`). Every completed task — and every text subtask — distills 1-3 reusable facts via LLM and stores them as leaf nodes in a fractal SQLite tree:

```
L0  raw memories  (facts, code snippets, observations)
L1  concept clusters
L2  domain clusters
L3  meta clusters
L4  root
```

Nodes are placed by cosine similarity (`mxbai-embed-large`, JOIN_THRESH=0.52). Overgrown clusters split by k-means. Queries use direct leaf scan for collections under 2 000 entries (~83% P@1), falling back to beam search for larger DBs.

The **Memory tab** on the dashboard shows the live hierarchy, recent memories with lineage breadcrumbs, and a semantic search box.

```python
from core.fractal_memory import context_for_task, insert

# Inject relevant prior knowledge into any agent prompt
ctx = context_for_task("build a FastAPI microservice with auth")

# Store a fact manually
insert("argon2-cffi is the recommended password hasher for Python", tags=["security"])
```

See [docs/FRACTAL_MEMORY_WHITEPAPER.md](docs/FRACTAL_MEMORY_WHITEPAPER.md) for benchmarks and design rationale.

---

## CLI Usage

```bash
# Start the dashboard
python3 ollamagi.py serve [--port 8000]

# Run a flow from the command line
python3 ollamagi.py run "Build a REST API for todo management"

# Launch a bug bounty flow
python3 ollamagi.py bounty example.com --platform hackerone
```

---

## Flow Types

| Type | Agents Used |
|---|---|
| `agent_development` | architect, coder, installer, refiner |
| `product_development` | researcher, adviser, monetizer, coder |
| `research` | researcher, refiner, adviser |
| `security` | pentester, researcher, reflector |
| `general` | generator, coder, refiner |

OllamAGI auto-detects the type from your objective, or you can set it via the Run tab.

---

## Configuration Reference

All config is via environment variables. See [`.env.example`](.env.example) for the full list.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_CTX` | `32768` | Context window tokens |
| `MODEL_SINGLE` | `vaultbox/qwen3.5-uncensored:27b` | Model used for all agent roles |
| `SSH_KEY` | `~/.ssh/ollamagi_agent` | Agent SSH key |
| `SSH_HOST` | `172.17.0.1` | Host IP from inside containers |
| `SSH_USER` | `$USER` | SSH username |

---

## Troubleshooting

**Ollama shows unreachable in dashboard**
OllamAGI checks Ollama server-side (not from your browser), so the SSH tunnel doesn't need to expose port 11434. If it still fails, confirm `ollama serve` is running.

**Flow stuck / no LLM calls**
Flows have a 10-minute Ollama timeout. If the model is very slow, increase `MAX_TASK_TIMEOUT` in `.env`. Check `journalctl -u ollama` for errors.

**Container exec fails**
Make sure Docker is running and your user is in the `docker` group: `sudo usermod -aG docker $USER` (then log out/in).

**Packages fail to install in containers (Rust errors)**
This happens when a package like pydantic v2 tries to compile Rust code without a toolchain. OllamAGI instructs agents to use `--prefer-binary` to avoid this, but some packages have no binary wheel for newer Python. Pin to older versions or use alternatives.

---

## Screenshots

> Add screenshots here after first public run.

---

## Roadmap

See [docs/roadmap.md](docs/roadmap.md).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [SECURITY.md](SECURITY.md).

## License

MIT — see [LICENSE](LICENSE).
