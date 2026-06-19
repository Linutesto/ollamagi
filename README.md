# OllamAGI

**Local-first autonomous agent platform powered by Ollama.**

OllamAGI turns any machine with a GPU into a self-contained AI agent that can plan, code, research, scrape, and deploy — all without sending your data to a cloud API. Give it an objective; it breaks the work into tasks, spins up Docker containers, executes code, retries failures automatically, and feeds everything it learns back into a persistent memory system.

---

## Why OllamAGI?

Most AI agent frameworks assume cloud LLMs. OllamAGI is built from the ground up for **local Ollama models**:

- Zero API costs — your GPU does the work
- No data leaves your machine
- Works offline
- Mobile-accessible dashboard over SSH tunnel
- Deep memory via [Hermes](docs/architecture.md#hermes-memory) (optional but powerful)

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
| **Hermes memory** | Completed tasks feed beliefs back into a 100-table SQLite brain |
| **Mobile dashboard** | Full-featured mobile web UI, works over SSH tunnel from Termux |
| **Live terminal** | Execute commands in agent containers from the dashboard |
| **Token tracking** | Per-flow and session-level token usage with reset button |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Web Dashboard (mobile-first SPA)                       │
│  WebSocket live updates · Stop · Steer · Terminal       │
├─────────────────────────────────────────────────────────┤
│  FastAPI Server  (port 7654)                            │
├─────────────────────────────────────────────────────────┤
│  Orchestrator                                           │
│  Flow → Tasks → Subtasks                                │
│  Model router · Auto-fix · Replan · Memory distill      │
├─────────────────────────────────────────────────────────┤
│  Agent Roles                                            │
│  primary_agent · architect · coder · researcher         │
│  installer · adviser · reflector · monetizer · pentester│
├─────────────────────────────────────────────────────────┤
│  Executor                                               │
│  Docker containers (python / debian / kali)             │
│  SSH host access · /work bind-mount                     │
├─────────────────────────────────────────────────────────┤
│  Memory (optional)                                      │
│  Hermes SQLite — beliefs, memories, goals, RAG          │
└─────────────────────────────────────────────────────────┘
         ↕  Ollama  (local, port 11434)
```

See [docs/architecture.md](docs/architecture.md) for full detail.

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
# Edit .env — at minimum set MODEL_ORCHESTRATOR to a model you have pulled
```

### 3. Pull an Ollama model

```bash
ollama pull qwen2.5:7b        # minimum viable (fast, lower quality)
ollama pull qwen2.5:32b       # recommended orchestrator
ollama pull qwen2.5-coder:32b # recommended for code tasks
```

### 4. Start

```bash
python3 ollamagi.py serve
# Dashboard: http://localhost:7654
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

OllamAGI is designed to be used from a phone. In Termux:

```bash
ssh -L 7654:localhost:7654 youruser@yourserver -N
```

Then open `http://localhost:7654` in your mobile browser.

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

## Hermes Memory (optional)

OllamAGI integrates with Hermes — a 100-table SQLite cognitive memory system. Every completed task extracts beliefs that inform future flows.

Hermes is **optional** — OllamAGI works without it. If `HERMES_DB` points to a non-existent file, memory features are silently disabled.

---

## CLI Usage

```bash
# Start the dashboard
python3 ollamagi.py serve [--port 7654]

# Run a flow from the command line
python3 ollamagi.py run "Build a REST API for todo management"

# Launch a bug bounty flow
python3 ollamagi.py bounty example.com --platform hackerone

# Query Hermes memory
python3 ollamagi.py memory "web scraping techniques"
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
| `MODEL_ORCHESTRATOR` | `qwen2.5:32b` | Planning/reasoning model |
| `MODEL_CODER` | `qwen2.5-coder:32b` | Code generation model |
| `MODEL_FAST` | `qwen2.5:7b` | Fast/cheap calls |
| `HERMES_DB` | `~/.hermes/cognitive_memory.sqlite` | Hermes memory path |
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
