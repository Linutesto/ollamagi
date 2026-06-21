"""OllamAGI orchestrator — flow lifecycle with stop/steer/autofix/replan/memory agents."""
import uuid
import time
import json
import re
import ast
import hashlib
import fnmatch
import subprocess
import threading
import tomllib
from dataclasses import dataclass, field
from typing import Callable
from pathlib import Path

from core.config import WORKSPACE_DIR, MAX_TASK_TIMEOUT
from core.agents import run_agent, FLOW_TYPE_ROLES, normalize_role_name
from core.fractal_memory import context_for_task, store_from_result as store_belief
from core.model_router import chat, get_tokens, cancel_flow, register_stop_event, register_llm_callback, FlowStoppedException, interruptible_sleep
from executor.docker_manager import create_container, exec_script, exec_python, stop_container, sync_workspace

MAX_RETRIES = 2  # auto-fix attempts per subtask before giving up
MAX_AUTO_REPLANS = 2


@dataclass
class Subtask:
    id: str
    task_id: str
    title: str
    description: str
    agent: str
    status: str = "created"   # created|running|finished|failed|retrying|superseded
    result: str = ""
    output: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    container_type: str = "python"
    needs_container: bool = False
    attempts: int = 0
    artifacts: list[str] = field(default_factory=list)
    validation: str = ""
    deliverable_kind: str = "auto"
    expected_artifacts: list[str] = field(default_factory=list)


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
    memory_items_stored: int = 0
    replan_count: int = 0
    repair_count: int = 0
    error: str = ""
    validation: str = ""
    compact_context: bool = False


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
    entry = {"flow_id": flow_id, "msg": msg, "level": level, "ts": time.time()}
    for cb in _log_callbacks.get(flow_id, []):
        try:
            cb(entry)
        except Exception:
            pass
    try:
        log_path = WORKSPACE_DIR / flow_id / "flow_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _read_jsonl(path: Path) -> list[dict]:
    entries = []
    if not path.exists():
        return entries
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return entries


def get_flow_transcript(flow_id: str) -> dict | None:
    """Return full untruncated flow data + unified event timeline for the transcript viewer."""
    flow = _flows.get(flow_id)
    work_dir = WORKSPACE_DIR / flow_id

    # Merge log lines + LLM call records into a single timeline sorted by ts
    log_entries = _read_jsonl(work_dir / "flow_log.jsonl")
    llm_entries = _read_jsonl(work_dir / "llm_calls.jsonl")
    events = sorted(log_entries + llm_entries, key=lambda e: e.get("ts", 0))

    if flow is None:
        # Flow not in memory (server restarted) — return events only
        return {"flow_id": flow_id, "events": events, "tasks": []}

    def _full_subtask(s: Subtask) -> dict:
        return {
            "id": s.id, "title": s.title, "description": s.description,
            "agent": s.agent, "status": s.status,
            "result": s.result, "output": s.output,
            "attempts": s.attempts, "artifacts": s.artifacts,
            "validation": s.validation, "deliverable_kind": s.deliverable_kind,
            "started_at": s.started_at, "finished_at": s.finished_at,
        }

    def _full_task(t: Task) -> dict:
        return {
            "id": t.id, "title": t.title, "description": t.description,
            "agent": t.agent, "status": t.status,
            "result": t.result,
            "started_at": t.started_at, "finished_at": t.finished_at,
            "subtasks": [_full_subtask(s) for s in t.subtasks],
        }

    return {
        "flow_id": flow_id,
        "title": flow.title,
        "objective": flow.objective,
        "status": flow.status,
        "events": events,
        "tasks": [_full_task(t) for t in flow.tasks],
    }

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
        "memory_items_stored": flow.memory_items_stored,
        "replan_count": flow.replan_count,
        "repair_count": flow.repair_count,
        "error": flow.error,
        "validation": flow.validation,
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
                        "description": s.description,
                        "agent": s.agent,
                        "status": s.status,
                        "result": s.result[:300] if s.result else "",
                        "attempts": s.attempts,
                        "artifacts": s.artifacts,
                        "validation": s.validation,
                        "deliverable_kind": s.deliverable_kind,
                        "expected_artifacts": s.expected_artifacts,
                        "needs_container": s.needs_container,
                        "container_type": s.container_type,
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
    # Security first — unambiguous intent
    if any(w in obj for w in ["pentest", "hack", "security audit", "vuln", "exploit", "bug bounty", "ctf"]):
        return "security"
    # Research first — objectives that START with a research verb are always research flows,
    # even if they mention agents, SaaS, or products as the topic being researched
    _RESEARCH_VERBS = ("research ", "analyze ", "analyse ", "study ", "explore ",
                       "survey ", "compare ", "investigate ", "audit ", "identify ")
    if any(obj.startswith(v) for v in _RESEARCH_VERBS):
        return "research"
    # Agent/AI development — must be about BUILDING an agent, not USING one
    _AGENT_BUILD = ["build an agent", "build a bot", "create an agent", "make an agent",
                    "autonomous agent", "llm agent", "ai agent system", "build a chatbot"]
    if any(w in obj for w in _AGENT_BUILD):
        return "agent_development"
    # Scraping/automation — check before product to avoid "scrape product prices" → product
    if any(w in obj for w in ["scrape", "crawl", "automate", "automation", "scheduler", "workflow"]):
        return "automation"
    # Data engineering — check before general/research
    if any(w in obj for w in ["pipeline", "etl", "dataset", "data engineering", "analytics", "transform data"]):
        return "data_engineering"
    # DevOps — unambiguous infra keywords
    if any(w in obj for w in ["deploy", "kubernetes", "ci/cd", "devops", "infrastructure as code", "nginx config", "systemd unit"]):
        return "devops"
    # Content/writing
    if any(w in obj for w in ["write a blog", "write an article", "write a report", "write documentation", "write a whitepaper", "write a readme"]):
        return "content"
    # Research — broader check (not just at start)
    if any(w in obj for w in ["research ", "analyze ", "study ", "explore ", "survey ", "compare "]):
        return "research"
    # Product/business — broader keywords last in high-intent group
    if any(w in obj for w in ["product", "saas", "revenue", "monetize", "business", "roi", "sell", "startup"]):
        return "product_development"
    # Broader agent/tool/bot pattern — only when clearly building something
    if any(w in obj for w in ["build a", "create a", "make a"]) and any(
            w in obj for w in ["agent", "tool", "bot", "chatbot", "llm", "ai system"]):
        return "agent_development"
    # Broader infra patterns
    if any(w in obj for w in ["docker", "monitoring", "systemd", "nginx"]):
        return "devops"
    # Broader content
    if any(w in obj for w in ["write", "article", "blog", "documentation", "readme", "report", "whitepaper", "content"]):
        return "content"
    return "general"

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return text.strip()


def _object_list(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _role_for_flow(role_name: str, flow_type: str) -> str:
    allowed = FLOW_TYPE_ROLES.get(flow_type, FLOW_TYPE_ROLES["general"])
    return normalize_role_name(role_name, allowed)


def _objective_constraints(objective: str, flow_type: str) -> str:
    """Deterministic guardrails that keep generated workflows locally testable."""
    import datetime as _dt
    lower = (objective or "").lower()
    from core.config import SSH_HOST, SSH_USER
    _now = _dt.datetime.now()
    _today = _now.strftime("%Y-%m-%d")
    _ts    = _now.strftime("%Y-%m-%d %H:%M")
    rules = [
        f"CURRENT DATE/TIME: {_ts} (local). Always use this exact date in reports, logs, and "
        f"filenames — never use a hardcoded or guessed date like 2023-10-27.",
        "Do not introduce Redis, PostgreSQL, Kafka, cloud services, or browser daemons unless the "
        "user explicitly requested them.",
        "Optional integrations must use an adapter/interface with an in-memory or local-file fallback.",
        "All validation must be deterministic and offline. Never require credentials, a running daemon, "
        "a public website, or a provider API. Exception: SSH to the pre-authorized host "
        f"({SSH_HOST}, user {SSH_USER}) is always allowed — the key is at /root/.ssh/id_ed25519.",
        "Every third-party runtime import must be declared in requirements.txt or pyproject.toml.",
        "git is NOT pre-installed in Python containers. Install it safely (apt-get update first, no check=True):\n"
        "  subprocess.run(['apt-get','update','-qq'], capture_output=True)\n"
        "  subprocess.run(['apt-get','install','-y','-qq','git'], capture_output=True)\n"
        "Alternative (no apt needed): download a GitHub ZIP with requests.get('https://codeload.github.com/{user}/{repo}/zip/refs/heads/main').\n"
        "GitHub repo names are case-sensitive. Never use check=True for apt-get commands.",
    ]
    # Self-referential objectives about OllamAGI's own logs/flows
    _SELF_LOG_KEYWORDS = ("agent log", "agent execution log", "flow log", "execution log",
                          "failure pattern", "ollamagi log", "recurring failure", "task failure",
                          "last 7 days", "last seven days", "analyze.*log", "analyse.*log",
                          "identify.*failure", "identify.*pattern")
    if any(kw in lower for kw in _SELF_LOG_KEYWORDS):
        rules.append(
            "OLLAMAGI DATA ACCESS: This objective requires reading OllamAGI's own execution history.\n"
            f"The OllamAGI workspace is at: {WORKSPACE_DIR}  (accessible in-container via the host home mount)\n"
            "Data structures:\n"
            f"  - Per-flow log:      {WORKSPACE_DIR}/<flow-id>/flow_log.jsonl\n"
            f"    Fields: {{flow_id, msg, level, ts}}  where ts is a Unix timestamp (float)\n"
            f"  - Per-flow metadata: {WORKSPACE_DIR}/<flow-id>/flow.json\n"
            f"    Fields: id, objective, status, created_at (Unix ts), flow_type, tasks[]\n"
            "To collect logs from the last 7 days:\n"
            "  import json, pathlib, time\n"
            f"  ws = pathlib.Path('{WORKSPACE_DIR}')\n"
            "  cutoff = time.time() - 7 * 86400\n"
            "  rows = []\n"
            "  for log_file in ws.glob('*/flow_log.jsonl'):\n"
            "      for line in log_file.read_text(errors='replace').splitlines():\n"
            "          try:\n"
            "              e = json.loads(line)\n"
            "              if e.get('ts', 0) >= cutoff: rows.append(e)\n"
            "          except Exception: pass\n"
            "  pathlib.Path('/work/raw_logs.jsonl').write_text('\\n'.join(json.dumps(r) for r in rows))\n"
            "Also read flow.json for task-level pass/fail: flow['tasks'][i]['status'] in 'success'|'failed'|'running'."
        )
    # System audit / host inspection objectives: inject SSH data-collection rule
    _AUDIT_KEYWORDS = ("audit", "health report", "workstation", "system check", "inspect.*system",
                       "performance issue", "storage issue", "service issue", "security issue")
    if any(kw in lower for kw in _AUDIT_KEYWORDS):
        rules.append(
            f"HOST DATA COLLECTION: This task audits the local host ({SSH_HOST}). "
            "The script runs inside a Debian container — do NOT run df/free/systemctl/journalctl directly "
            "(those show container data, not host data). "
            f"MUST collect all host metrics via SSH. Use paramiko (always pre-installed):\n"
            f"  import paramiko, json, pathlib\n"
            f"  client = paramiko.SSHClient()\n"
            f"  client.set_missing_host_key_policy(paramiko.AutoAddPolicy())\n"
            f"  client.connect('{SSH_HOST}', username='{SSH_USER}', key_filename='/root/.ssh/id_ed25519', timeout=10)\n"
            f"  def ssh(cmd):\n"
            f"      _, out, err = client.exec_command(cmd, timeout=30)\n"
            f"      return out.read().decode() + err.read().decode()\n"
            f"  memory   = ssh('free -h')\n"
            f"  disk     = ssh('df -h')\n"
            f"  services = ssh('systemctl --failed --no-pager 2>&1 || true')\n"
            f"  journal  = ssh('journalctl -p err -n 50 --no-pager 2>&1 || true')\n"
            f"  security = ssh('last -n 20 2>&1; ss -tlnp 2>&1')\n"
            f"  client.close()\n"
            "Save all results to /work/ as .json. SSH key is pre-configured — no password needed."
        )
    if flow_type == "agent_development":
        rules.append(
            "The final application must provide --self-test or --dry-run that exercises its core "
            "behavior with fixtures/fakes and exits successfully without external network access."
        )
    if flow_type == "research":
        rules.append(
            "Research must produce a durable cited report or dataset saved to /work/. "
            "Individual unavailable sources must be skipped and recorded rather than failing the workflow."
        )
    if flow_type == "product_development":
        rules.append(
            "Product work must end in a durable decision document saved to /work/ with assumptions, "
            "risks, validation steps, and a concrete next action."
        )
    if flow_type == "security":
        from core.config import SSH_HOST, SSH_USER
        rules.append(
            f"TARGET HOST: {SSH_HOST} (Fedora host, authorized pentest scope). "
            f"SSH user: {SSH_USER}. SSH key at /root/.ssh/id_ed25519 is pre-configured.\n"
            "Security flows MUST follow this active pentest workflow — do NOT generate schema, "
            "documentation, or Python validation tasks:\n"
            "  1. Reconnaissance — nmap port+OS scan → reports/nmap.txt, reports/nmap.xml\n"
            "  2. Enumeration — nikto + gobuster → reports/nikto.txt, reports/gobuster.txt\n"
            "  3. Vulnerability assessment — nuclei + searchsploit → reports/nuclei.txt, reports/searchsploit.txt\n"
            "  4. Exploitation — within authorized scope → reports/exploits.txt\n"
            "  5. Report — executive summary with CVSS scores → reports/pentest_report.md\n"
            "All tasks: agent='pentester', needs_container=true, container_type='pentest'.\n"
            "ARTIFACT RULES: scan output must end in .txt or .xml — NEVER .json for raw tool output.\n"
            "Scan tools (nmap/nikto/nuclei/gobuster/searchsploit) do NOT produce JSON natively.\n"
            "Tool errors and empty findings are valid outcomes — never skip the final report."
        )
    if flow_type == "data_engineering":
        rules.append(
            "Data pipelines must process data in chunks/streams for large inputs. "
            "Always validate input schema before transforming. Produce a data quality summary "
            "(row counts, null rates, sample rows) alongside the output artifact."
        )
    if flow_type == "devops":
        rules.append(
            "All infrastructure configs must be idempotent. Secrets via environment variables only. "
            "Every service must have a health check. Validate configs syntactically (nginx -t, "
            "docker compose config --quiet) before declaring success."
        )
    if flow_type == "automation":
        rules.append(
            "Automation scripts must terminate after one bounded cycle during validation. "
            "Never start an infinite polling loop or daemon during build/test. "
            "Validate against local fixtures or public unauthenticated endpoints."
        )
    if flow_type == "content":
        rules.append(
            "Content must be saved as a file to /work/ (Markdown preferred). "
            "Cite sources inline. Completeness over brevity — no placeholder sections."
        )
    if any(term in lower for term in ("git clone", "github", "gitlab", "clone", "repository", "repo")):
        rules.append(
            "Install git before use: subprocess.run(['apt-get','install','-y','-qq','git'], check=True, capture_output=True). "
            "Always check that the cloned directory exists after git clone — 'fatal:' in output means the clone failed. "
            "If the clone target directory already exists, skip cloning (idempotent). "
            "Prefer HTTPS clones for public repos. For the OllamAGI repo: "
            "https://github.com/Linutesto/ollamagi (lowercase — GitHub is case-sensitive)."
        )
    if any(term in lower for term in ("email", "smtp", "imap")):
        rules.append(
            "Email automation must test parsing, rules, and send/fetch adapters using local .eml fixtures "
            "and fake SMTP/IMAP transports. Redis may be optional but must never be required for self-test."
        )
    if any(term in lower for term in ("telegram", "discord", "slack")):
        rules.append(
            "Chat bots must test handlers with fake events/messages and must not start polling or validate "
            "tokens during self-test."
        )
    if any(term in lower for term in ("api integration", "webhook")):
        rules.append(
            "API integrations must validate with a fake transport or localhost fixture server, including "
            "success, timeout, retry, and malformed-response cases."
        )
    if any(term in lower for term in ("data pipeline", "etl")):
        rules.append(
            "Data pipelines must include local input fixtures and deterministically validate transformed output."
        )
    if any(term in lower for term in ("web automation", "browser automation")):
        rules.append(
            "Web automation must validate against local HTML/localhost fixtures and not require a live site."
        )
    if any(term in lower for term in ("scraper", "crawler", "scrape", "crawl")):
        rules.append(
            "Crawlers must validate with local HTML fixtures or localhost and handle expected failures offline."
        )
    if any(term in lower for term in ("trading", "crypto", "exchange")):
        rules.append(
            "Trading systems must default to paper mode with deterministic market fixtures and no private endpoints."
        )
    return "\n".join(f"- {rule}" for rule in rules)


_FATAL_OUTPUT_PATTERNS = (
    r"(?im)^traceback \(most recent call last\):",
    r"(?im)command not found",
    r"(?im)no such file or directory",
    r"(?im)file not (?:created|found)",
    r"(?im)^fatal(?:\s*:|\s+-)",
    r"(?im)^unhandled (?:error|exception)",
    r"(?im)^error:\s*(?:required|failed|cannot|unable|missing)",
)


def _is_transient_path(path: Path) -> bool:
    return (
        "__pycache__" in path.parts
        or path.suffix.lower() in {".pyc", ".pyo", ".log", ".tmp"}
        or path.name in {".DS_Store"}
    )


def _execution_failed(exit_code: int, output: str) -> bool:
    """Classify process failure without treating handled ERROR log records as fatal."""
    if exit_code != 0:
        return True
    return any(re.search(pattern, output or "") for pattern in _FATAL_OUTPUT_PATTERNS)


def _detect_referenced_flow_ids(objective: str) -> list[str]:
    """Find 8-char flow ID prefixes in objective text that match existing workspaces."""
    if not WORKSPACE_DIR.exists():
        return []
    existing = {d.name for d in WORKSPACE_DIR.iterdir() if d.is_dir()}
    found, seen = [], set()
    for candidate in re.findall(r'\b([0-9a-f]{8,32})\b', objective.lower()):
        for name in existing:
            if name.startswith(candidate) and name not in seen:
                found.append(name)
                seen.add(name)
    return found


def _build_cross_flow_context(ref_flow_ids: list[str]) -> str:
    """Build a context block describing referenced flows and their artifacts."""
    if not ref_flow_ids:
        return ""
    KEY_NAMES = {"main.py", "app.py", "index.py", "server.py", "pipeline.py",
                 "readme.md", "report.md", "analysis.md", "output.md"}
    sections = [
        "## Referenced Projects\n"
        "*Files from these projects are available in the new flow's workspace at "
        "/work/_context/{flow_id}/ — read them before writing code that builds on prior work.*\n"
    ]
    skip = {"flow.json", "flow_log.jsonl", "llm_calls.jsonl"}
    for flow_id in ref_flow_ids:
        work_dir = WORKSPACE_DIR / flow_id
        if not work_dir.exists():
            continue
        meta = {}
        meta_file = work_dir / "flow.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
            except Exception:
                pass
        title = (meta.get("title") or meta.get("objective") or flow_id)[:70]
        status = meta.get("status", "?")
        ftype = meta.get("flow_type", "?")
        files = [
            (str(f.relative_to(work_dir)), f.stat().st_size, f)
            for f in sorted(work_dir.rglob("*"))
            if f.is_file() and f.name not in skip
        ]
        sections.append(f"### [{flow_id}] {title}")
        sections.append(f"Status: {status} | Type: {ftype}")
        if files:
            for rel, sz, _ in files[:15]:
                sections.append(f"  /work/_context/{flow_id}/{rel}  ({sz:,} B)")
            if len(files) > 15:
                sections.append(f"  … and {len(files)-15} more files")
        # Inline preview of first key file found
        for rel, sz, fpath in files:
            if fpath.name.lower() in KEY_NAMES and sz < 6000:
                try:
                    content = fpath.read_text(errors="replace")[:2500]
                    lang = "python" if fpath.suffix == ".py" else ""
                    sections.append(f"\n```{lang}\n# {rel}\n{content}\n```")
                    break
                except Exception:
                    pass
        sections.append("")
    return "\n".join(sections) if len(sections) > 1 else ""


def _copy_referenced_workspaces(flow_id: str, ref_flow_ids: list[str]):
    """Copy referenced flow workspaces into _context/ so agents can read them."""
    import shutil
    skip = {"flow.json", "flow_log.jsonl", "llm_calls.jsonl"}
    context_root = WORKSPACE_DIR / flow_id / "_context"
    context_root.mkdir(parents=True, exist_ok=True)
    for ref_id in ref_flow_ids:
        src = WORKSPACE_DIR / ref_id
        if not src.exists():
            continue
        dst = context_root / ref_id
        dst.mkdir(exist_ok=True)
        for src_file in src.rglob("*"):
            if src_file.is_file() and src_file.name not in skip:
                rel = src_file.relative_to(src)
                dst_file = dst / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(src_file, dst_file)
                except Exception:
                    pass


def _workspace_inventory(flow_id: str, limit: int = 80) -> str:
    paths = []
    for path in sync_workspace(flow_id):
        if path.is_file() and path.name != "flow.json" and not _is_transient_path(path):
            try:
                paths.append(str(path.relative_to(WORKSPACE_DIR / flow_id)))
            except ValueError:
                continue
    if not paths:
        return "WORKSPACE FILES: none yet"
    return "WORKSPACE FILES:\n" + "\n".join(f"- /work/{p}" for p in sorted(paths)[:limit])


def _workspace_snapshot(flow_id: str) -> dict[str, tuple[int, str]]:
    """Capture content identity for workspace files, excluding orchestrator state."""
    root = WORKSPACE_DIR / flow_id
    snapshot = {}
    if not root.exists():
        return snapshot
    for path in root.rglob("*"):
        if (
            not path.is_file()
            or path.name == "flow.json"
            or _is_transient_path(path.relative_to(root))
        ):
            continue
        try:
            data = path.read_bytes()
            rel = str(path.relative_to(root))
            snapshot[rel] = (len(data), hashlib.sha256(data).hexdigest())
        except (OSError, ValueError):
            continue
    return snapshot


_PROOF_ONLY_PATTERN = re.compile(
    r"\b(test|validate|verify|check|inspect|review|install|probe)\b",
    re.IGNORECASE,
)
_SOURCE_ACTION_PATTERN = re.compile(
    r"\b(build|create|develop|generate|implement|write|scaffold|refactor|add|integrate)\b",
    re.IGNORECASE,
)
_SOURCE_NOUN_PATTERN = re.compile(
    r"\b(code|codebase|script|module|application|app|bot|agent|api|service|cli|logic)\b",
    re.IGNORECASE,
)
_DOC_PATTERN = re.compile(r"\b(readme|documentation|docs?)\b", re.IGNORECASE)
_CONFIG_PATTERN = re.compile(r"\b(config|configuration|settings)\b", re.IGNORECASE)
_REPORT_PATTERN = re.compile(
    r"\b(report|results?|findings|analysis|summary|dataset|export)\b",
    re.IGNORECASE,
)
_SOURCE_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".sh"}
_CONFIG_SUFFIXES = {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env"}
_REPORT_SUFFIXES = {".md", ".json", ".txt", ".csv", ".html", ".xml"}
_DATASET_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".xml", ".html", ".sqlite", ".db", ".parquet"}
_DEPENDENCY_NAMES = {
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "package-lock.json", "go.mod", "cargo.toml",
}
_DELIVERABLE_KINDS = {
    "auto", "text", "source", "documentation", "configuration",
    "dependency", "report", "dataset", "test", "artifact", "none",
    "scan",  # pentest scan output — file must exist but empty = no findings (valid)
}
_PENTEST_SCAN_TOOLS_RE = re.compile(
    r'\b(nmap|nikto|gobuster|ffuf|nuclei|searchsploit|masscan|enum4linux|'
    r'whatweb|smbmap|hydra|sqlmap|wpscan|dirb|dirbuster|feroxbuster|nessus|'
    r'openvas|metasploit|msfconsole|crackmapexec|rpcclient|smbclient)\b',
    re.IGNORECASE,
)
_PENTEST_REPORT_KEYWORDS_RE = re.compile(
    r'\b(report|summary|executive|compile|assign|cvss|remediation|assessment)\b',
    re.IGNORECASE,
)

def _normalize_scan_artifacts(artifacts: list[str]) -> list[str]:
    """Rewrite .json extensions to .txt for scan tool output — scan tools don't output JSON."""
    result = []
    for a in artifacts:
        p = Path(a)
        if p.suffix.lower() == '.json':
            result.append(str(p.with_suffix('.txt')))
        else:
            result.append(a)
    return result

_TOOL_REQUIRED_PATTERN = re.compile(
    r"\b(inspect|read|open|analy[sz]e|review|check|verify|list|search)\b.*"
    r"\b(existing|workspace|file|code|codebase|script|artifact|directory|log)\b",
    re.IGNORECASE,
)


def _normalize_expected_artifacts(value) -> list[str]:
    """Keep only safe /work-relative paths or glob patterns."""
    if not isinstance(value, list):
        return []
    normalized = []
    for raw in value:
        if not isinstance(raw, str):
            continue
        path = raw.strip().replace("\\", "/")
        if path.startswith("/work/"):
            path = path[6:]
        elif path.startswith("/"):
            continue
        path = path.lstrip("/")
        if not path or path == "flow.json":
            continue
        if any(part in {"", ".", ".."} for part in path.split("/")):
            continue
        normalized.append(path)
    return sorted(set(normalized))


def _infer_expected_artifacts(text: str) -> list[str]:
    """Extract explicit /work paths and common deliverable filenames."""
    candidates = re.findall(
        r"(?:/work/)?(?:[\w.-]+/)*[\w.-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|sh|"
        r"md|txt|json|jsonl|ya?ml|toml|ini|cfg|conf|csv|tsv|xml|html|sqlite|db|parquet)",
        text or "",
        flags=re.IGNORECASE,
    )
    return _normalize_expected_artifacts(candidates)


def _infer_deliverable_kind(title: str, description: str, needs_container: bool = False) -> str:
    """Conservative fallback for old/preplanned tasks that lack a contract."""
    text = f"{title}\n{description}"
    writes_artifact = bool(re.search(
        r"\b(build|create|draft|generate|implement|produce|save|scaffold|update|write|export)\b",
        text,
        re.I,
    ))
    if _PROOF_ONLY_PATTERN.search(text) and not writes_artifact:
        return "test"
    if _DOC_PATTERN.search(text) and writes_artifact:
        return "documentation"
    if re.search(r"\b(requirements(?:\.txt)?|pyproject|package manifest|dependency manifest)\b", text, re.I):
        return "dependency"
    if _CONFIG_PATTERN.search(text) and writes_artifact:
        return "configuration"
    if writes_artifact and re.search(
        r"\b(dataset|crawl(?:ed)? data|scrap(?:ed|ing) data|csv|jsonl|sqlite|parquet)\b",
        text,
        re.I,
    ):
        return "dataset"
    if _REPORT_PATTERN.search(text) and writes_artifact:
        return "report"
    if _SOURCE_ACTION_PATTERN.search(text) and _SOURCE_NOUN_PATTERN.search(text):
        return "source"
    if needs_container and re.search(r"\b(create|write|save|export|produce|download)\b", text, re.I):
        return "artifact"
    return "text" if not needs_container else "test"


def _subtask_contract(subtask: Subtask) -> tuple[str, list[str]]:
    kind = (subtask.deliverable_kind or "auto").strip().lower()
    if kind not in _DELIVERABLE_KINDS or kind == "auto":
        kind = _infer_deliverable_kind(
            subtask.title, subtask.description, subtask.needs_container
        )
    expected = _normalize_expected_artifacts(subtask.expected_artifacts)
    if not expected:
        expected = _infer_expected_artifacts(f"{subtask.title}\n{subtask.description}")
    return kind, expected


def _matching_paths(paths: list[Path], patterns: list[str]) -> list[Path]:
    if not patterns:
        return []
    return [
        path for path in paths
        if any(fnmatch.fnmatch(path.as_posix(), pattern) for pattern in patterns)
    ]


def _missing_patterns(paths: list[Path], patterns: list[str]) -> list[str]:
    return [
        pattern for pattern in patterns
        if not any(fnmatch.fnmatch(path.as_posix(), pattern) for path in paths)
    ]


def _kind_matches(kind: str, path: Path) -> bool:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if kind == "source":
        return suffix in _SOURCE_SUFFIXES
    if kind == "documentation":
        return name.startswith("readme") or suffix in {".md", ".rst"}
    if kind == "configuration":
        return suffix in _CONFIG_SUFFIXES or name == ".env"
    if kind == "dependency":
        return name in _DEPENDENCY_NAMES
    if kind == "report":
        return suffix in _REPORT_SUFFIXES
    if kind == "dataset":
        return suffix in _DATASET_SUFFIXES
    return kind == "artifact"


def _validate_artifact(path: Path) -> str | None:
    """Return an error for malformed artifacts, otherwise None."""
    try:
        size = path.stat().st_size
        if size == 0 and path.name != "__init__.py":
            return "file is empty"
        suffix = path.suffix.lower()
        if suffix == ".py":
            ast.parse(path.read_text(errors="replace"), filename=str(path))
        elif suffix == ".json":
            json.loads(path.read_text(errors="replace"))
        elif suffix == ".toml":
            tomllib.loads(path.read_text(errors="replace"))
        elif suffix in (".sh", ".bash"):
            result = subprocess.run(
                ["bash", "-n", str(path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return result.stderr.strip() or "invalid shell syntax"
    except (
        OSError, SyntaxError, json.JSONDecodeError,
        tomllib.TOMLDecodeError, subprocess.SubprocessError,
    ) as exc:
        return str(exc)
    return None


def _external_python_imports(python_files: list[Path], root: Path) -> set[str]:
    """Return imported top-level modules not provided by stdlib or this project."""
    local_modules = {
        path.stem for path in python_files
    } | {
        path.name for path in root.iterdir() if path.is_dir()
    }
    external = set()
    stdlib = getattr(__import__("sys"), "stdlib_module_names", set())
    for path in python_files:
        try:
            tree = ast.parse(path.read_text(errors="replace"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".", 1)[0]]
            for name in names:
                if name not in stdlib and name not in local_modules:
                    external.add(name)
    return external


_IMPORT_DISTRIBUTIONS = {
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "discord": "discord.py",
    "dotenv": "python-dotenv",
    "telegram": "python-telegram-bot",
    "telebot": "pytelegrambotapi",
    "yaml": "pyyaml",
}


def _undeclared_imports(
    imports: set[str], manifests: list[Path]
) -> set[str]:
    manifest_text = "\n".join(
        path.read_text(errors="replace").lower()
        for path in manifests
        if path.stat().st_size < 1_000_000
    )
    compact = re.sub(r"[-_.]+", "", manifest_text)
    undeclared = set()
    for module in imports:
        distribution = _IMPORT_DISTRIBUTIONS.get(module, module).lower()
        if re.sub(r"[-_.]+", "", distribution) not in compact:
            undeclared.add(module)
    return undeclared


def _agent_capability_errors(objective: str, source_text: str) -> list[str]:
    lower = objective.lower()
    errors = []
    if not any(
        marker in source_text
        for marker in ("--self-test", "--self_test", "--dry-run", "--dry_run")
    ):
        errors.append("agent project has no explicit offline --self-test/--dry-run mode")
    if "logging" in lower and "logging" not in source_text:
        errors.append("agent project does not implement logging")
    if "error handling" in lower and not any(
        marker in source_text for marker in ("try:", "except ", "exception")
    ):
        errors.append("agent project does not implement explicit error handling")
    if any(term in lower for term in ("email", "smtp", "imap")):
        if not any(
            marker in source_text
            for marker in ("imaplib", "smtplib", "emailmessage", "email.message")
        ):
            errors.append("email automation project has no email transport/message implementation")
        if not any(
            marker in source_text
            for marker in ("fake", "mock", "fixture", ".eml", "inmemory", "in_memory")
        ):
            errors.append("email automation project has no offline email fixture/fake transport")
    if "telegram" in lower and "from telegram" not in source_text:
        errors.append("Telegram project does not use a Telegram SDK")
    if "discord" in lower and "import discord" not in source_text:
        errors.append("Discord project does not use a Discord SDK")
    if "slack" in lower and not any(
        marker in source_text for marker in ("slack_sdk", "slack_bolt")
    ):
        errors.append("Slack project does not use a Slack SDK")
    if "api integration" in lower and not any(
        marker in source_text for marker in ("requests", "httpx", "aiohttp", "urllib")
    ):
        errors.append("API integration project has no HTTP client implementation")
    if any(term in lower for term in ("data pipeline", "etl")) and not any(
        marker in source_text for marker in ("csv", "json", "sqlite", "transform")
    ):
        errors.append("data pipeline project has no local input/transform/output implementation")
    if any(term in lower for term in ("web automation", "browser automation")):
        if not any(
            marker in source_text
            for marker in ("playwright", "selenium", "beautifulsoup", "requests")
        ):
            errors.append("web automation project has no browser/page automation implementation")
        if not any(
            marker in source_text
            for marker in ("fixture", "localhost", "127.0.0.1", "mock")
        ):
            errors.append("web automation project has no deterministic local page fixture")
    return errors


def _validate_execution(
    flow_id: str,
    subtask: Subtask,
    before: dict[str, tuple[int, str]],
    output: str,
) -> tuple[bool, list[str], str]:
    """Validate observable effects instead of trusting exit code or model claims."""
    after = _workspace_snapshot(flow_id)
    changed = sorted(
        path for path, identity in after.items()
        if before.get(path) != identity
    )
    meaningful = [
        path for path in changed
        if Path(path).name not in {".env"} and not _is_transient_path(Path(path))
    ]

    errors = []
    kind, expected = _subtask_contract(subtask)
    for rel in meaningful:
        # scan kind: empty output files are valid (means "no findings" — a legitimate result)
        if kind == "scan":
            continue
        error = _validate_artifact(WORKSPACE_DIR / flow_id / rel)
        if error:
            errors.append(f"/work/{rel}: {error}")

    weak_output = not output.strip() or output.strip() in {"(done)", "done", "ok"}
    changed_paths = [Path(path) for path in meaningful]
    expected_changed = _matching_paths(changed_paths, expected)
    missing_expected = _missing_patterns(changed_paths, expected)

    # scan kind: missing expected artifacts are still OK (tool ran, nothing to write is valid)
    if missing_expected and kind not in {"test", "text", "none", "scan"}:
        errors.append(
            "expected artifact was not created or modified: "
            + ", ".join(f"/work/{path}" for path in missing_expected)
        )
    if kind in {
        "source", "documentation", "configuration", "dependency",
        "report", "dataset", "artifact",
    } and not meaningful:
        errors.append("no deliverable file was created or modified")
    elif kind == "scan" and weak_output and not meaningful:
        errors.append("scan task produced no output and created no files — tool likely failed to run")
    elif kind == "test" and weak_output and not meaningful:
        errors.append("no artifact or meaningful verification output was produced")
    elif kind not in {"text", "none", "scan"} and weak_output and not meaningful:
        errors.append("execution produced no observable result")

    if kind in {
        "source", "documentation", "configuration", "dependency", "report", "dataset",
    } and not expected_changed and not any(_kind_matches(kind, path) for path in changed_paths) and kind != "scan":
        messages = {
            "source": "task requires source code, but no source file was created or modified",
            "documentation": "task requires documentation, but no README or documentation file changed",
            "configuration": "task requires configuration, but no configuration file changed",
            "dependency": "task requires dependencies, but no dependency manifest changed",
            "report": "task requires report/results output, but no report artifact changed",
            "dataset": "task requires structured data, but no dataset artifact changed",
        }
        errors.append(messages[kind])

    if errors:
        report = "Validation failed:\n- " + "\n- ".join(errors)
        return False, changed, report

    evidence = []
    if meaningful:
        evidence.append(
            "Artifacts created/modified: "
            + ", ".join(f"/work/{path}" for path in meaningful[:20])
        )
    if output.strip() and not weak_output:
        evidence.append("Process returned meaningful output")
    return True, changed, "Validation passed. " + "; ".join(evidence)


def _validate_text_result(result: str) -> tuple[bool, str]:
    text = (result or "").strip()
    if not text or text.lower() in {"done", "(done)", "ok"}:
        return False, "agent produced no substantive result"
    unsupported_claims = (
        r"(?i)\bi(?:'ve| have) created\b",
        r"(?i)\bsaved (?:it|the file|output) to /work",
        r"(?i)\bcreated at /work/",
        r"(?i)\bi(?:'ll| will) inspect\b.*\b(?:cat|ls|grep)\b",
    )
    if any(re.search(pattern, text) for pattern in unsupported_claims):
        return False, "text-only agent claimed filesystem/tool actions it did not execute"
    return True, "substantive text result produced"


def _workspace_matches_deliverable(subtask: Subtask, flow_id: str) -> tuple[bool, list[str], str]:
    """Check whether the final workspace satisfies an explicit subtask contract."""
    snapshot = _workspace_snapshot(flow_id)
    paths = [Path(path) for path in snapshot]
    kind, expected = _subtask_contract(subtask)
    if kind in {"text", "test", "none"} and not expected:
        return False, [], f"{kind} contract cannot be recovered from workspace files alone"
    missing = _missing_patterns(paths, expected)
    matches = _matching_paths(paths, expected) if expected else [
        path for path in paths if _kind_matches(kind, path)
    ]
    if missing or not matches:
        target = (
            ", ".join(f"/work/{path}" for path in missing)
            if missing else kind
        )
        return False, [], f"final workspace does not satisfy {target} deliverable"
    malformed = [
        f"/work/{path}: {error}"
        for path in matches
        if (error := _validate_artifact(WORKSPACE_DIR / flow_id / path))
    ]
    if malformed:
        return False, [str(path) for path in matches], "; ".join(malformed)
    return True, sorted(str(path) for path in matches), "final workspace contains required deliverables"


def _reconcile_task_status(task: Task, flow: Flow) -> tuple[bool, str, list[str]]:
    """Recover a task when later subtasks produced its required final deliverables."""
    if task.status != "failed":
        return False, "", []
    successful_after_failure = False
    seen_failure = False
    for subtask in task.subtasks:
        if subtask.status == "failed":
            seen_failure = True
        elif seen_failure and subtask.status == "finished" and (
            subtask.artifacts or subtask.validation.startswith("Validation passed")
        ):
            successful_after_failure = True
    if not successful_after_failure:
        return False, "", []

    evidence = []
    reports = []
    for subtask in task.subtasks:
        if subtask.status != "failed":
            continue
        valid, matched, report = _workspace_matches_deliverable(subtask, flow.id)
        if not valid:
            return False, report, matched
        evidence.extend(matched)
        reports.append(report)
    return True, "; ".join(sorted(set(reports))), sorted(set(evidence))


def _validate_flow_deliverables(flow: Flow) -> tuple[bool, str]:
    """Apply objective-level checks after all task-level validation."""
    root = WORKSPACE_DIR / flow.id
    files = [
        path for path in root.rglob("*")
        if (
            path.is_file()
            and path.name != "flow.json"
            and not _is_transient_path(path.relative_to(root))
        )
    ] if root.exists() else []
    errors = []

    for path in files:
        if path.name == ".env" or path.suffix.lower() in (".log", ".tmp"):
            continue
        error = _validate_artifact(path)
        if error:
            errors.append(f"/work/{path.relative_to(root)}: {error}")

    contract_errors = []
    for task in flow.tasks:
        for subtask in task.subtasks:
            # Failed branches are historical evidence, not active final requirements.
            # A replan may intentionally replace their file layout and implementation.
            if subtask.status != "finished":
                continue
            kind, expected = _subtask_contract(subtask)
            if kind in {"text", "test", "none"} and not expected:
                continue
            valid, _, report = _workspace_matches_deliverable(subtask, flow.id)
            if not valid:
                contract_errors.append(f"{subtask.title}: {report}")
    if contract_errors:
        errors.append("unsatisfied deliverable contracts: " + "; ".join(contract_errors))

    if flow.flow_type == "agent_development":
        code_files = [
            path for path in files
            if path.suffix.lower() in (".py", ".js", ".ts", ".go", ".rs", ".sh")
        ]
        if not code_files:
            errors.append("agent-development flow produced no source code")
        if not any(path.name.lower().startswith("readme") for path in files):
            errors.append("agent-development flow produced no README")
        source_bytes = sum(path.stat().st_size for path in code_files)
        if code_files and source_bytes < 800:
            errors.append(
                f"generated source is only {source_bytes} bytes; placeholder code is not a functional agent"
            )
        substantial_python = False
        for path in code_files:
            if path.suffix.lower() != ".py":
                continue
            try:
                tree = ast.parse(path.read_text(errors="replace"), filename=str(path))
                definitions = sum(
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    for node in ast.walk(tree)
                )
                if path.stat().st_size >= 500 and definitions >= 2:
                    substantial_python = True
                    break
            except (OSError, SyntaxError):
                continue
        if any(path.suffix.lower() == ".py" for path in code_files) and not substantial_python:
            errors.append(
                "Python project has no substantial module with at least two functions/classes"
            )
        python_files = [path for path in code_files if path.suffix.lower() == ".py"]
        external_imports = _external_python_imports(python_files, root)
        dependency_manifests = [
            path for path in files
            if path.name.lower() in _DEPENDENCY_NAMES
        ]
        if external_imports and not dependency_manifests:
            errors.append(
                "Python project imports third-party modules but has no dependency manifest: "
                + ", ".join(sorted(external_imports))
            )
        elif external_imports:
            undeclared = _undeclared_imports(external_imports, dependency_manifests)
            if undeclared:
                errors.append(
                    "dependency manifest does not declare imported module(s): "
                    + ", ".join(sorted(undeclared))
                )
        objective_lower = flow.objective.lower()
        source_text = "\n".join(
            path.read_text(errors="replace")
            for path in code_files
            if path.stat().st_size < 1_000_000
        ).lower()
        errors.extend(_agent_capability_errors(flow.objective, source_text))
        if any(term in objective_lower for term in ("telegram", "discord", "slack")):
            if re.search(
                r"\b\d{6,12}:[a-z0-9_-]{20,}\b",
                source_text,
                re.IGNORECASE,
            ):
                errors.append("Telegram project contains a hardcoded token-like credential")
        if any(term in objective_lower for term in ("trading", "crypto", "exchange")):
            forbidden_hosts = (
                "api.trading-platform.com",
                "api.example.com",
                "your-api-host",
                "your_api_host",
            )
            used_forbidden = [host for host in forbidden_hosts if host in source_text]
            if used_forbidden:
                errors.append(
                    "trading project contains invented API host(s): " + ", ".join(used_forbidden)
                )
            if not any(
                marker in source_text
                for marker in ("paper", "dry_run", "dry-run", "mock", "fixture", "sandbox")
            ):
                errors.append("trading project has no explicit paper/dry-run validation mode")
        if any(term in objective_lower for term in ("scraper", "crawler", "scrape", "crawl")):
            if not any(
                marker in source_text
                for marker in (
                    "self-test", "self_test", "fixture", "mock", "localhost",
                    "127.0.0.1", "http.server",
                )
            ):
                errors.append(
                    "crawler project has no deterministic local fixture/self-test mode"
                )
            external_test_hosts = [
                host for host in ("httpbin.org", "test.org", "demo.net")
                if host in source_text
            ]
            if external_test_hosts and not any(
                marker in source_text for marker in ("self-test", "self_test", "fixture", "localhost")
            ):
                errors.append(
                    "crawler validation depends on external test host(s): "
                    + ", ".join(external_test_hosts)
                )
        if code_files and any(path.suffix.lower() == ".py" for path in code_files):
            runtime_valid, runtime_report = _validate_python_project_runtime(
                flow.id, flow.objective
            )
            if not runtime_valid:
                errors.append(runtime_report)
    elif flow.flow_type == "research":
        if not any(
            path.suffix.lower() in {".md", ".json", ".csv", ".html"}
            for path in files
        ):
            errors.append("research flow produced no durable report or dataset artifact")
    elif flow.flow_type == "product_development":
        if not any(
            path.suffix.lower() in {".md", ".json", ".csv"}
            for path in files
        ):
            errors.append("product-development flow produced no durable plan/report artifact")
    elif flow.flow_type == "security":
        if not any(
            path.suffix.lower() in {".md", ".json", ".html"}
            for path in files
        ):
            errors.append("security flow produced no durable assessment report")

    executable_subtasks = [
        subtask
        for task in flow.tasks
        for subtask in task.subtasks
        if subtask.needs_container
    ]
    unvalidated = [
        subtask.title for subtask in executable_subtasks
        if subtask.status == "finished"
        and not subtask.validation.startswith("Validation passed")
    ]
    if unvalidated:
        errors.append(
            "executable subtasks lack passing validation: " + ", ".join(unvalidated)
        )

    if errors:
        return False, "Flow validation failed:\n- " + "\n- ".join(errors)
    return True, f"Flow validation passed with {len(files)} workspace file(s)."


def _recovered_after_replan(flow: Flow) -> bool:
    """Allow a successful replan to supersede earlier failed approaches."""
    if flow.replan_count < 1:
        return False
    last_failed = max(
        (index for index, task in enumerate(flow.tasks) if task.status == "failed"),
        default=-1,
    )
    later = flow.tasks[last_failed + 1:]
    if not later or any(task.status != "finished" for task in later):
        return False
    return any(
        subtask.artifacts or subtask.validation.startswith("Validation passed")
        for task in later
        for subtask in task.subtasks
        if subtask.status == "finished"
    )


def _supersede_failed_attempts(flow: Flow) -> None:
    """Preserve failed output as history while reflecting a validated replacement."""
    for task in flow.tasks:
        if task.status == "failed":
            task.status = "finished"
            if task.result:
                task.result = (
                    "FINAL RECOVERY: Superseded by the validated final workspace.\n\n"
                    + task.result
                )
        for subtask in task.subtasks:
            if subtask.status == "failed":
                subtask.status = "superseded"
                if subtask.result:
                    subtask.result = (
                        "SUPERSEDED: The final workspace passed deterministic validation.\n\n"
                        + subtask.result
                    )
                subtask.validation = (
                    subtask.validation
                    or "Superseded by deterministic final workspace validation"
                )


def _validate_python_project_runtime(
    flow_id: str, objective: str = ""
) -> tuple[bool, str]:
    """Run a bounded smoke test in the same Python container used by agents."""
    code = r'''
import ast
import os
import pathlib
import subprocess
import sys
import textwrap

root = pathlib.Path("/work")
objective = os.environ.get("OLLAMAGI_OBJECTIVE", "").lower()
python_files = sorted(root.rglob("*.py"))
if not python_files:
    print("No Python files found", file=sys.stderr)
    raise SystemExit(1)

for path in python_files:
    try:
        ast.parse(path.read_text(errors="replace"), filename=str(path))
    except Exception as exc:
        print(f"Syntax validation failed for {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)

requirements = root / "requirements.txt"
if requirements.exists() and requirements.stat().st_size:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--prefer-binary", "-r", str(requirements)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        print("Dependency installation failed:", file=sys.stderr)
        print((result.stdout + result.stderr)[-3000:], file=sys.stderr)
        raise SystemExit(1)
elif (root / "pyproject.toml").exists():
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--prefer-binary", "."],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        print("Project installation failed:", file=sys.stderr)
        print((result.stdout + result.stderr)[-3000:], file=sys.stderr)
        raise SystemExit(1)

entry = next(
    (root / name for name in ("main.py", "app.py", "bot.py", "run.py") if (root / name).exists()),
    None,
)
if entry is None:
    print(f"Validated syntax for {len(python_files)} Python file(s); no entrypoint found")
    raise SystemExit(0)

source = entry.read_text(errors="replace").lower()
if "--self-test" in source or "--self_test" in source:
    mode = "--self-test"
elif "--dry-run" in source or "--dry_run" in source:
    mode = "--dry-run"
else:
    print(
        f"Agent entrypoint {entry.name} has no explicit --self-test or --dry-run mode",
        file=sys.stderr,
    )
    raise SystemExit(1)

guard_dir = pathlib.Path("/tmp/ollamagi-network-guard")
guard_dir.mkdir(parents=True, exist_ok=True)
(guard_dir / "sitecustomize.py").write_text(textwrap.dedent("""
    import socket
    _real_getaddrinfo = socket.getaddrinfo
    _real_socket = socket.socket
    _allowed = {"localhost", "127.0.0.1", "::1"}

    def _guarded_getaddrinfo(host, *args, **kwargs):
        if host not in _allowed:
            raise OSError(f"external network disabled during validation: {host}")
        return _real_getaddrinfo(host, *args, **kwargs)

    class _GuardedSocket(_real_socket):
        def connect(self, address):
            host = address[0] if isinstance(address, tuple) and address else address
            if host not in _allowed:
                raise OSError(f"external network disabled during validation: {host}")
            return super().connect(address)

    socket.getaddrinfo = _guarded_getaddrinfo
    socket.socket = _GuardedSocket
"""))

env = os.environ.copy()
env.update({
    "PYTHONPATH": f"{guard_dir}:{root}",
    "OLLAMAGI_VALIDATE": "1",
    "OLLAMAGI_OBJECTIVE": objective,
    "DRY_RUN": "1",
    "OFFLINE": "1",
    "PAPER_TRADING": "1",
    "TELEGRAM_BOT_TOKEN": "000000000:OFFLINE_VALIDATION_TOKEN",
    "DISCORD_TOKEN": "OFFLINE_VALIDATION_TOKEN",
    "SLACK_BOT_TOKEN": "OFFLINE_VALIDATION_TOKEN",
    "LOG_DIR": "/tmp/ollamagi-validation-logs",
})
try:
    result = subprocess.run(
        [sys.executable, str(entry), mode],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
except subprocess.TimeoutExpired:
    print(f"Entrypoint {entry.name} did not terminate within 45 seconds", file=sys.stderr)
    raise SystemExit(1)

combined = (result.stdout + result.stderr)[-4000:]
print(combined)
if result.returncode != 0:
    print(f"Entrypoint {entry.name} exited with code {result.returncode}", file=sys.stderr)
    raise SystemExit(1)
print(f"Runtime smoke test passed for {entry.name} using {mode} with external network disabled")
'''
    container = None
    try:
        container = create_container(flow_id, "final-validation", "python")
        escaped_objective = json.dumps(objective)
        code_with_objective = code.replace(
            'objective = os.environ.get("OLLAMAGI_OBJECTIVE", "").lower()',
            f"objective = {escaped_objective}.lower()",
        )
        exit_code, output = exec_python(
            container, code_with_objective, timeout=MAX_TASK_TIMEOUT
        )
        if exit_code != 0:
            return False, f"Python runtime smoke test failed: {output[-2000:]}"
        return True, output[-1000:] or "Python runtime smoke test passed"
    except Exception as exc:
        return False, f"Python runtime smoke test error: {exc}"
    finally:
        if container:
            stop_container(container)


def _trim_history(history: list[dict], max_tail: int = 8) -> list[dict]:
    """Keep the objective (first entry) + the last max_tail messages to avoid context overflow."""
    if len(history) <= max_tail + 1:
        return history
    return [history[0]] + history[-max_tail:]


_COMPACT_MARKER = "[CONTEXT SUMMARY]"
_COMPACT_THRESHOLD = 10   # start compacting once history exceeds this many entries
_COMPACT_TAIL = 6         # always keep the last N raw messages verbatim


def _compact_history(history: list[dict], flow_id: str) -> list[dict]:
    """LLM-summarize old history into one compressed message, keeping the last few raw entries.

    Layout returned: [objective, compact_summary_msg, ...last_N_raw]
    When history is already compacted (has a summary), the prior summary is included in the
    next summarization so nothing is lost across multiple compaction rounds.
    """
    if len(history) <= _COMPACT_THRESHOLD:
        return history

    objective = history[0]
    tail = history[-_COMPACT_TAIL:]
    to_compress = history[1:-_COMPACT_TAIL]

    lines = []
    for m in to_compress:
        content = m["content"]
        role = m["role"]
        # If a prior compact summary is in the list, include it in full
        if content.startswith(_COMPACT_MARKER):
            lines.append(content)
        else:
            lines.append(f"{role}: {content[:400]}")

    summary_input = "\n".join(lines)
    summary = chat(
        [{"role": "user", "content":
          f"Summarize the following completed subtask history into 1-2 sentences per item. "
          f"Preserve key outputs, file paths, and error messages. Be concise.\n\n{summary_input}"}],
        task_type="fast",
        flow_id=flow_id,
    )
    summary_msg = {"role": "assistant",
                   "content": f"{_COMPACT_MARKER}: {summary}"}
    return [objective, summary_msg] + tail


# ── Planning ──────────────────────────────────────────────────────────────────
def _generate_tasks(flow: Flow, mem_ctx: str) -> list[Task]:
    roles = FLOW_TYPE_ROLES.get(flow.flow_type, FLOW_TYPE_ROLES["general"])
    role_list = ", ".join(roles)
    system = (
        f"You are decomposing a '{flow.flow_type}' flow into tasks.\n"
        f"Available agents: {role_list}\n"
        "Return ONLY valid JSON array with keys: id(int), title, description, agent, "
        "needs_container(bool), container_type('pentest'|'python'|'generic')\n"
        "3-5 tasks. Concrete and actionable. The user objective is authoritative; memory context is "
        "optional and must not add infrastructure or requirements the user did not request.\n"
        "For agent-development flows always include implementation, persistent dependencies, README, "
        "and one deterministic offline self-test task. Avoid redundant architecture/refinement passes.\n"
        f"NON-NEGOTIABLE CONSTRAINTS:\n{_objective_constraints(flow.objective, flow.flow_type)}\n"
        "No markdown."
    )
    user_parts = [f"OBJECTIVE: {flow.objective}"]
    if flow.flow_type == "security":
        from core.config import SSH_HOST, SSH_USER
        user_parts.append(
            f"TARGET: {SSH_HOST} (authorized Fedora host). SSH user: {SSH_USER}. "
            "All tasks: agent='pentester', needs_container=true, container_type='pentest'. "
            "Generate ACTIVE testing tasks only — NO schema/validation/Python tasks:\n"
            "  Task 1: Recon — nmap full port scan + OS detection → reports/nmap.txt, reports/nmap.xml\n"
            "  Task 2: Enumeration — nikto + gobuster on HTTP ports → reports/nikto.txt, reports/gobuster.txt\n"
            "  Task 3: Vuln assessment — nuclei + searchsploit → reports/nuclei.txt, reports/searchsploit.txt\n"
            "  Task 4: Exploitation — targeted exploits for found CVEs → reports/exploits.txt\n"
            "  Task 5: Report — executive summary with CVSS scores → reports/pentest_report.md\n"
            "IMPORTANT: All scan output paths end in .txt or .xml — NEVER .json"
        )
    if mem_ctx:
        user_parts.append(mem_ctx)
    raw = chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": "\n\n".join(user_parts)}],
        task_type="orchestrator", flow_id=flow.id,
        max_tokens=1200, think=False,
    )
    raw = _strip_fences(raw.strip())
    try:
        task_defs = _object_list(json.loads(raw))
        if not task_defs:
            raise ValueError("planner returned no task objects")
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
            agent=_role_for_flow(td.get("agent", "primary_agent"), flow.flow_type),
        ))
    return tasks


def _generate_subtasks(task: Task, flow: Flow, mem_ctx: str) -> list[Subtask]:
    roles = FLOW_TYPE_ROLES.get(flow.flow_type, FLOW_TYPE_ROLES["general"])
    system = (
        "You are the Generator agent. Break this task into 2-5 concrete subtasks.\n"
        f"Agent must be exactly one of: {', '.join(roles)}.\n"
        "Return ONLY JSON array. Every object must contain: id(int), title, description, agent, "
        "needs_container(bool), container_type('pentest'|'python'|'generic'), "
        "deliverable_kind, expected_artifacts(array of /work-relative paths or glob patterns).\n"
        "deliverable_kind must be exactly one of: text, source, documentation, configuration, "
        "dependency, report, dataset, test, artifact, none, scan.\n"
        "Use scan for pentest tool output files (nmap, nikto, gobuster, nuclei, searchsploit) — "
        "empty output is accepted as 'no findings found'.\n"
        "Use source only when that subtask must create/modify executable source code. "
        "Use documentation for README/Markdown/docs, dependency for requirements/pyproject/package "
        "manifests, configuration for config files, report for prose findings, dataset for scraped "
        "or structured data, test for validation/inspection that may succeed without modifying files, "
        "text for reasoning-only output, and artifact for another required file type.\n"
        "List exact expected paths whenever filenames are known, for example ['README.md'], "
        "['requirements.txt'], ['bot.py'], or ['reports/*.json']. Use [] only for text/test/none.\n"
        "Do not list generated logs, __pycache__, or transient validation output as expected artifacts "
        "unless the subtask's sole purpose is explicitly to validate that output.\n"
        "Use needs_container=false for planning, analysis, synthesis, and advice.\n"
        "Use agent='coder' and container_type='python' for scripts, web requests, parsing, "
        "file generation, or structured-data transformations.\n"
        "Do not add a subtask that writes directly to memory; the orchestrator handles that.\n"
        "For applications that normally need credentials or live services, require deterministic "
        "paper/dry-run behavior with mock fixtures or public unauthenticated endpoints.\n"
        "Never invent API hostnames, credentials, or require live authenticated trading during build validation.\n"
        "Do not use target runtime dependencies merely to generate configuration files; write JSON with "
        "the standard library and YAML/TOML as plain text unless the dependency is already available.\n"
        "A task that says mock/self-test must not connect to localhost services such as Redis/Postgres.\n"
        f"NON-NEGOTIABLE CONSTRAINTS:\n{_objective_constraints(flow.objective, flow.flow_type)}\n"
        "Every container subtask must tolerate missing prior artifacts: inspect /work first, "
        "reuse equivalent files when present, and create required parent directories.\n"
        "No markdown."
    )
    user = f"TASK: {task.title}\n\nDESCRIPTION: {task.description}"
    if flow.flow_type == "security":
        from core.config import SSH_HOST, SSH_USER
        user += (
            f"\n\nPENTEST CONTEXT: All subtasks must use agent='pentester', "
            f"needs_container=true, container_type='pentest'. "
            f"TARGET HOST: {SSH_HOST} (SSH user: {SSH_USER}). "
            "Generate subtasks that run actual Kali tools (nmap, nikto, nuclei, gobuster, etc.) "
            "against the target. "
            "DELIVERABLE KINDS:\n"
            "  - Scan tool subtasks (nmap, nikto, gobuster, nuclei, searchsploit): deliverable_kind='scan'\n"
            "  - Final summary/report subtask only: deliverable_kind='report'\n"
            "EXPECTED ARTIFACTS — STRICT RULES:\n"
            "  - Scan tools output TEXT, not JSON. ALL scan artifact paths MUST end in .txt or .xml.\n"
            "  - NEVER use .json extension for nmap, nikto, nuclei, gobuster, or searchsploit output.\n"
            "  - Use flat simple names: 'reports/nmap.txt', 'reports/nikto.txt', 'reports/gobuster.txt',\n"
            "    'reports/nuclei.txt', 'reports/searchsploit.txt'\n"
            "  - nmap may also produce 'reports/nmap.xml' (nmap -oX format is valid XML, not JSON)\n"
            "  - Report artifacts may use .md or .txt: 'reports/pentest_report.md'"
        )
    if mem_ctx:
        user += f"\n\n{mem_ctx}"
    raw = chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        task_type="orchestrator", flow_id=flow.id,
        max_tokens=1000, think=False,
    )
    raw = _strip_fences(raw.strip())
    try:
        sub_defs = _object_list(json.loads(raw))
        if not sub_defs:
            raise ValueError("planner returned no subtask objects")
    except Exception:
        agent = _role_for_flow(task.agent, flow.flow_type)
        needs_container = agent in ("coder", "installer", "pentester")
        return [Subtask(id=f"{task.id}-s1", task_id=task.id, title=task.title,
                        description=task.description, agent=agent,
                        needs_container=needs_container,
                        container_type="pentest" if agent == "pentester" else "python",
                        deliverable_kind=_infer_deliverable_kind(
                            task.title, task.description, needs_container
                        ),
                        expected_artifacts=_infer_expected_artifacts(
                            f"{task.title}\n{task.description}"
                        ))]
    subtasks = []
    for sd in sub_defs:
        agent = _role_for_flow(sd.get("agent", task.agent), flow.flow_type)
        needs_container = bool(sd.get("needs_container", False))
        container_type = sd.get("container_type", "python")
        if container_type not in ("pentest", "python", "generic"):
            container_type = "python"
        if agent in ("coder", "installer", "pentester"):
            needs_container = True
        if agent in ("primary_agent", "generator", "refiner", "adviser", "architect", "monetizer"):
            needs_container = False
        action_text = f"{sd.get('title', '')} {sd.get('description', '')}"
        if _TOOL_REQUIRED_PATTERN.search(action_text):
            agent = "coder" if "coder" in roles else roles[0]
            needs_container = True
            container_type = "python"
        deliverable_kind = str(sd.get("deliverable_kind", "auto")).strip().lower()
        if deliverable_kind not in _DELIVERABLE_KINDS or deliverable_kind == "auto":
            deliverable_kind = _infer_deliverable_kind(
                sd.get("title", ""), sd.get("description", ""), needs_container
            )
        expected_artifacts = _normalize_expected_artifacts(sd.get("expected_artifacts"))
        if not expected_artifacts:
            expected_artifacts = _infer_expected_artifacts(action_text)
        # Pentest normalization: force scan kind for tool runs; rewrite .json → .txt for scan output
        if agent == "pentester":
            title_str = sd.get("title", "")
            is_final_report = bool(_PENTEST_REPORT_KEYWORDS_RE.search(title_str))
            if not is_final_report and deliverable_kind != "report":
                deliverable_kind = "scan"
            if deliverable_kind == "scan":
                expected_artifacts = _normalize_scan_artifacts(expected_artifacts)
        if deliverable_kind == "text":
            needs_container = False
            if agent in ("coder", "installer"):
                agent = next(
                    (
                        candidate for candidate in (
                            "architect", "researcher", "refiner", "primary_agent"
                        )
                        if candidate in roles
                    ),
                    roles[0],
                )
        elif deliverable_kind in {
            "source", "documentation", "configuration", "dependency",
            "report", "dataset", "artifact",
        }:
            needs_container = True
            if agent not in ("coder", "installer", "pentester", "data_engineer", "devops"):
                # source/artifact kind always needs a code-generation agent
                if deliverable_kind == "source":
                    agent = "coder"  # hard-force for source regardless of roles
                elif "coder" in roles:
                    agent = "coder"
                elif "pentester" in roles:
                    agent = "pentester"
                    container_type = "pentest"
                else:
                    agent = roles[0]
        subtasks.append(Subtask(
            id=f"{task.id}-s{sd.get('id', len(subtasks)+1)}",
            task_id=task.id,
            title=sd.get("title", "Subtask"),
            description=sd.get("description", ""),
            agent=agent,
            needs_container=needs_container,
            container_type=container_type,
            deliverable_kind=deliverable_kind,
            expected_artifacts=expected_artifacts,
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
        "needs_container(bool), container_type}. 1-3 tasks. Reuse valid artifacts and repair the "
        "smallest remaining gap. Do not redesign the project or add new infrastructure.\n"
        f"NON-NEGOTIABLE CONSTRAINTS:\n{_objective_constraints(flow.objective, flow.flow_type)}"
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
        max_tokens=1200, think=False,
    )
    raw = _strip_fences(raw.strip())
    try:
        defs = _object_list(json.loads(raw))
        if not defs:
            raise ValueError("replanner returned no task objects")
    except Exception:
        return []
    tasks = []
    for i, td in enumerate(defs):
        tasks.append(Task(
            id=f"r{from_idx + i + 1}",
            flow_id=flow.id,
            title=td.get("title", "Task"),
            description=td.get("description", ""),
            agent=_role_for_flow(td.get("agent", "primary_agent"), flow.flow_type),
        ))
    return tasks


# ── Auto-fix ──────────────────────────────────────────────────────────────────
def _read_work_sources(flow_id: str | None, max_files: int = 4, max_bytes: int = 8000) -> str:
    """Return content of small Python source files from the flow workspace for cross-file debugging."""
    if not flow_id:
        return ""
    root = WORKSPACE_DIR / flow_id
    if not root.exists():
        return ""
    budget = max_bytes
    parts = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "ollamagi_task.py" or _is_transient_path(path):
            continue
        try:
            size = path.stat().st_size
            if size == 0 or size > 20_000:
                continue
            content = path.read_text(errors="replace")[:budget]
            budget -= len(content)
            rel = path.relative_to(root)
            parts.append(f"=== /work/{rel} ===\n{content}")
            if len(parts) >= max_files or budget <= 0:
                break
        except OSError:
            continue
    return "\n\n".join(parts)


def _read_work_data(flow_id: str | None, max_bytes: int = 12000) -> str:
    """Return content of small data/report files from the workspace so text agents can use real data."""
    if not flow_id:
        return ""
    root = WORKSPACE_DIR / flow_id
    if not root.exists():
        return ""
    skip = {"flow.json", "flow_log.jsonl", "llm_calls.jsonl"}
    data_suffixes = {".json", ".txt", ".md", ".csv", ".xml"}
    budget = max_bytes
    parts = []
    for path in sorted(root.rglob("*"), key=lambda p: p.stat().st_size if p.is_file() else 0):
        if not path.is_file() or path.name in skip or _is_transient_path(path):
            continue
        if path.suffix.lower() not in data_suffixes:
            continue
        try:
            size = path.stat().st_size
            if size == 0 or size > 80_000:
                continue
            content = path.read_text(errors="replace")[:budget]
            budget -= len(content)
            rel = path.relative_to(root)
            parts.append(f"=== /work/{rel} ===\n{content}")
            if budget <= 0:
                break
        except OSError:
            continue
    return "\n\n".join(parts)


def _fix_python(code: str, error_output: str, description: str, flow_id: str | None) -> str:
    workspace = _workspace_inventory(flow_id) if flow_id else "WORKSPACE FILES: unavailable"
    # Include /work source files so the LLM can fix bugs in generated application code,
    # not just in the build script wrapper (e.g. LocalFileHandler overwriting self.directory)
    work_sources = _read_work_sources(flow_id)
    prompt = (
        "Fix this Python build script that failed. Return ONLY corrected Python code — no markdown.\n"
        "Preserve useful files already present in /work. The corrected build script must modify "
        "the actual deliverable source files in /work and use bounded offline/paper-mode validation.\n"
        "GIT: If the error involves git/clone — git is NOT pre-installed. Add this before any git command:\n"
        "  import subprocess\n"
        "  subprocess.run(['apt-get','update','-qq'], capture_output=True)  # MUST run before install\n"
        "  subprocess.run(['apt-get','install','-y','-qq','git'], capture_output=True)  # no check=True\n"
        "  # If apt fails (exit 100), fall back to ZIP download instead of crashing.\n"
        "  Then verify: assert pathlib.Path('/work/repo-name').is_dir(), 'clone failed'\n"
        "  GitHub repo names are case-sensitive. OllamAGI: https://github.com/Linutesto/ollamagi\n"
        "  If the directory already exists, skip cloning to stay idempotent.\n"
        "Persist every third-party dependency in /work/requirements.txt or pyproject.toml; installing "
        "a package only inside the temporary build container is not a deliverable. Match imports to "
        "the correct distribution (for example `from telegram ...` requires `python-telegram-bot`, "
        "while `telebot` requires `pyTelegramBotAPI`; never mix their APIs).\n"
        "Credentialed bots must implement --self-test or --dry-run that exercises handlers with mocks "
        "without building a live polling client or contacting Telegram/Discord/Slack.\n"
        "Any Redis, database, SMTP, IMAP, queue, browser, or provider integration must be optional "
        "during validation and replaced by an in-memory/local-file fake. A mode called mock or "
        "self-test must never connect to localhost infrastructure.\n"
        "The BUILD SCRIPT itself must parse as Python. Never embed generated multi-line files inside "
        "triple-quoted strings because their docstrings will terminate the outer string. Write file "
        "content using a list of ordinary quoted lines joined with '\\n', JSON-decoded strings, or "
        "another syntax-safe method. Do not use ''' or \\\"\\\"\\\" anywhere in the build script.\n"
        "PYTHON SYNTAX RULES — common LLM mistakes:\n"
        "  - `with` / `async with` blocks do NOT support `else`. Only `for`, `while`, and `try` support `else`.\n"
        "  - `async for` and `async with` follow the same else-clause rules as their sync counterparts.\n"
        "  - Variables assigned inside `async with` are NOT in scope after the block unless assigned before.\n"
        "If the error is a SyntaxError in a /work/*.py file (not the build script itself), the build "
        "script is writing BROKEN CONTENT to that file. Fix the content string/lines that get written.\n"
        "For web crawlers, use local HTML fixtures or a temporary localhost HTTP server rather than "
        "external test domains. Handled 404/timeouts are test evidence, not fatal build failures.\n"
        "IMPORTANT — when a /work/*.py application fails at runtime (e.g. local HTTP server returns 404 "
        "for fixtures, imports fail, attribute errors), the build script MUST rewrite that source file "
        "with the bug fixed. Common HTTP server fixture bug: never set self.directory before calling "
        "super().__init__(); always pass directory= to super() instead:\n"
        "  class Handler(SimpleHTTPRequestHandler):\n"
        "    def __init__(self, *args, **kw): super().__init__(*args, directory='/work/fixtures', **kw)\n"
        "To install packages use (--break-system-packages REQUIRED on modern Debian):\n"
        "  import sys,subprocess; subprocess.run([sys.executable,'-m','pip','install','--break-system-packages','--prefer-binary','pkg'],check=False,capture_output=True)\n"
        "If error is 'externally-managed-environment' — add --break-system-packages to the pip call.\n"
        "If the error is 'No URL provided', 'usage: ollamagi_task.py', or any argparse help/usage output — "
        "the ENTIRE FAILING CODE is the application itself, not a build script. The application was run "
        "as /tmp/ollamagi_task.py with no arguments, so argparse printed help and exited.\n"
        "FIX: restructure the code as a BUILD SCRIPT that writes the application to /work/app.py, "
        "then runs a syntax check. The build script itself must NOT be the application:\n"
        "  CORRECT build script pattern:\n"
        "    import pathlib, ast, sys\n"
        "    code_lines = ['#!/usr/bin/env python3', 'import argparse', '...rest of app...']\n"
        "    pathlib.Path('/work/crawler.py').write_text('\\n'.join(code_lines))\n"
        "    ast.parse(pathlib.Path('/work/crawler.py').read_text())\n"
        "    print('Written /work/crawler.py')\n"
        "  WRONG: writing application code that calls argparse.parse_args() at module level\n\n"
        f"TASK: {description}\n\n"
        f"{workspace}\n\n"
        + (f"CURRENT /work SOURCE FILES:\n{work_sources}\n\n" if work_sources else "")
        + (f"CURRENT /work DATA FILES (exact schemas — use these, don't guess):\n{_read_work_data(flow_id, max_bytes=6000)}\n\n"
           if flow_id else "")
        + f"ERROR OUTPUT:\n{error_output[:2000]}\n\n"
        f"FAILING CODE (first 3000 chars):\n{code[:3000]}"
    )
    fixed = chat([{"role": "user", "content": prompt}], task_type="coder", flow_id=flow_id)
    fixed = _strip_fences(fixed)
    if not fixed.strip():
        # Context budget exhausted — thinking tokens consumed all of num_predict. Use minimal prompt.
        _mini = (
            f"Task: {description[:200]}\n"
            f"Error: {error_output[:400]}\n"
            "Write a Python build script that creates the required files in /work/ using "
            "pathlib.Path.write_text(). Return ONLY raw Python code."
        )
        fixed = _strip_fences(chat([{"role": "user", "content": _mini}],
                                   task_type="coder", max_tokens=4096, flow_id=flow_id))
    return fixed


def _fix_bash_pentest(script: str, error_output: str, description: str,
                      flow_id: str | None,
                      expected_artifacts: list[str] | None = None) -> str:
    """Fix prompt for pentester bash scripts — emphasizes real tool execution."""
    from core.config import SSH_HOST, SSH_USER
    artifacts_block = ""
    if expected_artifacts:
        paths = "\n".join(f"  /work/{a}" for a in expected_artifacts)
        artifacts_block = f"\nREQUIRED OUTPUT FILES (create these exact paths):\n{paths}\n"
    prompt = (
        "Fix this Kali Linux bash script that failed. Return ONLY corrected bash — no markdown.\n"
        "CRITICAL: Run ACTUAL Kali tools against the target. NEVER write synthetic/fake data.\n"
        "An empty output file is VALID — it means 'no findings'. Do NOT fill empty files with made-up data.\n"
        f"TARGET: {SSH_HOST}  SSH: {SSH_USER}@{SSH_HOST}  (key: /root/.ssh/id_ed25519)\n"
        f"{artifacts_block}"
        "TOOL RULES:\n"
        "- nmap: -oN file.txt -oX file.xml  (NO -oJ flag — nmap has no JSON output)\n"
        "- nuclei: nuclei -as -target TARGET -o file.txt || true  (text output only, no JSON flag)\n"
        "- gobuster: gobuster dir ... || true; if empty: echo 'No findings' >> file.txt\n"
        "- searchsploit: searchsploit --colour never SERVICE VERSION 2>&1 | tee file.txt || true\n"
        "  (NEVER wrap searchsploit output in JSON — ANSI codes corrupt it)\n"
        "- ALL output paths end in .txt or .xml — NEVER .json for scan output\n"
        "- After each tool: verify the file was created, add 'no findings' line if empty\n"
        "- Use '|| true' after every scan tool to prevent exit on empty results\n\n"
        f"TASK: {description}\n\n"
        f"ERROR:\n{error_output[:2000]}\n\n"
        f"FAILING SCRIPT:\n{script[:8000]}"
    )
    fixed = chat([{"role": "user", "content": prompt}], task_type="coder", flow_id=flow_id)
    return _strip_fences(fixed)

def _fix_bash(script: str, error_output: str, description: str, flow_id: str | None) -> str:
    prompt = (
        "Fix this bash script that failed. Do NOT use heredocs. Return ONLY corrected bash — no markdown.\n"
        "If error is 'externally-managed-environment': add --break-system-packages to all pip calls.\n"
        "pip install syntax: pip3 install --break-system-packages --prefer-binary PKG\n\n"
        f"TASK: {description}\n\n"
        f"ERROR OUTPUT:\n{error_output[:2000]}\n\n"
        f"FAILING SCRIPT:\n{script[:3000]}"
    )
    fixed = chat([{"role": "user", "content": prompt}], task_type="coder", flow_id=flow_id)
    return _strip_fences(fixed)


def _python_syntax_error(code: str) -> str | None:
    try:
        ast.parse(code, filename="/tmp/ollamagi_task.py")
    except SyntaxError as exc:
        line = (exc.text or "").rstrip()
        pointer = " " * max((exc.offset or 1) - 1, 0) + "^"
        return (
            f"Python build-script syntax error at line {exc.lineno}: {exc.msg}\n"
            f"{line}\n{pointer}"
        )
    return None


# ── Memory agent ──────────────────────────────────────────────────────────────
def _memory_distill(flow_id: str, task_title: str, result: str) -> int:
    """Extract key facts from a task result and store in fractal memory."""
    if not result or len(result) < 60:
        return 0
    prompt = (
        f"Extract 1-3 concise, reusable facts or learnings from this task result.\n"
        f"Task: {task_title}\nResult: {result[:2000]}\n\n"
        "Return ONLY a JSON array of strings (the facts). No markdown. Max 3 items."
    )
    try:
        raw = chat(
            [{"role": "user", "content": prompt}],
            task_type="fast",
            flow_id=flow_id,
            max_tokens=256,
            timeout_s=60,
        )
    except Exception:
        return 0
    raw = _strip_fences(raw.strip())
    try:
        facts = json.loads(raw)
        if isinstance(facts, list):
            count = 0
            for f in facts[:3]:
                if isinstance(f, str) and len(f) > 20:
                    store_belief(f, flow_id=flow_id, confidence=0.72)
                    count += 1
            return count
    except Exception:
        pass
    return 0


# ── Subtask execution with auto-fix retry ─────────────────────────────────────
def _execute_subtask(subtask: Subtask, flow: Flow, task: Task,
                     history: list[dict], log_fn: Callable,
                     flow_id: str | None = None) -> str:
    mem_ctx = context_for_task(subtask.description)
    deliverable_kind, expected_artifacts = _subtask_contract(subtask)
    contract_text = (
        f"DELIVERABLE CONTRACT: kind={deliverable_kind}; expected="
        f"{', '.join('/work/' + path for path in expected_artifacts) or 'no fixed path'}"
    )
    import datetime as _dt
    _now_str = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    extra_ctx = (
        f"CURRENT DATE/TIME: {_now_str}\n"
        f"FLOW: {flow.objective}\nTASK: {task.title}\n"
        f"{contract_text}\n"
        f"NON-NEGOTIABLE CONSTRAINTS:\n"
        f"{_objective_constraints(flow.objective, flow.flow_type)}\n"
        f"{_workspace_inventory(flow.id)}\n"
    )
    if mem_ctx:
        extra_ctx += f"\n{mem_ctx}"

    direct_bundle_targets = (
        expected_artifacts
        if deliverable_kind in {"configuration", "dependency"}
        and expected_artifacts
        and all(
            Path(path).suffix.lower() in {
                ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env", ".txt"
            }
            or Path(path).name in _DEPENDENCY_NAMES
            or Path(path).name.startswith(".env")
            for path in expected_artifacts
        )
        else []
    )
    if direct_bundle_targets:
        before = _workspace_snapshot(flow.id)
        prompt = (
            "Return a single valid JSON object mapping each requested /work-relative path to its "
            "complete text file contents. Values must be JSON strings. No fences or explanation.\n"
            f"SUBTASK: {subtask.title}\n{subtask.description}\n"
            f"FLOW OBJECTIVE: {flow.objective}\n"
            f"PATHS: {json.dumps(direct_bundle_targets)}\n"
            f"CONSTRAINTS:\n{_objective_constraints(flow.objective, flow.flow_type)}"
        )
        bundle = None
        bundle_error = ""
        for bundle_attempt in range(MAX_RETRIES + 1):
            raw = chat(
                history + [{"role": "user", "content": prompt + (
                    f"\nPREVIOUS JSON ERROR: {bundle_error}" if bundle_error else ""
                )}],
                task_type="analysis",
                flow_id=flow_id,
                max_tokens=4000,
                think=False,
            )
            try:
                candidate = json.loads(_strip_fences(raw))
                if not isinstance(candidate, dict):
                    raise ValueError("bundle is not an object")
                for rel in direct_bundle_targets:
                    content = candidate.get(rel)
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError(f"missing content for {rel}")
                bundle = candidate
                break
            except Exception as exc:
                bundle_error = str(exc)
        if bundle is None:
            subtask.validation = f"direct artifact bundle was invalid: {bundle_error}"
            return f"[FAILED — validation]\n{subtask.validation}"
        for rel in direct_bundle_targets:
            target = WORKSPACE_DIR / flow.id / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(bundle[rel].rstrip() + "\n")
        subtask.attempts = 1
        valid, artifacts, validation = _validate_execution(
            flow.id, subtask, before, "Wrote configuration/dependency artifact bundle"
        )
        subtask.artifacts = artifacts
        subtask.validation = validation
        if not valid:
            return f"[FAILED — validation]\n{validation}"
        return (
            "Wrote " + ", ".join(f"/work/{path}" for path in direct_bundle_targets)
            + f"\n\n[VALIDATION] {validation}"
        )

    direct_text_target = (
        expected_artifacts[0]
        if deliverable_kind in {"documentation", "report"}
        and len(expected_artifacts) == 1
        and Path(expected_artifacts[0]).suffix.lower() in {".md", ".txt", ".rst", ".html"}
        else None
    )
    if direct_text_target:
        before = _workspace_snapshot(flow.id)
        prompt = (
            f"Create the complete contents of /work/{direct_text_target} for this subtask.\n"
            f"SUBTASK: {subtask.title}\n{subtask.description}\n\n"
            f"FLOW OBJECTIVE: {flow.objective}\n"
            f"{_workspace_inventory(flow.id)}\n\n"
            "Return ONLY the final file contents. Do not include Markdown code fences, preambles, "
            "claims about saving the file, shell commands, or Python writer code."
        )
        content = chat(
            history + [{"role": "user", "content": prompt}],
            task_type="analysis",
            flow_id=flow_id,
            max_tokens=4000,
            think=False,
        )
        content = _strip_fences(content)
        if not content.strip():
            subtask.validation = "direct artifact generation produced no content"
            return f"[FAILED — validation]\n{subtask.validation}"
        target = WORKSPACE_DIR / flow.id / direct_text_target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content.rstrip() + "\n")
        subtask.attempts = 1
        valid, artifacts, validation = _validate_execution(
            flow.id, subtask, before, f"Wrote /work/{direct_text_target}"
        )
        subtask.artifacts = artifacts
        subtask.validation = validation
        if not valid:
            return f"[FAILED — validation]\n{validation}"
        return f"Wrote /work/{direct_text_target}\n\n[VALIDATION] {validation}"

    if not subtask.needs_container:
        # Include real data files from the workspace so the agent doesn't hallucinate
        work_data = _read_work_data(flow.id)
        if work_data:
            extra_ctx += f"\n\nWORKSPACE DATA FILES (USE THIS DATA — do NOT invent values):\n{work_data}"
        extra_ctx += (
            "\nIMPORTANT: This is a text-only reasoning call. You cannot execute shell commands, "
            "read file contents, or create files. Use ONLY the data provided in WORKSPACE DATA FILES "
            "above — never invent metrics, statistics, or findings. If data is missing, say so explicitly."
        )
        import datetime as _dt3
        _date_banner = f"[TODAY IS {_dt3.datetime.now().strftime('%Y-%m-%d')}. Use this exact date in all output — never write 2023 or any other year.]\n\n"
        messages = history + [{"role": "user", "content": _date_banner + subtask.description}]
        result = run_agent(subtask.agent, messages, extra_ctx, flow_id=flow_id)
        valid, validation = _validate_text_result(result)
        subtask.validation = validation
        if not valid:
            return f"[FAILED — validation]\n{validation}"
        return result

    use_python = subtask.container_type == "python" or subtask.agent in ("coder", "installer")
    before = _workspace_snapshot(flow.id)

    # Injected before every Python build script: installs /work/requirements.txt so
    # later subtasks (e.g. "run self-test") can import packages declared by earlier ones.
    import datetime as _dt2
    _today_var = _dt2.datetime.now().strftime("%Y-%m-%d")
    _ts_var    = _dt2.datetime.now().strftime("%Y-%m-%d %H:%M")
    _REQ_PREAMBLE = (
        f"# ── Real current date (do NOT use any other date in output) ──\n"
        f"import datetime as _datetime_mod\n"
        f"TODAY = '{_today_var}'  # YYYY-MM-DD\n"
        f"NOW   = '{_ts_var}'    # YYYY-MM-DD HH:MM\n\n"
        "import subprocess as _sp, sys as _sys, pathlib as _pl\n"
        "_req = _pl.Path('/work/requirements.txt')\n"
        "if _req.exists() and _req.stat().st_size > 0:\n"
        "    _sp.run([_sys.executable, '-m', 'pip', 'install', '--prefer-binary',\n"
        "             '--break-system-packages', '-q',\n"
        "             '-r', str(_req)], check=False, capture_output=True)\n"
        "del _sp, _sys, _pl, _req\n\n"
        # web_search() helper: SearxNG primary (better quality, no rate-limits),
        # DuckDuckGo as fallback. Agents just call web_search('query').
        "def web_search(query, max_results=10, fetch_pages=False):\n"
        "    import requests as _rq, json as _js\n"
        "    results = []\n"
        "    try:\n"
        "        r = _rq.get('http://host.docker.internal:4000/search',\n"
        "                    params={'q': query, 'format': 'json', 'language': 'en'},\n"
        "                    timeout=12)\n"
        "        results = [{'title': x.get('title',''), 'href': x.get('url',''),\n"
        "                    'body': x.get('content','')} for x in r.json().get('results',[])][:max_results]\n"
        "    except Exception:\n"
        "        pass\n"
        "    if not results:\n"
        "        try:\n"
        "            from duckduckgo_search import DDGS\n"
        "            results = list(DDGS().text(query, max_results=max_results))\n"
        "        except Exception:\n"
        "            pass\n"
        "    if fetch_pages and results:\n"
        "        import requests as _rq2\n"
        "        from bs4 import BeautifulSoup as _BS\n"
        "        for item in results:\n"
        "            try:\n"
        "                html = _rq2.get(item['href'], timeout=8,\n"
        "                               headers={'User-Agent': 'Mozilla/5.0'}).text\n"
        "                soup = _BS(html, 'lxml')\n"
        "                for tag in soup(['script','style','nav','footer']):\n"
        "                    tag.decompose()\n"
        "                item['page_text'] = ' '.join(soup.get_text().split())[:4000]\n"
        "            except Exception:\n"
        "                item['page_text'] = ''\n"
        "    return results\n\n"
    )

    # Inject data file content so code agents know the exact JSON/CSV schemas they'll read.
    # Without this, they guess structures blind and fail silently on wrong keys/indices.
    _work_data_ctx = _read_work_data(flow.id, max_bytes=8000)

    _data_section = (
        f"WORKSPACE DATA FILES — read these exact structures, do not guess schemas:\n{_work_data_ctx}\n\n"
        if _work_data_ctx else ""
    )

    if use_python:
        code_prompt = (
            f"TODAY IS {_ts_var}. Use this exact date/time in any reports, headers, or filenames.\n\n"
            f"Write a Python 3 script that accomplishes this subtask:\n\n"
            f"{subtask.description}\n\n"
            f"{contract_text}\n\n"
            f"{_data_section}"
            "Environment: Python 3.11 container. Pre-installed: requests, httpx, aiohttp, rich, "
            "beautifulsoup4, lxml, python-dotenv, pyyaml, toml, psutil, loguru, colorama, "
            "selenium, webdriver-manager, duckduckgo-search.\n"
            "Any /work/requirements.txt present from earlier subtasks is automatically "
            "installed before your script runs — no need to pip-install those again.\n"
            "GIT: git is NOT pre-installed. Install it safely (no check=True for apt-get):\n"
            "  import subprocess\n"
            "  subprocess.run(['apt-get','update','-qq'], capture_output=True)\n"
            "  subprocess.run(['apt-get','install','-y','-qq','git'], capture_output=True)\n"
            "  subprocess.run(['git','clone','https://github.com/user/repo','/work/repo'], check=True)\n"
            "  assert pathlib.Path('/work/repo').is_dir(), 'clone failed'\n"
            "  # To skip re-cloning if already present: if not Path('/work/repo').exists(): ...\n"
            "  # GitHub repo names are CASE-SENSITIVE. OllamAGI repo: https://github.com/Linutesto/ollamagi\n\n"
            "WEB SEARCH: A web_search() function is pre-injected into every script.\n"
            "  results = web_search('your query here', max_results=10)\n"
            "  # Each result: {'title': str, 'href': str, 'body': str}\n"
            "  # To also fetch and extract the full page text of results:\n"
            "  results = web_search('your query here', max_results=5, fetch_pages=True)\n"
            "  # Then access result['page_text'] for full article content\n"
            "  # Uses SearxNG (aggregates Google+Bing+DDG) with DuckDuckGo as fallback\n\n"
            "Rules:\n"
            "- Save ALL output files to /work/ using open('/work/filename', 'w')\n"
            "- You are writing a BUILD SCRIPT, not merely the final application body\n"
            "- The BUILD SCRIPT must be valid Python before execution\n"
            "- NEVER place generated multi-line source inside triple-quoted strings; nested "
            "docstrings will break the build script. Use lists of ordinary quoted lines joined "
            "with '\\n' or JSON-decoded string literals. Do not use triple quotes anywhere\n"
            "- PYTHON SYNTAX: `with` / `async with` blocks do NOT support `else`. Only `for`, `while`, "
            "and `try` blocks support `else`. This is a common mistake — never write `async with ...: ... else: ...`\n"
            "- Variables that must be used after a `with` block must be declared before it or assigned "
            "outside the block (e.g. `result = None` before `with`, then `result = value` inside)\n"
            "- For implementation tasks, the build script MUST write the requested source files "
            "(.py/.js/etc.) into /work; executing code only from /tmp does not count\n"
            "- Preserve and improve existing /work source files instead of replacing them with placeholders\n"
            "- Persist every third-party runtime dependency in /work/requirements.txt or pyproject.toml; "
            "temporary pip installation alone does not count\n"
            "- Keep SDK imports and distributions consistent. For Telegram prefer "
            "python-telegram-bot>=21,<22 with imports from telegram/telegram.ext. Never install the "
            "unrelated `telegram` package and never mix python-telegram-bot with telebot/pyTelegramBotAPI\n"
            "- Install missing packages with pip — MUST include --break-system-packages (required on modern Debian):\n"
            "  import sys, subprocess\n"
            "  subprocess.run([sys.executable, '-m', 'pip', 'install', '--prefer-binary', '--break-system-packages', 'pkg'], check=False, capture_output=True)\n"
            "- NEVER use pydantic v2 — use pydantic v1 (pip install 'pydantic<2') or plain dataclasses instead\n"
            "- NEVER use packages that require Rust compilation (polars, cryptography>=42, pydantic-core, etc.)\n"
            "- Prefer packages that have pre-built wheels: requests, httpx, bs4, lxml, playwright, selenium\n"
            "- Print progress to stdout so the user can track execution\n"
            "- Inspect /work before assuming an input filename or schema\n"
            "- Validate loaded JSON types before iterating; inputs may be lists or objects\n"
            "- Create parent directories before writing files\n"
            "- Exit non-zero when required output cannot be produced; never print Error and exit 0\n"
            "- This is a BUILD/TEST subtask: the script MUST terminate on its own\n"
            "- NEVER run an infinite loop, daemon, server, scheduler, or long-lived bot process\n"
            "- CRITICAL STRUCTURE RULE: You are writing a BUILD SCRIPT that runs as /tmp/ollamagi_task.py. "
            "The build script's job is to CREATE /work/crawler.py (or whatever the deliverable is) as a "
            "separate file. The build script itself is NOT the final application.\n"
            "  CORRECT: build script writes the crawler to /work/crawler.py using pathlib.Path.write_text()\n"
            "  WRONG: build script contains argparse, main(), or CLI logic at module level — that makes\n"
            "  the build script the application, so running it without args shows help and exits 1.\n"
            "- IMPORTANT: If the deliverable_kind is 'source', your job is to WRITE the source files to /work "
            "and verify them with ast.parse — do NOT run the generated application. The orchestrator runs "
            "it separately with the correct flags. Running it without args will always fail.\n"
            "- If you must run a generated application to validate it, ALWAYS pass '--self-test' or "
            "'--dry-run': subprocess.run([sys.executable, '/work/app.py', '--self-test'], ...). "
            "NEVER invoke it without arguments — it will error because it needs a URL or target.\n"
            "- For autonomous apps, implement a bounded --self-test or --dry-run mode and execute one cycle only\n"
            "- Credentialed integrations such as Telegram, Discord, or Slack MUST make self-test/dry-run "
            "fully offline: test handlers with fake update/message objects and never start polling, "
            "construct a live client session, validate a token remotely, or contact provider APIs\n"
            "- Use placeholder/default configuration safely; do not repeatedly retry missing credentials\n"
            "- Never contact invented domains such as api.example.com or api.trading-platform.com\n"
            "- Never require API keys for validation; use paper mode, local fixtures, mocks, or public "
            "unauthenticated market-data endpoints\n"
            "- Do not call private/account endpoints (balances, currencies, orders) during validation\n"
            "- For scrapers/crawlers, validate against local HTML fixtures or a local temporary HTTP server; "
            "do not depend on test.org/httpbin.org/external network availability\n"
            "- Scraper/crawler deliverables must expose a bounded --self-test or equivalent local-fixture mode\n"
            "- Expected HTTP failures (404, timeout fixtures) must be handled and logged but should not make "
            "the build script fail when recovery behavior is the feature under test\n"
            "- Validate the actual generated source file from /work, not a separate throwaway implementation\n"
            "- Wrap the main logic in try/except and print any errors clearly\n"
            "- Return ONLY raw Python code — NO markdown fences, NO explanation"
        )
        # For source deliverables: explicitly tell the coder to write the file, not run it
        if subtask.deliverable_kind == "source" and subtask.expected_artifacts:
            _src_paths = "\n".join(f"  pathlib.Path('/work/{a}').write_text(code_str)" for a in subtask.expected_artifacts)
            code_prompt += (
                "\n\nSOURCE DELIVERABLE — CRITICAL RULES:\n"
                "Your job is to WRITE the source files into /work/, not to execute their logic.\n"
                "Step 1: Write a stub of each file in the FIRST lines of your build script:\n"
                + "\n".join(f"  pathlib.Path('/work/{a}').write_text('# stub')" for a in subtask.expected_artifacts)
                + "\nStep 2: Generate the final content as a Python string, then overwrite:\n"
                + _src_paths
                + "\nStep 3: Validate with ast.parse() for .py files. Never run the generated app directly."
            )
        messages = history + [{"role": "user", "content": code_prompt}]
        # Always use a code-generation agent for Python execution — planning agents produce
        # orchestration text, not runnable scripts
        _CODE_AGENTS = {"coder", "installer", "data_engineer", "devops", "pentester"}
        code_gen_agent = subtask.agent if subtask.agent in _CODE_AGENTS else "coder"
        code = _strip_fences(run_agent(code_gen_agent, messages, extra_ctx, flow_id=flow_id))
        if not code.strip():
            # Model returned empty — thinking consumed all tokens or context was too large.
            # Retry with a minimal self-contained prompt (no history, no extra_ctx).
            _artifacts_hint = ", ".join(f"/work/{a}" for a in (subtask.expected_artifacts or []))
            _mini_prompt = (
                f"Task: {subtask.description[:400]}\n"
                f"Write files: {_artifacts_hint or 'required /work/ files'}\n"
                "Write a Python build script that creates these files with pathlib.Path.write_text().\n"
                "Return ONLY raw Python code. No markdown."
            )
            code = _strip_fences(run_agent(code_gen_agent,
                                           [{"role": "user", "content": _mini_prompt}],
                                           flow_id=flow_id))

        for attempt in range(MAX_RETRIES + 1):
            subtask.attempts = attempt + 1
            container = None
            try:
                syntax_error = _python_syntax_error(code)
                if syntax_error:
                    if attempt < MAX_RETRIES:
                        log_fn(
                            f"  ⚠ generated build script has invalid Python syntax — "
                            f"auto-fixing (attempt {attempt+1}/{MAX_RETRIES})…",
                            "warn",
                        )
                        code = _fix_python(
                            code, syntax_error, subtask.description, flow_id
                        )
                        continue
                    return f"[FAILED after {attempt+1} attempts]\n{syntax_error}"
                container = create_container(flow.id, f"{subtask.id}-a{attempt}", subtask.container_type)
                exit_code, output = exec_python(container, _REQ_PREAMBLE + code, timeout=MAX_TASK_TIMEOUT)
                execution_failed = _execution_failed(exit_code, output)
                if not execution_failed:
                    valid, artifacts, validation = _validate_execution(
                        flow.id, subtask, before, output
                    )
                    subtask.artifacts = artifacts
                    subtask.validation = validation
                    if valid:
                        visible_output = output[-3500:] if output else ""
                        return f"{visible_output}\n\n[VALIDATION] {validation}".strip()
                    output = f"{output}\n\n{validation}".strip()
                elif exit_code == 0:
                    output += "\nDetected fatal error text despite exit code 0."
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
                    return f"[FAILED — container error]\n{e}"
            finally:
                if container:
                    stop_container(container)
        return "[FAILED — max retries exceeded]"

    else:
        is_pentest = subtask.container_type == "pentest" or subtask.agent == "pentester"
        if is_pentest:
            from core.config import SSH_HOST, SSH_USER
            script_prompt = (
                f"Write a bash script for this penetration testing subtask:\n{subtask.description}\n"
                f"{contract_text}\n"
                f"## Environment: Kali Linux — authorized to pentest {SSH_HOST}\n"
                "ALL tools PRE-INSTALLED — do NOT apt-get install. Do NOT fabricate data.\n"
                "Available: nmap masscan nikto gobuster ffuf nuclei sqlmap hydra john hashcat\n"
                "  searchsploit enum4linux smbclient rpcclient curl wget netcat openssl whatweb\n"
                f"## Target: {SSH_HOST}  |  SSH: {SSH_USER}@{SSH_HOST} (key: /root/.ssh/id_ed25519)\n"
                "## CRITICAL TOOL USAGE\n"
                "nmap:\n"
                "  - Use: nmap -T4 -sV -O -oN /work/reports/nmap.txt -oX /work/reports/nmap.xml TARGET\n"
                "  - nmap has NO JSON output flag (-oJ does not exist). Use -oA for all 3 formats.\n"
                "  - Never write or echo fake nmap JSON — run the real tool\n"
                "nuclei:\n"
                "  - nuclei -as -target TARGET -o /work/reports/nuclei.txt || true\n"
                "  - '-as' = auto-scan mode. Output is TEXT (.txt). nuclei has NO JSON output flag.\n"
                "  - If -as is unavailable: nuclei -t /usr/share/nuclei-templates/ -target TARGET -o FILE || true\n"
                "gobuster:\n"
                "  - gobuster dir -u http://TARGET:PORT -w /usr/share/wordlists/dirb/common.txt \\\n"
                "      -o /work/reports/gobuster.txt -t 20 --no-error || true\n"
                "  - gobuster exits non-zero when nothing found — always use || true\n"
                "  - If output file empty or missing: echo 'No directories found' >> /work/reports/gobuster.txt\n"
                "searchsploit:\n"
                "  - Parse nmap output to find service+version, then run:\n"
                "    searchsploit --colour never SERVICE VERSION 2>&1 | tee /work/reports/searchsploit.txt || true\n"
                "  - Output is PLAIN TEXT. NEVER wrap in JSON or python — ANSI codes break JSON.\n"
                "  - If no results: echo 'searchsploit: no exploits found for installed versions' >> file.txt\n"
                "nikto:\n"
                "  - nikto -h http://TARGET:PORT -o /work/reports/nikto.txt -Format txt || true\n"
                "## OUTPUT RULES\n"
                "- ALL expected_artifacts paths end in .txt or .xml — NEVER .json for scan output\n"
                "- Write to EXACTLY the paths listed in DELIVERABLE CONTRACT (no renaming, no extension changes)\n"
                "- Create /work/reports/ before writing: mkdir -p /work/reports\n"
                "- Verify each expected file exists after writing:\n"
                "  for f in EXPECTED_PATHS; do [ -f \"$f\" ] || echo \"WARNING: $f not created\" >&2; done\n"
                "- Empty scan result = valid. Never fabricate data. If tool finds nothing:\n"
                "  echo 'RESULT: no findings' >> /work/reports/TOOL.txt\n"
                "- Exit non-zero ONLY on true failure (tool missing, host unreachable)\n"
                "- The script MUST terminate. No daemons, no msfconsole interactive sessions.\n"
                "Return ONLY the bash script — no markdown."
            )
        else:
            script_prompt = (
                f"Write a bash script for this subtask:\n{subtask.description}\n"
                f"{contract_text}\n"
                "Environment: Debian Linux container with apt-get. "
                "python3 and pip3 may not be pre-installed — if you need them, add: "
                "apt-get install -y -qq python3 python3-pip 2>/dev/null || true\n"
                "pip install MUST use --break-system-packages (modern Debian requires it):\n"
                "  pip3 install --break-system-packages --prefer-binary PKG\n"
                "Do NOT use heredocs. Do NOT embed Python code inline in bash.\n"
                "Do NOT assume any tool is available — install what you need via apt-get.\n"
                "Start with strict error handling. Inspect /work for equivalent input files before failing.\n"
                "Create parent directories before writing. Exit non-zero if required output is missing.\n"
                "The script MUST terminate. Never launch a daemon, server, watcher, scheduler, or infinite loop.\n"
                "For long-running applications, run only one bounded smoke-test cycle.\n"
                "Never require credentials or contact invented/private API endpoints during validation.\n"
                "Save all outputs to /work/. Return ONLY the bash script, no markdown."
            )
        messages = history + [{"role": "user", "content": script_prompt}]
        bash_gen_agent = subtask.agent if subtask.agent in _CODE_AGENTS else "coder"
        script = _strip_fences(run_agent(bash_gen_agent, messages, extra_ctx, flow_id=flow_id))

        for attempt in range(MAX_RETRIES + 1):
            subtask.attempts = attempt + 1
            container = None
            try:
                container = create_container(flow.id, f"{subtask.id}-a{attempt}", subtask.container_type)
                exit_code, output = exec_script(container, script, timeout=MAX_TASK_TIMEOUT)
                execution_failed = _execution_failed(exit_code, output)
                if not execution_failed:
                    valid, artifacts, validation = _validate_execution(
                        flow.id, subtask, before, output
                    )
                    subtask.artifacts = artifacts
                    subtask.validation = validation
                    if valid:
                        visible_output = output[-3500:] if output else ""
                        return f"{visible_output}\n\n[VALIDATION] {validation}".strip()
                    output = f"{output}\n\n{validation}".strip()
                elif exit_code == 0:
                    output += "\nDetected fatal error text despite exit code 0."
                if attempt < MAX_RETRIES:
                    log_fn(f"  ⚠ exit {exit_code} — auto-fixing bash (attempt {attempt+1}/{MAX_RETRIES})…", "warn")
                    if is_pentest:
                        script = _fix_bash_pentest(
                            script, output, subtask.description, flow_id,
                            expected_artifacts=subtask.expected_artifacts,
                        )
                    else:
                        script = _fix_bash(script, output, subtask.description, flow_id)
                else:
                    return f"[FAILED after {attempt+1} attempts]\n{output[-2000:]}"
            except Exception as e:
                if attempt < MAX_RETRIES:
                    log_fn(f"  ⚠ container error — retrying: {e}", "warn")
                else:
                    return f"[FAILED — container error]\n{e}"
            finally:
                if container:
                    stop_container(container)
        return "[FAILED — max retries exceeded]"


def _extract_missing_artifacts(report: str) -> list[str]:
    """Pull /work-relative paths of missing files from a validation report string."""
    # Match any /work/... path with any extension (including .log, .txt, .json, etc.)
    hits = re.findall(r'/work/([\w.\-/]+\.[\w]+)', report)
    # Deduplicate, exclude internal meta-files
    _meta = {"flow_log.jsonl", "llm_calls.jsonl"}
    seen: dict[str, None] = {}
    for h in hits:
        if h not in _meta:
            seen[h] = None
    return list(seen)


def _repair_final_deliverables(
    flow: Flow,
    validation_report: str,
    log_fn: Callable,
    max_repairs: int = 2,
) -> tuple[bool, str]:
    """Give the coder bounded chances to repair the real workspace after final validation."""
    report = validation_report
    for repair_index in range(1, max_repairs + 1):
        # Extract the exact missing paths so the repair subtask is contracted to them
        # and the coder is told explicitly what to create.
        missing_artifacts = _extract_missing_artifacts(report)
        missing_list = (
            "\n".join(f"  - /work/{p}" for p in missing_artifacts)
            if missing_artifacts
            else "(see report above)"
        )

        task = Task(
            id=f"repair-{repair_index}",
            flow_id=flow.id,
            title=f"Repair final deliverables ({repair_index}/{max_repairs})",
            description=(
                "Inspect the existing /work project, fix every issue from deterministic final "
                "validation, and run a bounded offline smoke test. Modify the actual project files; "
                "do not create a separate replacement implementation."
            ),
            agent="coder",
            status="running",
            started_at=time.time(),
        )
        subtask = Subtask(
            id=f"{task.id}-s1",
            task_id=task.id,
            title="Repair and revalidate workspace",
            description=(
                f"Deterministic validation failed with:\n{report}\n\n"
                f"CRITICAL — these files are MISSING and MUST be created by this script:\n"
                f"{missing_list}\n\n"
                "Write a Python script that:\n"
                "1. Creates every missing file listed above with real, non-empty content\n"
                "2. Repairs any other issues from the validation report (imports, entrypoints, "
                "configuration, documentation, dependency manifests)\n"
                "3. Verifies every missing file now exists at the correct path before exiting\n\n"
                "The script must write the missing files directly — do not just run the original "
                "validation script and hope it creates them. Use open(..., 'w') or pathlib.Path.write_text()."
            ),
            agent="coder",
            needs_container=True,
            container_type="python",
            deliverable_kind="artifact",
            expected_artifacts=missing_artifacts,
        )
        task.subtasks = [subtask]
        flow.tasks.append(task)
        flow.repair_count += 1
        log_fn(f"Final validation repair {repair_index}/{max_repairs} started", "warn")
        result = _execute_subtask(subtask, flow, task, [], log_fn, flow.id)
        failed = result.startswith("[FAILED") or _execution_failed(0, result)
        subtask.result = result
        subtask.status = "failed" if failed else "finished"
        subtask.finished_at = time.time()
        task.status = subtask.status
        task.result = result
        task.finished_at = time.time()
        _save(flow)

        valid, report = _validate_flow_deliverables(flow)
        if valid:
            log_fn(f"Final validation repair {repair_index} succeeded", "warn")
            return True, report
        log_fn(f"Final validation repair {repair_index} did not resolve all issues", "error")
    return False, report


# ── Main flow runner ──────────────────────────────────────────────────────────
def run_flow(objective: str, flow_type: str | None = None,
             broadcast: Callable | None = None,
             tasks: list[dict] | None = None,
             base_flows: list[str] | None = None,
             compact_context: bool = False) -> Flow:
    flow_id = uuid.uuid4().hex[:8]
    detected_type = flow_type or _detect_flow_type(objective)
    title = objective[:60] + ("..." if len(objective) > 60 else "")

    flow = Flow(id=flow_id, title=title, objective=objective, flow_type=detected_type,
                compact_context=compact_context)
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

    def _on_llm_call(entry: dict):
        # Write full entry (including messages) to llm_calls.jsonl
        try:
            llm_log_path = WORKSPACE_DIR / flow_id / "llm_calls.jsonl"
            with llm_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        # Broadcast a lightweight version via WS (truncate messages/response for network)
        if broadcast:
            ws_entry = {
                **entry,
                "messages": [
                    {**m, "content": m.get("content", "")[:600]}
                    for m in (entry.get("messages") or [])
                ],
                "response": (entry.get("response") or "")[:8000],
            }
            broadcast(ws_entry)

    register_llm_callback(flow_id, _on_llm_call)

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

    # Resolve cross-flow references (explicit + auto-detected from objective)
    auto_ref = _detect_referenced_flow_ids(objective)
    explicit_ref = [r for r in (base_flows or []) if r]
    ref_flow_ids = list(dict.fromkeys(explicit_ref + auto_ref))  # dedup, explicit first
    cross_ctx = ""
    if ref_flow_ids:
        cross_ctx = _build_cross_flow_context(ref_flow_ids)
        _copy_referenced_workspaces(flow_id, ref_flow_ids)
        log(f"Cross-flow context: {len(ref_flow_ids)} project(s) loaded — {', '.join(r[:8] for r in ref_flow_ids)}")

    mem_ctx = context_for_task(objective)
    if mem_ctx:
        log(f"Memory: {len(mem_ctx.splitlines())} relevant entries loaded")

    # Combined context: cross-flow first so agents see referenced code before general knowledge
    mem_ctx = "\n\n".join(filter(None, [cross_ctx, mem_ctx]))

    try:
        flow.status = "running"
        if tasks:
            flow.tasks = [
                Task(
                    id=f"t{td.get('id', i + 1)}",
                    flow_id=flow.id,
                    title=td.get("title", f"Task {i + 1}"),
                    description=td.get("description", ""),
                    agent=_role_for_flow(
                        td.get("agent") or (
                            "pentester" if td.get("type") == "pentest" else "coder"
                        ),
                        flow.flow_type,
                    ),
                )
                for i, td in enumerate(_object_list(tasks))
            ]
            if not flow.tasks:
                raise ValueError("preplanned task list contained no valid task objects")
            log(f"primary_agent: using {len(flow.tasks)} preplanned tasks")
        else:
            log("primary_agent: decomposing objective…")
            for _attempt in range(3):
                try:
                    flow.tasks = _generate_tasks(flow, mem_ctx)
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
                    task.subtasks = _generate_subtasks(task, flow, mem_ctx)
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
                    _hist = (
                        _compact_history(conversation_history, flow_id)
                        if flow.compact_context
                        else _trim_history(conversation_history)
                    )
                    result = _execute_subtask(
                        subtask, flow, task, _hist, log, flow_id
                    )
                    failed = result.startswith("[FAILED") or _execution_failed(0, result)
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
                        # Text agents produce rich knowledge — store immediately
                        if not subtask.needs_container and subtask.result and len(subtask.result) > 80:
                            _n = _memory_distill(flow_id, subtask.title, subtask.result)
                            if _n:
                                flow.memory_items_stored += _n
                                log(f"  memory: {_n} fact(s) from {subtask.agent}")
                except Exception as e:
                    subtask.result = str(e)
                    subtask.status = "failed"
                    task_had_failure = True
                    log(f"  ✗ {subtask.title[:60]}: {e}", "error")

                subtask.finished_at = time.time()
                task_results.append({
                    "subtask": subtask.title,
                    "status": subtask.status,
                    "result": subtask.result[:1000],
                    "artifacts": list(subtask.artifacts),
                    "validation": subtask.validation,
                })
                conversation_history.append({
                    "role": "assistant",
                    "content": f"Subtask '{subtask.title}': {subtask.result[:500]}"
                })
                _save(flow)

            task.status = "failed" if task_had_failure else "finished"
            recovered_task, recovery_report, recovery_artifacts = _reconcile_task_status(
                task, flow
            )
            if recovered_task:
                task.status = "finished"
                task_had_failure = False
                log(
                    f"task [{task.id}] recovered by later validated deliverables: "
                    + ", ".join(f"/work/{path}" for path in recovery_artifacts[:10]),
                    "warn",
                )
            evidence_lines = []
            for result in task_results:
                artifact_text = (
                    ", ".join(f"/work/{path}" for path in result["artifacts"])
                    if result["artifacts"] else "none"
                )
                evidence_lines.append(
                    f"- {result['subtask']} | status={result['status']} | "
                    f"artifacts={artifact_text} | validation={result['validation'] or 'none'} | "
                    f"output={result['result'][:300]}"
                )
            synthesis_prompt = (
                "Produce a concise evidence-based task summary.\n"
                "Do not claim a file, feature, test, or capability exists unless it appears "
                "in the evidence below. Explicitly report failed validation.\n\n"
                f"TASK: {task.title}\n"
                f"DETERMINISTIC STATUS: {task.status}\n"
                f"EVIDENCE:\n" + "\n".join(evidence_lines)
            )
            narrative = run_agent(
                "primary_agent",
                [{"role": "user", "content": synthesis_prompt}],
                flow_id=flow_id,
            )
            verified_artifacts = sorted({
                path for result in task_results
                if result["status"] == "finished"
                for path in result["artifacts"]
            })
            task.result = (
                f"VERIFIED STATUS: {task.status.upper()}\n"
                f"VERIFIED ARTIFACTS: "
                f"{', '.join('/work/' + p for p in sorted(set(verified_artifacts + recovery_artifacts))) or 'none'}\n"
                + (f"RECOVERY: {recovery_report}\n" if recovered_task else "")
                + "\n"
                f"{narrative}"
            )
            task.finished_at = time.time()
            log(f"task [{task.id}] {'failed' if task_had_failure else 'done'}")
            all_results.append({
                "task": task.title,
                "result": task.result,
                "status": "success" if task.status == "finished" else "failed",
            })

            if task.status == "finished":
                n = _memory_distill(flow_id, task.title, task.result)
                if n:
                    flow.memory_items_stored += n
                    log(f"  memory agent: {n} belief(s) stored in fractal memory")
                completed_tasks.append(task)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if (
                    consecutive_failures >= 2
                    and task_idx + 1 < len(flow.tasks)
                    and flow.replan_count < MAX_AUTO_REPLANS
                ):
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

        deliverables_valid, flow.validation = _validate_flow_deliverables(flow)
        if not deliverables_valid and not _is_stopped(flow.id):
            deliverables_valid, flow.validation = _repair_final_deliverables(
                flow, flow.validation, log
            )
        if deliverables_valid:
            log(flow.validation)
        else:
            log(flow.validation, "error")
            flow.error = flow.validation
        historical_failures = any(
            task.status == "failed"
            or any(subtask.status == "failed" for subtask in task.subtasks)
            for task in flow.tasks
        )
        if deliverables_valid and historical_failures:
            flow.validation += (
                " Earlier failed attempts remain in history but were superseded by the "
                "deterministically validated final workspace."
            )
            log("Final workspace validation superseded earlier failed attempts", "warn")
            _supersede_failed_attempts(flow)
        flow.status = "finished" if deliverables_valid else "failed"

        # Store validated flow knowledge in fractal memory
        if flow.status == "finished":
            log("Storing validated knowledge in fractal memory…")
            try:
                from core.fractal_memory import store_from_result
                for r in all_results:
                    if r.get("status") == "success" and r.get("result"):
                        store_from_result(
                            f"{r['task']}: {r['result'][:500]}",
                            flow_id=flow_id,
                            confidence=0.8,
                        )
                        flow.memory_items_stored += 1
                log(f"Stored {flow.memory_items_stored} items in fractal memory")
            except Exception as e:
                log(f"Fractal memory storage skipped: {e}", "warn")

        flow.finished_at = time.time()
        _save(flow)
        log(f"Flow {flow_id} {flow.status} — workspace: {WORKSPACE_DIR / flow_id}")
        if broadcast:
            broadcast({"type": "flow_done", "flow_id": flow_id,
                       "status": flow.status, "data": _flow_to_dict(flow)})

    except (SystemExit, KeyboardInterrupt, FlowStoppedException):
        _mark_stopped()

    except Exception as e:
        flow.status = "failed"
        flow.error = f"{type(e).__name__}: {e}"
        flow.finished_at = time.time()
        _save(flow)
        log(f"Flow failed: {e}", "error")
        if broadcast:
            broadcast({"type": "flow_done", "flow_id": flow_id,
                       "status": "failed", "data": _flow_to_dict(flow)})

    return flow
