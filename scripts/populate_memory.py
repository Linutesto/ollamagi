"""
Populate fractal memory with synthetic but useful agent knowledge.
Covers: Python, Docker, web, databases, security, AI patterns, deployment,
        testing, data, business, OllamAGI system knowledge.
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.fractal_memory import insert, stats

CORPUS = {

# ─────────────────────────────────────────────────────────────────────────────
"python_async": [
    "Use asyncio.gather() to run multiple coroutines concurrently; it returns results in order even if tasks finish out of order.",
    "asyncio.TaskGroup (Python 3.11+) cancels all sibling tasks if any one raises an exception — safer than gather() for critical flows.",
    "aiofiles provides async file I/O; never open() inside a coroutine without it or you block the event loop.",
    "Use asyncio.Semaphore to limit concurrent HTTP requests; without it 1000 concurrent aiohttp requests will exhaust file descriptors.",
    "httpx.AsyncClient should be reused across requests via async context manager, not recreated per request — reuse handles connection pooling.",
    "asyncio.wait_for() wraps a coroutine with a timeout and raises asyncio.TimeoutError on expiry — use this for any network call.",
    "Running sync code in an async context: loop.run_in_executor(None, blocking_func) offloads to a thread pool without blocking the event loop.",
    "async generators (async def + yield) let you stream large datasets without loading everything into memory.",
    "asyncio.Queue is the correct way to implement producer-consumer in async code; use maxsize to apply backpressure.",
],

"python_patterns": [
    "dataclasses.field(default_factory=list) is the correct way to set mutable defaults in dataclasses; never use default=[] directly.",
    "Use __slots__ in hot-path classes to reduce memory by 40-50% and speed attribute access by eliminating __dict__.",
    "functools.lru_cache(maxsize=None) caches pure function calls; use functools.cache as the simpler alias in Python 3.9+.",
    "pathlib.Path is safer than os.path for all file operations; Path.read_text() / write_text() replaces open+read+close patterns.",
    "contextlib.suppress(FileNotFoundError) is cleaner than try/except/pass for expected errors you want to ignore.",
    "Use itertools.islice() to take the first N items from any iterator without consuming it all.",
    "typing.Protocol enables structural subtyping — define what methods an object needs without inheritance.",
    "collections.defaultdict(list) is faster than checking 'if key in dict' before appending.",
    "subprocess.run(capture_output=True, text=True) is the modern replacement for Popen; check returncode, not just stdout.",
    "Use json.dumps(obj, default=str) to safely serialize objects with dates/datetimes/Paths without custom encoders.",
    "Walrus operator := in list comprehensions: [y for x in data if (y := transform(x)) is not None] avoids calling transform twice.",
    "contextlib.contextmanager lets you write context managers as generators — yield once, cleanup runs in finally block.",
],

"python_packaging": [
    "requirements.txt should pin exact versions (==) for reproducibility; use pip-tools or poetry for dependency resolution.",
    "pyproject.toml is the modern standard (PEP 517/518); setup.py is legacy — avoid for new projects.",
    "Use __all__ in __init__.py to control what 'from package import *' exports and to document the public API.",
    "importlib.resources (Python 3.9+) is the correct way to access package data files; never use __file__ hacks.",
    "Virtual environments: always use python -m venv, never install to system Python for projects.",
],

"docker_patterns": [
    "Multi-stage builds: use a build stage with all compilers, then COPY --from=build only the final binary to a slim runtime image.",
    "COPY . . always comes after RUN pip install to maximize cache hits — dependency layer rarely changes.",
    "Use .dockerignore to exclude .git, __pycache__, *.pyc, .env, node_modules — reduces build context by 90% on typical projects.",
    "Health check: HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8080/health || exit 1",
    "Never run containers as root; add 'USER nobody' or create a dedicated user with adduser --no-create-home.",
    "Docker volumes for databases: always use named volumes (not bind mounts) for /var/lib/postgresql/data to survive container restarts.",
    "Environment variable injection: use --env-file .env at runtime, never bake secrets into image layers.",
    "Resource limits: --memory=512m --cpus=0.5 prevent a rogue container from starving the host.",
    "docker compose depends_on with condition: service_healthy waits for the health check to pass, not just container start.",
    "Use ENTRYPOINT for the fixed command and CMD for default arguments — allows overriding args without rewriting the entrypoint.",
    "Alpine images are small but use musl libc which can cause subtle incompatibilities; debian-slim is safer for Python.",
    "exec form ENTRYPOINT ['python', 'app.py'] receives signals (SIGTERM) directly; shell form wraps in /bin/sh which swallows signals.",
],

"web_scraping": [
    "Always set a User-Agent header mimicking a real browser; many sites block the default requests/python-httpx user agent.",
    "httpx with http2=True enables HTTP/2 multiplexing — faster for sites that support it, fewer connections needed.",
    "Use BeautifulSoup with lxml parser (fastest): BeautifulSoup(html, 'lxml') — install lxml separately.",
    "Playwright is required for JavaScript-heavy sites; use page.wait_for_selector() not time.sleep() to wait for elements.",
    "Respect robots.txt and crawl-delay; parse it with urllib.robotparser.RobotFileParser.",
    "Rotate requests through a small delay (0.5-2s random sleep) to avoid rate limits and IP bans.",
    "CSS selectors in BeautifulSoup: soup.select('div.article > h2 a') — faster and more readable than find_all chains.",
    "For pagination, look for 'next' links or page parameters; use a seen-URLs set to avoid infinite loops.",
    "Store scraped data incrementally to disk (append to JSONL) — never hold all results in memory for large crawls.",
    "Playwright's page.evaluate() executes JavaScript in the page context — useful for extracting data from JS variables.",
    "Use fake_useragent library to rotate realistic user agent strings across requests.",
    "aiohttp with asyncio.Semaphore(10) gives async scraping with concurrency control — 10x faster than sync requests.",
],

"api_integration": [
    "Always implement exponential backoff with jitter for retries: wait = base * 2^attempt + random(0, 1).",
    "OAuth2: store refresh tokens securely; access tokens expire (usually 1h), refresh tokens last weeks/months.",
    "Use httpx.Client as a context manager to ensure connection pooling and cleanup: 'with httpx.Client() as client:'",
    "Rate limit headers: X-RateLimit-Remaining and Retry-After tell you when to pause; respect them to avoid 429 bans.",
    "Webhook security: always verify HMAC signatures (usually SHA256 of body with shared secret) before processing.",
    "Idempotency keys: include a unique ID per request for payment/mutation APIs — prevents duplicate charges on retry.",
    "API versioning: prefer URL versioning (/v1/) for stability; Accept header versioning works but is harder to test.",
    "Pagination patterns: cursor-based (next_cursor) is better than offset-based for large/changing datasets.",
    "Cache API responses with TTL based on Cache-Control headers; use requests_cache or a simple dict+timestamp.",
    "REST vs GraphQL: use REST for simple CRUD, GraphQL when you need to fetch related objects in one round trip.",
],

"sqlite_patterns": [
    "WAL mode (PRAGMA journal_mode=WAL) allows concurrent reads while writing — essential for any multi-threaded app.",
    "PRAGMA busy_timeout=5000 makes SQLite wait up to 5s instead of immediately raising 'database is locked'.",
    "Use parameterized queries always: cursor.execute('SELECT * FROM t WHERE id=?', (id,)) — never f-strings with user data.",
    "FTS5 virtual table: CREATE VIRTUAL TABLE t_fts USING fts5(content, content=t, content_rowid=id) with triggers for sync.",
    "JSON1 extension (built into SQLite 3.38+): json_extract(data, '$.key') lets you query JSON columns without deserializing in Python.",
    "Indexes on (created_at DESC) dramatically speed up 'ORDER BY created_at DESC LIMIT N' queries on large tables.",
    "sqlite3.connect(':memory:') for tests — no cleanup needed, isolated, and 10x faster than disk.",
    "VACUUM recompacts the database file after many deletes; run it periodically or use auto_vacuum=INCREMENTAL.",
    "row_factory = sqlite3.Row makes results accessible by column name: row['name'] instead of row[0].",
    "Transactions: wrap bulk inserts in 'with conn:' (context manager) for automatic commit/rollback.",
    "EXPLAIN QUERY PLAN SELECT ... shows whether SQLite is using your indexes — use this before adding indexes blindly.",
    "Partial indexes: CREATE INDEX idx ON t(col) WHERE status='active' — smaller and faster for filtered queries.",
],

"security_patterns": [
    "Never log passwords, API keys, tokens, or PII — sanitize exception messages before logging.",
    "Use secrets.token_hex(32) for generating secure random tokens; random.random() is NOT cryptographically secure.",
    "bcrypt or argon2-cffi for password hashing — never MD5/SHA1/SHA256 alone without salt and stretching.",
    "SQL injection: always use parameterized queries; never concatenate user input into SQL strings.",
    "SSRF prevention: whitelist allowed domains/IPs before making outbound requests based on user-supplied URLs.",
    "Path traversal: use Path(base_dir / user_path).resolve().relative_to(base_dir.resolve()) and catch ValueError.",
    "Environment variables for secrets, never hardcode: os.environ.get('API_KEY') with a clear error if missing.",
    "JWT: always verify signature and expiry; never decode without verification (jwt.decode(token, key, algorithms=['HS256'])).",
    "CORS: only whitelist specific origins, never use '*' for APIs that handle authentication.",
    "Rate limiting on auth endpoints: 5 failed attempts → 15m lockout — prevents brute force.",
    "Use subprocess with a list (not shell=True) to prevent shell injection: subprocess.run(['ls', user_path]).",
    "Content-Security-Policy header prevents XSS by restricting which scripts can run on your pages.",
],

"testing_patterns": [
    "pytest fixtures with scope='session' reuse expensive setup (DB connections, containers) across all tests.",
    "unittest.mock.patch as decorator: @patch('module.ClassName') replaces the class for the duration of the test.",
    "Use tmp_path pytest fixture for temp directories — automatically cleaned up, no manual teardown needed.",
    "Property-based testing with hypothesis: @given(st.text()) generates hundreds of inputs to find edge cases automatically.",
    "Parametrize tests: @pytest.mark.parametrize('input,expected', [(1,2),(2,4)]) runs the test with multiple inputs.",
    "pytest-asyncio for async tests: mark with @pytest.mark.asyncio and use async def test_foo().",
    "Mocking time: freezegun library's @freeze_time('2024-01-01') makes datetime.now() return a fixed value.",
    "Test isolation: each test should set up its own state and not rely on test execution order.",
    "Coverage: aim for 80%+ line coverage; 100% is diminishing returns — focus on critical paths.",
    "Integration tests against real dependencies (use testcontainers-python to spin up actual Postgres/Redis).",
],

"deployment_patterns": [
    "Systemd unit file: Type=notify with sd_notify lets systemd know when your service is ready (not just started).",
    "Use gunicorn with uvicorn workers for FastAPI production: gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app",
    "nginx as reverse proxy in front of uvicorn: handles SSL termination, static files, and connection limits.",
    "Health check endpoint /health returns 200 with {'status':'ok'} — used by load balancers and orchestrators.",
    "Graceful shutdown: handle SIGTERM to finish in-flight requests before exiting (FastAPI does this automatically).",
    "Log to stdout/stderr in production (12-factor app), not to files — let the container runtime handle log aggregation.",
    "Use environment-specific config files: .env.production, .env.staging — load with python-dotenv.",
    "Secrets in production: use systemd credentials, Docker secrets, or Vault — never plain .env files on production.",
    "Zero-downtime deploy: update the container then let nginx upstream pick up the new container (or use rolling update).",
    "Backup before every deploy: database dump + workspace snapshot, kept for at least 7 days.",
],

"data_processing": [
    "For DataFrames: prefer polars over pandas for speed (10-100x faster); pandas is fine for <100k rows.",
    "csv.DictReader is stdlib and handles most CSV files; pandas read_csv is better for large files with dtype inference.",
    "JSONL (one JSON object per line) is better than JSON arrays for streaming/appending large datasets.",
    "Use itertools.chain.from_iterable to flatten nested lists without materializing intermediate lists.",
    "For large file processing: process line by line with a generator, never load the whole file into memory.",
    "sqlite3 is the best local data store for structured results — query with SQL, no server needed.",
    "orjson is 3-10x faster than json for serialization; drop-in replacement for most use cases.",
    "When merging datasets: always check for duplicate keys and null values before joining.",
    "Chunked processing: process data in batches of 1000 rows to balance memory usage and I/O efficiency.",
    "Use hashlib.sha256(data).hexdigest() to deduplicate records by content hash before inserting.",
],

"llm_agent_patterns": [
    "Chain-of-thought prompting: ask the model to 'think step by step' before giving a final answer — improves reasoning accuracy.",
    "System prompts should define role, constraints, and output format; user messages provide the specific task.",
    "Few-shot examples in the prompt dramatically improve structured output compliance (JSON, code, lists).",
    "Temperature 0.0-0.2 for structured/deterministic tasks (code, JSON); 0.3-0.7 for creative/diverse outputs.",
    "Tool calling: define tools as JSON schemas; the model decides which tool to call and with what arguments.",
    "RAG (Retrieval-Augmented Generation): embed the query, fetch top-K relevant chunks, inject into prompt context.",
    "Prompt injection defense: never concatenate user input directly into system prompts; use a separator and sanitize.",
    "Token budgeting: max_tokens controls response length; leave headroom by estimating ~4 chars per token.",
    "Streaming responses (stream=True) gives better perceived latency for long outputs — show tokens as they arrive.",
    "Self-consistency: generate 3-5 answers with temperature>0, then take the majority vote for better accuracy.",
    "Structured outputs: JSON mode or constrained decoding (outlines library) guarantees schema-valid responses.",
    "Context window management: summarize completed conversation turns to stay within limits while retaining key facts.",
    "For multi-step tasks, break into subtasks and pass only the relevant context to each LLM call — don't pass everything.",
    "Ollama local inference: /api/chat for chat completions, /api/embed for embeddings, /api/generate for raw completions.",
    "qwen3 35b excels at code and reasoning; use it for complex planning, architecture, and code generation tasks.",
],

"embeddings_rag": [
    "mxbai-embed-large produces 1024-dim embeddings; cosine similarity is the standard distance metric for retrieval.",
    "Chunk text at sentence or paragraph boundaries, not at fixed character counts — preserves semantic coherence.",
    "Optimal chunk size for RAG: 200-500 tokens; larger chunks have more context but lower retrieval precision.",
    "Hybrid search combines dense (embedding) and sparse (BM25/FTS) retrieval for better recall than either alone.",
    "MMR (Maximal Marginal Relevance) selects diverse results when multiple chunks would say the same thing.",
    "Reranking: use a cross-encoder to rerank the top-20 retrieved chunks to top-5 — much more accurate than bi-encoder alone.",
    "Store embeddings as binary-packed float32 (struct.pack) — 4x more compact than JSON, same precision.",
    "Embedding models are trained with specific templates; mxbai-embed-large uses 'Represent this sentence: {text}' prefix for passages.",
],

"fastapi_patterns": [
    "Use Depends() for dependency injection — shared database connections, auth, rate limiting all belong here.",
    "Pydantic models as request/response schemas: automatic validation, serialization, and OpenAPI doc generation.",
    "BackgroundTasks.add_task() runs work after the response is sent — good for notifications, cache updates.",
    "FastAPI lifespan context manager (asynccontextmanager) replaces startup/shutdown events for resource setup.",
    "StreamingResponse with an async generator streams large responses without buffering the full body.",
    "Use APIRouter to organize routes by domain; include_router() adds them to the main app with a prefix.",
    "HTTPException(status_code=422, detail=[...]) returns standard validation error format.",
    "WebSocket: await ws.send_json() / ws.receive_json() for structured real-time bidirectional communication.",
    "CORS middleware: app.add_middleware(CORSMiddleware, allow_origins=['https://mysite.com'], allow_credentials=True)",
    "Mount static files: app.mount('/static', StaticFiles(directory='static'), name='static')",
],

"git_automation": [
    "GitPython library: repo = git.Repo('.'); repo.index.add(['file.py']); repo.index.commit('message').",
    "Use subprocess(['git', ...]) for git operations in scripts — more predictable than GitPython for simple ops.",
    "git log --oneline --since='1 week ago' shows recent commits; parse with splitlines() and split(' ', 1).",
    "Pre-commit hooks: .git/hooks/pre-commit bash script runs automatically before each commit — use for linting.",
    "GitHub Actions: on push to main, run tests, build Docker image, push to registry, deploy — full CI/CD in YAML.",
],

"ollamagi_system": [
    "OllamAGI flows run in Docker containers with /work bind-mounted; all output files must be written to /work/.",
    "web_search() is pre-injected into every Python agent script — use it directly without importing anything.",
    "Flow types: agent_development, product_development, research, security, general — each routes to different agent roles.",
    "Agents: primary_agent (orchestrates), researcher (finds info), coder (writes code), architect (designs systems), adviser (strategy).",
    "Fractal memory stores knowledge from each flow; it is automatically queried at flow start to inject relevant prior knowledge.",
    "context_for_task(description) returns relevant prior memories as a formatted string ready to inject into prompts.",
    "SearxNG runs at localhost:4000 and aggregates Google, Bing, and DuckDuckGo — use it for research tasks.",
    "Docker containers get host.docker.internal mapped to the host — use this to reach SearxNG and Ollama from inside containers.",
    "The reflector agent analyzes failures and extracts lessons — its output is logged but not automatically re-used.",
    "MAX_RETRIES=2 means each subtask gets up to 3 execution attempts with auto-fix between attempts.",
    "Store important discoveries as memories using store_from_result() — they will be available to future flows on similar topics.",
    "Subtask deliverable_kind determines validation: 'source' requires a .py file, 'report' requires .md/.json, 'test' allows no file.",
    "The coder agent writes BUILD SCRIPTS that create deliverable files in /work — not the applications themselves.",
    "Ollama is at localhost:11434; models include qwen3:35b for reasoning, deepseek-coder for code, nomic-embed-text for embeddings.",
    "Agent memory writes happen: after each text subtask, after each task summary, and at flow end for all validated results.",
],

"business_patterns": [
    "SaaS pricing: freemium converts 2-5% to paid; usage-based billing aligns cost with value and reduces churn.",
    "MVP validation: 10 paying customers at $100/mo = $12k ARR = strong enough signal to invest further.",
    "API-first products: build the API before the UI — it forces clean abstractions and enables future integrations.",
    "Recurring revenue: subscriptions beat one-time sales for predictability; monthly churn should stay below 2%.",
    "Distribution channels: solo dev → productized service first (high margin, no code), then SaaS as you learn the problem.",
    "Webhook-first architecture lets customers integrate with their existing tools without you building every integration.",
    "Open-source core + paid hosted/support (open-core model) drives adoption; OSS acts as top-of-funnel marketing.",
    "Time-to-value: the faster a user gets their first win, the higher the activation rate — optimize onboarding ruthlessly.",
],

"networking_linux": [
    "netstat -tlnp shows listening ports with PIDs; ss -tlnp is the modern replacement.",
    "curl -v URL shows request/response headers — use for debugging API calls.",
    "tcpdump -i any port 80 captures all HTTP traffic — essential for debugging container networking issues.",
    "Linux file descriptor limits: ulimit -n 65536 and /etc/security/limits.conf for persistent change.",
    "iptables -L shows firewall rules; ufw is the simpler frontend — 'ufw allow 8080' opens a port.",
    "DNS resolution inside Docker: use the container name as hostname when services are on the same Docker network.",
    "Port forwarding: ssh -L 8080:localhost:8080 user@remote-host forwards remote port to local machine.",
    "socat TCP-LISTEN:8080,reuseaddr,fork TCP:target:80 proxies TCP connections — useful for debugging.",
],

"markdown_reports": [
    "Research reports should have: Executive Summary, Methodology, Findings (with sources), Analysis, Recommendations, Appendix.",
    "Use tables for comparisons: | Feature | Option A | Option B | is more scannable than prose lists.",
    "Link sources inline [text](url) in Markdown; never cite 'various sources' — be specific.",
    "Technical writeups: code blocks with language tags (```python) get syntax highlighting in most renderers.",
    "Decision documents: include Assumptions, Risks, Alternatives Considered, and a clear Next Action at the end.",
    "Mermaid diagrams in Markdown (```mermaid) render as flowcharts in GitHub, GitLab, and Obsidian.",
],

}

def main():
    print(f"Starting memory population — {sum(len(v) for v in CORPUS.values())} total entries across {len(CORPUS)} domains\n")
    s0 = stats()
    print(f"Current state: {s0['total_nodes']} nodes ({s0['by_level'].get(0,0)} leaves)\n")

    total_inserted = 0
    for domain, entries in CORPUS.items():
        domain_count = 0
        for content in entries:
            try:
                node_id = insert(
                    content,
                    flow_id="",
                    tags=[domain],
                    metadata={"source": "synthetic_seed", "domain": domain},
                )
                domain_count += 1
                total_inserted += 1
                print(f"  [{total_inserted:3d}] [{domain}] id={node_id}  '{content[:60]}…'")
            except Exception as e:
                print(f"  ERROR inserting [{domain}]: {e}")
        print(f"  → {domain_count} entries stored for [{domain}]\n")

    s1 = stats()
    print(f"\n{'═'*60}")
    print(f"  Done. {total_inserted} memories inserted.")
    print(f"  Nodes: {s0['total_nodes']} → {s1['total_nodes']} (+{s1['total_nodes']-s0['total_nodes']})")
    print(f"  Leaves (L0): {s0['by_level'].get(0,0)} → {s1['by_level'].get(0,0)}")
    for lvl in sorted(s1['by_level']):
        before = s0['by_level'].get(lvl, 0)
        after  = s1['by_level'].get(lvl, 0)
        label  = {0:'leaves',1:'concepts',2:'domains',3:'meta',4:'root'}.get(lvl, f'L{lvl}')
        print(f"  Level {lvl} ({label}): {before} → {after}")
    print(f"  DB size: {s1.get('db_size_kb',0)} KB")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
