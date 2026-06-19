# Release Checklist

Use this before every public release or when opening the repo to new contributors.

## Security & Privacy

- [ ] No real usernames in source code (`grep -r "yan" --include="*.py"` → none)
- [ ] No private IPs in source code (192.168.x.x, 100.x.x.x Tailscale ranges)
- [ ] `workspace/` is in `.gitignore` and not committed
- [ ] `CURRENT_CIVILIZATION_MAP.md` is in `.gitignore` and not committed
- [ ] No `.env` files committed (only `.env.example`)
- [ ] No SSH keys in the repo
- [ ] No API tokens, passwords, or credentials in any file
- [ ] `docker-compose.yml` uses `${ENV_VAR}` substitution, not hardcoded paths
- [ ] Agent system prompts contain no personal info

## Code Quality

- [ ] `python3 -c "import core.orchestrator; import core.model_router; import api.server; print('OK')"` passes
- [ ] `bash scripts/health_check.sh` passes on a clean start
- [ ] All config reads from env vars with sensible defaults (`core/config.py`)
- [ ] `HOME_DIR` in `docker_manager.py` reads from `OLLAMAGI_USER_HOME` env var

## Documentation

- [ ] `README.md` — quickstart works on a fresh machine
- [ ] `.env.example` — all config options documented
- [ ] `docs/architecture.md` — up to date with current code
- [ ] `docs/roadmap.md` — reflects actual next priorities
- [ ] `SECURITY.md` — responsible disclosure contact is current
- [ ] `CONTRIBUTING.md` — dev setup instructions work

## First-user Experience

- [ ] `pip3 install -r requirements.txt` installs all dependencies
- [ ] `python3 ollamagi.py serve` starts with a single command
- [ ] Dashboard loads at `http://localhost:7654`
- [ ] Running a simple flow ("print hello world in python") works end-to-end
- [ ] Stop button stops the flow within 1 second
- [ ] Token counts appear in flow detail after completion

## Git

- [ ] `git status` is clean (no untracked secrets or generated files)
- [ ] `git log --oneline -5` — commit messages are meaningful
- [ ] Branch is up to date with main

## Manual Review Required

The following items cannot be fully automated and require human review before each release:

1. **`CURRENT_CIVILIZATION_MAP.md`** — This file contains personal machine details (IPs, container names). It is git-ignored but verify it is not staged: `git status CURRENT_CIVILIZATION_MAP.md`

2. **`workspace/` directory** — Agent-generated files may contain sensitive data (API keys agents found, credentials.yaml files, .env files in generated projects). The directory is git-ignored but confirm: `git status workspace/`

3. **Agent system prompts in `core/agents.py`** — Review each prompt for personal context that may have crept back in.

4. **`core/config.py` MODELS dict** — Default model names use public Ollama Hub model names, not personal/private fine-tunes. Confirm these exist on the public hub or document them as user-supplied.

5. **Dashboard System tab** — Hardware info now loads dynamically from env vars. Confirm the tab shows generic defaults ("—") when `HW_CPU`, `HW_RAM`, `HW_GPU` are not set.

6. **SSH info in dashboard** — The SSH access box now shows `SSH_USER@SSH_HOST` from config. Confirm it does not leak private network addresses when defaults are used.
