# Contributing to OllamAGI

Thanks for your interest. OllamAGI is a working local prototype open-sourced for the community — contributions that improve reliability, portability, or usability are very welcome.

## What we're looking for

- Bug fixes (especially around Docker, Ollama edge cases, mobile UI)
- Support for more Linux distros (Ubuntu, Arch, NixOS)
- New agent roles
- Better error messages
- Tests
- Documentation improvements

## What to avoid

- Breaking the single-command startup
- Adding cloud API dependencies
- Requiring root / elevated privileges beyond Docker
- Large refactors without prior discussion

## Development setup

```bash
git clone https://github.com/yourname/ollamagi.git
cd ollamagi
pip3 install -r requirements.txt
cp .env.example .env   # edit as needed
python3 ollamagi.py serve
```

## Making changes

1. Fork https://github.com/linutesto/ollamagi and create a feature branch
2. Make your changes
3. Run the health check: `bash scripts/health_check.sh`
4. Open a PR with a clear description of what changed and why

## Code style

- Python 3.11+, no type-ignore comments
- Keep functions short and named after what they do
- No comments explaining *what* the code does — only *why* if it's non-obvious
- New agent roles go in `core/agents.py`
- New API endpoints go in `api/server.py`

## Adding a new agent role

1. Add an `AgentRole` to `core/agents.py`
2. Add it to `ALL_ROLES`
3. Optionally add it to relevant `FLOW_TYPE_ROLES` entries
4. Document it in `docs/architecture.md`

## Questions

Open a GitHub Discussion rather than an issue for general questions.
