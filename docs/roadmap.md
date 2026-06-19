# Roadmap

OllamAGI is a working prototype open-sourced for the community. The roadmap reflects what's most valuable for real daily use.

## Now (v0.1 — current release)

- [x] Flow engine: objective → tasks → subtasks → Docker execution
- [x] 10 agent roles with model routing
- [x] Auto-fix with up to 2 retries per subtask
- [x] Immediate flow stop (~500ms)
- [x] Mid-flow steering (inject prompt, triggers replan)
- [x] Automatic replanning after consecutive failures
- [x] Mobile-first dashboard (6 tabs)
- [x] Live terminal in dashboard
- [x] Token tracking (per-flow, session, all-time)
- [x] Hermes memory integration (beliefs, RAG)
- [x] Token counter persistence across restarts
- [x] Quick launch presets with config sheets

## Near-term (v0.2)

- [ ] Dockerfile for easy containerized deployment
- [ ] Ubuntu/Debian setup script (alongside Fedora)
- [ ] Flow templates: save and reuse successful flow structures
- [ ] Workspace file browser in dashboard (view/download agent outputs)
- [ ] Per-agent streaming output in the dashboard
- [ ] Model capability auto-detection (don't require manual `.env` model config)

## Medium-term (v0.3)

- [ ] Scheduled flows (cron-style recurring tasks)
- [ ] Flow chaining (output of one flow becomes input of next)
- [ ] Agent-to-agent communication within a flow
- [ ] Plugin system for custom agent roles
- [ ] Simple authentication for shared/team deployments

## Long-term ideas

- Multi-machine execution (agent containers on remote hosts)
- Flow marketplace (share/import flow templates)
- Native macOS support (without Docker bind-mount hacks)
- Voice interface via Whisper for mobile flow creation

## Not planned

- Cloud LLM support (by design — local-first)
- GUI desktop app (the web dashboard is the UI)
- Windows support (WSL2 may work but untested)

---

Want to contribute to any of these? See [CONTRIBUTING.md](../CONTRIBUTING.md).
