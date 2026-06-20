#!/usr/bin/env python3
"""OllamAGI — CLI entry point."""
import sys
import argparse
import json
from pathlib import Path
import uvicorn

def cmd_run(args):
    from core.orchestrator import run_flow
    from core.config import WORKSPACE_DIR
    flow = run_flow(args.objective)
    print(f"\n[done] flow {flow.id} — {flow.status}")
    print(f"[done] workspace: {WORKSPACE_DIR / flow.id}")

def cmd_bounty(args):
    from flows.bug_bounty import BugBountyConfig, build_tasks, get_objective
    from core.orchestrator import run_flow
    scope = args.scope if args.scope else [args.target]
    cfg = BugBountyConfig(
        target=args.target,
        scope=scope,
        out_of_scope=args.oos if args.oos else [],
        platform=args.platform,
        program_name=args.program or args.target,
    )
    objective = get_objective(cfg)
    tasks = build_tasks(cfg)
    print(f"[ollamagi] launching bug bounty: {args.target} on {args.platform}")
    print(f"[ollamagi] {len(tasks)} tasks planned")
    flow = run_flow(objective, tasks=tasks)
    print(f"\n[done] report: ~/ollamagi/workspace/{flow.id}/reports/report.md")

def _cleanup_stale_flows():
    """On startup, mark any flow still showing 'running' as 'stopped' — they can't be live."""
    import time
    from pathlib import Path
    ws = Path(__file__).parent / "workspace"
    if not ws.exists():
        return
    fixed = 0
    for d in ws.iterdir():
        f = d / "flow.json"
        if not f.exists():
            continue
        try:
            d = json.loads(f.read_text())
            if d.get("status") == "running":
                d["status"] = "stopped"
                if not d.get("finished_at"):
                    d["finished_at"] = time.time()
                f.write_text(json.dumps(d, indent=2))
                fixed += 1
        except Exception:
            pass
    if fixed:
        print(f"[ollamagi] cleaned up {fixed} stale running flow(s)")

def cmd_serve(args):
    from api.server import app
    _cleanup_stale_flows()
    print(f"[ollamagi] dashboard: http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")

def cmd_memory(args):
    from core.memory_bridge import get_relevant_context, get_goals
    if args.query:
        results = get_relevant_context(args.query)
        for r in results:
            print(f"[{r['type'].upper()}] {r['content'][:120]}")
    else:
        goals = get_goals()
        print(f"Active goals: {len(goals)}")
        for g in goals[:5]:
            print(f"  • [{g['priority']}] {g['title']}")

def main():
    p = argparse.ArgumentParser(prog="ollamagi", description="OllamAGI — Local AI Civilization")
    sub = p.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Run a flow from an objective")
    run_p.add_argument("objective", help="What to do")

    bb_p = sub.add_parser("bounty", help="Launch a bug bounty flow")
    bb_p.add_argument("target", help="Target domain (e.g. example.com)")
    bb_p.add_argument("--scope", nargs="+", help="In-scope domains")
    bb_p.add_argument("--oos", nargs="+", help="Out-of-scope domains")
    bb_p.add_argument("--platform", default="hackerone", help="Platform (hackerone/intigriti/bugcrowd)")
    bb_p.add_argument("--program", help="Program name")

    serve_p = sub.add_parser("serve", help="Start web dashboard")
    serve_p.add_argument("--port", type=int, default=7654)

    mem_p = sub.add_parser("memory", help="Query cognitive memory")
    mem_p.add_argument("query", nargs="?", help="Search query")

    args = p.parse_args()
    if args.cmd == "run":       cmd_run(args)
    elif args.cmd == "bounty":  cmd_bounty(args)
    elif args.cmd == "serve":   cmd_serve(args)
    elif args.cmd == "memory":  cmd_memory(args)
    else:                       p.print_help()

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
