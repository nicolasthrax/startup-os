"""Command-line entry point: `uv run startup-os ...`."""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .runtime import MissingAPIKey, Runtime, TaskOutcome


def _print_outcome(o: TaskOutcome, rt: Runtime) -> None:
    print(f"\ntask {o.task_id}: {o.status.upper()}")
    if o.reason:
        print(f"  reason: {o.reason}")
    for p in o.written:
        print(f"  ✓ {rt.cfg.paths.company / p}")
    for f in o.failed:
        print(f"  ✗ {f['seed']}: {f['error']}")
    print(f"  run log: {o.log_path}")


def cmd_ideas(args, rt: Runtime) -> int:
    rt.call_fn()  # fail fast if the API key is missing
    print(f"Research agent: generating {args.n} idea briefs…")
    o = rt.ideas(args.n)
    _print_outcome(o, rt)
    _print_usage(rt)
    return 0 if o.status == "done" else 2


def cmd_run_due(args, rt: Runtime) -> int:
    rt.call_fn()
    outs = rt.run_due()
    if not outs:
        print("no deferred tasks are due")
    for o in outs:
        _print_outcome(o, rt)
    return 0


def cmd_queue(args, rt: Runtime) -> int:
    tasks = rt.scheduler.list_tasks(None if args.all else "deferred")
    if not tasks:
        print("queue empty")
    for t in tasks:
        print(f"{t['id']}  {t['status']:<9} {t['agent']:<9} {t['kind']} {json.dumps(t['payload'])}"
              f"  run_after={t['run_after']}  {t['reason'] or ''}")
    return 0


def _print_usage(rt: Runtime) -> None:
    print(f"\nbudget {rt.scheduler.today()} (UTC):")
    print(f"  {'agent':<10}{'used':>6}{'limit':>7}{'ok':>5}{'429':>5}{'err':>5}{'tokens':>9}")
    for r in rt.scheduler.usage_summary():
        print(f"  {r['agent']:<10}{r['requests']:>6}{r['limit']:>7}{r['ok']:>5}"
              f"{r['rate_limited']:>5}{r['errors']:>5}{r['tokens']:>9}")


def cmd_status(args, rt: Runtime) -> int:
    _print_usage(rt)
    print(f"\ndeferred tasks: {len(rt.scheduler.list_tasks('deferred'))}")
    print("recent memory commits:")
    for s in rt.memory.log(5):
        print(f"  {s}")
    return 0


def cmd_search(args, rt: Runtime) -> int:
    for h in rt.index.search(args.query, args.limit):
        print(f"{h['score']:>7}  {h['path']}\n         {h['snippet']}")
    return 0


def cmd_models(args, rt: Runtime) -> int:
    """List OpenRouter's currently free models (a read-only GET; costs nothing)."""
    import urllib.request

    with urllib.request.urlopen(f"{rt.cfg.llm.base_url}/models", timeout=30) as r:
        data = json.load(r)["data"]
    free = sorted(m["id"] for m in data if m["id"].endswith(":free"))
    configured = set(rt.cfg.llm.models)
    for m in free:
        print(("* " if m in configured else "  ") + m)
    missing = configured - set(free)
    if missing:
        print("\nconfigured but not currently offered:", ", ".join(sorted(missing)))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="startup-os")
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ideas", help="Research agent: generate N idea briefs")
    s.add_argument("--n", type=int, default=5)
    s.set_defaults(fn=cmd_ideas)
    sub.add_parser("run-due", help="resume deferred tasks whose time has come").set_defaults(fn=cmd_run_due)
    s = sub.add_parser("queue", help="list deferred tasks")
    s.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_queue)
    sub.add_parser("status", help="budget usage and recent memory commits").set_defaults(fn=cmd_status)
    s = sub.add_parser("search", help="full-text search the company memory")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=5)
    s.set_defaults(fn=cmd_search)
    sub.add_parser("models", help="list OpenRouter :free models").set_defaults(fn=cmd_models)
    args = p.parse_args(argv)
    if getattr(args, "n", 1) < 1:
        p.error("--n must be >= 1")
    try:
        rt = Runtime(load_config(args.config))
        return args.fn(args, rt)
    except MissingAPIKey as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
