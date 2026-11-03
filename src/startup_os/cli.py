"""Command-line entry point: `uv run startup-os ...`."""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from datetime import datetime

from .approvals import ApprovalError
from .runtime import CommandError, MissingAPIKey, Runtime, TaskOutcome
from .tickets import TicketError
from .venture import TransitionError


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


def cmd_pick(args, rt: Runtime) -> int:
    r = rt.pick(args.brief, reason=args.reason)
    print(f"Picked {r['venture']} ({r['brief']}); approval {r['approval']} recorded.")
    print(f"  stage: Ideation → Validation   decision: company/{r['decision']}")
    print(f"  CEO created {r['ticket']} for research ({'written by CEO' if r['ticket_source'] == 'ceo' else 'template; LLM unavailable'})")
    print("  next: startup-os validate")
    return 0


def cmd_validate(args, rt: Runtime) -> int:
    v = rt.venture.load()
    print(f"Research agent (validation mode): validating {v.name}…")
    o = rt.validate()
    _print_outcome(o, rt)
    if o.status == "done":
        from . import frontmatter
        meta, _ = frontmatter.parse(rt.memory.read(o.extra["report_path"]))
        print(f"  go/no-go: weighted {meta['go_weighted']:g}/10 (shown as {meta['go_score']}; threshold "
              f"{meta['go_threshold']}) → {meta['recommendation']}")
        print("  next: startup-os plan")
    return 0 if o.status == "done" else 2


def cmd_research(args, rt: Runtime) -> int:
    results = rt.research(args.ticket, args.limit)
    if not results:
        print("no open research tickets")
        return 0
    rc = 0
    for r in results:
        if r["status"] == "skipped":
            print(f"{r['ticket']}: skipped — {r['reason']}")
            continue
        o = r["outcome"]
        print(f"{r['ticket']}:", end="")
        _print_outcome(o, rt)
        rc = rc or (0 if o.status == "done" else 2)
    return rc


def cmd_plan(args, rt: Runtime) -> int:
    print("CEO agent: writing this week's plan…")
    o = rt.plan()
    _print_outcome(o, rt)
    for n in o.extra.get("notes") or []:
        print(f"  • {n}")
    if o.status == "done":
        print(f"  awaiting approval {o.extra['approval']} — see `startup-os inbox`")
    return 0 if o.status == "done" else 2


def _ago(iso: str | None, now: datetime) -> str:
    if not iso:
        return "—"
    secs = (datetime.fromisoformat(iso) - now).total_seconds()
    h = abs(secs) / 3600
    s = f"{h:.0f}h" if h >= 1 else f"{abs(secs) / 60:.0f}m"
    return f"in {s}" if secs > 0 else f"{s} ago"


def cmd_inbox(args, rt: Runtime) -> int:
    now = rt.now()
    if rt.auto_approved:
        print("auto-approved just now: " + ", ".join(f"{a['ref']} ({a['title']})" for a in rt.auto_approved))
    pend = rt.approvals.list(status="pending")
    print(f"INBOX — {len(pend)} pending\n")
    for a in pend:
        print(f"{a['ref']:<6} {a['kind']:<14} {a['title']}")
        print(f"       requested by {a['requested_by']}, {_ago(a['created_at'], now)}")
        if a["kind"] == "plan_approval":
            print(f"       auto-approves {_ago(a['auto_approve_at'], now)} ({a['auto_approve_at'][:16]}Z) unless rejected")
            print(f"       file: company/{a['payload']['path']}")
        elif a["kind"] == "spend_request":
            p = a["payload"]
            print(f"       never auto-approves; blocks {', '.join(p.get('blocks') or []) or 'no tickets'}")
            print(f"       why: {p['justification']}")
        else:
            print(f"       brief: {a['payload'].get('brief')}")
        print()
    if not pend:
        print("nothing to decide.\n")
    print("approve: startup-os approve <id>    reject: startup-os reject <id> --reason \"...\"")
    if args.all:
        print("\nrecent decisions:")
        for a in [x for x in rt.approvals.list() if x["status"] != "pending"][-10:]:
            print(f"  {a['ref']:<6} {a['status']:<13} {a['kind']:<14} {a['title']}  (by {a['decided_by']})")
    blocked = rt.tickets.list(status="blocked")
    if blocked:
        print("\nblocked tickets:")
        for t in blocked:
            print(f"  {t.id} [{t.owner}] {t.title} — {t.blocked_reason}")
    return 0


def cmd_approve(args, rt: Runtime) -> int:
    a = rt.approve(args.id, reason=args.reason)
    print(f"{a['ref']} {a['kind']} approved: {a['title']}")
    if a["kind"] == "spend_request":
        print("  dependent tickets unblocked; nothing is spent automatically.")
    return 0


def cmd_reject(args, rt: Runtime) -> int:
    a = rt.reject(args.id, reason=args.reason)
    print(f"{a['ref']} {a['kind']} rejected: {a['title']}  (reason: {a['reason']})")
    return 0


def cmd_tickets(args, rt: Runtime) -> int:
    ts = rt.tickets.list(status=None if args.all else ("todo", "blocked", "in_progress"))
    for t in ts:
        print(f"{t.id}  {t.status:<9} {t.owner:<12} {t.title}" + (f"  [{t.blocked_reason}]" if t.blocked_reason else ""))
    if not ts:
        print("no tickets")
    return 0


def cmd_venture(args, rt: Runtime) -> int:
    v = rt.venture.load()
    print(f"venture: {v.name or '(none)'}   stage: {v.stage.value}   idea: {v.idea or '—'}")
    for h in v.history:
        print(f"  {h['at']}  {h['from']} → {h['to']}  ({h['decision']})")
    return 0


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
    s = sub.add_parser("pick", help="pick an idea brief as the venture (→ Validation)")
    s.add_argument("brief", help="path or name of a brief in company/research/ideas/")
    s.add_argument("--reason", default=None)
    s.set_defaults(fn=cmd_pick)
    sub.add_parser("validate", help="Research agent: validation report for the picked idea").set_defaults(fn=cmd_validate)
    sub.add_parser("plan", help="CEO agent: write this week's plan").set_defaults(fn=cmd_plan)
    s = sub.add_parser("research", help="Research agent: execute open research tickets (any type)")
    s.add_argument("--ticket", default=None, help="run only this ticket, e.g. T-0002")
    s.add_argument("--limit", type=int, default=None, help="run at most N tickets")
    s.set_defaults(fn=cmd_research)
    s = sub.add_parser("inbox", help="pending approvals")
    s.add_argument("--all", action="store_true", help="also show recent decisions")
    s.set_defaults(fn=cmd_inbox)
    s = sub.add_parser("approve", help="approve an inbox item")
    s.add_argument("id")
    s.add_argument("--reason", default=None)
    s.set_defaults(fn=cmd_approve)
    s = sub.add_parser("reject", help="reject an inbox item")
    s.add_argument("id")
    s.add_argument("--reason", required=True)
    s.set_defaults(fn=cmd_reject)
    s = sub.add_parser("tickets", help="list open tickets")
    s.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_tickets)
    sub.add_parser("venture", help="show venture and stage history").set_defaults(fn=cmd_venture)
    args = p.parse_args(argv)
    if getattr(args, "n", 1) < 1:
        p.error("--n must be >= 1")
    try:
        rt = Runtime(load_config(args.config))
        for a in rt.invalid_plans:
            print(f"note: rejected {a['ref']} ({a['title']}): {a['reason']}")
        if rt.auto_approved and args.cmd != "inbox":
            print("note: auto-approved " + ", ".join(a["ref"] for a in rt.auto_approved) + " (48h passed)")
        if rt.recovered:
            print(f"note: requeued {len(rt.recovered)} interrupted task(s) "
                  f"({', '.join(rt.recovered)}); run `startup-os run-due` to resume them.")
        return args.fn(args, rt)
    except (MissingAPIKey, CommandError, ApprovalError, TransitionError, TicketError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
