"""Wiring: memory, index, scheduler, approvals, venture, tickets, spend gate and
the agent graphs, built from config. Every graph run is a task in the scheduler's
queue, checkpointed by LangGraph, and deferred (not dropped) when the budget or
all models are exhausted."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from . import frontmatter
from .agents import ceo as ceo_agent
from .agents.research import ResearchDeps, build_research_graph, recursion_limit
from .agents.validation import ValidationDeps, build_validation_graph
from .approvals import ApprovalError, ApprovalQueue
from .budget import BudgetExceeded, Scheduler
from .config import Config
from .guards import LoopGuard
from .llm import AllModelsFailed, CallFn, ModelRouter, openrouter_call_fn
from .memory import MemoryRepo
from .runlog import NullRunLog, RunLog
from .search_index import SearchIndex
from .spend import SpendGate
from .tickets import OPEN, TicketStore
from .tools import Toolbox, web_search
from .venture import Decision, Stage, TransitionError, Venture


class MissingAPIKey(Exception):
    pass


class CommandError(Exception):
    """A user-facing error (bad state or arguments)."""


@dataclass
class TaskOutcome:
    task_id: str
    status: str  # done | deferred | failed
    written: list[str]
    failed: list[dict]
    reason: str | None
    run_id: str
    log_path: str
    extra: dict = field(default_factory=dict)


TASK_AGENT = {"ideas": "research", "validate": "research", "plan": "ceo"}


class Runtime:
    def __init__(self, cfg: Config, *, call_fn: CallFn | None = None, web_search_fn=None,
                 scheduler: Scheduler | None = None, now=None):
        self.cfg = cfg
        self.now = now or (lambda: datetime.now(timezone.utc))
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self.index = SearchIndex(cfg.data_dir / "index.db", cfg.company_dir)
        self.memory = MemoryRepo(cfg.company_dir, self.index)
        self.memory.ensure()
        self.scheduler = scheduler or Scheduler.from_config(cfg, now=self.now)
        self.approvals = ApprovalQueue(cfg.data_dir / "state.db", now=self.now, on_decided=self._on_decided)
        self.venture = Venture(self.memory, now=self.now)
        self.tickets = TicketStore(self.memory, now=self.now)
        self.spend_gate = SpendGate(self.approvals)
        self._call_fn = call_fn
        backends = cfg.web_search.backends
        self.web_search_fn = web_search_fn or (lambda q, n: web_search(q, n, backends))
        self.recovered = self.scheduler.recover_interrupted(stale_after_s=cfg.llm.timeout_s * 2)
        self.auto_approved = self.approvals.process_due()
        self._log: RunLog = NullRunLog()
        self._last_pick: dict | None = None

    # ------------------------------------------------------------ plumbing --
    def call_fn(self) -> CallFn:
        if self._call_fn is None:
            key = self.cfg.api_key()
            if not key:
                raise MissingAPIKey(
                    f"{self.cfg.llm.api_key_env} is not set. Copy .env.example to .env and add your key.")
            self._call_fn = openrouter_call_fn(self.cfg.llm.base_url, key, self.cfg.llm.timeout_s)
        return self._call_fn

    def _checkpointer(self) -> SqliteSaver:
        conn = sqlite3.connect(self.cfg.data_dir / "checkpoints.db", check_same_thread=False)
        return SqliteSaver(conn)

    def router(self, log: RunLog) -> ModelRouter:
        cfg = self.cfg
        return ModelRouter(cfg.llm.models, self.scheduler, self.call_fn(), run_log=log,
                           defer_hours=cfg.llm.defer_hours_on_failure,
                           call_kwargs={"temperature": cfg.llm.temperature, "max_tokens": cfg.llm.max_tokens})

    def toolbox(self, log: RunLog, agent: str) -> Toolbox:
        return Toolbox(self.index, web_enabled=self.cfg.web_search.enabled,
                       web_max_results=self.cfg.web_search.max_results, web_search_fn=self.web_search_fn,
                       run_log=log, spend_gate=self.spend_gate, agent=agent)

    def ceo_deps(self, log: RunLog) -> ceo_agent.CEODeps:
        g = self.cfg.guards
        return ceo_agent.CEODeps(
            router=self.router(log), memory=self.memory, venture=self.venture, tickets=self.tickets,
            approvals=self.approvals, spend_gate=self.spend_gate, scheduler=self.scheduler,
            guard=LoopGuard(self.cfg.ceo.max_steps, g.max_identical_tool_calls),
            go_threshold=self.cfg.validation.go_threshold, agents=sorted(self.cfg.budget.agents),
            now=self.now, run_log=log)

    # --------------------------------------------------------------- tasks --
    def _graph_for(self, task: dict, log: RunLog):
        cfg, kind = self.cfg, task["kind"]
        if kind == "ideas":
            guard = LoopGuard(cfg.guards.max_steps_per_task, cfg.guards.max_identical_tool_calls,
                              cfg.guards.max_tool_calls_per_task)
            deps = ResearchDeps(self.router(log), self.memory, self.toolbox(log, "research"), guard, log)
            n = task["payload"]["n"]
            return (build_research_graph(deps, self._checkpointer()), {"n": n, "run_id": log.run_id},
                    recursion_limit(n, cfg.guards.max_steps_per_task))
        if kind == "validate":
            v = cfg.validation
            guard = LoopGuard(v.max_steps, cfg.guards.max_identical_tool_calls, v.max_tool_calls)
            deps = ValidationDeps(self.router(log), self.memory, self.toolbox(log, "research"), guard,
                                  self.venture, self.tickets, v.go_threshold, log)
            return (build_validation_graph(deps, self._checkpointer()),
                    {"ticket_id": task["payload"]["ticket"], "run_id": log.run_id}, 2 * v.max_steps + 10)
        if kind == "plan":
            return (ceo_agent.build_plan_graph(self.ceo_deps(log), self._checkpointer()),
                    {"run_id": log.run_id}, cfg.ceo.max_steps + 10)
        raise ValueError(f"unknown task kind {kind!r}")

    def run_task(self, task: dict, *, resume: bool = False) -> TaskOutcome:
        log = RunLog(self.cfg.logs_dir, task["kind"], task_id=task["id"], thread_id=task["thread_id"],
                     payload=task["payload"], resume=resume, models=self.cfg.llm.models)
        self._log = log
        graph, init, limit = self._graph_for(task, log)
        run_cfg = {"configurable": {"thread_id": task["thread_id"]}, "recursion_limit": limit}
        self.scheduler.set_status(task["id"], "running")
        try:
            state = graph.invoke(None if resume else init, run_cfg)
        except (BudgetExceeded, AllModelsFailed) as e:
            self.scheduler.defer(task["id"], e.retry_at, str(e))
            snap = graph.get_state(run_cfg).values
            log.close("deferred", reason=str(e), retry_at=e.retry_at)
            return TaskOutcome(task["id"], "deferred", snap.get("written", []), snap.get("failed", []),
                               str(e), log.run_id, str(log.path))
        except KeyboardInterrupt:
            self.scheduler.defer(task["id"], self.scheduler.now(), "interrupted by user")
            log.close("interrupted")
            raise
        except Exception as e:
            self.scheduler.set_status(task["id"], "failed", reason=repr(e)[:500])
            log.close("failed", error=repr(e))
            raise
        finally:
            self._log = NullRunLog()
        self.scheduler.set_status(task["id"], "done")
        written = state.get("written") or [p for p in (state.get("report_path"), state.get("plan_path")) if p]
        extra = {k: state.get(k) for k in ("approval", "notes", "report_path", "plan_path") if state.get(k)}
        log.close("done", written=written, **extra)
        return TaskOutcome(task["id"], "done", written, state.get("failed", []), None, log.run_id,
                           str(log.path), extra)

    # backwards-compatible name
    def run_research(self, task: dict, *, resume: bool = False) -> TaskOutcome:
        return self.run_task(task, resume=resume)

    def _new_task(self, kind: str, payload: dict) -> dict:
        return self.scheduler.create_task(TASK_AGENT[kind], kind, payload)

    def run_due(self) -> list[TaskOutcome]:
        return [self.run_task(t, resume=True) for t in self.scheduler.due_tasks() if t["kind"] in TASK_AGENT]

    # ------------------------------------------------------------ commands --
    def ideas(self, n: int) -> TaskOutcome:
        return self.run_task(self._new_task("ideas", {"n": n}))

    def resolve_brief(self, path: str) -> str:
        """Accept company/research/ideas/x.md, research/ideas/x.md, x.md, x, or an absolute path."""
        p = Path(path)
        if p.is_absolute():
            try:
                p = p.resolve().relative_to(self.memory.root)
            except ValueError:
                raise CommandError(f"{path} is not inside {self.memory.root}") from None
        cands = [p]
        if p.parts and p.parts[0] == Path(self.cfg.paths.company).name:
            cands.append(Path(*p.parts[1:]))
        cands += [Path("research/ideas") / p.name, Path("research/ideas") / f"{p.name}.md"]
        for c in cands:
            try:
                if c.suffix == ".md" and self.memory.exists(c):
                    return str(c)
            except Exception:  # noqa: BLE001 - path escapes repo
                continue
        matches = sorted((self.memory.root / "research/ideas").glob(f"*{p.stem}*.md"))
        if len(matches) == 1:
            return str(matches[0].relative_to(self.memory.root))
        raise CommandError(f"no idea brief matches {path!r}")

    def pick(self, path: str, reason: str | None = None) -> dict:
        """Human picks an idea: recorded as an approved idea_pick, which moves the stage to Validation."""
        brief = self.resolve_brief(path)
        st = self.venture.load()
        if st.stage != Stage.IDEATION:
            raise CommandError(f"venture is in {st.stage.value} ({st.name}); a new pick is only possible in "
                               "Ideation. Move back to Ideation first.")
        self.call_fn()  # the CEO writes the validation ticket
        title = frontmatter.parse(self.memory.read(brief))[0].get("title", Path(brief).stem)
        a = self.approvals.create("idea_pick", f"Pick {title}", {"brief": brief, "why": reason},
                                  requested_by="human")
        self.approvals.approve(a["id"], by="human", reason=reason or "picked via `startup-os pick`")
        return self._last_pick

    def _do_pick(self, brief: str, why: str | None, approval: dict) -> dict:
        meta, body = frontmatter.parse(self.memory.read(brief))
        title = meta.get("title", Path(brief).stem)
        others = [b for b in ceo_agent.list_briefs(self.memory) if b["path"] != brief]
        others.sort(key=lambda b: -(b["score"] or 0))
        m = re.search(r"## Demand signal[^\n]*\n+(.+?)(?:\n\n|\Z)", body, re.DOTALL)
        reason = why or f"Picked by the human ({approval['ref']})."
        reason += f" Brief demand score {meta.get('demand_score')}/10" + (f": {m.group(1).strip()[:400]}" if m else ".")
        decision = Decision(
            what=f"Pick \"{title}\" ({brief}) as the venture and start validation.",
            why=reason,
            alternatives=[f"{b['title']} ({b['score']}/10) — {b['path']}" for b in others] or ["Generate more ideas first"],
            decided_by=f"human via {approval['ref']}")
        rec = self.venture.transition(Stage.VALIDATION, decision, agent="ceo", name=title, idea=brief)
        log = self._log if not isinstance(self._log, NullRunLog) else RunLog(self.cfg.logs_dir, "pick", brief=brief)
        ticket, source = ceo_agent.validation_ticket(self.ceo_deps(log), venture=title, brief_path=brief)
        self.tickets.save(ticket, agent="ceo", message=f"create validation ticket {ticket.id} for {title}")
        if log is not self._log:
            log.close("done", ticket=ticket.id, decision=rec, ticket_source=source)
        return {"venture": title, "brief": brief, "decision": rec, "ticket": ticket.id,
                "ticket_source": source, "approval": approval["ref"]}

    def validate(self) -> TaskOutcome:
        st = self.venture.load()
        if st.stage != Stage.VALIDATION:
            raise CommandError(f"venture is in {st.stage.value}; validate needs the Validation stage (run `pick`).")
        open_v = [t for t in self.tickets.list(owner="research", kind="validation", status=("todo",))
                  if t.venture == st.name]
        if not open_v:
            raise CommandError(f"no open research validation ticket for {st.name}")
        self.call_fn()
        return self.run_task(self._new_task("validate", {"ticket": open_v[0].id, "venture": st.name}))

    def plan(self) -> TaskOutcome:
        self.call_fn()
        return self.run_task(self._new_task("plan", {"week": ceo_agent.iso_week(self.now())}))

    def approve(self, aid: str, reason: str | None = None) -> dict:
        return self.approvals.approve(aid, by="human", reason=reason)

    def reject(self, aid: str, reason: str) -> dict:
        return self.approvals.reject(aid, reason=reason, by="human")

    # ----------------------------------------------------- approval effects --
    def _on_decided(self, a: dict) -> None:
        """Apply the effects of a decision, then log it to company/approvals.log."""
        approved = a["status"] in ("approved", "auto_approved")
        kind, p = a["kind"], a["payload"]
        if kind == "plan_approval" and a["status"] != "superseded":
            self._set_plan_status(p, a)
            if approved:
                self.scheduler.set_share_override(p.get("budget_split") or None, source=f"plan {p['week']} ({a['ref']})")
            else:
                stale = [t.model_copy(update={"status": "cancelled", "blocked_reason": f"plan rejected ({a['ref']})"})
                         for t in self.tickets.list(status=OPEN) if t.id in p.get("tickets", [])]
                if stale:
                    self.memory.write_many(self.tickets.files_for(stale), agent="system",
                                           message=f"cancel tickets of rejected plan {p['week']}")
        elif kind == "spend_request":
            # Approval only unblocks dependent tickets; nothing is ever spent in this slice.
            self.tickets.refresh(self.approvals.status)
        elif kind == "idea_pick" and approved:
            self._last_pick = self._do_pick(p["brief"], p.get("why"), a)
        line = f"{a['ref']}\t{a['kind']}\t{a['status']}\tby={a['decided_by']}\t{a['title']}"
        if a.get("reason"):
            line += f"\treason={a['reason']}"
        self.memory.append_approval(line, agent=a["decided_by"] if a["decided_by"] != "human" else "human")

    def _set_plan_status(self, p: dict, a: dict) -> None:
        rel = p.get("path")
        if not rel or not self.memory.exists(rel):
            return
        meta, body = frontmatter.parse(self.memory.read(rel))
        meta["status"] = a["status"]
        meta["decided_at"] = a["decided_at"]
        meta["decided_by"] = a["decided_by"]
        if a.get("reason"):
            meta["decision_reason"] = a["reason"]
        self.memory.write(rel, frontmatter.dump(meta, body), agent="system",
                          message=f"plan {p['week']} {a['status']} ({a['ref']})")
