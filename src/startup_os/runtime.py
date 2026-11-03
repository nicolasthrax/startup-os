"""Wiring: builds the memory repo, index, scheduler, router and graphs from config."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from langgraph.checkpoint.sqlite import SqliteSaver

from .agents.research import AGENT, ResearchDeps, build_research_graph, recursion_limit
from .budget import BudgetExceeded, Scheduler
from .config import Config
from .guards import LoopGuard
from .llm import AllModelsFailed, CallFn, ModelRouter, openrouter_call_fn
from .memory import MemoryRepo
from .runlog import RunLog
from .search_index import SearchIndex
from .tools import Toolbox, duckduckgo_search


class MissingAPIKey(Exception):
    pass


@dataclass
class TaskOutcome:
    task_id: str
    status: str  # done | deferred | failed
    written: list[str]
    failed: list[dict]
    reason: str | None
    run_id: str
    log_path: str


class Runtime:
    def __init__(self, cfg: Config, *, call_fn: CallFn | None = None, web_search_fn=duckduckgo_search,
                 scheduler: Scheduler | None = None):
        self.cfg = cfg
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self.index = SearchIndex(cfg.data_dir / "index.db", cfg.company_dir)
        self.memory = MemoryRepo(cfg.company_dir, self.index)
        self.memory.ensure()
        self.scheduler = scheduler or Scheduler.from_config(cfg)
        self._call_fn = call_fn
        self.web_search_fn = web_search_fn

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

    def run_research(self, task: dict, *, resume: bool = False) -> TaskOutcome:
        cfg = self.cfg
        n = task["payload"]["n"]
        log = RunLog(cfg.logs_dir, "ideas", task_id=task["id"], thread_id=task["thread_id"],
                     n=n, resume=resume, models=cfg.llm.models)
        router = ModelRouter(
            cfg.llm.models, self.scheduler, self.call_fn(), run_log=log,
            defer_hours=cfg.llm.defer_hours_on_failure,
            call_kwargs={"temperature": cfg.llm.temperature, "max_tokens": cfg.llm.max_tokens},
        )
        toolbox = Toolbox(self.index, web_enabled=cfg.web_search.enabled,
                          web_max_results=cfg.web_search.max_results,
                          web_search_fn=self.web_search_fn, run_log=log)
        guard = LoopGuard(cfg.guards.max_steps_per_task, cfg.guards.max_identical_tool_calls)
        graph = build_research_graph(ResearchDeps(router, self.memory, toolbox, guard, log),
                                     checkpointer=self._checkpointer())
        run_cfg = {"configurable": {"thread_id": task["thread_id"]},
                   "recursion_limit": recursion_limit(n, cfg.guards.max_steps_per_task)}
        self.scheduler.set_status(task["id"], "running")
        try:
            state = graph.invoke(None if resume else {"n": n, "run_id": log.run_id}, run_cfg)
        except (BudgetExceeded, AllModelsFailed) as e:
            self.scheduler.defer(task["id"], e.retry_at, str(e))
            snap = graph.get_state(run_cfg).values
            log.close("deferred", reason=str(e), retry_at=e.retry_at)
            return TaskOutcome(task["id"], "deferred", snap.get("written", []), snap.get("failed", []),
                               str(e), log.run_id, str(log.path))
        except Exception as e:
            self.scheduler.set_status(task["id"], "failed", reason=repr(e)[:500])
            log.close("failed", error=repr(e))
            raise
        self.scheduler.set_status(task["id"], "done")
        log.close("done", written=state.get("written", []), failed=state.get("failed", []))
        return TaskOutcome(task["id"], "done", state.get("written", []), state.get("failed", []),
                           None, log.run_id, str(log.path))

    def ideas(self, n: int) -> TaskOutcome:
        task = self.scheduler.create_task(AGENT, "ideas", {"n": n})
        return self.run_research(task)

    def run_due(self) -> list[TaskOutcome]:
        out = []
        for t in self.scheduler.due_tasks():
            if t["kind"] == "ideas":
                out.append(self.run_research(t, resume=True))
        return out
