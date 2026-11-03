"""Call-budget scheduler: per-agent daily quotas, a per-minute token bucket,
a usage log and a queue of deferred tasks, all in SQLite."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    day TEXT NOT NULL,
    agent TEXT NOT NULL,
    model TEXT,
    status TEXT NOT NULL,          -- reserved | ok | rate_limited | error | interrupted
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    latency_ms INTEGER,
    run_id TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS usage_day_agent ON usage(day, agent);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    status TEXT NOT NULL,          -- running | deferred | done | failed
    run_after TEXT,
    reason TEXT,
    pid INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class BudgetExceeded(Exception):
    def __init__(self, agent: str, used: int, limit: int, retry_at: datetime):
        super().__init__(
            f"daily budget for agent '{agent}' exhausted ({used}/{limit}); retry at {retry_at.isoformat()}"
        )
        self.agent, self.used, self.limit, self.retry_at = agent, used, limit, retry_at


class TokenBucket:
    """Classic token bucket: `capacity` tokens, refilled at `rate_per_min`/60 per second."""

    def __init__(
        self,
        rate_per_min: float,
        capacity: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.rate = rate_per_min / 60.0
        self.capacity = float(capacity if capacity is not None else rate_per_min)
        self.tokens = self.capacity
        self.clock, self.sleep = clock, sleep
        self.last = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now

    def try_acquire(self) -> bool:
        with self._lock:
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def acquire(self) -> float:
        """Block until a token is available. Returns seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return waited
                wait = (1 - self.tokens) / self.rate
            self.sleep(wait)
            waited += wait


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Scheduler:
    def __init__(
        self,
        db_path: Path,
        daily_requests: int,
        per_minute: int,
        shares: dict[str, float],
        *,
        now: Callable[[], datetime] = _utcnow,
        bucket: TokenBucket | None = None,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.daily_requests = daily_requests
        self.shares = shares
        self.now = now
        self.bucket = bucket or TokenBucket(per_minute)
        with self._conn() as c:
            c.executescript(SCHEMA)
            cols = {r[1] for r in c.execute("PRAGMA table_info(tasks)")}
            if "pid" not in cols:  # migrate DBs created before the pid column
                c.execute("ALTER TABLE tasks ADD COLUMN pid INTEGER")

    @classmethod
    def from_config(cls, cfg, **kw) -> "Scheduler":
        return cls(
            cfg.data_dir / "state.db",
            cfg.budget.daily_requests,
            cfg.budget.per_minute,
            cfg.budget.agents,
            **kw,
        )

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    # -- quota -------------------------------------------------------------
    def agent_limit(self, agent: str) -> int:
        return int(self.daily_requests * self.shares.get(agent, 0) / 100)

    def today(self) -> str:
        return self.now().date().isoformat()

    def next_day_start(self) -> datetime:
        n = self.now()
        return datetime(n.year, n.month, n.day, tzinfo=timezone.utc) + timedelta(days=1)

    def used_today(self, agent: str) -> int:
        with self._conn() as c:
            return c.execute(
                "SELECT COUNT(*) FROM usage WHERE day = ? AND agent = ?",
                (self.today(), agent),
            ).fetchone()[0]

    def remaining_today(self, agent: str) -> int:
        return max(0, self.agent_limit(agent) - self.used_today(agent))

    def acquire(self, agent: str, run_id: str | None = None) -> int:
        """Reserve one request for `agent`. Blocks on the per-minute bucket;
        raises BudgetExceeded if the agent's daily share is spent.
        Returns a usage-row id to pass to `record`."""
        limit, used = self.agent_limit(agent), self.used_today(agent)
        if used >= limit:
            raise BudgetExceeded(agent, used, limit, self.next_day_start())
        self.bucket.acquire()
        now = self.now()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO usage(ts, day, agent, status, run_id) VALUES (?, ?, ?, 'reserved', ?)",
                (now.isoformat(), now.date().isoformat(), agent, run_id),
            )
            return cur.lastrowid

    def record(
        self,
        usage_id: int,
        *,
        model: str,
        status: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE usage SET model=?, status=?, prompt_tokens=?, completion_tokens=?, "
                "total_tokens=?, latency_ms=?, error=? WHERE id=?",
                (
                    model, status, prompt_tokens, completion_tokens,
                    prompt_tokens + completion_tokens, latency_ms,
                    (error or None) and error[:500], usage_id,
                ),
            )

    def usage_summary(self, day: str | None = None) -> list[dict]:
        day = day or self.today()
        with self._conn() as c:
            rows = c.execute(
                "SELECT agent, COUNT(*) AS requests, SUM(status='ok') AS ok, "
                "SUM(status='rate_limited') AS rate_limited, SUM(status='error') AS errors, "
                "SUM(total_tokens) AS tokens FROM usage WHERE day = ? GROUP BY agent",
                (day,),
            ).fetchall()
        used = {r["agent"]: dict(r) for r in rows}
        agents = sorted(set(self.shares) | set(used))
        out = []
        for a in agents:
            r = used.get(a, {"requests": 0, "ok": 0, "rate_limited": 0, "errors": 0, "tokens": 0})
            out.append({"agent": a, "limit": self.agent_limit(a), **{k: r[k] or 0 for k in
                        ("requests", "ok", "rate_limited", "errors", "tokens")}})
        return out

    # -- task queue ----------------------------------------------------------
    def create_task(self, agent: str, kind: str, payload: dict, thread_id: str | None = None) -> dict:
        tid = uuid.uuid4().hex[:12]
        now = self.now().isoformat()
        thread_id = thread_id or f"{agent}-{tid}"
        with self._conn() as c:
            c.execute(
                "INSERT INTO tasks(id, agent, kind, payload, thread_id, status, pid, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?)",
                (tid, agent, kind, json.dumps(payload), thread_id, os.getpid(), now, now),
            )
        return self.get_task(tid)

    def get_task(self, task_id: str) -> dict:
        with self._conn() as c:
            r = c.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not r:
            raise KeyError(task_id)
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        return d

    def set_status(self, task_id: str, status: str, *, run_after: datetime | None = None,
                   reason: str | None = None) -> None:
        pid = os.getpid() if status == "running" else None
        with self._conn() as c:
            c.execute(
                "UPDATE tasks SET status=?, run_after=?, reason=?, pid=?, updated_at=? WHERE id=?",
                (status, run_after.isoformat() if run_after else None, reason, pid,
                 self.now().isoformat(), task_id),
            )

    def defer(self, task_id: str, run_after: datetime, reason: str) -> None:
        """Queue a task to run later instead of dropping it."""
        self.set_status(task_id, "deferred", run_after=run_after, reason=reason)

    def list_tasks(self, status: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM tasks", ()
        if status:
            q, args = q + " WHERE status = ?", (status,)
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY created_at", args).fetchall()
        return [dict(r) | {"payload": json.loads(r["payload"])} for r in rows]

    def due_tasks(self) -> list[dict]:
        now = self.now()
        return [
            t for t in self.list_tasks("deferred")
            if t["run_after"] is None or datetime.fromisoformat(t["run_after"]) <= now
        ]

    # -- crash recovery -------------------------------------------------------
    @staticmethod
    def _pid_alive(pid: int | None) -> bool:
        if not pid:
            return False
        if pid == os.getpid():
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def recover_interrupted(self, stale_after_s: float = 180) -> list[str]:
        """Requeue tasks left 'running' by a process that died, and close out usage
        rows left 'reserved' by an in-flight call that never returned. Interrupted
        tasks resume from their checkpoint via `run-due`. Returns requeued task ids."""
        now = self.now()
        requeued = []
        for t in self.list_tasks("running"):
            if not self._pid_alive(t.get("pid")):
                self.defer(t["id"], now, "interrupted: process exited mid-run")
                requeued.append(t["id"])
        cutoff = (now - timedelta(seconds=stale_after_s)).isoformat()
        with self._conn() as c:
            c.execute(
                "UPDATE usage SET status='interrupted', error='no response recorded' "
                "WHERE status='reserved' AND ts < ?", (cutoff,),
            )
        return requeued
