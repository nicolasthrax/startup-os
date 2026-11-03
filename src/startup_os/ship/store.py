"""SQLite records for the ship pipeline: Claude Code sessions (daily cap), build runs and steps."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable

SCHEMA = """
CREATE TABLE IF NOT EXISTS claude_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, day TEXT NOT NULL, ticket TEXT NOT NULL, venture TEXT,
    status TEXT NOT NULL,            -- started | ok | failed | timeout
    duration_s REAL, transcript TEXT
);
CREATE TABLE IF NOT EXISTS build_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket TEXT NOT NULL, venture TEXT, dry_run INTEGER NOT NULL,
    started_at TEXT NOT NULL, finished_at TEXT,
    status TEXT,                     -- shipped | dry_run_ok | failed | blocked_spend | blocked_config | capped
    commit_sha TEXT, transcript TEXT, preview_url TEXT, prod_url TEXT, log TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS build_steps (
    run_id INTEGER NOT NULL, idx INTEGER NOT NULL, step TEXT NOT NULL,
    status TEXT NOT NULL,            -- pass | fail | skip
    duration_s REAL, detail TEXT, url TEXT
);
"""


class ShipStore:
    def __init__(self, db_path: Path, now: Callable[[], datetime]):
        self.db_path, self.now = Path(db_path), now
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    # -- Claude Code session cap ------------------------------------------------
    def sessions_today(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM claude_sessions WHERE day = ?",
                             (self.now().date().isoformat(),)).fetchone()[0]

    def start_session(self, ticket: str, venture: str) -> int:
        now = self.now()
        with self._conn() as c:
            return c.execute("INSERT INTO claude_sessions(ts, day, ticket, venture, status) VALUES (?,?,?,?, 'started')",
                             (now.isoformat(), now.date().isoformat(), ticket, venture)).lastrowid

    def end_session(self, sid: int, status: str, duration_s: float, transcript: str | None) -> None:
        with self._conn() as c:
            c.execute("UPDATE claude_sessions SET status=?, duration_s=?, transcript=? WHERE id=?",
                      (status, duration_s, transcript, sid))

    # -- runs ------------------------------------------------------------------
    def start_run(self, ticket: str, venture: str, dry_run: bool, log: str) -> int:
        with self._conn() as c:
            return c.execute("INSERT INTO build_runs(ticket, venture, dry_run, started_at, log) VALUES (?,?,?,?,?)",
                             (ticket, venture, int(dry_run), self.now().isoformat(), log)).lastrowid

    def update_run(self, run_id: int, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE build_runs SET {cols} WHERE id=?", (*fields.values(), run_id))

    def add_step(self, run_id: int, idx: int, step: str, status: str, duration_s: float,
                 detail: str = "", url: str | None = None) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO build_steps VALUES (?,?,?,?,?,?,?)",
                      (run_id, idx, step, status, round(duration_s, 2), detail[:2000], url))

    def run(self, run_id: int) -> dict:
        with self._conn() as c:
            r = dict(c.execute("SELECT * FROM build_runs WHERE id=?", (run_id,)).fetchone())
            r["steps"] = [dict(x) for x in c.execute("SELECT * FROM build_steps WHERE run_id=? ORDER BY idx", (run_id,))]
        return r

    def runs(self, ticket: str | None = None) -> list[dict]:
        q, a = "SELECT id FROM build_runs", ()
        if ticket:
            q, a = q + " WHERE ticket=?", (ticket,)
        with self._conn() as c:
            ids = [r[0] for r in c.execute(q + " ORDER BY id", a)]
        return [self.run(i) for i in ids]
