"""Human approval queue (SQLite, data/state.db).

Kinds:
- plan_approval: auto-approves 48h after creation unless decided earlier.
- spend_request: never auto-approves and never expires into approval; tickets
  that depend on it stay blocked until a human approves it.
- idea_pick: picking the venture's idea.
Every decision is also appended to company/approvals.log (committed)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

KINDS = ("plan_approval", "spend_request", "idea_pick")
AUTO_APPROVE_AFTER = {"plan_approval": timedelta(hours=48)}   # nothing else ever auto-approves
DECIDED = ("approved", "auto_approved", "rejected", "superseded")

SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    requested_by TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | approved | auto_approved | rejected | superseded
    created_at TEXT NOT NULL,
    auto_approve_at TEXT,
    decided_at TEXT,
    decided_by TEXT,
    reason TEXT
);
"""


class ApprovalError(Exception):
    pass


def parse_id(s: str | int) -> int:
    s = str(s).strip().upper()
    return int(s[2:] if s.startswith("A-") else s)


class ApprovalQueue:
    def __init__(self, db_path: Path, now: Callable[[], datetime] | None = None,
                 on_decided: Callable[[dict], None] | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.on_decided = on_decided  # hook: apply effects + log
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    @staticmethod
    def _row(r) -> dict:
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        d["ref"] = f"A-{d['id']}"
        return d

    def create(self, kind: str, title: str, payload: dict, *, requested_by: str) -> dict:
        if kind not in KINDS:
            raise ApprovalError(f"unknown approval kind {kind!r}")
        now = self.now()
        auto = AUTO_APPROVE_AFTER.get(kind)
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO approvals(kind, title, requested_by, payload, created_at, auto_approve_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (kind, title, requested_by, json.dumps(payload), now.isoformat(),
                 (now + auto).isoformat() if auto else None),
            )
            new_id = cur.lastrowid
        return self.get(new_id)

    def get(self, aid: int | str) -> dict:
        with self._conn() as c:
            r = c.execute("SELECT * FROM approvals WHERE id = ?", (parse_id(aid),)).fetchone()
        if not r:
            raise ApprovalError(f"no approval A-{aid}")
        return self._row(r)

    def status(self, aid: int) -> str | None:
        try:
            return self.get(aid)["status"]
        except ApprovalError:
            return None

    def list(self, status: str | None = None, kind: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM approvals WHERE 1=1", []
        if status:
            q += " AND status = ?"; args.append(status)
        if kind:
            q += " AND kind = ?"; args.append(kind)
        with self._conn() as c:
            return [self._row(r) for r in c.execute(q + " ORDER BY id", args)]

    def _decide(self, aid: int | str, status: str, by: str, reason: str | None) -> dict:
        a = self.get(aid)
        if a["status"] != "pending":
            raise ApprovalError(f"{a['ref']} is already {a['status']}")
        with self._conn() as c:
            c.execute("UPDATE approvals SET status=?, decided_at=?, decided_by=?, reason=? WHERE id=?",
                      (status, self.now().isoformat(), by, reason, a["id"]))
        a = self.get(a["id"])
        if self.on_decided:
            try:
                self.on_decided(a)
            except BaseException:
                # The effect failed: undo the decision so the item stays actionable.
                with self._conn() as c:
                    c.execute("UPDATE approvals SET status='pending', decided_at=NULL, decided_by=NULL, "
                              "reason=NULL WHERE id=?", (a["id"],))
                raise
        return a

    def approve(self, aid: int | str, *, by: str = "human", reason: str | None = None) -> dict:
        return self._decide(aid, "approved", by, reason)

    def reject(self, aid: int | str, *, reason: str, by: str = "human") -> dict:
        if not reason or not reason.strip():
            raise ApprovalError("a rejection needs a reason")
        return self._decide(aid, "rejected", by, reason)

    def supersede(self, aid: int | str, *, reason: str) -> dict:
        return self._decide(aid, "superseded", "system", reason)

    def process_due(self) -> list[dict]:
        """Auto-approve due plan approvals. Spend requests and idea picks are never touched."""
        now = self.now()
        out = []
        for a in self.list(status="pending", kind="plan_approval"):
            if a["auto_approve_at"] and datetime.fromisoformat(a["auto_approve_at"]) <= now:
                out.append(self._decide(a["id"], "auto_approved", "system",
                                        "no human decision within 48h"))
        return out
