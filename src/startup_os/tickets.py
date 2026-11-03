"""Tickets: company/tickets/T-0001-<slug>.md, Markdown with YAML front matter.

Owners are `research` (executed by the Research agent) or `claude-code` (created
only; not executed in this slice). `depends_on` lists ticket ids (T-0001) and
approval ids (A-3); a ticket is blocked until its tickets are done and its
approvals are approved."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Callable, Literal

from pydantic import BaseModel, Field, field_validator

from . import frontmatter
from .memory import MemoryRepo

TICKETS_DIR = "tickets"
Owner = Literal["research", "claude-code"]
Status = Literal["todo", "blocked", "in_progress", "done", "cancelled"]
OPEN = ("todo", "blocked", "in_progress")
_ID = re.compile(r"^T-\d{4,}$")
_DEP = re.compile(r"^(T-\d{4,}|A-\d+)$")


class TicketError(Exception):
    pass


class Ticket(BaseModel):
    id: str
    title: str = Field(min_length=3)
    owner: Owner
    status: Status = "todo"
    kind: str = "task"
    depends_on: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(min_length=1)
    created: str | None = None
    created_by: str = "ceo"
    plan: str | None = None
    venture: str | None = None
    blocked_reason: str | None = None
    result: str | None = None
    description: str = ""          # Markdown body, not front matter

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not _ID.match(v):
            raise ValueError(f"bad ticket id {v!r} (expected T-0001)")
        return v

    @field_validator("depends_on", mode="before")
    @classmethod
    def _deps(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",")]
        out = [str(x).strip() for x in v if str(x).strip()]
        for d in out:
            if not _DEP.match(d):
                raise ValueError(f"bad dependency {d!r} (expected T-0001 or A-3)")
        return out

    @field_validator("acceptance_criteria", mode="before")
    @classmethod
    def _ac(cls, v):
        if isinstance(v, str):
            v = [v]
        return [str(x).strip() for x in (v or []) if str(x).strip()]

    def to_markdown(self) -> str:
        meta = self.model_dump(exclude={"description"}, exclude_none=True)
        body = [f"# {self.id}: {self.title}", "", self.description.strip() or "_No description._", "",
                "## Acceptance criteria", ""] + [f"- [ ] {c}" for c in self.acceptance_criteria]
        if self.result:
            body += ["", "## Result", "", self.result]
        return frontmatter.dump(meta, "\n".join(body) + "\n")

    @classmethod
    def from_markdown(cls, text: str) -> "Ticket":
        meta, body = frontmatter.parse(text)
        if not meta:
            raise TicketError("ticket has no front matter")
        m = re.search(r"\A#[^\n]*\n+(.*?)(?:\n## Acceptance criteria|\Z)", body, re.DOTALL)
        desc = m.group(1).strip() if m else ""
        return cls(**meta, description="" if desc == "_No description._" else desc)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:50] or "ticket"


class TicketStore:
    def __init__(self, memory: MemoryRepo, now: Callable[[], datetime] | None = None):
        self.memory = memory
        self.now = now or (lambda: datetime.now(timezone.utc))

    @property
    def dir(self):
        return self.memory.root / TICKETS_DIR

    def _paths(self) -> dict[str, str]:
        out = {}
        for p in sorted(self.dir.glob("T-*.md")):
            out[p.name.split("-")[0] + "-" + p.name.split("-")[1]] = str(p.relative_to(self.memory.root))
        return out

    def list(self, *, owner: str | None = None, status: str | tuple | None = None,
             kind: str | None = None) -> list[Ticket]:
        out = []
        for rel in self._paths().values():
            t = Ticket.from_markdown(self.memory.read(rel))
            if owner and t.owner != owner:
                continue
            if status and t.status not in ((status,) if isinstance(status, str) else status):
                continue
            if kind and t.kind != kind:
                continue
            out.append(t)
        return out

    def get(self, tid: str) -> Ticket:
        rel = self._paths().get(tid)
        if not rel:
            raise TicketError(f"no ticket {tid}")
        return Ticket.from_markdown(self.memory.read(rel))

    def path_of(self, tid: str) -> str:
        return self._paths()[tid]

    def next_id(self) -> str:
        nums = [int(k.split("-")[1]) for k in self._paths()]
        return f"T-{(max(nums) + 1) if nums else 1:04d}"

    def new(self, **fields) -> Ticket:
        """Build (not save) a ticket with the next free id."""
        fields.setdefault("created", self.now().isoformat(timespec="seconds"))
        return Ticket(id=fields.pop("id", None) or self.next_id(), **fields)

    def files_for(self, tickets: list[Ticket]) -> dict[str, str]:
        """Map path → content, for committing several tickets together."""
        existing = self._paths()
        return {existing.get(t.id, f"{TICKETS_DIR}/{t.id}-{_slug(t.title)}.md"): t.to_markdown()
                for t in tickets}

    def save(self, ticket: Ticket, *, agent: str, message: str) -> str:
        return self.memory.write_many(self.files_for([ticket]), agent=agent, message=message)

    def blocking(self, t: Ticket, approval_status: Callable[[int], str | None]) -> list[str]:
        """Reasons this ticket is blocked (empty if it can start)."""
        reasons = []
        for dep in t.depends_on:
            if dep.startswith("A-"):
                st = approval_status(int(dep[2:]))
                if st not in ("approved", "auto_approved"):
                    reasons.append(f"{dep} is {st or 'missing'}")
            else:
                try:
                    dt = self.get(dep)
                except TicketError:
                    reasons.append(f"{dep} is missing")
                    continue
                if dt.status != "done":
                    reasons.append(f"{dep} is {dt.status}")
        return reasons

    def refresh(self, approval_status: Callable[[int], str | None], *, agent: str = "system") -> list[str]:
        """Recompute todo/blocked for open tickets; commit changes. Returns changed ids."""
        changed = []
        for t in self.list(status=("todo", "blocked")):
            reasons = self.blocking(t, approval_status)
            status = "blocked" if reasons else "todo"
            reason = "; ".join(reasons) or None
            if (status, reason) != (t.status, t.blocked_reason):
                changed.append(t.model_copy(update={"status": status, "blocked_reason": reason}))
        if changed:
            self.memory.write_many(self.files_for(changed), agent=agent,
                                   message="update ticket status: " + ", ".join(
                                       f"{t.id} {t.status}" for t in changed))
        return [t.id for t in changed]
