"""Venture state (company/venture.md), owned by the CEO agent.

Stages: Ideation → Validation → Build → Launch → Iterate. Every stage change
is written together with a decision record in company/decisions/ (one commit)."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field, field_validator

from . import frontmatter
from .memory import MemoryRepo

VENTURE_FILE = "venture.md"


class Stage(str, Enum):
    IDEATION = "Ideation"
    VALIDATION = "Validation"
    BUILD = "Build"
    LAUNCH = "Launch"
    ITERATE = "Iterate"


# Forward one step, Iterate loops back to Build, and any stage can go back to
# Ideation (no-go / pivot / kill).
ALLOWED: dict[Stage, set[Stage]] = {
    Stage.IDEATION: {Stage.VALIDATION},
    Stage.VALIDATION: {Stage.BUILD, Stage.IDEATION},
    Stage.BUILD: {Stage.LAUNCH, Stage.IDEATION},
    Stage.LAUNCH: {Stage.ITERATE, Stage.IDEATION},
    Stage.ITERATE: {Stage.BUILD, Stage.IDEATION},
}


class TransitionError(Exception):
    pass


class Decision(BaseModel):
    """A decision record: what was decided, why, and which alternatives were considered."""
    what: str = Field(min_length=3)
    why: str = Field(min_length=3)
    alternatives: list[str] = Field(min_length=1)
    decided_by: str = "ceo"

    @field_validator("alternatives", mode="before")
    @classmethod
    def _alts(cls, v):
        if isinstance(v, str):
            v = [v]
        return [a for a in (v or []) if str(a).strip()]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "decision"


class VentureState(BaseModel):
    stage: Stage = Stage.IDEATION
    name: str | None = None
    idea: str | None = None          # path of the picked brief, relative to company/
    updated: str | None = None
    history: list[dict] = Field(default_factory=list)


class Venture:
    def __init__(self, memory: MemoryRepo, now: Callable[[], datetime] | None = None):
        self.memory = memory
        self.now = now or (lambda: datetime.now(timezone.utc))

    def load(self) -> VentureState:
        if not self.memory.exists(VENTURE_FILE):
            return VentureState()
        meta, _ = frontmatter.parse(self.memory.read(VENTURE_FILE))
        meta = {k: v for k, v in meta.items() if k in VentureState.model_fields}
        return VentureState(**meta)

    @staticmethod
    def render(state: VentureState) -> str:
        meta = state.model_dump(mode="json")
        lines = [f"# Venture: {state.name or '(none selected)'}", "",
                 f"**Stage:** {state.stage.value}", ""]
        if state.idea:
            lines += [f"**Idea brief:** [{state.idea}]({state.idea})", ""]
        lines += ["Stages: " + " → ".join(
            f"**{s.value}**" if s == state.stage else s.value for s in Stage), ""]
        if state.history:
            lines += ["## Stage history", "", "| When (UTC) | From | To | Decision |", "|---|---|---|---|"]
            lines += [f"| {h['at']} | {h['from']} | {h['to']} | [{h['decision']}]({h['decision']}) |"
                      for h in state.history]
            lines.append("")
        return frontmatter.dump(meta, "\n".join(lines))

    def transition(self, to: Stage | str, decision: Decision, *, agent: str = "ceo",
                   name: str | None = None, idea: str | None = None) -> str:
        """Change stage. Writes venture.md and decisions/<date>-<slug>.md in one commit.
        Returns the decision record path."""
        to = Stage(to)
        if not isinstance(decision, Decision):
            raise TransitionError("a stage change requires a Decision record")
        state = self.load()
        if to not in ALLOWED[state.stage]:
            raise TransitionError(f"cannot move from {state.stage.value} to {to.value}; "
                                  f"allowed: {sorted(s.value for s in ALLOWED[state.stage])}")
        now = self.now()
        stamp = now.isoformat(timespec="seconds")
        rel = self.memory.unique_path(
            f"decisions/{now.date().isoformat()}-stage-{_slug(state.stage.value)}-to-{_slug(to.value)}.md")
        record = render_decision(decision, title=f"Stage change: {state.stage.value} → {to.value}",
                                 at=stamp, extra={"type": "stage_change", "from": state.stage.value,
                                                  "to": to.value, "venture": name or state.name})
        new = state.model_copy(update={
            "stage": to, "updated": stamp,
            "name": name if name is not None else state.name,
            "idea": idea if idea is not None else state.idea,
            "history": state.history + [{"at": stamp, "from": state.stage.value, "to": to.value,
                                         "decision": rel}],
        })
        if to == Stage.IDEATION:  # back to ideation clears the venture
            new.name, new.idea = None, None
        self.memory.write_many({VENTURE_FILE: self.render(new), rel: record}, agent=agent,
                               message=f"stage {state.stage.value} → {to.value}: {decision.what[:70]}")
        return rel


def render_decision(d: Decision, *, title: str, at: str, extra: dict | None = None) -> str:
    meta = {"title": title, "date": at, "decided_by": d.decided_by, **(extra or {})}
    body = [f"# {title}", "", "## What", "", d.what, "", "## Why", "", d.why, "",
            "## Alternatives considered", ""] + [f"- {a}" for a in d.alternatives] + [""]
    return frontmatter.dump(meta, "\n".join(body))
