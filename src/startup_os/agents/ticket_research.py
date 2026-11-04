"""Research agent, ticket mode: executes any open `research` ticket that is not a
validation ticket, and writes a research note.

start → agent ⇄ tool (guarded) → [force_final] → write_note → END"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

from .. import frontmatter
from ..guards import LoopGuard
from ..llm import ModelRouter
from ..memory import MemoryRepo
from ..paths import filter_paths
from ..runlog import NullRunLog, RunLog
from ..tickets import TicketStore
from ..tools import Toolbox
from .loop import LoopState, ToolLoop, fresh_loop_state
from .research import slugify

AGENT = "research"
NOTES_DIR = "research/notes"


class Finding(BaseModel):
    finding: str
    evidence: str = ""
    source: str | None = None


class CriterionResult(BaseModel):
    criterion: str
    status: Literal["met", "partial", "unmet"]
    notes: str = ""

    @field_validator("status", mode="before")
    @classmethod
    def _s(cls, v):
        v = str(v).strip().lower()
        return {"done": "met", "pass": "met", "yes": "met", "partially met": "partial",
                "fail": "unmet", "not met": "unmet", "no": "unmet"}.get(v, v)


class ResearchNote(BaseModel):
    summary: str
    findings: list[Finding] = Field(min_length=1)
    criteria: list[CriterionResult] = Field(default_factory=list)
    recommendation: str = ""
    related_files: list[str] = Field(default_factory=list)   # company/ paths; validated
    evidence_links: list[str] = Field(default_factory=list)


NOTE_SCHEMA = """{
  "summary": "3-5 sentences answering the ticket",
  "findings": [{"finding": "...", "evidence": "...", "source": "url from a tool result"}],
  "criteria": [{"criterion": "each acceptance criterion, verbatim", "status": "met|partial|unmet", "notes": "..."}],
  "recommendation": "what the CEO should do next",
  "related_files": ["existing company/ file paths you relied on, e.g. research/validation/<file>.md"],
  "evidence_links": ["urls you actually saw in tool results"]
}"""


class TicketGraphState(LoopState, total=False):
    ticket_id: str
    run_id: str
    report_path: str | None


@dataclass
class TicketResearchDeps:
    router: ModelRouter
    memory: MemoryRepo
    toolbox: Toolbox
    guard: LoopGuard
    tickets: TicketStore
    run_log: RunLog | None = None


def system_prompt(toolbox: Toolbox, max_tools: int | None) -> str:
    return f"""You are the Research agent of Startup OS, executing a research ticket from the CEO.
Answer the ticket rigorously and skeptically using web search and the company memory.

Tools (at most {max_tools or 'a few'} calls):
{toolbox.describe()}

Reply with EXACTLY ONE JSON object. Either a tool call:
{{"action": "tool", "tool": "web_search", "args": {{"query": "..."}}}}
or the final note:
{{"action": "final", "note": {NOTE_SCHEMA}}}

Rules: never repeat an identical query; only cite URLs that appeared in tool results; only list
related_files that exist in company/ (unknown paths are dropped); mark each acceptance criterion
met/partial/unmet honestly. You cannot contact people."""


def render_note(note: ResearchNote, *, ticket, run_id: str, models: list[str], retrieved: list[str],
                related: list[str], dropped: list[dict]) -> str:
    retrieved = list(dict.fromkeys(retrieved))
    cited_only = [u for u in dict.fromkeys(note.evidence_links) if u not in retrieved]
    met = sum(c.status == "met" for c in note.criteria)
    meta = {"title": f"{ticket.id}: {ticket.title}", "ticket": ticket.id, "agent": AGENT,
            "date": datetime.now(timezone.utc).date().isoformat(), "run_id": run_id,
            "models": sorted(set(models)), "criteria_met": f"{met}/{len(note.criteria)}",
            "evidence": "web" if retrieved else "reasoning-only", "related_files": related}
    L = [f"# {ticket.id}: {ticket.title}", "", note.summary, ""]
    if not retrieved:
        L += ["> ⚠️ **Reasoning only** — no web evidence was retrieved.", ""]
    L += ["## Findings", ""]
    L += [f"- {f.finding}" + (f" — {f.evidence}" if f.evidence else "") + (f" ([source]({f.source}))" if f.source else "")
          for f in note.findings]
    if note.criteria:
        L += ["", "## Acceptance criteria", "", "| Criterion | Status | Notes |", "|---|---|---|"]
        L += [f"| {c.criterion} | {c.status} | {c.notes} |" for c in note.criteria]
    if note.recommendation:
        L += ["", "## Recommendation", "", note.recommendation]
    if related:
        L += ["", "## Related files", ""] + [f"- [{p}](../../{p})" for p in related]
    if dropped:
        L += ["", "Dropped references (not found in company/):", ""] + [f"- `{d['path']}` ({d['reason']})" for d in dropped]
    L += ["", "## Evidence links", ""] + ([f"- {u}" for u in retrieved] or ["- _none retrieved_"])
    if cited_only:
        L += ["", "Cited by the model but not seen in tool results (unverified):", ""] + [f"- {u}" for u in cited_only]
    return frontmatter.dump(meta, "\n".join(L) + "\n")


def build_ticket_graph(deps: TicketResearchDeps, checkpointer=None):
    log = deps.run_log or NullRunLog()
    loop = ToolLoop(agent=AGENT, router=deps.router, toolbox=deps.toolbox, guard=deps.guard,
                    schema=ResearchNote, final_key="note", log=log)

    def start(state: TicketGraphState) -> dict:
        t = deps.tickets.get(state["ticket_id"])
        inputs, _ = filter_paths(deps.memory, t.inputs, agent=AGENT, item=f"{t.id} inputs", log=log)
        ctx = "\n\n".join(f"### {p}\n{deps.memory.read(p)[:4000]}" for p in inputs)
        user = (f"Ticket {t.id}: {t.title}\n\n{t.description}\n\nAcceptance criteria:\n"
                + "\n".join(f"- {c}" for c in t.acceptance_criteria)
                + (f"\n\nInput files:\n{ctx}" if ctx else ""))
        log.event("node", node="ticket_start", ticket=t.id)
        return fresh_loop_state([{"role": "system", "content": system_prompt(deps.toolbox, deps.guard.max_tool_calls)},
                                 {"role": "user", "content": user}])

    def write_note(state: TicketGraphState) -> dict:
        if not state.get("final"):
            raise RuntimeError(f"research ticket produced no valid note: {state.get('error')}")
        note = ResearchNote.model_validate(state["final"])
        t = deps.tickets.get(state["ticket_id"])
        related, dropped = filter_paths(deps.memory, note.related_files, agent=AGENT,
                                        item=f"{t.id} related_files", log=log)
        content = render_note(note, ticket=t, run_id=state.get("run_id", log.run_id), models=state["models_used"],
                              retrieved=state["sources"], related=related, dropped=dropped)
        date = datetime.now(timezone.utc).date().isoformat()
        rel = deps.memory.unique_path(f"{NOTES_DIR}/{date}-{t.id.lower()}-{slugify(t.title)[:40]}.md")
        met = sum(c.status == "met" for c in note.criteria)
        done = t.model_copy(update={"status": "done", "blocked_reason": None,
                                    "result": f"Note: [{rel}](../{rel}) — criteria met {met}/{len(note.criteria)}"})
        sha = deps.memory.write_many({rel: content, **deps.tickets.files_for([done])}, agent=AGENT,
                                     message=f"research note for {t.id} ({met}/{len(note.criteria)} criteria met); "
                                             f"close {t.id}")
        log.event("note_written", path=rel, commit=sha, ticket=t.id, dropped_paths=dropped)
        return {"report_path": rel}

    g = StateGraph(TicketGraphState)
    g.add_node("start", start)
    g.add_node("agent", loop.agent_node)
    g.add_node("tool", loop.tool_node)
    g.add_node("force_final", loop.force_final_node)
    g.add_node("write_note", write_note)
    g.add_edge(START, "start")
    g.add_edge("start", "agent")
    g.add_conditional_edges("agent", loop.route, {"tool": "tool", "agent": "agent",
                                                  "force_final": "force_final", "done": "write_note"})
    g.add_edge("tool", "agent")
    g.add_edge("force_final", "write_note")
    g.add_edge("write_note", END)
    return g.compile(checkpointer=checkpointer)
