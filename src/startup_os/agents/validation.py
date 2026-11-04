"""Research agent, validation mode: turns the picked idea into a go/no-go report.

start → agent ⇄ tool (guarded) → [force_final] → write_report → END"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from datetime import datetime, timezone
from typing import Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

from .. import frontmatter
from ..guards import LoopGuard
from ..llm import ModelRouter
from ..memory import MemoryRepo
from ..runlog import NullRunLog, RunLog
from ..tickets import TicketStore
from ..tools import Toolbox
from ..venture import Venture
from .loop import LoopState, ToolLoop, fresh_loop_state
from .research import slugify

AGENT = "research"
VALIDATION_DIR = "research/validation"


def _lower(v):
    return v.strip().lower() if isinstance(v, str) else v


class Teardown(BaseModel):
    name: str
    url: str | None = None
    pricing: str = "unknown"
    strengths: str
    weaknesses: str
    positioning: str = ""


class PricePoint(BaseModel):
    vendor: str
    plan: str = ""
    price: str
    source: str | None = None


class PricingScan(BaseModel):
    price_points: list[PricePoint] = Field(min_length=1)
    summary: str
    suggested_price: str


class Signal(BaseModel):
    signal: str
    strength: Literal["weak", "moderate", "strong"]
    source: str | None = None

    @field_validator("strength", mode="before")
    @classmethod
    def _s(cls, v):
        v = _lower(v)
        return {"medium": "moderate", "high": "strong", "low": "weak"}.get(v, v)


class KillCheck(BaseModel):
    criterion: str
    status: Literal["pass", "fail", "unknown"]
    evidence: str = ""

    @field_validator("status", mode="before")
    @classmethod
    def _s(cls, v):
        v = _lower(v)
        return {"met": "pass", "passed": "pass", "failed": "fail", "not met": "fail"}.get(v, v)


class Factor(BaseModel):
    factor: str
    weight: float = Field(gt=0)        # any scale; normalised by the sum of weights
    score: float = Field(ge=1, le=10)
    reasoning: str = ""

    @field_validator("weight", "score", mode="before")
    @classmethod
    def _num(cls, v):
        if isinstance(v, str):
            m = re.match(r"^\s*(\d+(?:\.\d+)?)", v)
            return float(m.group(1)) if m else v
        return v


def weighted_score(breakdown: list["Factor"]) -> Decimal:
    """Exact weighted mean of factor scores (Decimal, so 4.55 stays 4.55)."""
    total_w = sum(Decimal(str(f.weight)) for f in breakdown)
    return sum(Decimal(str(f.weight)) * Decimal(str(f.score)) for f in breakdown) / total_w


def round_half_up(x: Decimal | float) -> int:
    return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def is_go(weighted: Decimal | float, threshold: float) -> bool:
    """The decision uses the unrounded weighted score: 6.95 is NO-GO at threshold 7."""
    return Decimal(str(weighted)) >= Decimal(str(threshold))


class GoNoGo(BaseModel):
    breakdown: list[Factor] = Field(min_length=1)
    reasoning: str

    @property
    def weighted(self) -> Decimal:
        return weighted_score(self.breakdown)

    @property
    def score(self) -> int:
        return round_half_up(self.weighted)


class ValidationReport(BaseModel):
    summary: str
    competitor_teardown: list[Teardown] = Field(min_length=2)
    pricing_scan: PricingScan
    demand_signals: list[Signal] = Field(min_length=1)
    kill_criteria_check: list[KillCheck] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    go_no_go: GoNoGo
    evidence_links: list[str] = Field(default_factory=list)


REPORT_SCHEMA = """{
  "summary": "3-5 sentences: what we learned",
  "competitor_teardown": [{"name": "...", "url": "...", "pricing": "...", "strengths": "...", "weaknesses": "...", "positioning": "..."}],
  "pricing_scan": {"price_points": [{"vendor": "...", "plan": "...", "price": "$X/unit/month", "source": "url"}], "summary": "...", "suggested_price": "..."},
  "demand_signals": [{"signal": "concrete observation", "strength": "weak|moderate|strong", "source": "url"}],
  "kill_criteria_check": [{"criterion": "from the brief", "status": "pass|fail|unknown", "evidence": "..."}],
  "risks": ["..."],
  "go_no_go": {"breakdown": [{"factor": "pain & budget", "weight": 30, "score": 1-10, "reasoning": "..."},
                             {"factor": "differentiation", "weight": 25, "score": 1-10, "reasoning": "..."}],
               "reasoning": "overall verdict"},
  "evidence_links": ["urls you actually saw in tool results"]
}"""


class ValidationGraphState(LoopState, total=False):
    ticket_id: str
    run_id: str
    report_path: str | None


@dataclass
class ValidationDeps:
    router: ModelRouter
    memory: MemoryRepo
    toolbox: Toolbox
    guard: LoopGuard
    venture: Venture
    tickets: TicketStore
    go_threshold: int
    run_log: RunLog | None = None


def system_prompt(toolbox: Toolbox, max_tools: int | None, threshold: int) -> str:
    return f"""You are the Research agent of Startup OS, in VALIDATION mode.
Decide whether the picked idea deserves to be built. Be skeptical: your job is to find
reasons it will fail before we spend weeks building it.

Tools (at most {max_tools or 'a few'} calls):
{toolbox.describe()}

Reply with EXACTLY ONE JSON object. Either a tool call:
{{"action": "tool", "tool": "web_search", "args": {{"query": "..."}}}}
or the final report:
{{"action": "final", "report": {REPORT_SCHEMA}}}

Rules:
- Spend searches on: named competitors and their pricing pages, reviews/complaints, and demand evidence
  (forums, job posts, search interest). Never repeat an identical query.
- Only list URLs in evidence_links / source fields that appeared in tool results.
- Go/no-go: score each factor 1-10 (1 = clearly do not build, 10 = overwhelming evidence) with a weight.
  Do NOT compute the total: the system computes the weighted score and builds only if it is >= {threshold}
  (unrounded). Most ideas should land 4-7. Unknown is not evidence of demand."""


def render_report(rep: ValidationReport, *, venture: str, idea: str | None, ticket: str, run_id: str,
                  models: list[str], retrieved: list[str], threshold: int) -> str:
    now = datetime.now(timezone.utc)
    weighted = rep.go_no_go.weighted
    go = is_go(weighted, threshold)
    w_str = f"{weighted.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP).normalize():f}"
    retrieved_set = list(dict.fromkeys(retrieved))
    cited_only = [u for u in dict.fromkeys(rep.evidence_links) if u not in retrieved_set]
    meta = {"title": f"Validation: {venture}", "venture": venture, "idea": idea, "ticket": ticket,
            "date": now.date().isoformat(), "agent": AGENT, "run_id": run_id,
            "models": sorted(set(models)), "go_score": rep.go_no_go.score, "go_weighted": float(weighted),
            "go_threshold": threshold,
            "recommendation": "GO" if go else "NO-GO",
            "evidence": "web" if retrieved_set else "reasoning-only"}
    L = [f"# Validation report: {venture}", "",
         f"**Recommendation: {'GO' if go else 'NO-GO'}** — weighted score {w_str}/10 "
         f"(shown as {rep.go_no_go.score}/10; go needs >= {threshold} unrounded)", "", rep.summary, ""]
    if not retrieved_set:
        L += ["> ⚠️ **Reasoning only** — no web evidence was retrieved; treat every claim as unverified.", ""]
    L += ["## Competitor teardown", "", "| Competitor | Pricing | Strengths | Weaknesses |", "|---|---|---|---|"]
    for c in rep.competitor_teardown:
        name = f"[{c.name}]({c.url})" if c.url else c.name
        L.append(f"| {name} | {c.pricing} | {c.strengths} | {c.weaknesses} |")
    L += ["", "## Pricing scan", "", "| Vendor | Plan | Price | Source |", "|---|---|---|---|"]
    L += [f"| {p.vendor} | {p.plan} | {p.price} | {p.source or '—'} |" for p in rep.pricing_scan.price_points]
    L += ["", rep.pricing_scan.summary, "", f"**Suggested price:** {rep.pricing_scan.suggested_price}", "",
          "## Demand signals", ""]
    L += [f"- **{s.strength}** — {s.signal}" + (f" ([source]({s.source}))" if s.source else "")
          for s in rep.demand_signals]
    if rep.kill_criteria_check:
        L += ["", "## Kill criteria check", "", "| Criterion | Status | Evidence |", "|---|---|---|"]
        L += [f"| {k.criterion} | {k.status} | {k.evidence} |" for k in rep.kill_criteria_check]
    if rep.risks:
        L += ["", "## Risks", ""] + [f"- {r}" for r in rep.risks]
    L += ["", f"## Go / no-go: {w_str}/10 → {'GO' if go else 'NO-GO'}", "",
          "| Factor | Weight | Score | Reasoning |", "|---|---|---|---|"]
    total_w = sum(f.weight for f in rep.go_no_go.breakdown)
    L += [f"| {f.factor} | {f.weight / total_w:.0%} | {f.score:g} | {f.reasoning} |" for f in rep.go_no_go.breakdown]
    L += ["", f"Weighted score {w_str} (computed by the system; rounded half up to {rep.go_no_go.score}). "
          f"Decision compares the unrounded score to the threshold {threshold}.", "",
          rep.go_no_go.reasoning, "", "## Evidence links", ""]
    L += [f"- {u}" for u in retrieved_set] or ["- _none retrieved_"]
    if cited_only:
        L += ["", "Cited by the model but not seen in tool results (unverified):", ""]
        L += [f"- {u}" for u in cited_only]
    return frontmatter.dump(meta, "\n".join(L) + "\n")


def latest_report(memory: MemoryRepo, venture: str | None) -> tuple[str, dict] | None:
    d = memory.root / VALIDATION_DIR
    best = None
    for p in sorted(d.glob("*.md")):
        meta, _ = frontmatter.parse(p.read_text(encoding="utf-8"))
        if venture and meta.get("venture") != venture:
            continue
        best = (str(p.relative_to(memory.root)), meta)
    return best


def build_validation_graph(deps: ValidationDeps, checkpointer=None):
    log = deps.run_log or NullRunLog()
    loop = ToolLoop(agent=AGENT, router=deps.router, toolbox=deps.toolbox, guard=deps.guard,
                    schema=ValidationReport, final_key="report", log=log)

    def start(state: ValidationGraphState) -> dict:
        v = deps.venture.load()
        ticket = deps.tickets.get(state["ticket_id"])
        brief = deps.memory.read(v.idea) if v.idea and deps.memory.exists(v.idea) else "(brief missing)"
        user = (f"Venture: {v.name}\n\nValidation ticket {ticket.id}: {ticket.title}\n{ticket.description}\n\n"
                "Acceptance criteria:\n" + "\n".join(f"- {c}" for c in ticket.acceptance_criteria)
                + f"\n\nIdea brief:\n{brief[:6000]}")
        log.event("node", node="validation_start", ticket=ticket.id, venture=v.name)
        return fresh_loop_state([
            {"role": "system", "content": system_prompt(deps.toolbox, deps.guard.max_tool_calls,
                                                        deps.go_threshold)},
            {"role": "user", "content": user}])

    def write_report(state: ValidationGraphState) -> dict:
        if not state.get("final"):
            raise RuntimeError(f"validation produced no valid report: {state.get('error')}")
        rep = ValidationReport.model_validate(state["final"])
        v = deps.venture.load()
        ticket = deps.tickets.get(state["ticket_id"])
        content = render_report(rep, venture=v.name, idea=v.idea, ticket=ticket.id,
                                run_id=state.get("run_id", log.run_id), models=state["models_used"],
                                retrieved=state["sources"], threshold=deps.go_threshold)
        date = datetime.now(timezone.utc).date().isoformat()
        rel = deps.memory.unique_path(f"{VALIDATION_DIR}/{date}-{slugify(v.name or 'venture')}.md")
        go = is_go(rep.go_no_go.weighted, deps.go_threshold)
        done = ticket.model_copy(update={
            "status": "done", "blocked_reason": None,
            "result": f"Report: [{rel}](../{rel}) — {rep.go_no_go.score}/10, {'GO' if go else 'NO-GO'}"})
        sha = deps.memory.write_many({rel: content, **deps.tickets.files_for([done])}, agent=AGENT,
                                     message=f"validation report for {v.name}: {rep.go_no_go.score}/10 "
                                             f"{'GO' if go else 'NO-GO'}; close {ticket.id}")
        log.event("report_written", path=rel, commit=sha, score=rep.go_no_go.score, go=go)
        return {"report_path": rel}

    g = StateGraph(ValidationGraphState)
    g.add_node("start", start)
    g.add_node("agent", loop.agent_node)
    g.add_node("tool", loop.tool_node)
    g.add_node("force_final", loop.force_final_node)
    g.add_node("write_report", write_report)
    g.add_edge(START, "start")
    g.add_edge("start", "agent")
    g.add_conditional_edges("agent", loop.route, {"tool": "tool", "agent": "agent",
                                                  "force_final": "force_final", "done": "write_report"})
    g.add_edge("tool", "agent")
    g.add_edge("force_final", "write_report")
    g.add_edge("write_report", END)
    return g.compile(checkpointer=checkpointer)
