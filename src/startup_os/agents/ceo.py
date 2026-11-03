"""CEO agent: owns the venture stage and writes the weekly plan.

plan graph:  gather → draft (JSON, validated, repaired; guarded) → write_plan → END

write_plan applies a proposed stage change (only if policy allows, with a
decision record), files spend requests through the spend gate, creates
tickets, writes plans/<ISO-week>.md and files a plan_approval."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError, field_validator

from .. import frontmatter
from ..approvals import ApprovalQueue
from ..budget import BudgetExceeded, Scheduler
from ..guards import LoopGuard, LoopGuardTripped
from ..llm import AllModelsFailed, ModelRouter
from ..memory import MemoryRepo
from ..runlog import NullRunLog, RunLog
from ..spend import SpendGate
from ..tickets import OPEN, Owner, Ticket, TicketStore
from ..venture import ALLOWED, Decision, Stage, TransitionError, Venture
from .research import IDEAS_DIR, extract_json
from .validation import latest_report

AGENT = "ceo"
PLANS_DIR = "plans"
MAX_TICKETS = 8


def iso_week(dt: datetime) -> str:
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


# ------------------------------------------------------------- schemas ---
class PlanTicket(BaseModel):
    key: str
    title: str = Field(min_length=3)
    owner: Owner
    description: str = ""
    acceptance_criteria: list[str] = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)   # other plan keys or existing T-xxxx
    needs_spend: list[str] = Field(default_factory=list)  # spend request keys

    @field_validator("owner", mode="before")
    @classmethod
    def _owner(cls, v):
        v = str(v).strip().lower().replace("_", "-").replace(" ", "-")
        return {"claude": "claude-code", "claudecode": "claude-code", "engineer": "claude-code"}.get(v, v)


class PlanSpend(BaseModel):
    key: str
    item: str
    amount_usd: float = Field(gt=0)
    justification: str
    vendor: str | None = None
    recurring: str | None = None


class StageChange(BaseModel):
    to: Stage
    what: str
    why: str
    alternatives: list[str] = Field(min_length=1)


class IdeaPickProposal(BaseModel):
    brief: str
    why: str


class PlanDraft(BaseModel):
    summary: str
    goals: list[str] = Field(min_length=1, max_length=5)
    tickets: list[PlanTicket] = Field(default_factory=list, max_length=MAX_TICKETS)
    budget_split: dict[str, float]
    budget_rationale: str = ""
    spend_requests: list[PlanSpend] = Field(default_factory=list)
    stage_change: StageChange | None = None
    idea_pick: IdeaPickProposal | None = None
    risks: list[str] = Field(default_factory=list)

    @field_validator("stage_change", "idea_pick", mode="before")
    @classmethod
    def _empty(cls, v):
        return None if v in ({}, "", "null", "none") else v


PLAN_SCHEMA = """{
  "summary": "2-4 sentences: where we are and the focus this week",
  "goals": ["1-5 measurable goals for this week"],
  "tickets": [{"key": "t1", "title": "...", "owner": "research|claude-code", "description": "...",
               "acceptance_criteria": ["testable"], "depends_on": ["t0 or T-0001"], "needs_spend": ["s1"]}],
  "budget_split": {"<agent>": percent, ...},
  "budget_rationale": "why this split",
  "spend_requests": [{"key": "s1", "item": "...", "amount_usd": 12.0, "vendor": "...", "recurring": "monthly|null", "justification": "..."}],
  "stage_change": null or {"to": "Build|Ideation|...", "what": "...", "why": "...", "alternatives": ["..."]},
  "idea_pick": null or {"brief": "research/ideas/<file>.md", "why": "..."},
  "risks": ["..."]
}"""


class PlanState(TypedDict, total=False):
    run_id: str
    week: str
    context: str
    messages: list[dict]
    steps: int
    draft: dict | None
    models_used: list[str]
    plan_path: str | None
    approval: str | None
    notes: list[str]


@dataclass
class CEODeps:
    router: ModelRouter
    memory: MemoryRepo
    venture: Venture
    tickets: TicketStore
    approvals: ApprovalQueue
    spend_gate: SpendGate
    scheduler: Scheduler
    guard: LoopGuard
    go_threshold: int
    agents: list[str]
    now: Callable[[], datetime]
    run_log: RunLog | None = None


# ------------------------------------------------------------- context ---
def list_briefs(memory: MemoryRepo) -> list[dict]:
    out = []
    for p in sorted((memory.root / IDEAS_DIR).glob("*.md")):
        meta, body = frontmatter.parse(p.read_text(encoding="utf-8"))
        m = re.search(r"^>\s*(.+)$", body, re.MULTILINE)
        out.append({"path": str(p.relative_to(memory.root)), "title": meta.get("title", p.stem),
                    "score": meta.get("demand_score"), "evidence": meta.get("evidence"),
                    "one_liner": m.group(1).strip() if m else ""})
    return out


def latest_file(memory: MemoryRepo, folder: str) -> str | None:
    files = sorted((memory.root / folder).glob("*.md"))
    return str(files[-1].relative_to(memory.root)) if files else None


def stage_policy(to: Stage, current: Stage, report: dict | None, threshold: int) -> str | None:
    """Return a reason the transition is NOT allowed, or None if it is."""
    if to not in ALLOWED[current]:
        return f"{current.value} → {to.value} is not an allowed transition"
    if current == Stage.VALIDATION and to == Stage.BUILD:
        if not report:
            return "Validation → Build needs a validation report"
        if (report.get("go_score") or 0) < threshold:
            return f"validation score {report.get('go_score')}/10 is below the go threshold {threshold}"
    if current == Stage.VALIDATION and to == Stage.IDEATION and report and \
            (report.get("go_score") or 0) >= threshold:
        return "validation says GO; dropping the idea needs a human decision"
    return None


def gather_context(deps: CEODeps) -> str:
    m = deps.memory
    v = deps.venture.load()
    parts = [f"Today: {deps.now().date().isoformat()} (ISO week {iso_week(deps.now())})",
             f"## venture.md\n{m.read('venture.md')}"]
    if v.stage == Stage.IDEATION:
        briefs = sorted(list_briefs(m), key=lambda b: -(b["score"] or 0))
        parts.append("## Latest research: idea briefs (no venture picked yet)\n" + "\n".join(
            f"- {b['path']} — {b['title']} ({b['score']}/10, {b['evidence']}): {b['one_liner']}" for b in briefs)
            if briefs else "## Latest research\n(no idea briefs yet)")
    else:
        rep = latest_report(m, v.name)
        if rep:
            parts.append(f"## Latest research: validation report ({rep[0]})\n{m.read(rep[0])[:7000]}")
        else:
            parts.append("## Latest research\nNo validation report yet for this venture.")
        if v.idea and m.exists(v.idea):
            parts.append(f"## Picked idea brief ({v.idea})\n{m.read(v.idea)[:4000]}")
    retro = latest_file(m, "retros")
    parts.append(f"## Last retro ({retro})\n{m.read(retro)[:3000]}" if retro else "## Last retro\nNone yet.")
    open_t = deps.tickets.list(status=OPEN)
    parts.append("## Open tickets\n" + ("\n".join(
        f"- {t.id} [{t.owner}, {t.status}] {t.title}" + (f" (blocked: {t.blocked_reason})" if t.blocked_reason else "")
        for t in open_t) or "None."))
    pend = deps.approvals.list(status="pending")
    parts.append("## Pending approvals\n" + ("\n".join(f"- {a['ref']} {a['kind']}: {a['title']}" for a in pend) or "None."))
    shares = deps.scheduler.effective_shares()
    parts.append(f"## Call budget\n{deps.scheduler.daily_requests} LLM requests/day, current split "
                 f"(from {deps.scheduler.share_source()}): " + json.dumps(shares) +
                 "\nUsage today: " + json.dumps({r['agent']: r['requests'] for r in deps.scheduler.usage_summary()}))
    return "\n\n".join(parts)


def system_prompt(deps: CEODeps) -> str:
    return f"""You are the CEO agent of Startup OS, an autonomous startup studio run by a tiny team of agents
supervised by one human. You own the venture stage (Ideation → Validation → Build → Launch → Iterate)
and write the weekly plan. Be concrete, frugal and skeptical.

Agents you can assign tickets to:
- research: market/competitor/customer research (web search + reasoning). Cannot talk to people.
- claude-code: writes and ships code. Tickets for it are queued for later; only create them when the
  stage is Build or later, or when a stage change to Build is part of this plan.

Rules:
- Stage changes need evidence. Validation → Build only if the latest validation report's go score is
  >= {deps.go_threshold}; if it is below, move back to Ideation (or keep validating if evidence is thin).
  Every stage change needs what/why/alternatives.
- In Ideation with no venture, you may propose idea_pick (a brief path) for the human to approve.
- budget_split: percentages per agent across exactly these agents: {deps.agents}. Sum <= 100.
- Spending money is never automatic. Only add spend_requests when a ticket truly cannot proceed
  without it (e.g. a domain, an API plan); keep amounts small and justified. Tickets that need money
  list the spend key in needs_spend and stay blocked until a human approves.
- No outbound messages to people, no ads, no deploys this week.
- At most {MAX_TICKETS} tickets; acceptance criteria must be testable.

Reply with ONE JSON object: {{"action": "final", "plan": {PLAN_SCHEMA}}}"""


# -------------------------------------------------------------- render ---
def render_plan(*, week: str, draft: PlanDraft, venture_name: str | None, stage: str,
                tickets: list[Ticket], spend: list[dict], approval: dict, shares_now: dict,
                daily: int, notes: list[str], run_id: str, models: list[str], created: str) -> str:
    meta = {"week": week, "status": "pending_approval", "approval": approval["ref"],
            "auto_approve_at": approval["auto_approve_at"], "venture": venture_name, "stage": stage,
            "created": created, "agent": AGENT, "run_id": run_id, "models": sorted(set(models)),
            "budget_split": draft.budget_split, "tickets": [t.id for t in tickets],
            "spend_requests": [s["ref"] for s in spend]}
    L = [f"# Plan {week} — {venture_name or 'no venture yet'} ({stage})", "", f"> {draft.summary}", "",
         "## Goals", ""] + [f"{i}. {g}" for i, g in enumerate(draft.goals, 1)]
    L += ["", "## Tickets", ""]
    if tickets:
        L += ["| ID | Owner | Title | Status | Depends on |", "|---|---|---|---|---|"]
        L += [f"| [{t.id}](../tickets/) | {t.owner} | {t.title} | {t.status} | {', '.join(t.depends_on) or '—'} |"
              for t in tickets]
        if any(t.owner == "claude-code" for t in tickets):
            L += ["", "_claude-code tickets are queued only; they are not executed in this slice._"]
    else:
        L.append("None.")
    L += ["", "## Call-budget split", "", f"Daily quota: {daily} requests.", "",
          "| Agent | Current | Proposed | Requests/day |", "|---|---|---|---|"]
    for a in sorted(set(shares_now) | set(draft.budget_split)):
        p = draft.budget_split.get(a, 0)
        L.append(f"| {a} | {shares_now.get(a, 0):g}% | {p:g}% | {int(daily * p / 100)} |")
    if draft.budget_rationale:
        L += ["", draft.budget_rationale]
    L += ["", "The proposed split takes effect when this plan is approved.", "", "## Spend requests", ""]
    if spend:
        L += ["| ID | Item | Amount (USD) | Status | Blocks |", "|---|---|---|---|---|"]
        L += [f"| {s['ref']} | {s['payload']['item']} | {s['payload']['amount_usd']:,.2f}"
              f"{' / ' + s['payload']['recurring'] if s['payload'].get('recurring') else ''} | "
              f"{s['status']} — needs human approval, never auto-approves | {', '.join(s['payload']['blocks']) or '—'} |"
              for s in spend]
    else:
        L.append("None.")
    if notes:
        L += ["", "## Stage and decisions", ""] + [f"- {n}" for n in notes]
    if draft.risks:
        L += ["", "## Risks", ""] + [f"- {r}" for r in draft.risks]
    L += ["", "## Approval", "",
          f"Pending as **{approval['ref']}**. Auto-approves at {approval['auto_approve_at']} (48h) unless decided:",
          "", f"    startup-os approve {approval['ref']}", f"    startup-os reject {approval['ref']} --reason \"...\"", ""]
    return frontmatter.dump(meta, "\n".join(L))


# --------------------------------------------------------------- graph ---
def build_plan_graph(deps: CEODeps, checkpointer=None):
    log = deps.run_log or NullRunLog()

    def gather(state: PlanState) -> dict:
        ctx = gather_context(deps)
        log.event("node", node="gather", chars=len(ctx))
        return {"week": iso_week(deps.now()), "context": ctx, "steps": 0, "draft": None,
                "models_used": [], "notes": [],
                "messages": [{"role": "system", "content": system_prompt(deps)},
                             {"role": "user", "content": f"Write this week's plan.\n\n{ctx}"}]}

    def draft(state: PlanState) -> dict:
        try:
            deps.guard.check_step(state["steps"])
        except LoopGuardTripped as e:
            raise RuntimeError(f"CEO could not produce a valid plan: {e}") from e
        res = deps.router.complete(AGENT, state["messages"])
        msgs = state["messages"] + [{"role": "assistant", "content": res.text}]
        upd: dict[str, Any] = {"steps": state["steps"] + 1, "models_used": state["models_used"] + [res.model]}
        try:
            obj = extract_json(res.text)
            d = PlanDraft.model_validate(obj.get("plan") if isinstance(obj.get("plan"), dict) else obj)
            problems = check_draft(d, deps)
            if not problems:
                return upd | {"messages": msgs, "draft": d.model_dump(mode="json")}
            feedback = "Fix these problems and resend the full JSON: " + "; ".join(problems)
        except ValueError as e:  # includes ValidationError
            feedback = f"Invalid plan JSON: {str(e)[:1500]}. Resend the full JSON object."
        return upd | {"messages": msgs + [{"role": "user", "content": feedback}]}

    def write_plan(state: PlanState) -> dict:
        d = PlanDraft.model_validate(state["draft"])
        week, notes = state["week"], []
        v = deps.venture.load()
        now = deps.now()

        # 1. stage change (policy-checked, with decision record)
        if d.stage_change and d.stage_change.to != v.stage:
            rep = latest_report(deps.memory, v.name)
            why_not = stage_policy(d.stage_change.to, v.stage, rep[1] if rep else None, deps.go_threshold)
            if why_not:
                notes.append(f"Proposed stage change to {d.stage_change.to.value} was **refused by policy**: {why_not}.")
            else:
                why = d.stage_change.why + (f" (evidence: {rep[0]}, score {rep[1].get('go_score')}/10)" if rep else "")
                rel = deps.venture.transition(d.stage_change.to, Decision(
                    what=d.stage_change.what, why=why, alternatives=d.stage_change.alternatives,
                    decided_by="ceo"), agent=AGENT)
                notes.append(f"Stage changed {v.stage.value} → {d.stage_change.to.value}; decision record: [{rel}](../{rel}).")
                v = deps.venture.load()

        # 2. idea pick proposal → approval (human decides)
        if d.idea_pick and v.stage == Stage.IDEATION:
            if deps.memory.exists(d.idea_pick.brief):
                a = deps.approvals.create("idea_pick", f"Pick {d.idea_pick.brief}",
                                          {"brief": d.idea_pick.brief, "why": d.idea_pick.why}, requested_by=AGENT)
                notes.append(f"Proposed idea pick {d.idea_pick.brief} as {a['ref']} (needs human approval).")
            else:
                notes.append(f"Ignored idea pick for missing brief {d.idea_pick.brief}.")

        # 3. supersede an earlier pending plan for the same week, cancel its open tickets
        for a in deps.approvals.list(status="pending", kind="plan_approval"):
            if a["payload"].get("week") == week:
                deps.approvals.supersede(a["id"], reason="replaced by a newer plan for the same week")
                stale = [t.model_copy(update={"status": "cancelled", "blocked_reason": "plan superseded"})
                         for t in deps.tickets.list(status=OPEN) if t.plan == week]
                if stale:
                    deps.memory.write_many(deps.tickets.files_for(stale), agent=AGENT,
                                           message=f"cancel tickets of superseded plan {week}")
                notes.append(f"Supersedes earlier plan approval {a['ref']}.")

        # 4. spend requests go through the spend gate only
        spend_ref: dict[str, str] = {}
        key_to_id: dict[str, str] = {}
        next_num = int(deps.tickets.next_id()[2:])
        for t in d.tickets:
            key_to_id[t.key] = f"T-{next_num:04d}"
            next_num += 1
        spend_rows = []
        for s in d.spend_requests:
            blocks = [key_to_id[t.key] for t in d.tickets if s.key in t.needs_spend]
            a = deps.spend_gate.request({**s.model_dump(exclude={"key"}), "blocks": blocks}, agent=AGENT)
            spend_ref[s.key] = a["ref"]
            spend_rows.append(a)

        # 5. tickets
        created = now.isoformat(timespec="seconds")
        tickets = []
        for t in d.tickets:
            deps_ = [key_to_id.get(x, x) for x in t.depends_on] + [spend_ref[x] for x in t.needs_spend if x in spend_ref]
            tickets.append(Ticket(id=key_to_id[t.key], title=t.title, owner=t.owner, kind="task",
                                  description=t.description, acceptance_criteria=t.acceptance_criteria,
                                  depends_on=deps_, created=created, created_by=AGENT, plan=week, venture=v.name))
        # status from dependencies (spend approvals are pending → blocked)
        known = {t.id: t for t in tickets}
        final = []
        for t in tickets:
            reasons = []
            for dep in t.depends_on:
                if dep.startswith("A-"):
                    st = deps.approvals.status(int(dep[2:]))
                    if st not in ("approved", "auto_approved"):
                        reasons.append(f"{dep} is {st}")
                elif dep in known or not dep.startswith("T-"):
                    reasons.append(f"{dep} is todo")
                else:
                    try:
                        if deps.tickets.get(dep).status != "done":
                            reasons.append(f"{dep} is {deps.tickets.get(dep).status}")
                    except Exception:  # noqa: BLE001
                        reasons.append(f"{dep} is missing")
            final.append(t.model_copy(update={"status": "blocked" if reasons else "todo",
                                              "blocked_reason": "; ".join(reasons) or None}))

        # 6. plan approval + plan file, committed with the tickets
        rel = f"{PLANS_DIR}/{week}.md"
        approval = deps.approvals.create("plan_approval", f"Plan {week}" + (f" — {v.name}" if v.name else ""),
                                         {"week": week, "path": rel, "budget_split": d.budget_split,
                                          "tickets": [t.id for t in final]}, requested_by=AGENT)
        content = render_plan(week=week, draft=d, venture_name=v.name, stage=v.stage.value, tickets=final,
                              spend=[deps.approvals.get(s["id"]) for s in spend_rows], approval=approval,
                              shares_now=deps.scheduler.effective_shares(), daily=deps.scheduler.daily_requests,
                              notes=notes, run_id=state.get("run_id", log.run_id),
                              models=state["models_used"], created=created)
        sha = deps.memory.write_many({rel: content, **deps.tickets.files_for(final)}, agent=AGENT,
                                     message=f"plan {week}: {len(final)} tickets, {len(spend_rows)} spend requests "
                                             f"(awaiting {approval['ref']})")
        log.event("plan_written", path=rel, commit=sha, approval=approval["ref"],
                  tickets=[t.id for t in final], spend=[s["ref"] for s in spend_rows], notes=notes)
        return {"plan_path": rel, "approval": approval["ref"], "notes": notes}

    def route(state: PlanState) -> str:
        return "write_plan" if state.get("draft") else "draft"

    g = StateGraph(PlanState)
    g.add_node("gather", gather)
    g.add_node("draft", draft)
    g.add_node("write_plan", write_plan)
    g.add_edge(START, "gather")
    g.add_edge("gather", "draft")
    g.add_conditional_edges("draft", route, {"draft": "draft", "write_plan": "write_plan"})
    g.add_edge("write_plan", END)
    return g.compile(checkpointer=checkpointer)


def check_draft(d: PlanDraft, deps: CEODeps) -> list[str]:
    problems = []
    unknown = set(d.budget_split) - set(deps.agents)
    if unknown:
        problems.append(f"budget_split has unknown agents {sorted(unknown)}; use only {deps.agents}")
    if sum(d.budget_split.values()) > 100 + 1e-9:
        problems.append(f"budget_split sums to {sum(d.budget_split.values()):g}%, must be <= 100")
    if any(v < 0 for v in d.budget_split.values()):
        problems.append("budget_split percentages must be >= 0")
    keys = [t.key for t in d.tickets]
    if len(set(keys)) != len(keys):
        problems.append("ticket keys must be unique")
    spend_keys = {s.key for s in d.spend_requests}
    for t in d.tickets:
        for dep in t.depends_on:
            if dep not in keys and not re.match(r"^T-\d{4,}$", dep):
                problems.append(f"ticket {t.key} depends on unknown {dep!r}")
        for s in t.needs_spend:
            if s not in spend_keys:
                problems.append(f"ticket {t.key} needs unknown spend request {s!r}")
    return problems


# ---------------------------------------------------- validation ticket ---
class TicketDraft(BaseModel):
    title: str
    description: str
    acceptance_criteria: list[str] = Field(min_length=2)


def validation_ticket(deps: CEODeps, *, venture: str, brief_path: str) -> tuple[Ticket, str]:
    """CEO writes the Research validation ticket for a freshly picked idea.
    Falls back to a template if the LLM is unavailable (budget/all models failed)."""
    brief = deps.memory.read(brief_path)
    meta, _ = frontmatter.parse(brief)
    msgs = [{"role": "system", "content": "You are the CEO agent. Reply with one JSON object only."},
            {"role": "user", "content": f"""We just picked "{venture}" to validate. Write ONE ticket for the Research
agent (web search + reasoning only; it cannot contact people) to produce a validation report:
competitor teardown, pricing scan, demand signals, kill-criteria check and a go/no-go score.
Make acceptance criteria specific to this idea.

Brief ({brief_path}):
{brief[:5000]}

JSON: {{"title": "...", "description": "...", "acceptance_criteria": ["...", "..."]}}"""}]
    source = "ceo"
    try:
        res = deps.router.complete(AGENT, msgs)
        td = TicketDraft.model_validate(extract_json(res.text))
    except (BudgetExceeded, AllModelsFailed, ValueError, ValidationError) as e:
        (deps.run_log or NullRunLog()).event("validation_ticket_fallback", error=str(e)[:300])
        source = "template"
        td = TicketDraft(
            title=f"Validate {venture}: competitors, pricing, demand, go/no-go",
            description=f"Validate the picked idea ({brief_path}) before any build work.",
            acceptance_criteria=[
                "Teardown of at least 4 named competitors with pricing, strengths and weaknesses",
                "Pricing scan with at least 3 public price points and a suggested price",
                "At least 3 demand signals with source links",
                "Each kill criterion from the brief marked pass / fail / unknown with evidence",
                f"Go/no-go score 1-10 with reasoning (go threshold {deps.go_threshold})"])
    t = deps.tickets.new(title=td.title, owner="research", kind="validation", description=td.description,
                         acceptance_criteria=td.acceptance_criteria, created_by=AGENT, venture=venture)
    return t, source
