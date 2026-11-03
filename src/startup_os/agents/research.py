"""Research agent: a LangGraph graph that generates N startup idea briefs.

seed ──► start_idea ──► agent ──► tool ──┐
                         ▲  │            │
                         │  └────────────┘ (loop, guarded)
                         │  ├─► force_final (guard tripped)
                         │  ▼
                         └─ write (commit brief) ──► END when all ideas done
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError, field_validator

from ..guards import LoopGuard, LoopGuardTripped
from ..llm import ModelRouter
from ..memory import MemoryRepo
from ..runlog import NullRunLog, RunLog
from ..tools import Toolbox

AGENT = "research"
IDEAS_DIR = "research/ideas"
TOOL_RESULT_CHARS = 3000


# ---------------------------------------------------------------- schema ---
class Competitor(BaseModel):
    name: str
    note: str = ""


class DemandSignal(BaseModel):
    score: int = Field(ge=1, le=10)
    reasoning: str

    @field_validator("score", mode="before")
    @classmethod
    def _coerce(cls, v: Any) -> Any:
        if isinstance(v, str) and re.match(r"^\s*\d+", v):
            return int(re.match(r"^\s*(\d+)", v).group(1))
        if isinstance(v, float):
            return round(v)
        return v


class IdeaBrief(BaseModel):
    title: str
    one_liner: str
    problem: str
    icp: str
    why_now: str
    competitors: list[Competitor] = Field(min_length=1)
    kill_criteria: list[str] = Field(min_length=1)
    demand_signal: DemandSignal

    @field_validator("competitors", mode="before")
    @classmethod
    def _competitors(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [{"name": c} if isinstance(c, str) else c for c in v]
        return v


class ResearchState(TypedDict, total=False):
    n: int
    run_id: str
    seeds: list[dict]
    idx: int
    messages: list[dict]
    steps: int
    tool_sigs: list[str]
    sources: list[str]
    web_hits: int
    web_attempts: int
    pending_tool: dict | None
    final: dict | None
    guard_note: str | None
    models_used: list[str]
    written: list[str]
    failed: list[dict]


@dataclass
class ResearchDeps:
    router: ModelRouter
    memory: MemoryRepo
    toolbox: Toolbox
    guard: LoopGuard
    run_log: RunLog | None = None


# ------------------------------------------------------------- helpers ---
def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply (handles code fences / prose)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    start = text.find("{")
    if start != -1:
        depth, in_str, esc = 0, False, False
        for i, ch in enumerate(text[start:], start):
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                elif ch != "\\":
                    esc = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object found in reply")


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "idea"


def existing_idea_titles(memory: MemoryRepo) -> list[str]:
    d = memory.root / IDEAS_DIR
    titles = []
    for p in sorted(d.glob("*.md")):
        m = re.search(r"^#\s+(.+)$", p.read_text(encoding="utf-8"), re.MULTILINE)
        if m:
            titles.append(m.group(1).strip())
    return titles


BRIEF_SCHEMA = """{
  "title": "short name",
  "one_liner": "one sentence pitch",
  "problem": "the painful problem, concretely",
  "icp": "ideal customer profile: who exactly buys, role, company size, budget owner",
  "why_now": "what changed recently that makes this possible or urgent",
  "competitors": [{"name": "...", "note": "how they solve it and their gap"}],
  "kill_criteria": ["specific, testable signal that would make us drop this idea", "..."],
  "demand_signal": {"score": 1-10 integer, "reasoning": "why this score, citing evidence"}
}"""


def system_prompt(toolbox: Toolbox, max_tools: int | None = None) -> str:
    web = toolbox.web_enabled
    return (f"""You are the Research agent of Startup OS, an autonomous startup studio.
Your job: turn a startup idea seed into a rigorous, skeptical idea brief.

You can call tools. Available tools:
{toolbox.describe()}

Reply with EXACTLY ONE JSON object and nothing else. Either a tool call:
{{"action": "tool", "tool": "{'web_search' if web else 'search'}", "args": {{"query": "..."}}}}
or, when you are done, the final brief:
{{"action": "final", "brief": {BRIEF_SCHEMA}}}

Rules:
- You have at most {{max_tools}} tool calls for this brief. {"Use web_search 1-3 times to ground competitors and demand evidence before finalising. " if web else "No web access is available: rely on your own knowledge and say so in the demand reasoning. "}Never repeat an identical tool call.
- Be honest and calibrated: the demand-signal score is 1 (no evidence anyone wants this) to 10 (strong, proven willingness to pay). Most ideas deserve 3-6.
- Name real competitors where you know them. Kill criteria must be measurable within weeks (e.g. "<5 of 20 interviewed ops leads report this pain monthly").
- Prefer ideas a tiny team can ship as a software MVP in weeks with no regulated activities.""").replace(
        "{max_tools}", str(max_tools) if max_tools else "a few")


def render_brief(brief: IdeaBrief, *, seed: dict, run_id: str, models: list[str],
                 web_evidence: bool, sources: list[str], web_note: str) -> str:
    date = datetime.now(timezone.utc).date().isoformat()
    evidence = "web" if web_evidence else "reasoning-only"
    fm = {
        "title": brief.title,
        "date": date,
        "agent": AGENT,
        "run_id": run_id,
        "models": sorted(set(models)),
        "evidence": evidence,
        "demand_score": brief.demand_signal.score,
        "status": "candidate",
    }
    lines = ["---", *(f"{k}: {json.dumps(v)}" for k, v in fm.items()), "---", "",
             f"# {brief.title}", "", f"> {brief.one_liner}", ""]
    if not web_evidence:
        lines += [f"> ⚠️ **Reasoning only** — {web_note} Competitors and the demand score "
                  "come from model knowledge and are unverified.", ""]
    lines += ["## Problem", "", brief.problem, "",
              "## Ideal customer profile (ICP)", "", brief.icp, "",
              "## Why now", "", brief.why_now, "",
              "## Competitors", ""]
    lines += [f"- **{c.name}**" + (f" — {c.note}" if c.note else "") for c in brief.competitors]
    lines += ["", "## Kill criteria", ""]
    lines += [f"- [ ] {k}" for k in brief.kill_criteria]
    lines += ["", f"## Demand signal: {brief.demand_signal.score}/10", "",
              brief.demand_signal.reasoning, ""]
    if sources:
        lines += ["## Sources", ""] + [f"- {u}" for u in dict.fromkeys(sources)] + [""]
    lines += ["---", f"_Seed: {seed.get('title')} — {seed.get('one_liner', '')}_", ""]
    return "\n".join(lines)


# --------------------------------------------------------------- graph ---
def build_research_graph(deps: ResearchDeps, checkpointer=None):
    router, memory, toolbox, guard = deps.router, deps.memory, deps.toolbox, deps.guard
    log = deps.run_log or NullRunLog()

    def seed(state: ResearchState) -> dict:
        n = state["n"]
        existing = existing_idea_titles(memory)
        avoid = "\n".join(f"- {t}" for t in existing[-50:]) or "(none yet)"
        messages = [
            {"role": "system", "content": "You are a startup idea generator. Reply with JSON only."},
            {"role": "user", "content": f"""Generate {n} distinct B2B or prosumer software startup ideas
that a 1-2 person team could ship as an MVP within weeks, with no regulated activities
(no lending, medical diagnosis, etc.). Favour specific, painful, frequent problems with a clear buyer.
Make them diverse across industries.

Do NOT repeat these existing ideas:
{avoid}

Reply with JSON: {{"ideas": [{{"title": "...", "one_liner": "...", "target_customer": "...", "angle": "why this could work"}}]}}"""},
        ]
        seeds: list[dict] = []
        for _ in range(2):  # one repair attempt
            res = router.complete(AGENT, messages)
            try:
                seeds = [s for s in extract_json(res.text).get("ideas", [])
                         if isinstance(s, dict) and s.get("title")]
            except ValueError:
                seeds = []
            if len(seeds) >= n:
                break
            messages += [{"role": "assistant", "content": res.text},
                         {"role": "user", "content": f"Invalid or too few ideas. Return exactly {n} ideas as the JSON object described."}]
        if not seeds:
            raise RuntimeError("could not generate idea seeds")
        seeds = seeds[:n]
        log.event("node", node="seed", seeds=[s["title"] for s in seeds])
        return {"seeds": seeds, "idx": 0, "written": state.get("written", []),
                "failed": state.get("failed", [])}

    def start_idea(state: ResearchState) -> dict:
        s = state["seeds"][state["idx"]]
        log.event("node", node="start_idea", idx=state["idx"], seed=s)
        return {
            "messages": [
                {"role": "system", "content": system_prompt(toolbox, guard.max_tool_calls)},
                {"role": "user", "content": "Research this idea seed and produce the brief:\n"
                                            + json.dumps(s, ensure_ascii=False, indent=2)},
            ],
            "steps": 0, "tool_sigs": [], "sources": [], "web_hits": 0, "web_attempts": 0,
            "pending_tool": None, "final": None, "guard_note": None, "models_used": [],
        }

    def agent(state: ResearchState) -> dict:
        steps = state["steps"]
        try:
            guard.check_step(steps)
        except LoopGuardTripped as e:
            log.event("guard", reason=str(e), idx=state["idx"])
            return {"guard_note": str(e)}
        res = router.complete(AGENT, state["messages"])
        msgs = state["messages"] + [{"role": "assistant", "content": res.text}]
        upd: dict = {"steps": steps + 1, "models_used": state["models_used"] + [res.model],
                     "pending_tool": None}
        try:
            obj = extract_json(res.text)
        except ValueError:
            return upd | {"messages": msgs + [{"role": "user", "content":
                          "Your reply was not a JSON object. Reply with exactly one JSON object."}]}
        action = obj.get("action") or ("final" if "brief" in obj else "tool" if "tool" in obj else None)
        if action == "tool":
            name, args = obj.get("tool"), obj.get("args") or {}
            try:
                sig = guard.check_tool(state["tool_sigs"], name, args)
            except LoopGuardTripped as e:
                log.event("guard", reason=str(e), idx=state["idx"])
                return upd | {"messages": msgs, "guard_note": str(e)}
            return upd | {"messages": msgs, "pending_tool": {"name": name, "args": args},
                          "tool_sigs": state["tool_sigs"] + [sig]}
        if action == "final":
            try:
                brief = IdeaBrief.model_validate(obj.get("brief") or {})
                return upd | {"messages": msgs, "final": brief.model_dump()}
            except ValidationError as e:
                return upd | {"messages": msgs + [{"role": "user", "content":
                              f"The brief failed validation: {e.errors(include_url=False)}. Fix it and reply with the full JSON."}]}
        return upd | {"messages": msgs + [{"role": "user", "content":
                      'Unknown action. Use {"action": "tool", ...} or {"action": "final", ...}.'}]}

    def tool(state: ResearchState) -> dict:
        call = state["pending_tool"]
        out = toolbox.run(call["name"], call["args"])
        result = json.dumps(out["result"], ensure_ascii=False, default=str)[:TOOL_RESULT_CHARS]
        used = len(state["tool_sigs"])
        if guard.max_tool_calls is None:
            budget = ""
        elif used >= guard.max_tool_calls:
            budget = f"\n\n[Tool budget used up ({used}/{guard.max_tool_calls}). Reply with the final brief now.]"
        else:
            budget = f"\n\n[Tool calls used: {used}/{guard.max_tool_calls}.]"
        upd: dict = {"pending_tool": None,
                     "messages": state["messages"] + [{"role": "user", "content":
                                 f"TOOL RESULT ({call['name']}): {result}{budget}"}]}
        if call["name"] == "web_search":
            upd["web_attempts"] = state["web_attempts"] + 1
            if out["ok"]:
                upd["web_hits"] = state["web_hits"] + 1
                upd["sources"] = state["sources"] + out["urls"][:3]
        return upd

    def force_final(state: ResearchState) -> dict:
        msgs = state["messages"] + [{"role": "user", "content":
               f"STOP: {state['guard_note']}. No more tools. Reply now with the final brief JSON "
               f'({{"action": "final", "brief": {{...}}}}) based on what you have.'}]
        res = router.complete(AGENT, msgs)
        try:
            obj = extract_json(res.text)
            brief = IdeaBrief.model_validate(obj.get("brief") or obj)
            return {"final": brief.model_dump(), "messages": msgs,
                    "models_used": state["models_used"] + [res.model]}
        except (ValueError, ValidationError) as e:
            seed_ = state["seeds"][state["idx"]]
            log.event("idea_failed", idx=state["idx"], seed=seed_, error=str(e)[:500])
            return {"final": None, "idx": state["idx"] + 1,
                    "failed": state["failed"] + [{"seed": seed_.get("title"), "error": str(e)[:300]}]}

    def write(state: ResearchState) -> dict:
        brief = IdeaBrief.model_validate(state["final"])
        seed_ = state["seeds"][state["idx"]]
        web_evidence = state["web_hits"] > 0
        if not toolbox.web_enabled:
            note = "web search is disabled in config."
        elif state["web_attempts"] and not state["web_hits"]:
            note = "web search was attempted but returned no usable results."
        else:
            note = "the agent did not use web search for this brief."
        content = render_brief(brief, seed=seed_, run_id=state.get("run_id", log.run_id),
                               models=state["models_used"], web_evidence=web_evidence,
                               sources=state["sources"], web_note=note)
        date = datetime.now(timezone.utc).date().isoformat()
        rel = memory.unique_path(f"{IDEAS_DIR}/{date}-{slugify(brief.title)}.md")
        sha = memory.write(rel, content, agent=AGENT,
                           message=f'add idea brief "{brief.title}" (demand {brief.demand_signal.score}/10, '
                                   f'{"web" if web_evidence else "reasoning-only"})')
        log.event("brief_written", path=rel, commit=sha, title=brief.title,
                  score=brief.demand_signal.score, steps=state["steps"])
        return {"written": state["written"] + [rel], "idx": state["idx"] + 1, "final": None}

    def after_agent(state: ResearchState) -> Literal["tool", "agent", "force_final", "write"]:
        if state.get("guard_note") and not state.get("final"):
            return "force_final"
        if state.get("final"):
            return "write"
        if state.get("pending_tool"):
            return "tool"
        return "agent"

    def next_idea(state: ResearchState) -> Literal["start_idea", "__end__"]:
        return "start_idea" if state["idx"] < len(state["seeds"]) else END

    def after_force(state: ResearchState) -> Literal["write", "start_idea", "__end__"]:
        if state.get("final"):
            return "write"
        return next_idea(state)

    g = StateGraph(ResearchState)
    for name, fn in [("seed", seed), ("start_idea", start_idea), ("agent", agent),
                     ("tool", tool), ("force_final", force_final), ("write", write)]:
        g.add_node(name, fn)
    g.add_edge(START, "seed")
    g.add_edge("seed", "start_idea")
    g.add_edge("start_idea", "agent")
    g.add_conditional_edges("agent", after_agent)
    g.add_edge("tool", "agent")
    g.add_conditional_edges("force_final", after_force)
    g.add_conditional_edges("write", next_idea)
    return g.compile(checkpointer=checkpointer)


def recursion_limit(n: int, max_steps: int) -> int:
    return n * (2 * max_steps + 6) + 10
