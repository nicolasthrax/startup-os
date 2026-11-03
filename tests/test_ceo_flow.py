"""End-to-end: pick → CEO validation ticket → validate → plan → approvals. LLMs mocked."""
import json

import pytest

from startup_os import frontmatter
from startup_os.config import load_config
from startup_os.runtime import CommandError, Runtime
from startup_os.venture import Stage

from conftest import HTTP429, ScriptedLLM

BRIEF = """---
title: "ReorderHub"
demand_score: 6
evidence: "web"
---

# ReorderHub

> B2B reorder portal for distributors.

## Kill criteria

- [ ] <5 of 20 distributors confirm the pain

## Demand signal: 6/10

Real pain, crowded market.
"""


def ticket_json():
    return json.dumps({"title": "Validate ReorderHub demand and pricing",
                       "description": "Teardown + pricing + demand.",
                       "acceptance_criteria": ["4 competitors with pricing", "go/no-go score"]})


def report_json(score):
    return json.dumps({"action": "final", "report": {
        "summary": "Pain is real.",
        "competitor_teardown": [
            {"name": "Ximple", "url": "https://ximple.example", "pricing": "$500/mo", "strengths": "ERP", "weaknesses": "slow"},
            {"name": "OroCommerce", "pricing": "custom", "strengths": "features", "weaknesses": "heavy"}],
        "pricing_scan": {"price_points": [{"vendor": "Ximple", "price": "$500/mo", "source": "https://ximple.example"}],
                         "summary": "$300-2k/mo", "suggested_price": "$400/mo"},
        "demand_signals": [{"signal": "40% of orders by phone", "strength": "Moderate", "source": "https://s.example"}],
        "kill_criteria_check": [{"criterion": "<5 of 20", "status": "Unknown", "evidence": "no interviews"}],
        "risks": ["integration effort"],
        "go_no_go": {"score": score, "reasoning": "because"},
        "evidence_links": ["https://s.example", "https://made-up.example"]}})


def plan_json(stage_to=None, spend=True, split=None):
    plan = {
        "summary": "Start building.", "goals": ["Ship skeleton"],
        "tickets": [
            {"key": "t1", "title": "Interview script research", "owner": "research",
             "acceptance_criteria": ["10 questions"]},
            {"key": "t2", "title": "Build reorder MVP", "owner": "Claude Code",
             "acceptance_criteria": ["tests pass"], "depends_on": ["t1"], "needs_spend": ["s1"] if spend else []}],
        "budget_split": split or {"research": 30, "ceo": 10, "engineer": 40, "product": 10, "reserve": 5},
        "budget_rationale": "build week",
        "spend_requests": [{"key": "s1", "item": "Domain reorderhub.io", "amount_usd": 12,
                            "justification": "landing page"}] if spend else [],
        "stage_change": {"to": stage_to, "what": "Start building", "why": "score >= 7",
                         "alternatives": ["Keep validating"]} if stage_to else None,
        "risks": ["scope creep"]}
    return json.dumps({"action": "final", "plan": plan})


def web(q, n):
    return [{"title": q, "url": "https://s.example", "snippet": "x"}]


@pytest.fixture
def rt(cfg_path, clock):
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["budget"]["agents"] = {"research": 40, "ceo": 25, "product": 15, "engineer": 15, "reserve": 5}
    cfg_path.write_text(yaml.safe_dump(cfg))
    llm = ScriptedLLM([])
    r = Runtime(load_config(cfg_path), call_fn=llm, web_search_fn=web, now=clock.now)
    r.memory.write("research/ideas/2026-11-03-reorderhub.md", BRIEF, agent="research", message="brief")
    r.memory.write("research/ideas/2026-11-03-other.md", BRIEF.replace("ReorderHub", "Other").replace("6", "4"),
                   agent="research", message="brief")
    r.llm = llm
    return r


def picked(rt):
    rt.llm.script = [ticket_json()]
    return rt.pick("company/research/ideas/2026-11-03-reorderhub.md")


def test_pick_moves_to_validation_with_decision_and_ticket(rt):
    r = picked(rt)
    v = rt.venture.load()
    assert v.stage == Stage.VALIDATION and v.name == "ReorderHub" and v.idea.endswith("reorderhub.md")
    meta, body = frontmatter.parse(rt.memory.read(r["decision"]))
    assert "Other (4/10)" in body and meta["decided_by"].startswith("human via A-")
    t = rt.tickets.get(r["ticket"])
    assert t.owner == "research" and t.kind == "validation" and t.title == "Validate ReorderHub demand and pricing"
    a = rt.approvals.get(r["approval"])
    assert a["kind"] == "idea_pick" and a["status"] == "approved"
    assert "idea_pick\tapproved" in rt.memory.read("approvals.log")
    with pytest.raises(CommandError):
        rt.pick("other")  # no second pick outside Ideation


def test_pick_falls_back_to_template_ticket_when_llm_fails(rt):
    rt.llm.script = [HTTP429("a"), HTTP429("b"), HTTP429("c")]
    r = rt.pick("reorderhub")
    assert r["ticket_source"] == "template" and rt.tickets.get(r["ticket"]).kind == "validation"


def test_validate_writes_report_and_closes_ticket(rt):
    r = picked(rt)
    rt.llm.script = [json.dumps({"action": "tool", "tool": "web_search", "args": {"query": "ximple pricing"}}),
                     report_json(8)]
    out = rt.validate()
    assert out.status == "done"
    meta, body = frontmatter.parse(rt.memory.read(out.extra["report_path"]))
    assert meta["go_score"] == 8 and meta["recommendation"] == "GO" and meta["go_threshold"] == 7
    for s in ["## Competitor teardown", "## Pricing scan", "## Demand signals", "## Evidence links",
              "https://s.example", "not seen in tool results", "https://made-up.example"]:
        assert s in body
    t = rt.tickets.get(r["ticket"])
    assert t.status == "done" and out.extra["report_path"] in t.result


def test_plan_after_go_moves_to_build_files_spend_and_tickets(rt, clock):
    picked(rt)
    rt.llm.script = [report_json(8)]
    rt.validate()
    rt.llm.script = [plan_json(stage_to="Build")]
    out = rt.plan()
    assert out.status == "done"
    assert rt.venture.load().stage == Stage.BUILD
    meta, body = frontmatter.parse(rt.memory.read("plans/2026-W45.md"))
    assert meta["status"] == "pending_approval" and meta["approval"] == out.extra["approval"]
    for s in ["## Goals", "## Tickets", "## Call-budget split", "## Spend requests", "decision record"]:
        assert s in body
    t1, t2 = (rt.tickets.get(i) for i in meta["tickets"])
    assert t1.owner == "research" and t1.status == "todo"
    assert t2.owner == "claude-code" and t2.status == "blocked"
    spend = rt.approvals.get(meta["spend_requests"][0])
    assert spend["kind"] == "spend_request" and spend["payload"]["blocks"] == [t2.id]
    assert set(t2.depends_on) == {t1.id, spend["ref"]}
    # plan auto-approves after 48h; spend request does not
    clock.advance(48 * 3600)
    rt2 = Runtime(rt.cfg, call_fn=rt.llm, web_search_fn=web, now=clock.now)
    assert [a["ref"] for a in rt2.auto_approved] == [out.extra["approval"]]
    assert frontmatter.parse(rt.memory.read("plans/2026-W45.md"))[0]["status"] == "auto_approved"
    assert rt2.approvals.get(spend["id"])["status"] == "pending"
    assert rt2.scheduler.agent_limit("engineer") == 400       # approved split now live (40% of 1000)
    # approving the spend request only unblocks the ticket once its ticket dependency is done too
    rt2.approve(spend["ref"])
    assert rt2.tickets.get(t2.id).blocked_reason == f"{t1.id} is todo"


def test_policy_refuses_build_below_threshold(rt):
    picked(rt)
    rt.llm.script = [report_json(5)]
    rt.validate()
    rt.llm.script = [plan_json(stage_to="Build", spend=False)]
    out = rt.plan()
    assert rt.venture.load().stage == Stage.VALIDATION
    assert any("refused by policy" in n for n in out.extra["notes"])


def test_rejecting_plan_cancels_its_tickets(rt):
    picked(rt)
    rt.llm.script = [plan_json(spend=False)]
    out = rt.plan()
    ids = frontmatter.parse(rt.memory.read("plans/2026-W45.md"))[0]["tickets"]
    rt.reject(out.extra["approval"], reason="focus on validation")
    assert all(rt.tickets.get(i).status == "cancelled" for i in ids)
    assert frontmatter.parse(rt.memory.read("plans/2026-W45.md"))[0]["status"] == "rejected"
    assert rt.scheduler.share_source() == "config.yaml"


def test_plan_repairs_invalid_draft(rt):
    bad = plan_json(spend=False, split={"research": 80, "ceo": 50})          # > 100%
    rt.llm.script = [bad, "not json", plan_json(spend=False)]
    out = rt.plan()
    assert out.status == "done"
    fixes = [m["content"] for _, msgs in rt.llm.calls[-1:] for m in msgs if m["role"] == "user"]
    assert any("sums to 130%" in f for f in fixes) and any("Invalid plan JSON" in f for f in fixes)


def test_replanning_same_week_supersedes(rt):
    rt.llm.script = [plan_json(spend=False), plan_json(spend=False)]
    first = rt.plan()
    second = rt.plan()
    assert rt.approvals.get(first.extra["approval"])["status"] == "superseded"
    assert rt.approvals.get(second.extra["approval"])["status"] == "pending"
    assert len(rt.approvals.list(status="pending", kind="plan_approval")) == 1
