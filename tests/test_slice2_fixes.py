"""Tests for: budget agents, path validation, `research` command, score rounding, human-only picks."""
import json
from decimal import Decimal

import pytest
import yaml

from startup_os import frontmatter
from startup_os.agents.ceo import stage_policy
from startup_os.agents.validation import Factor, is_go, round_half_up, weighted_score
from startup_os.approvals import ApprovalError
from startup_os.cli import main
from startup_os.config import load_config
from startup_os.paths import check_path
from startup_os.runtime import CommandError, Runtime
from startup_os.venture import Stage

from conftest import HTTP429, ScriptedLLM
from test_ceo_flow import BRIEF, plan_json, report_json, ticket_json, web


@pytest.fixture
def rt(cfg_path, clock):
    llm = ScriptedLLM([])
    r = Runtime(load_config(cfg_path), call_fn=llm, web_search_fn=web, now=clock.now)
    r.memory.write("research/ideas/2026-11-03-reorderhub.md", BRIEF, agent="research", message="brief")
    r.llm = llm
    return r


def run_events(path):
    return [json.loads(line) for line in open(path)]


# ---------------------------------------------------------------- 1. budget agents
@pytest.mark.parametrize("agents,ok", [
    ({"ceo": 25, "research": 65, "reserve": 10}, True),
    ({"ceo": 25, "research": 55, "reserve": 10, "engineer": 10}, False),
    ({"ceo": 25, "research": 65, "reserve": 5, "claude-code": 5}, False),
    ({"ceo": 30, "research": 70}, False),
])
def test_config_budget_agents_must_be_exactly_ceo_research_reserve(cfg_path, agents, ok):
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["budget"]["agents"] = agents
    cfg_path.write_text(yaml.safe_dump(cfg))
    if ok:
        load_config(cfg_path)
    else:
        with pytest.raises(ValueError):
            load_config(cfg_path)


def test_plan_naming_other_agents_is_rejected_then_repaired(rt):
    bad = plan_json(spend=False, split={"ceo": 20, "research": 60, "reserve": 10, "claude-code": 10})
    rt.llm.script = [bad, plan_json(spend=False)]
    out = rt.plan()
    feedback = [m["content"] for m in rt.llm.calls[-1][1] if m["role"] == "user"][-1]
    assert "names agents outside the budget: ['claude-code']" in feedback
    split = frontmatter.parse(rt.memory.read("plans/2026-W45.md"))[0]["budget_split"]
    assert set(split) == {"ceo", "research", "reserve"} and out.status == "done"


def test_plan_never_valid_fails_without_writing(rt):
    bad = plan_json(spend=False, split={"ceo": 20, "research": 60, "reserve": 10, "engineer": 10})
    rt.llm.script = [bad, bad, bad]                       # ceo.max_steps defaults to 3
    with pytest.raises(RuntimeError, match="could not produce a valid plan"):
        rt.plan()
    assert not rt.memory.exists("plans/2026-W45.md")
    assert rt.approvals.list(kind="plan_approval") == []


def test_pending_plan_with_invalid_split_is_rejected_not_auto_approved(rt, clock):
    a = rt.approvals.create("plan_approval", "Old plan", {"week": "2026-W45", "budget_split":
                            {"ceo": 25, "research": 40, "engineer": 15, "product": 15, "reserve": 5}},
                            requested_by="ceo")
    clock.advance(49 * 3600)
    rt2 = Runtime(rt.cfg, call_fn=rt.llm, web_search_fn=web, now=clock.now)
    a = rt2.approvals.get(a["id"])
    assert a["status"] == "rejected" and "engineer" in a["reason"] and a["decided_by"] == "system"
    assert rt2.auto_approved == []
    assert rt2.scheduler.share_source() == "config.yaml"


# ---------------------------------------------------------------- 2. path validation
def test_check_path(rt):
    m = rt.memory
    good = "research/ideas/2026-11-03-reorderhub.md"
    assert check_path(m, good) == (good, None)
    assert check_path(m, "company/" + good) == (good, None)
    assert check_path(m, str(m.root / good)) == (good, None)
    for bad, reason in [("research/ideas/nope.md", "no such file"), ("../config.yaml", "'..'"),
                        ("/etc/passwd", "outside company/"), (".git/config", ".git"), ("", "empty")]:
        rel, problem = check_path(m, bad)
        assert rel is None and reason in problem, (bad, problem)
    assert check_path(m, "venture.md", under="research/ideas")[1] == "not under research/ideas/"


def test_ceo_unknown_paths_invalidate_items_and_are_logged(rt):
    plan = json.loads(plan_json(spend=False))
    plan["plan"]["tickets"] = [
        {"key": "a", "title": "Use the validation report", "owner": "research", "acceptance_criteria": ["x"],
         "inputs": ["company/research/ideas/2026-11-03-reorderhub.md"]},
        {"key": "b", "title": "Read a made-up file", "owner": "research", "acceptance_criteria": ["x"],
         "inputs": ["research/validation/imaginary.md"]},
        {"key": "c", "title": "Follow up on b", "owner": "research", "acceptance_criteria": ["x"], "depends_on": ["b"]}]
    plan["plan"]["idea_pick"] = {"brief": "research/ideas/2026-11-03-made-up-pivot.md", "why": "pivot"}
    rt.llm.script = [json.dumps(plan)]
    out = rt.plan()
    meta = frontmatter.parse(rt.memory.read("plans/2026-W45.md"))[0]
    assert len(meta["tickets"]) == 1
    kept = rt.tickets.get(meta["tickets"][0])
    assert kept.title == "Use the validation report"
    assert kept.inputs == ["research/ideas/2026-11-03-reorderhub.md"]
    assert meta["recommended_pick"] is None
    ev = run_events(out.log_path)
    invalid = [(e["item"], e["path"]) for e in ev if e["type"] == "invalid_path"]
    assert ("idea_pick", "research/ideas/2026-11-03-made-up-pivot.md") in invalid
    assert ("ticket b", "research/validation/imaginary.md") in invalid
    assert any(e["type"] == "invalid_dependency" and e["item"] == "ticket c" for e in ev)
    assert any("Dropped ticket \"Read a made-up file\"" in n for n in out.extra["notes"])


def research_ticket(rt, title="Research pivot: HVAC distributors", inputs=None, **kw):
    t = rt.tickets.new(title=title, owner="research", kind=kw.pop("kind", "task"), description="Find out.",
                       acceptance_criteria=["3 competitors named", "pricing found"], inputs=inputs or [], **kw)
    rt.tickets.save(t, agent="ceo", message=f"create {t.id}")
    return t


def note_json(related=()):
    return json.dumps({"action": "final", "note": {
        "summary": "HVAC distributors are underserved.",
        "findings": [{"finding": "Few vertical portals", "evidence": "search", "source": "https://s.example"}],
        "criteria": [{"criterion": "3 competitors named", "status": "Met"},
                     {"criterion": "pricing found", "status": "partially met"}],
        "recommendation": "Write a brief.", "related_files": list(related),
        "evidence_links": ["https://s.example"]}})


def test_research_note_unknown_related_files_are_dropped_and_logged(rt):
    t = research_ticket(rt)
    rt.llm.script = [note_json(["research/ideas/2026-11-03-reorderhub.md", "research/notes/ghost.md"])]
    [r] = rt.research()
    meta, body = frontmatter.parse(rt.memory.read(r["outcome"].extra["report_path"]))
    assert meta["related_files"] == ["research/ideas/2026-11-03-reorderhub.md"]
    assert "`research/notes/ghost.md` (no such file in company/)" in body
    ev = run_events(r["outcome"].log_path)
    assert any(e["type"] == "invalid_path" and e["path"] == "research/notes/ghost.md" for e in ev)


# ---------------------------------------------------------------- 3. research command
def test_research_executes_open_research_tickets_of_any_kind(rt):
    t1 = research_ticket(rt, inputs=["research/ideas/2026-11-03-reorderhub.md"])
    t2 = research_ticket(rt, title="Another research question")
    cc = rt.tickets.new(title="Build it", owner="claude-code", acceptance_criteria=["x"])
    rt.tickets.save(cc, agent="ceo", message="cc")
    blocked = research_ticket(rt, title="Waits on approval", depends_on=["A-99"])
    rt.tickets.refresh(rt.approvals.status)
    rt.llm.script = [json.dumps({"action": "tool", "tool": "web_search", "args": {"query": "hvac portal"}}),
                     note_json(), note_json()]
    results = rt.research()
    assert [(r["ticket"], r["status"]) for r in results] == [(t1.id, "done"), (t2.id, "done")]
    assert rt.tickets.get(t1.id).status == "done" and "criteria met 1/2" in rt.tickets.get(t1.id).result
    assert rt.tickets.get(cc.id).status == "todo"                 # claude-code tickets are never executed
    assert rt.tickets.get(blocked.id).status == "blocked"
    first_prompt = rt.llm.calls[0][1][1]["content"]
    assert "Input files:" in first_prompt and "ReorderHub" in first_prompt   # inputs are read
    note = rt.memory.read(results[0]["outcome"].extra["report_path"])
    assert "## Findings" in note and "| pricing found | partial |" in note
    assert rt.scheduler.list_tasks()[0]["kind"] == "ticket"      # same scheduler / task queue


def test_research_skips_validation_ticket_for_other_venture_and_stops_on_deferral(rt):
    research_ticket(rt, title="Validate Old Venture", kind="validation", venture="Old")
    t2 = research_ticket(rt, title="Some question")
    t3 = research_ticket(rt, title="Later question")
    rt.llm.script = [HTTP429("a"), HTTP429("b"), HTTP429("c")]
    results = rt.research()
    assert results[0]["status"] == "skipped" and "Old" in results[0]["reason"]
    assert [(r["ticket"], r["status"]) for r in results[1:]] == [(t2.id, "deferred")]   # t3 not attempted
    assert rt.tickets.get(t2.id).status == "todo" and rt.tickets.get(t3.id).status == "todo"


def test_research_single_ticket_must_be_open_research(rt):
    cc = rt.tickets.new(title="Build it", owner="claude-code", acceptance_criteria=["x"])
    rt.tickets.save(cc, agent="ceo", message="cc")
    with pytest.raises(CommandError, match="owned by claude-code"):
        rt.research(cc.id)


# ---------------------------------------------------------------- 4. scoring
def F(w, s):
    return Factor(factor="f", weight=w, score=s)


def test_6_95_rounds_to_7_but_is_no_go_at_threshold_7():
    bd = [F(95, 7), F(5, 6)]                          # 0.95*7 + 0.05*6 = 6.95
    assert weighted_score(bd) == Decimal("6.95")
    assert round_half_up(weighted_score(bd)) == 7
    assert is_go(weighted_score(bd), 7) is False
    assert is_go(Decimal("7"), 7) is True


def test_round_half_up_not_bankers_or_float():
    bd = [F(0.30, 7), F(0.25, 3), F(0.25, 2), F(0.20, 6)]   # the real ReorderHub breakdown
    assert weighted_score(bd) == Decimal("4.55")
    assert round_half_up(weighted_score(bd)) == 5        # float/round() would give 4
    assert round_half_up(Decimal("2.5")) == 3            # round() gives 2


def test_validation_at_6_95_reports_no_go_and_blocks_build(rt):
    rt.llm.script = [ticket_json()]
    rt.pick("research/ideas/2026-11-03-reorderhub.md")
    rep = json.loads(report_json(7))
    rep["report"]["go_no_go"]["breakdown"] = [{"factor": "pain", "weight": 95, "score": 7},
                                              {"factor": "moat", "weight": 5, "score": 6}]
    rt.llm.script = [json.dumps(rep)]
    out = rt.validate()
    meta, body = frontmatter.parse(rt.memory.read(out.extra["report_path"]))
    assert meta["go_score"] == 7 and meta["go_weighted"] == 6.95 and meta["recommendation"] == "NO-GO"
    assert "weighted score 6.95/10 (shown as 7/10" in body
    assert stage_policy(Stage.BUILD, Stage.VALIDATION, meta, 7) is not None


# ---------------------------------------------------------------- 6. picks are the human's
def test_pick_requires_explicit_file_argument(cfg_path, capsys):
    with pytest.raises(SystemExit):
        main(["--config", str(cfg_path), "pick"])
    assert "required: brief" in capsys.readouterr().err


@pytest.mark.parametrize("arg", ["reorderhub", "2026-11-03-reorderhub", "venture.md", "research/ideas/nope.md"])
def test_pick_never_auto_selects(rt, arg):
    with pytest.raises(CommandError):
        rt.pick(arg)
    assert rt.venture.load().stage == Stage.IDEATION


def test_ceo_can_only_recommend_a_pick(rt):
    plan = json.loads(plan_json(spend=False))
    plan["plan"]["idea_pick"] = {"brief": "research/ideas/2026-11-03-reorderhub.md", "why": "best evidence"}
    rt.llm.script = [json.dumps(plan)]
    rt.plan()
    assert rt.approvals.list(kind="idea_pick") == []
    assert rt.venture.load().stage == Stage.IDEATION
    meta, body = frontmatter.parse(rt.memory.read("plans/2026-W45.md"))
    assert meta["recommended_pick"] == "research/ideas/2026-11-03-reorderhub.md"
    assert "startup-os pick research/ideas/2026-11-03-reorderhub.md" in body
    with pytest.raises(ApprovalError, match="agents may only recommend"):
        rt.approvals.create("idea_pick", "Pick X", {"brief": "x"}, requested_by="ceo")
