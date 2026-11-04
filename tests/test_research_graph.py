import json

from startup_os.budget import Scheduler
from startup_os.config import load_config
from startup_os.runtime import Runtime

from conftest import HTTP429, ScriptedLLM, brief_json, seeds_json


def tool(query, name="web_search"):
    return json.dumps({"action": "tool", "tool": name, "args": {"query": query}})


def web_ok(q, n):
    return [{"title": f"Result for {q}", "url": f"https://example.com/{len(q)}", "snippet": "..."}]


def web_down(q, n):
    raise ConnectionError("blocked")


def make_rt(cfg_path, script, web=web_ok, **kw):
    llm = ScriptedLLM(script)
    rt = Runtime(load_config(cfg_path), call_fn=llm, web_search_fn=web, **kw)
    return rt, llm


def test_ideas_writes_briefs_and_commits(cfg_path):
    rt, llm = make_rt(cfg_path, [
        seeds_json(2),
        tool("invoice chaser competitors"), brief_json("Invoice Chaser", 6),
        brief_json("Shift Swap", 4),
    ])
    out = rt.ideas(2)
    assert out.status == "done" and len(out.written) == 2
    first = rt.memory.read(out.written[0])
    for section in ["## Problem", "## Ideal customer profile", "## Why now", "## Competitors",
                    "## Kill criteria", "## Demand signal: 6/10", "## Sources"]:
        assert section in first
    assert 'evidence: "web"' in first and "Reasoning only" not in first
    second = rt.memory.read(out.written[1])
    assert 'evidence: "reasoning-only"' in second and "Reasoning only" in second
    assert rt.memory.log(2) == [
        'research: add idea brief "Shift Swap" (demand 4/10, reasoning-only)',
        'research: add idea brief "Invoice Chaser" (demand 6/10, web)',
    ]
    assert rt.index.search("invoice")[0]["path"] == out.written[0]
    # structured run log has prompt, model, tool calls and tokens
    events = [json.loads(l) for l in open(out.log_path)]
    types = [e["type"] for e in events]
    assert types[0] == "run_start" and types[-1] == "run_end"
    call = next(e for e in events if e["type"] == "llm_call")
    assert call["model"] == "m1:free" and call["prompt"] and call["prompt_tokens"] == 100
    assert any(e["type"] == "tool_call" and e["tool"] == "web_search" for e in events)


def test_web_failure_flags_reasoning_only(cfg_path):
    rt, _ = make_rt(cfg_path, [seeds_json(1), tool("x"), brief_json()], web=web_down)
    out = rt.ideas(1)
    text = rt.memory.read(out.written[0])
    assert "web search was attempted but returned no usable results" in text


def test_repeated_identical_tool_call_stops_loop(cfg_path):
    rt, llm = make_rt(cfg_path, [seeds_json(1), tool("same"), tool("SAME"), brief_json()])
    out = rt.ideas(1)
    assert out.status == "done" and len(out.written) == 1
    # the forced final call is told to stop
    assert "STOP: repeated identical tool call" in llm.calls[-1][1][-1]["content"]


def test_max_steps_forces_final(cfg_path):
    # max_steps_per_task = 4 in the test config
    rt, llm = make_rt(cfg_path, [seeds_json(1), tool("a"), tool("b"), tool("c"), tool("d"), brief_json()])
    out = rt.ideas(1)
    assert out.status == "done"
    assert "STOP: max steps reached" in llm.calls[-1][1][-1]["content"]


def test_unrecoverable_idea_is_skipped_not_fatal(cfg_path):
    rt, _ = make_rt(cfg_path, [seeds_json(2), tool("a"), tool("a"), "not json at all",
                               brief_json("Second")])
    out = rt.ideas(2)
    assert out.status == "done" and len(out.written) == 1 and len(out.failed) == 1


def test_all_models_failing_defers_task_and_resume_continues(cfg_path):
    rt, llm = make_rt(cfg_path, [
        seeds_json(2), brief_json("First"),
        HTTP429("a"), HTTP429("b"), HTTP429("c"),       # second idea: every model rate-limited
    ])
    out = rt.ideas(2)
    assert out.status == "deferred" and len(out.written) == 1
    task = rt.scheduler.get_task(out.task_id)
    assert task["status"] == "deferred" and task["run_after"]
    # Resume from the LangGraph checkpoint: no re-seeding, first brief not redone.
    llm.script = [brief_json("Second")]
    out2 = rt.run_research(task, resume=True)
    assert out2.status == "done" and len(out2.written) == 2
    assert rt.scheduler.get_task(out.task_id)["status"] == "done"
    assert len(list((rt.memory.root / "research/ideas").glob("*.md"))) == 2


def test_budget_exhaustion_defers_to_next_day(cfg_path, tmp_path, clock):
    cfg = load_config(cfg_path)
    sched = Scheduler(tmp_path / "s.db", 4, 1000, {"research": 50}, now=clock.now)  # 2 requests/day
    rt, _ = make_rt(cfg_path, [seeds_json(2), brief_json("First"), brief_json("Second")], scheduler=sched)
    out = rt.ideas(2)
    assert out.status == "deferred" and "daily budget" in out.reason
    assert sched.get_task(out.task_id)["run_after"].startswith("2026-11-04T00:00:00")
