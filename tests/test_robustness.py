"""Web-search backend fallback, per-brief tool budget, and crash/interrupt recovery."""
import json
import sys
import types
from datetime import timedelta

import pytest

from startup_os.config import load_config
from startup_os.guards import LoopGuard, LoopGuardTripped
from startup_os.llm import ModelRouter
from startup_os.runtime import Runtime
from startup_os.tools import web_search

from conftest import ScriptedLLM, brief_json, seeds_json


def fake_ddgs(monkeypatch, behaviour):
    """behaviour: backend -> list of results or Exception."""
    calls = []

    class DDGS:
        def text(self, query, max_results=5, backend="auto"):
            calls.append(backend)
            r = behaviour.get(backend, Exception("No results found."))
            if isinstance(r, Exception):
                raise r
            return r

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=DDGS))
    return calls


def test_web_search_falls_back_across_backends(monkeypatch):
    calls = fake_ddgs(monkeypatch, {"duckduckgo": Exception("No results found."), "brave": [],
                                    "google": [{"title": "T", "href": "https://x.io", "body": "b"}]})
    hits = web_search("q", 3, ["duckduckgo", "brave", "google", "bing"])
    assert calls == ["duckduckgo", "brave", "google"]
    assert hits == [{"title": "T", "url": "https://x.io", "snippet": "b", "backend": "google"}]


def test_web_search_raises_when_all_backends_fail(monkeypatch):
    fake_ddgs(monkeypatch, {})
    with pytest.raises(Exception, match="No results"):
        web_search("q", 3, ["duckduckgo", "google"])


def test_tool_budget_guard():
    g = LoopGuard(max_steps=10, max_identical_tool_calls=2, max_tool_calls=2)
    sigs = [g.check_tool([], "web_search", {"query": "a"})]
    sigs.append(g.check_tool(sigs, "web_search", {"query": "b"}))
    with pytest.raises(LoopGuardTripped, match="tool budget used up"):
        g.check_tool(sigs, "web_search", {"query": "c"})


def tool(q):
    return json.dumps({"action": "tool", "tool": "web_search", "args": {"query": q}})


def test_agent_is_told_budget_and_forced_to_finalise(cfg_path):
    # test config: max_tool_calls_per_task defaults to 4
    llm = ScriptedLLM([seeds_json(1), tool("a"), tool("b"), tool("c"), tool("d"), tool("e"), brief_json()])
    rt = Runtime(load_config(cfg_path), call_fn=llm,
                 web_search_fn=lambda q, n: [{"title": q, "url": f"https://e.com/{q}"}])
    cfg = rt.cfg.model_copy(deep=True)
    cfg.guards.max_steps_per_task = 20
    rt.cfg = cfg
    out = rt.ideas(1)
    assert out.status == "done"
    msgs = llm.calls[-1][1]
    assert any("Tool budget used up (4/4)" in m["content"] for m in msgs)
    assert "STOP: tool budget used up" in msgs[-1]["content"]


def test_stale_running_task_is_requeued_and_reserved_rows_closed(make_scheduler, clock):
    s = make_scheduler()
    t = s.create_task("research", "ideas", {"n": 1})
    s.acquire("research")                                     # in-flight call, never recorded
    with s._conn() as c:
        c.execute("UPDATE tasks SET pid = 999999999 WHERE id = ?", (t["id"],))  # dead process
    clock.advance(600)
    assert s.recover_interrupted(stale_after_s=180) == [t["id"]]
    assert s.get_task(t["id"])["status"] == "deferred"
    assert [d["id"] for d in s.due_tasks()] == [t["id"]]
    with s._conn() as c:
        assert c.execute("SELECT status FROM usage").fetchone()[0] == "interrupted"


def test_live_task_is_not_requeued(make_scheduler):
    s = make_scheduler()
    t = s.create_task("research", "ideas", {"n": 1})          # pid = this process
    assert s.recover_interrupted() == []
    assert s.get_task(t["id"])["status"] == "running"


def test_keyboard_interrupt_records_usage_and_defers_task(cfg_path):
    llm = ScriptedLLM([seeds_json(1), KeyboardInterrupt()])
    rt = Runtime(load_config(cfg_path), call_fn=llm, web_search_fn=lambda q, n: [])
    with pytest.raises(KeyboardInterrupt):
        rt.ideas(1)
    task = rt.scheduler.list_tasks()[0]
    assert task["status"] == "deferred" and task["reason"] == "interrupted by user"
    with rt.scheduler._conn() as c:
        statuses = [r[0] for r in c.execute("SELECT status FROM usage ORDER BY id")]
    assert statuses == ["ok", "interrupted"]
    llm.script = [brief_json()]
    assert rt.run_research(task, resume=True).status == "done"
