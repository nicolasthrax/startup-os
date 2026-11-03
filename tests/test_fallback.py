from datetime import timedelta

import pytest

from startup_os.budget import BudgetExceeded
from startup_os.llm import AllModelsFailed, ModelRouter

from conftest import HTTP429, ScriptedLLM


def router(sched, llm, clock, models=("a:free", "b:free", "c:free")):
    return ModelRouter(list(models), sched, llm, defer_hours=1, now=clock.now)


def test_first_model_success(make_scheduler, clock):
    llm = ScriptedLLM(["hello"])
    res = router(make_scheduler(), llm, clock).complete("research", [{"role": "user", "content": "hi"}])
    assert res.model == "a:free" and res.text == "hello"
    assert [c[0] for c in llm.calls] == ["a:free"]


def test_429_falls_back_to_next_model(make_scheduler, clock):
    s = make_scheduler()
    llm = ScriptedLLM([HTTP429("rate limited"), "ok from b"])
    res = router(s, llm, clock).complete("research", [])
    assert res.model == "b:free"
    assert [a["status"] for a in res.attempts] == ["rate_limited", "ok"]
    row = [r for r in s.usage_summary() if r["agent"] == "research"][0]
    assert row["requests"] == 2 and row["rate_limited"] == 1 and row["ok"] == 1


def test_errors_and_empty_replies_fall_back(make_scheduler, clock):
    llm = ScriptedLLM([RuntimeError("500 upstream"), "   ", "finally"])
    res = router(make_scheduler(), llm, clock).complete("research", [])
    assert res.model == "c:free"
    assert [a["status"] for a in res.attempts] == ["error", "error", "ok"]


def test_all_models_fail_raises_with_retry_time(make_scheduler, clock):
    llm = ScriptedLLM([HTTP429("x"), HTTP429("y"), RuntimeError("z")])
    with pytest.raises(AllModelsFailed) as ei:
        router(make_scheduler(), llm, clock).complete("research", [])
    assert ei.value.retry_at == clock.now() + timedelta(hours=1)
    assert [e["status"] for e in ei.value.errors] == ["rate_limited", "rate_limited", "error"]


def test_budget_exhaustion_stops_fallback_chain(make_scheduler, clock):
    s = make_scheduler(daily=2, shares={"research": 100})
    llm = ScriptedLLM([HTTP429("x"), HTTP429("y"), "never reached"])
    with pytest.raises(BudgetExceeded):
        router(s, llm, clock).complete("research", [])
    assert len(llm.calls) == 2
