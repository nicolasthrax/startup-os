from datetime import datetime, timezone

import pytest

from startup_os.budget import BudgetExceeded, TokenBucket


def test_token_bucket_allows_burst_then_waits(clock):
    b = TokenBucket(20, clock=clock.monotonic, sleep=clock.sleep)
    for _ in range(20):
        assert b.acquire() == 0
    assert not b.try_acquire()
    waited = b.acquire()  # 21st request waits for one token: 60/20 = 3s
    assert waited == pytest.approx(3.0)


def test_token_bucket_refills_over_time(clock):
    b = TokenBucket(20, clock=clock.monotonic, sleep=clock.sleep)
    for _ in range(20):
        b.acquire()
    clock.advance(30)  # half a minute → 10 tokens
    assert sum(b.try_acquire() for _ in range(15)) == 10


def test_agent_limits_split_by_percentage(make_scheduler):
    s = make_scheduler(daily=1000, shares={"research": 40, "ceo": 25})
    assert s.agent_limit("research") == 400
    assert s.agent_limit("ceo") == 250
    assert s.agent_limit("unknown") == 0


def test_daily_quota_enforced_per_agent(make_scheduler):
    s = make_scheduler(daily=10, shares={"research": 30, "ceo": 70})
    for _ in range(3):
        s.record(s.acquire("research"), model="m", status="ok")
    with pytest.raises(BudgetExceeded) as ei:
        s.acquire("research")
    assert ei.value.retry_at == datetime(2026, 11, 4, tzinfo=timezone.utc)
    s.acquire("ceo")  # other agents unaffected


def test_quota_resets_next_day(make_scheduler, clock):
    s = make_scheduler(daily=2, shares={"research": 100})
    s.acquire("research"); s.acquire("research")
    with pytest.raises(BudgetExceeded):
        s.acquire("research")
    clock.advance(24 * 3600)
    s.acquire("research")


def test_usage_logged_to_sqlite(make_scheduler):
    s = make_scheduler()
    uid = s.acquire("research", run_id="r1")
    s.record(uid, model="m1:free", status="ok", prompt_tokens=10, completion_tokens=5, latency_ms=12)
    uid2 = s.acquire("research")
    s.record(uid2, model="m2:free", status="rate_limited", error="429")
    row = [r for r in s.usage_summary() if r["agent"] == "research"][0]
    assert row == {"agent": "research", "limit": 50, "requests": 2, "ok": 1,
                   "rate_limited": 1, "errors": 0, "tokens": 15}


def test_per_minute_limit_throttles_not_fails(make_scheduler, clock):
    s = make_scheduler(daily=1000, per_minute=5, shares={"research": 100})
    for _ in range(7):
        s.acquire("research")
    assert sum(clock.slept) == pytest.approx(2 * 60 / 5)


def test_over_budget_task_is_queued_for_next_day(make_scheduler, clock):
    s = make_scheduler(daily=1, shares={"research": 100})
    task = s.create_task("research", "ideas", {"n": 3})
    s.acquire("research")
    with pytest.raises(BudgetExceeded) as ei:
        s.acquire("research")
    s.defer(task["id"], ei.value.retry_at, str(ei.value))
    t = s.get_task(task["id"])
    assert t["status"] == "deferred" and t["payload"] == {"n": 3}
    assert s.due_tasks() == []                # not dropped, not due yet
    clock.advance(12 * 3600)                  # past midnight UTC
    assert [d["id"] for d in s.due_tasks()] == [task["id"]]
