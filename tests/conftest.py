from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from startup_os.budget import Scheduler, TokenBucket
from startup_os.llm import CallResponse


class FakeClock:
    """Controls both wall time (for quotas) and monotonic time (for the bucket)."""

    def __init__(self, start: datetime = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)):
        self.dt = start
        self.mono = 0.0
        self.slept: list[float] = []

    def now(self) -> datetime:
        return self.dt

    def monotonic(self) -> float:
        return self.mono

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.mono += s

    def advance(self, seconds: float) -> None:
        from datetime import timedelta

        self.dt += timedelta(seconds=seconds)
        self.mono += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def make_scheduler(tmp_path, clock):
    def _make(daily=100, per_minute=20, shares=None):
        bucket = TokenBucket(per_minute, clock=clock.monotonic, sleep=clock.sleep)
        return Scheduler(tmp_path / "state.db", daily, per_minute,
                         shares or {"research": 50, "ceo": 50}, now=clock.now, bucket=bucket)
    return _make


class ScriptedLLM:
    """Fake call_fn. `script` items: str reply, Exception to raise, or a callable(model, messages)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[tuple[str, list[dict]]] = []

    def __call__(self, model, messages, **kw):
        self.calls.append((model, messages))
        if not self.script:
            raise AssertionError("ScriptedLLM ran out of replies")
        item = self.script.pop(0)
        if callable(item) and not isinstance(item, BaseException):
            item = item(model, messages)
        if isinstance(item, BaseException):
            raise item
        return CallResponse(text=item, prompt_tokens=100, completion_tokens=50)


class HTTP429(Exception):
    status_code = 429


def brief_json(title="Invoice Chaser", score=6) -> str:
    return json.dumps({"action": "final", "brief": {
        "title": title, "one_liner": "Automated AR follow-ups for agencies.",
        "problem": "Agencies wait 60+ days to get paid.",
        "icp": "Ops lead at 10-50 person agencies.",
        "why_now": "LLMs write polite, context-aware nudges.",
        "competitors": [{"name": "Chaser", "note": "UK-focused"}, "Upflow"],
        "kill_criteria": ["<5 of 20 agencies report >30 day delays"],
        "demand_signal": {"score": score, "reasoning": "Frequent complaints in forums."},
    }})


def seeds_json(n: int) -> str:
    return json.dumps({"ideas": [{"title": f"Idea {i}", "one_liner": f"one liner {i}"} for i in range(n)]})


@pytest.fixture
def cfg_path(tmp_path) -> Path:
    cfg = {
        "paths": {"company": "company", "data": "data", "logs": "logs"},
        "llm": {"models": ["m1:free", "m2:free", "m3:free"], "defer_hours_on_failure": 1},
        "budget": {"daily_requests": 1000, "per_minute": 1000, "agents": {"research": 50}},
        "guards": {"max_steps_per_task": 4, "max_identical_tool_calls": 2},
        "web_search": {"enabled": True, "max_results": 3},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p
