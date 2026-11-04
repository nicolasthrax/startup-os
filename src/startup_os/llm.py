"""LLM access through OpenRouter's OpenAI-compatible API with a model fallback chain.

Every attempt (including fallbacks) is charged to the agent's budget via the
Scheduler. On a 429 or any error the next model is tried; when the list is
exhausted AllModelsFailed is raised so the caller can defer the task."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Protocol

from .budget import Scheduler
from .runlog import NullRunLog, RunLog


@dataclass
class CallResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class CallFn(Protocol):
    def __call__(self, model: str, messages: list[dict], **kw) -> CallResponse: ...


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    attempts: list[dict] = field(default_factory=list)


class AllModelsFailed(Exception):
    def __init__(self, errors: list[dict], retry_at: datetime):
        summary = "; ".join(f"{e['model']}: {e['status']}" for e in errors)
        super().__init__(f"all models failed ({summary}); retry at {retry_at.isoformat()}")
        self.errors, self.retry_at = errors, retry_at


def is_rate_limit(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 429:
        return True
    try:
        import openai

        if isinstance(exc, openai.RateLimitError):
            return True
    except ImportError:  # pragma: no cover
        pass
    return "429" in str(exc) or "rate limit" in str(exc).lower()


class ModelRouter:
    def __init__(
        self,
        models: list[str],
        scheduler: Scheduler,
        call_fn: CallFn,
        *,
        run_log: RunLog | None = None,
        defer_hours: float = 1,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        call_kwargs: dict | None = None,
    ):
        if not models:
            raise ValueError("at least one model required")
        self.models = list(models)
        self.scheduler = scheduler
        self.call_fn = call_fn
        self.run_log = run_log or NullRunLog()
        self.defer_hours = defer_hours
        self.now = now
        self.call_kwargs = call_kwargs or {}

    def complete(self, agent: str, messages: list[dict], **kw) -> LLMResult:
        """Try each model in order. Raises BudgetExceeded (from the scheduler)
        or AllModelsFailed."""
        attempts: list[dict] = []
        kwargs = {**self.call_kwargs, **kw}
        for model in self.models:
            usage_id = self.scheduler.acquire(agent, run_id=self.run_log.run_id)
            t0 = time.monotonic()
            try:
                resp = self.call_fn(model, messages, **kwargs)
                if not resp.text or not resp.text.strip():
                    raise RuntimeError("empty completion")
            except BaseException as exc:
                if not isinstance(exc, Exception):  # KeyboardInterrupt / SystemExit
                    self.scheduler.record(usage_id, model=model, status="interrupted",
                                          error=repr(exc))
                    raise
                # any other failure → next model
                status = "rate_limited" if is_rate_limit(exc) else "error"
                latency = int((time.monotonic() - t0) * 1000)
                self.scheduler.record(usage_id, model=model, status=status,
                                      latency_ms=latency, error=repr(exc))
                attempts.append({"model": model, "status": status, "error": repr(exc)[:300]})
                self.run_log.event("llm_error", agent=agent, model=model, status=status,
                                   error=repr(exc)[:1000], latency_ms=latency)
                continue
            latency = int((time.monotonic() - t0) * 1000)
            self.scheduler.record(usage_id, model=model, status="ok",
                                  prompt_tokens=resp.prompt_tokens,
                                  completion_tokens=resp.completion_tokens, latency_ms=latency)
            attempts.append({"model": model, "status": "ok"})
            self.run_log.event(
                "llm_call", agent=agent, model=model, prompt=messages, response=resp.text,
                prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                latency_ms=latency, fallbacks_tried=len(attempts) - 1,
            )
            return LLMResult(resp.text, model, resp.prompt_tokens, resp.completion_tokens, attempts)
        raise AllModelsFailed(attempts, self.now() + timedelta(hours=self.defer_hours))


def openrouter_call_fn(base_url: str, api_key: str, timeout_s: float = 90) -> CallFn:
    from openai import OpenAI

    client = OpenAI(
        base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0,
        default_headers={"X-Title": "Startup OS"},
    )

    def call(model: str, messages: list[dict], **kw) -> CallResponse:
        r = client.chat.completions.create(model=model, messages=messages, **kw)
        # OpenRouter can return HTTP 200 with an error payload and no choices.
        if not getattr(r, "choices", None):
            err = getattr(r, "error", None) or (r.model_extra or {}).get("error")
            raise RuntimeError(f"no choices in response: {err}")
        text = r.choices[0].message.content or ""
        u = r.usage
        return CallResponse(
            text=text,
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
        )

    return call
