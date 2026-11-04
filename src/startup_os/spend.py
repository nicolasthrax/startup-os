"""Spend gate.

Agents never hold a reference to anything that can spend money. A payment-capable
tool is registered through `SpendGate.wrap`, which returns a stand-in: calling it
files a `spend_request` approval and returns its id. The real function is kept
private and is never executed in this slice (spending is out of scope) — even an
approved request only unblocks the tickets that depend on it."""
from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field, field_validator

from .approvals import ApprovalQueue


def payment_capable(fn: Callable) -> Callable:
    """Mark a function as able to spend money. The Toolbox refuses to expose it directly."""
    fn.__payment_capable__ = True
    return fn


def is_payment_capable(fn: Callable) -> bool:
    return bool(getattr(fn, "__payment_capable__", False))


class SpendRequest(BaseModel):
    item: str = Field(min_length=2)
    amount_usd: float = Field(gt=0)
    justification: str = Field(min_length=3)
    vendor: str | None = None
    recurring: str | None = None          # e.g. "monthly"
    blocks: list[str] = Field(default_factory=list)   # ticket ids that wait on this
    tool: str | None = None
    tool_args: dict[str, Any] | None = None

    @field_validator("amount_usd", mode="before")
    @classmethod
    def _amount(cls, v):
        if isinstance(v, str):
            v = v.replace("$", "").replace(",", "").strip()
        return v


class SpendGate:
    def __init__(self, approvals: ApprovalQueue):
        self.approvals = approvals
        self._held: dict[str, Callable] = {}   # payment-capable fns, never called by agents

    def request(self, req: SpendRequest | dict, *, agent: str) -> dict:
        req = SpendRequest.model_validate(req)
        title = f"${req.amount_usd:,.2f} — {req.item}" + (f" ({req.vendor})" if req.vendor else "")
        return self.approvals.create("spend_request", title, req.model_dump(), requested_by=agent)

    def wrap(self, name: str, fn: Callable, *, describe_cost: Callable[[dict], dict] | None = None):
        """Return the stand-in an agent may call instead of `fn`."""
        self._held[name] = fn

        def stand_in(args: dict, *, agent: str) -> dict:
            cost = (describe_cost or (lambda a: {}))(args)
            a = self.request({
                "item": cost.get("item") or f"{name} call",
                "amount_usd": cost.get("amount_usd") or args.get("amount_usd") or args.get("amount") or 0.01,
                "justification": args.get("justification") or cost.get("justification") or "requested by agent",
                "vendor": cost.get("vendor"),
                "blocks": args.get("blocks") or [],
                "tool": name,
                "tool_args": {k: v for k, v in args.items() if k != "blocks"},
            }, agent=agent)
            return {"ok": False, "spend_request": a["ref"],
                    "result": f"{name} is payment-capable: filed spend request {a['ref']} for human approval. "
                              f"Nothing was spent. Tickets that need it must depend on {a['ref']}."}

        stand_in.__wrapped_payment_tool__ = name
        return stand_in
