"""Tools available to agents: memory search (FTS5) and free web search."""
from __future__ import annotations

import json
from typing import Any, Callable

from .runlog import NullRunLog, RunLog
from .search_index import SearchIndex


DEFAULT_BACKENDS = ("duckduckgo", "google", "brave", "bing", "mojeek")


def web_search(query: str, max_results: int = 5,
               backends: tuple[str, ...] | list[str] = DEFAULT_BACKENDS) -> list[dict]:
    """Free web search via `ddgs`. Tries each backend in order and returns the first
    non-empty result set; raises the last error if every backend fails."""
    from ddgs import DDGS

    last_exc: Exception | None = None
    for backend in backends:
        try:
            results = DDGS().text(query, max_results=max_results, backend=backend) or []
        except Exception as exc:  # noqa: BLE001 - "No results found", blocks, timeouts
            last_exc = exc
            continue
        if results:
            return [{"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body"),
                     "backend": backend} for r in results]
    if last_exc:
        raise last_exc
    return []


duckduckgo_search = web_search  # backwards-compatible name


class Toolbox:
    def __init__(
        self,
        index: SearchIndex,
        *,
        web_enabled: bool = True,
        web_max_results: int = 5,
        web_search_fn: Callable[[str, int], list[dict]] | None = duckduckgo_search,
        run_log: RunLog | None = None,
        spend_gate=None,
        agent: str = "research",
    ):
        self.index = index
        self.web_enabled = web_enabled and web_search_fn is not None
        self.web_max_results = web_max_results
        self.web_search_fn = web_search_fn
        self.run_log = run_log or NullRunLog()
        self.spend_gate = spend_gate
        self.agent = agent
        self._extra: dict[str, tuple[Callable, str]] = {}
        if spend_gate is not None:
            self._extra["request_spend"] = (
                lambda args, agent: self._request_spend(args, agent),
                "request_spend(item, amount_usd, justification, vendor?, blocks?): the ONLY way to spend "
                "money. Files a spend request for human approval; nothing is spent.")

    def register(self, name: str, fn: Callable, description: str, *, payment_capable: bool = False) -> None:
        """Add a tool. Payment-capable tools are only ever exposed through the spend gate."""
        from .spend import is_payment_capable

        if payment_capable or is_payment_capable(fn):
            if self.spend_gate is None:
                raise PermissionError(f"refusing to register payment-capable tool {name!r} without a spend gate")
            fn = self.spend_gate.wrap(name, fn)
            description += " [payment-capable: calling it files a spend request instead]"
        else:
            inner = fn
            fn = lambda args, agent: inner(**args)  # noqa: E731
        self._extra[name] = (fn, description)

    def _request_spend(self, args: dict, agent: str) -> dict:
        a = self.spend_gate.request(args, agent=agent)
        return {"ok": True, "spend_request": a["ref"],
                "result": f"filed spend request {a['ref']}; awaiting human approval. Nothing was spent."}

    def describe(self) -> str:
        lines = ['- search(query): full-text search over the company memory repo (past briefs, decisions, plans).']
        if self.web_enabled:
            lines.append("- web_search(query): free web search. Use it to find competitors, "
                         "pricing, market data and demand evidence. Returns titles, URLs and snippets.")
        lines += [f"- {d}" for _, d in self._extra.values()]
        return "\n".join(lines)

    def names(self) -> list[str]:
        return ["search"] + (["web_search"] if self.web_enabled else []) + list(self._extra)

    def search(self, query: str) -> list[dict]:
        return self.index.search(query)

    def run(self, name: str, args: dict) -> dict[str, Any]:
        """Run a tool. Returns {"ok", "result", "urls"}; never raises."""
        query = str((args or {}).get("query", "")).strip()
        out: dict[str, Any]
        if name in self._extra:
            fn, _ = self._extra[name]
            try:
                res = fn(args or {}, agent=self.agent)
                out = res if isinstance(res, dict) and "ok" in res else {"ok": True, "result": res}
            except Exception as exc:  # noqa: BLE001
                out = {"ok": False, "result": f"{name} failed: {exc!r}"[:500]}
            out.setdefault("urls", [])
        elif name not in self.names():
            out = {"ok": False, "result": f"unknown tool '{name}'. Available: {self.names()}", "urls": []}
        elif not query:
            out = {"ok": False, "result": "missing 'query' argument", "urls": []}
        elif name == "search":
            hits = self.search(query)
            out = {"ok": True, "result": hits or "no matches", "urls": []}
        else:
            try:
                hits = self.web_search_fn(query, self.web_max_results)
                out = {"ok": bool(hits), "result": hits or "no results",
                       "urls": [h["url"] for h in hits if h.get("url")]}
            except Exception as exc:  # noqa: BLE001
                out = {"ok": False, "result": f"web search failed: {exc!r}"[:500], "urls": []}
        self.run_log.event("tool_call", tool=name, args=args, ok=out["ok"],
                           result=json.dumps(out["result"], ensure_ascii=False, default=str)[:4000])
        return out
