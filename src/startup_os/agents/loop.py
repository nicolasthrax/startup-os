"""A reusable guarded tool loop for LangGraph agents that reply in JSON.

The model replies with either {"action": "tool", "tool": ..., "args": {...}} or
{"action": "final", "<final_key>": {...}}. The final payload is validated against
a Pydantic schema; validation errors are fed back. Loop guards (max steps, max
tool calls, repeated identical calls) route to a single forced final step."""
from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ValidationError

from ..guards import LoopGuard, LoopGuardTripped
from ..llm import ModelRouter
from ..runlog import NullRunLog, RunLog
from ..tools import Toolbox
from .research import extract_json

TOOL_RESULT_CHARS = 3000


class LoopState(TypedDict, total=False):
    messages: list[dict]
    steps: int
    tool_sigs: list[str]
    sources: list[str]
    web_hits: int
    web_attempts: int
    pending_tool: dict | None
    final: dict | None
    guard_note: str | None
    models_used: list[str]
    error: str | None


def fresh_loop_state(messages: list[dict]) -> dict:
    return {"messages": messages, "steps": 0, "tool_sigs": [], "sources": [], "web_hits": 0,
            "web_attempts": 0, "pending_tool": None, "final": None, "guard_note": None,
            "models_used": [], "error": None}


class ToolLoop:
    def __init__(self, *, agent: str, router: ModelRouter, toolbox: Toolbox, guard: LoopGuard,
                 schema: type[BaseModel], final_key: str, log: RunLog | None = None):
        self.agent, self.router, self.toolbox, self.guard = agent, router, toolbox, guard
        self.schema, self.final_key = schema, final_key
        self.log = log or NullRunLog()

    def _validate(self, obj: dict) -> dict:
        payload = obj.get(self.final_key) if isinstance(obj.get(self.final_key), dict) else obj
        return self.schema.model_validate(payload).model_dump(mode="json")

    def agent_node(self, state: LoopState) -> dict:
        steps = state["steps"]
        try:
            self.guard.check_step(steps)
        except LoopGuardTripped as e:
            self.log.event("guard", agent=self.agent, reason=str(e))
            return {"guard_note": str(e)}
        res = self.router.complete(self.agent, state["messages"])
        msgs = state["messages"] + [{"role": "assistant", "content": res.text}]
        upd: dict[str, Any] = {"steps": steps + 1, "models_used": state["models_used"] + [res.model],
                               "pending_tool": None}
        try:
            obj = extract_json(res.text)
        except ValueError:
            return upd | {"messages": msgs + [{"role": "user", "content":
                          "Your reply was not a JSON object. Reply with exactly one JSON object."}]}
        action = obj.get("action") or ("final" if self.final_key in obj else "tool" if "tool" in obj else None)
        if action == "tool":
            name, args = obj.get("tool"), obj.get("args") or {}
            try:
                sig = self.guard.check_tool(state["tool_sigs"], name, args)
            except LoopGuardTripped as e:
                self.log.event("guard", agent=self.agent, reason=str(e))
                return upd | {"messages": msgs, "guard_note": str(e)}
            return upd | {"messages": msgs, "pending_tool": {"name": name, "args": args},
                          "tool_sigs": state["tool_sigs"] + [sig]}
        if action == "final":
            try:
                return upd | {"messages": msgs, "final": self._validate(obj)}
            except ValidationError as e:
                return upd | {"messages": msgs + [{"role": "user", "content":
                              f"The {self.final_key} failed validation: {e.errors(include_url=False)}. "
                              "Fix it and reply with the full JSON."}]}
        return upd | {"messages": msgs + [{"role": "user", "content":
                      f'Unknown action. Use {{"action": "tool", ...}} or {{"action": "final", "{self.final_key}": ...}}.'}]}

    def tool_node(self, state: LoopState) -> dict:
        call = state["pending_tool"]
        out = self.toolbox.run(call["name"], call["args"])
        result = json.dumps(out["result"], ensure_ascii=False, default=str)[:TOOL_RESULT_CHARS]
        used, cap = len(state["tool_sigs"]), self.guard.max_tool_calls
        note = "" if cap is None else (
            f"\n\n[Tool budget used up ({used}/{cap}). Reply with the final JSON now.]" if used >= cap
            else f"\n\n[Tool calls used: {used}/{cap}.]")
        upd: dict[str, Any] = {"pending_tool": None, "messages": state["messages"] + [
            {"role": "user", "content": f"TOOL RESULT ({call['name']}): {result}{note}"}]}
        if call["name"] == "web_search":
            upd["web_attempts"] = state["web_attempts"] + 1
            if out["ok"]:
                upd["web_hits"] = state["web_hits"] + 1
                upd["sources"] = state["sources"] + out.get("urls", [])[:5]
        return upd

    def force_final_node(self, state: LoopState) -> dict:
        msgs = state["messages"] + [{"role": "user", "content":
               f"STOP: {state['guard_note']}. No more tools. Reply now with the final JSON "
               f'({{"action": "final", "{self.final_key}": {{...}}}}) based on what you have.'}]
        res = self.router.complete(self.agent, msgs)
        models = state["models_used"] + [res.model]
        try:
            return {"final": self._validate(extract_json(res.text)), "messages": msgs, "models_used": models}
        except (ValueError, ValidationError) as e:
            self.log.event("loop_failed", agent=self.agent, error=str(e)[:500])
            return {"final": None, "error": str(e)[:500], "messages": msgs, "models_used": models}

    @staticmethod
    def route(state: LoopState) -> Literal["tool", "agent", "force_final", "done"]:
        if state.get("final"):
            return "done"
        if state.get("guard_note"):
            return "force_final"
        if state.get("pending_tool"):
            return "tool"
        return "agent"
