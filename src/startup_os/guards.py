"""Loop guards for agent tool loops. State lives in the (checkpointed) graph
state, so these are pure checks."""
from __future__ import annotations

import json


class LoopGuardTripped(Exception):
    pass


def tool_signature(name: str, args: dict) -> str:
    return f"{name}:{json.dumps(args or {}, sort_keys=True, ensure_ascii=False).lower()}"


class LoopGuard:
    def __init__(self, max_steps: int, max_identical_tool_calls: int = 2,
                 max_tool_calls: int | None = None):
        self.max_steps = max_steps
        self.max_identical = max_identical_tool_calls
        self.max_tool_calls = max_tool_calls

    def check_step(self, steps: int) -> None:
        if steps >= self.max_steps:
            raise LoopGuardTripped(f"max steps reached ({steps}/{self.max_steps})")

    def check_tool(self, previous_sigs: list[str], name: str, args: dict) -> str:
        sig = tool_signature(name, args)
        if self.max_tool_calls is not None and len(previous_sigs) >= self.max_tool_calls:
            raise LoopGuardTripped(f"tool budget used up ({len(previous_sigs)}/{self.max_tool_calls})")
        if previous_sigs.count(sig) + 1 >= self.max_identical:
            raise LoopGuardTripped(f"repeated identical tool call: {sig}")
        return sig
