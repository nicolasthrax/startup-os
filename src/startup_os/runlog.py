"""Structured JSONL run logs: one file per run in logs/runs/, plus logs/runs.jsonl index."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RunLog:
    def __init__(self, logs_dir: Path, command: str, run_id: str | None = None, **meta: Any):
        self.logs_dir = Path(logs_dir)
        (self.logs_dir / "runs").mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = self.logs_dir / "runs" / f"{stamp}_{command}_{self.run_id}.jsonl"
        self.command = command
        self.totals = {"llm_calls": 0, "llm_errors": 0, "tool_calls": 0,
                       "prompt_tokens": 0, "completion_tokens": 0}
        self.event("run_start", command=command, **meta)

    def event(self, type: str, **data: Any) -> None:
        if type == "llm_call":
            self.totals["llm_calls"] += 1
            self.totals["prompt_tokens"] += data.get("prompt_tokens", 0) or 0
            self.totals["completion_tokens"] += data.get("completion_tokens", 0) or 0
        elif type == "llm_error":
            self.totals["llm_errors"] += 1
        elif type == "tool_call":
            self.totals["tool_calls"] += 1
        rec = {"ts": _now(), "run_id": self.run_id, "type": type, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def close(self, status: str, **data: Any) -> None:
        self.event("run_end", status=status, totals=self.totals, **data)
        with (self.logs_dir / "runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": _now(), "run_id": self.run_id, "command": self.command,
                                "status": status, "log": str(self.path), **self.totals,
                                **data}, default=str) + "\n")


class NullRunLog(RunLog):
    def __init__(self):  # noqa: D401 - no files
        self.run_id = "null"
        self.totals = {}

    def event(self, type: str, **data: Any) -> None:
        pass

    def close(self, status: str, **data: Any) -> None:
        pass
