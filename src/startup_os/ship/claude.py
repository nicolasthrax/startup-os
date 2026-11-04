"""Headless Claude Code runner (`claude -p`). Mocked in tests."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .secrets import clean_env
from .shell import CmdResult, Shell, run


@dataclass
class ClaudeResult:
    ok: bool
    exit_code: int
    transcript: str
    duration_s: float
    summary: str
    timed_out: bool = False
    cost_usd: float | None = None


class ClaudeRunnerProto(Protocol):
    def __call__(self, prompt: str, *, cwd: Path, transcript: Path, timeout_s: float) -> ClaudeResult: ...


class ClaudeRunner:
    def __init__(self, cfg, shell: Shell = run):
        self.cfg, self.shell = cfg, shell

    def command(self, prompt: str) -> list[str]:
        c = self.cfg
        cmd = [c.command, "-p", prompt, "--output-format", "stream-json", "--verbose",
               "--max-turns", str(c.max_turns), "--permission-mode", c.permission_mode]
        if c.allowed_tools:
            cmd += ["--allowedTools", ",".join(c.allowed_tools)]
        if c.disallowed_tools:
            cmd += ["--disallowedTools", ",".join(c.disallowed_tools)]
        return cmd

    def __call__(self, prompt: str, *, cwd: Path, transcript: Path, timeout_s: float) -> ClaudeResult:
        transcript.parent.mkdir(parents=True, exist_ok=True)
        # No project secrets in Claude Code's environment; its own auth vars are kept.
        res: CmdResult = self.shell(self.command(prompt), cwd=cwd, env=clean_env(), timeout_s=timeout_s,
                                    stdout_path=transcript)
        summary, cost, is_error = "", None, False
        for line in reversed(transcript.read_text(encoding="utf-8", errors="replace").splitlines()
                             if transcript.exists() else []):
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "result":
                summary = str(ev.get("result") or "")[:2000]
                cost = ev.get("total_cost_usd")
                is_error = bool(ev.get("is_error")) or ev.get("subtype", "success") != "success"
                break
        if res.timed_out:
            summary = f"timed out after {timeout_s / 60:.0f} min. " + summary
        elif not summary:
            summary = res.tail(800)
        return ClaudeResult(ok=res.ok and not is_error, exit_code=res.code, transcript=str(transcript),
                            duration_s=res.duration_s, summary=summary, timed_out=res.timed_out, cost_usd=cost)
