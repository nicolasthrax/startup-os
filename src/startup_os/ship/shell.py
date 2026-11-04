"""Subprocess helper with timeouts (mocked in tests)."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class CmdResult:
    code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.code == 0 and not self.timed_out

    def tail(self, n: int = 1500) -> str:
        return (self.stdout + "\n" + self.stderr).strip()[-n:]


class Shell(Protocol):
    def __call__(self, cmd: list[str], *, cwd: Path, env: dict[str, str], timeout_s: float,
                 stdout_path: Path | None = None) -> CmdResult: ...


def run(cmd: list[str], *, cwd: Path, env: dict[str, str], timeout_s: float,
        stdout_path: Path | None = None) -> CmdResult:
    """Run a command in its own process group; kill the whole group on timeout."""
    t0 = time.monotonic()
    out_f = open(stdout_path, "w", encoding="utf-8") if stdout_path else subprocess.PIPE
    try:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out_f, stderr=subprocess.PIPE, text=True,
                             start_new_session=True)
        try:
            stdout, stderr = p.communicate(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            stdout, stderr = p.communicate()
            timed_out = True
    except FileNotFoundError as e:
        return CmdResult(127, "", str(e), time.monotonic() - t0)
    finally:
        if stdout_path:
            out_f.close()
    if stdout_path:
        stdout = Path(stdout_path).read_text(encoding="utf-8", errors="replace")[-20000:]
    return CmdResult(p.returncode if not timed_out else -9, stdout or "", stderr or "",
                     time.monotonic() - t0, timed_out)
