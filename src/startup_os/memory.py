"""The company memory repo: a Git repo of Markdown where every write is a commit."""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .search_index import SearchIndex

DIRS = ["plans", "decisions", "research", "research/ideas", "tickets", "retros"]

VENTURE_TEMPLATE = """# Venture

_No venture selected yet. The Research agent writes idea briefs to `research/ideas/`;
a later stage will pick one and record the decision in `decisions/`._
"""


class MemoryError(Exception):
    pass


class MemoryRepo:
    def __init__(self, root: Path, index: SearchIndex | None = None):
        self.root = Path(root).resolve()
        self.index = index

    # -- git -------------------------------------------------------------
    def _git(self, *args: str) -> str:
        res = subprocess.run(
            ["git", *args], cwd=self.root, capture_output=True, text=True
        )
        if res.returncode != 0:
            raise MemoryError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
        return res.stdout.strip()

    def head(self) -> str:
        return self._git("rev-parse", "HEAD")

    def log(self, n: int = 20) -> list[str]:
        return self._git("log", f"-{n}", "--pretty=%s").splitlines()

    # -- lifecycle ---------------------------------------------------------
    def ensure(self) -> bool:
        """Create the repo layout on first run. Returns True if it was created."""
        created = not (self.root / ".git").exists()
        self.root.mkdir(parents=True, exist_ok=True)
        if created:
            self._git("init", "-q", "-b", "main")
            self._git("config", "user.name", "Startup OS")
            self._git("config", "user.email", "startup-os@localhost")
            self._git("config", "commit.gpgsign", "false")
        for d in DIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)
            keep = self.root / d / ".gitkeep"
            if not any((self.root / d).iterdir()):
                keep.touch()
        venture = self.root / "venture.md"
        if not venture.exists():
            venture.write_text(VENTURE_TEMPLATE)
        approvals = self.root / "approvals.log"
        if not approvals.exists():
            approvals.write_text("")
        if self._git("status", "--porcelain"):
            self._git("add", "-A")
            self._git("commit", "-q", "-m", "system: initialise company memory repo")
        if self.index:
            self.index.rebuild()
        return created

    # -- writes ------------------------------------------------------------
    def _resolve(self, rel_path: str | Path) -> Path:
        p = (self.root / rel_path).resolve()
        if self.root not in p.parents:
            raise MemoryError(f"path escapes memory repo: {rel_path}")
        if ".git" in p.relative_to(self.root).parts:
            raise MemoryError("refusing to write inside .git")
        return p

    def _commit(self, path: Path, agent: str, message: str) -> str:
        rel = str(path.relative_to(self.root))
        self._git("add", "--", rel)
        if not self._git("diff", "--cached", "--name-only"):
            return self.head()  # no change
        self._git("commit", "-q", "-m", f"{agent}: {message}", "-m", f"File: {rel}")
        sha = self.head()
        if self.index:
            self.index.rebuild()
        return sha

    def write(self, rel_path: str | Path, content: str, *, agent: str, message: str) -> str:
        """Write a file and commit it. Returns the commit SHA."""
        p = self._resolve(rel_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return self._commit(p, agent, message)

    def append_approval(self, line: str, *, agent: str) -> str:
        p = self._resolve("approvals.log")
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with p.open("a", encoding="utf-8") as f:
            f.write(f"{ts}\t{agent}\t{line}\n")
        return self._commit(p, agent, f"approval log: {line[:60]}")

    def read(self, rel_path: str | Path) -> str:
        return self._resolve(rel_path).read_text(encoding="utf-8")

    def unique_path(self, rel_path: str) -> str:
        p = Path(rel_path)
        candidate, i = p, 2
        while (self.root / candidate).exists():
            candidate = p.with_name(f"{p.stem}-{i}{p.suffix}")
            i += 1
        return str(candidate)
