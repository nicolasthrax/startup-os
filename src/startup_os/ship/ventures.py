"""Venture code repos: ventures/<slug>/, each its own Git repo (git-ignored by Startup OS)."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

GITIGNORE = "node_modules/\n.next/\n.vercel/\n.env\n.env.*\ncoverage/\nplaywright-report/\ntest-results/\n"
README = """# {name}

Venture code repo managed by Startup OS. Tickets are implemented by headless Claude Code
(`startup-os build`); the ship bar (build, tests, preview, smoke, migrations, promote, health)
is run by Startup OS, not by Claude Code.

`ship.json` (written by Claude Code when needed) declares anything that needs money:

    {{"paid_requirements": [{{"item": "Custom domain", "amount_usd": 12, "why": "..."}}]}}
"""


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:40] or "venture"


class VentureRepo:
    def __init__(self, root: Path, name: str):
        self.name = name
        self.slug = slugify(name)
        self.path = Path(root) / self.slug

    def git(self, *args: str) -> str:
        r = subprocess.run(["git", *args], cwd=self.path, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
        return r.stdout.strip()

    def ensure(self) -> bool:
        created = not (self.path / ".git").exists()
        self.path.mkdir(parents=True, exist_ok=True)
        if created:
            self.git("init", "-q", "-b", "main")
            self.git("config", "user.name", "Startup OS")
            self.git("config", "user.email", "startup-os@localhost")
            self.git("config", "commit.gpgsign", "false")
            (self.path / ".gitignore").write_text(GITIGNORE)
            (self.path / "README.md").write_text(README.format(name=self.name))
            self.git("add", "-A")
            self.git("commit", "-q", "-m", "Initialise venture repo")
        return created

    def head(self) -> str:
        return self.git("rev-parse", "HEAD")

    def dirty(self) -> bool:
        return bool(self.git("status", "--porcelain"))

    def commit_all(self, message: str) -> str | None:
        """Commit everything Claude Code changed. Returns the SHA, or None if nothing changed."""
        if not self.dirty():
            return None
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.head()

    def manifest(self) -> dict:
        p = self.path / "ship.json"
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"ship.json is not valid JSON: {e}") from e

    def package_json(self) -> dict:
        p = self.path / "package.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def migrations(self) -> list[Path]:
        return sorted((self.path / "supabase" / "migrations").glob("*.sql"))
