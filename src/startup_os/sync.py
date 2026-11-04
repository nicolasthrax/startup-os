"""Persist company/ to a private Git remote so it survives fresh sessions.

- The remote URL (config `company.remote`) never contains the token.
- The token (env `company.token_env`, default COMPANY_REPO_TOKEN) reaches git only as an
  in-memory config value via GIT_CONFIG_* env vars: never on disk, never on argv, never logged.
- Pull before every command, push after every commit. Never force-push.
- No remote or token → one warning per process, then local-only.
- A conflicting pull aborts the merge and raises SyncConflict; nothing is overwritten."""
from __future__ import annotations

import base64
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

INIT_SUBJECT = "system: initialise company memory repo"
BRANCH = "main"
_warned = False


class SyncError(Exception):
    pass


class SyncConflict(SyncError):
    pass


@dataclass
class GitResult:
    code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.code == 0


Git = Callable[[list[str], Path, dict[str, str]], GitResult]


def run_git(args: list[str], cwd: Path, env: dict[str, str]) -> GitResult:
    r = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True)
    return GitResult(r.returncode, r.stdout.strip(), r.stderr.strip())


def warn_once(msg: str) -> None:
    global _warned
    if not _warned:
        _warned = True
        print(f"warning: {msg}", file=sys.stderr)


@dataclass
class CompanySync:
    root: Path
    remote: str | None
    token: str | None
    git: Git = run_git
    warn: Callable[[str], None] = warn_once
    pushed: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, root: Path, remote: str | None, token_env: str,
                    environ: dict[str, str] | None = None, **kw) -> "CompanySync":
        env = os.environ if environ is None else environ
        return cls(Path(root), remote or None, env.get(token_env) or None, **kw)

    # ----------------------------------------------------------- plumbing --
    @property
    def enabled(self) -> bool:
        if self.remote and self.token:
            return True
        missing = "company.remote in config.yaml" if not self.remote else "the COMPANY_REPO_TOKEN env var"
        self.warn(f"company/ sync is off ({missing} is not set); company memory stays local-only "
                  "and will not survive a fresh session")
        return False

    def _secrets(self) -> list[str]:
        t = self.token or ""
        return [s for s in (t, self._basic()) if s]

    def _basic(self) -> str:
        return base64.b64encode(f"x-access-token:{self.token}".encode()).decode() if self.token else ""

    def _scrub(self, text: str) -> str:
        for s in self._secrets():
            text = text.replace(s, "***")
        return text

    def _env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_CONFIG_")}
        env.update(GIT_TERMINAL_PROMPT="0", GIT_CONFIG_COUNT="1",
                   GIT_CONFIG_KEY_0="http.extraHeader",
                   GIT_CONFIG_VALUE_0=f"Authorization: Basic {self._basic()}")
        return env

    def _run(self, *args: str, check: bool = True, remote: bool = False) -> GitResult:
        env = self._env() if remote else {k: v for k, v in os.environ.items() if not k.startswith("GIT_CONFIG_")}
        r = self.git(list(args), self.root, env)
        r = GitResult(r.code, self._scrub(r.stdout), self._scrub(r.stderr))
        if check and not r.ok:
            raise SyncError(f"git {args[0]} failed: {r.stderr or r.stdout}")
        return r

    def _is_ancestor(self, a: str, b: str) -> bool:
        return self._run("merge-base", "--is-ancestor", a, b, check=False).ok

    def _only_init_commit(self) -> bool:
        r = self._run("log", "--pretty=%s", check=False)
        return r.ok and r.stdout.splitlines() == [INIT_SUBJECT]

    # --------------------------------------------------------------- pull --
    def pull(self) -> str:
        """Fetch and merge the remote. Returns what happened; raises SyncConflict on a conflict."""
        if not self.enabled:
            return "local-only"
        r = self._run("fetch", "--quiet", self.remote, BRANCH, check=False, remote=True)
        if not r.ok:
            if "couldn't find remote ref" in r.stderr.lower():
                return "remote is empty"
            raise SyncError(f"fetching {self.remote} failed: {r.stderr}")
        remote = self._run("rev-parse", "FETCH_HEAD").stdout
        head = self._run("rev-parse", "HEAD").stdout
        if head == remote or self._is_ancestor(remote, head):
            return "up to date"
        if self._is_ancestor(head, remote):
            self._run("merge", "--ff-only", "--quiet", remote)
            return f"fast-forwarded to {remote[:10]}"
        if not self._run("merge-base", head, remote, check=False).ok:
            if self._only_init_commit():      # fresh local repo: adopt the remote history
                self._run("reset", "--hard", "--quiet", remote)
                return f"adopted remote history at {remote[:10]}"
            raise SyncConflict(f"company/ and {self.remote} have unrelated histories; nothing was changed. "
                               "Resolve by hand (never force-push).")
        m = self._run("merge", "--no-edit", "--quiet", "-m", "sync: merge company remote", remote, check=False)
        if not m.ok:
            files = self._run("diff", "--name-only", "--diff-filter=U", check=False).stdout.split()
            self._run("merge", "--abort", check=False)
            raise SyncConflict(f"pull from {self.remote} conflicts in {', '.join(files) or '(unknown files)'}; "
                               "merge aborted, local company/ unchanged. Resolve by hand, then run "
                               "`startup-os sync` (never force-push).")
        return f"merged {remote[:10]}"

    # --------------------------------------------------------------- push --
    def push(self) -> str:
        """Push HEAD to the remote branch. A rejected push pulls (merge) and retries once; never forces."""
        if not self.enabled:
            return "local-only"
        for attempt in (1, 2):
            r = self._run("push", "--quiet", self.remote, f"HEAD:refs/heads/{BRANCH}", check=False, remote=True)
            if r.ok:
                sha = self._run("rev-parse", "HEAD").stdout
                self.pushed.append(sha)
                return f"pushed {sha[:10]}"
            rejected = any(s in r.stderr for s in ("rejected", "non-fast-forward", "fetch first"))
            if not rejected or attempt == 2:
                raise SyncError(f"push to {self.remote} failed: {r.stderr}")
            self.pull()   # raises SyncConflict on a real conflict
        raise SyncError("unreachable")

    def sync(self) -> tuple[str, str]:
        return self.pull(), self.push()
