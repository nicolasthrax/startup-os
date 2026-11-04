"""company/ sync: pull at start, push after every commit, warn-once local-only, conflicts stop,
never force-push, token never on disk/argv/logs. The remote is a local bare repo (no network)."""
import subprocess
from pathlib import Path

import pytest
import yaml

from startup_os import cli, sync as sync_mod
from startup_os.config import load_config
from startup_os.runtime import Runtime
from startup_os.ship.secrets import clean_env
from startup_os.sync import CompanySync, SyncConflict, run_git

TOKEN = "ghp_company-secret-token-1234567890"


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture(autouse=True)
def _reset_warning(monkeypatch):
    monkeypatch.setattr(sync_mod, "_warned", False)


@pytest.fixture
def remote(tmp_path):
    bare = tmp_path / "company-remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True)
    return bare


def make_cfg(tmp_path, name, remote=None):
    """A separate checkout ("session") with its own company/, data/ and logs/."""
    root = tmp_path / name
    root.mkdir()
    cfg = {"paths": {"company": "company", "data": "data", "logs": "logs"},
           "llm": {"models": ["m1:free"]},
           "budget": {"daily_requests": 100, "per_minute": 100, "agents": {"research": 50, "ceo": 30, "reserve": 20}}}
    if remote is not None:
        cfg["company"] = {"remote": str(remote)}
    (root / "config.yaml").write_text(yaml.safe_dump(cfg))
    return root / "config.yaml"


_DATES = iter(f"2026-01-{d:02d}T12:00:00Z" for d in range(1, 29))


def session(tmp_path, name, remote, clock, **kw):
    """A new session. Each gets its own init-commit date, as real sessions created at different
    times do (otherwise identical init commits make the histories look related)."""
    import os
    date = next(_DATES)
    old = {k: os.environ.get(k) for k in ("GIT_AUTHOR_DATE", "GIT_COMMITTER_DATE")}
    os.environ.update(GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    try:
        cfg = load_config(make_cfg(tmp_path, name, remote))
        rt = Runtime(cfg, now=clock.now, **kw)
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return rt


def add_ticket(rt, title="Hello page"):
    return rt.add_ticket(title=title, owner="claude-code", venture="hello", criteria=['The home page / shows "Hi"'])


def remote_head(remote):
    return git(remote, "rev-parse", "main")


# ------------------------------------------------------------------ local-only
def test_no_remote_warns_once_and_stays_local(tmp_path, clock, capsys, monkeypatch):
    monkeypatch.setenv("COMPANY_REPO_TOKEN", TOKEN)
    rt = session(tmp_path, "a", None, clock)
    add_ticket(rt)
    session(tmp_path, "b", None, clock)
    err = capsys.readouterr().err
    assert err.count("company/ sync is off") == 1 and "company.remote" in err
    assert rt.pulled == "local-only"


def test_missing_token_warns_once_and_continues(tmp_path, clock, capsys, monkeypatch, remote):
    monkeypatch.delenv("COMPANY_REPO_TOKEN", raising=False)
    rt = session(tmp_path, "a", remote, clock)
    add_ticket(rt)
    add_ticket(rt, "Second")
    err = capsys.readouterr().err
    assert err.count("company/ sync is off") == 1 and "COMPANY_REPO_TOKEN" in err
    assert git(remote, "branch", "--list") == ""                      # nothing pushed


# ------------------------------------------------------------------ push / pull
def test_every_commit_is_pushed_and_a_fresh_session_pulls_it(tmp_path, clock, monkeypatch, remote):
    monkeypatch.setenv("COMPANY_REPO_TOKEN", TOKEN)
    a = session(tmp_path, "a", remote, clock)
    tid = add_ticket(a)
    assert remote_head(remote) == a.memory.head()                     # pushed right after the commit
    tid2 = add_ticket(a, "Second")
    assert remote_head(remote) == a.memory.head()
    # a brand-new session (empty company/) starts from the remote's state
    b = session(tmp_path, "b", remote, clock)
    assert b.pulled.startswith("adopted remote history")
    assert b.memory.head() == a.memory.head()
    assert [t.id for t in b.tickets.list()] == [tid, tid2]
    assert b.index.search("Second")                                   # index rebuilt after the pull
    # and a's next command picks up b's work
    tid3 = add_ticket(b, "Third")
    a2 = Runtime(a.cfg, now=clock.now)
    assert a2.pulled.startswith("fast-forwarded") and tid3 in [t.id for t in a2.tickets.list()]


def test_diverged_but_compatible_histories_merge_and_push(tmp_path, clock, monkeypatch, remote):
    monkeypatch.setenv("COMPANY_REPO_TOKEN", TOKEN)
    a = session(tmp_path, "a", remote, clock)
    add_ticket(a)
    b = session(tmp_path, "b", remote, clock)
    b.memory.write("retros/b.md", "from b\n", agent="test", message="b note")
    # a commits offline: its push is rejected, so it pulls (merges) and retries once
    a.memory.write("retros/a.md", "from a\n", agent="test", message="a note")
    assert remote_head(remote) == a.memory.head()
    assert (a.memory.root / "retros" / "b.md").exists()
    assert git(remote, "merge-base", "--is-ancestor", b.memory.head(), "main") == ""   # b's work kept


def test_conflicting_pull_stops_and_never_force_pushes(tmp_path, clock, monkeypatch, remote):
    monkeypatch.setenv("COMPANY_REPO_TOKEN", TOKEN)
    a = session(tmp_path, "a", remote, clock)
    a.memory.write("retros/x.md", "base\n", agent="test", message="base")
    b = session(tmp_path, "b", remote, clock)
    b.memory.write("retros/x.md", "b's version\n", agent="test", message="b edit")
    pushed_by_b = remote_head(remote)
    with pytest.raises(SyncConflict, match="retros/x.md"):
        a.memory.write("retros/x.md", "a's version\n", agent="test", message="a edit")
    assert remote_head(remote) == pushed_by_b                         # remote untouched: no force-push
    assert (a.memory.root / "retros" / "x.md").read_text() == "a's version\n"   # local work kept
    assert git(a.memory.root, "status", "--porcelain") == ""          # merge aborted cleanly
    # starting a new command on a reports the conflict and stops
    with pytest.raises(SyncConflict):
        Runtime(a.cfg, now=clock.now)


def test_unrelated_histories_with_local_work_stop(tmp_path, clock, monkeypatch, remote):
    monkeypatch.delenv("COMPANY_REPO_TOKEN", raising=False)
    local = session(tmp_path, "a", None, clock)                        # local-only work first
    add_ticket(local)
    monkeypatch.setenv("COMPANY_REPO_TOKEN", TOKEN)
    other = session(tmp_path, "b", remote, clock)
    add_ticket(other)
    cfg = local.cfg.model_copy(update={"company": local.cfg.company.model_copy(update={"remote": str(remote)})})
    with pytest.raises(SyncConflict, match="unrelated histories"):
        Runtime(cfg, now=clock.now)
    assert remote_head(remote) == other.memory.head()


def test_force_push_is_never_used(tmp_path, clock, monkeypatch, remote):
    monkeypatch.setenv("COMPANY_REPO_TOKEN", TOKEN)
    seen = []
    def spy(args, cwd, env):
        seen.append(args)
        return run_git(args, cwd, env)
    cfg = load_config(make_cfg(tmp_path, "a", remote))
    s = CompanySync.from_config(cfg.company_dir, cfg.company.remote, cfg.company.token_env, git=spy)
    rt = Runtime(cfg, now=clock.now, sync=s)
    add_ticket(rt)
    pushes = [a for a in seen if a[0] == "push"]
    assert pushes and not any(f in a for a in pushes for f in ("--force", "-f", "--force-with-lease", "--mirror"))
    assert not any(a[-1].startswith("+") for a in pushes)


# ------------------------------------------------------------------ token hygiene
def test_token_only_in_env_never_on_disk_argv_or_logs(tmp_path, clock, monkeypatch, remote):
    monkeypatch.setenv("COMPANY_REPO_TOKEN", TOKEN)
    seen = []
    def spy(args, cwd, env):
        seen.append((args, env))
        return run_git(args, cwd, env)
    cfg = load_config(make_cfg(tmp_path, "a", remote))
    s = CompanySync.from_config(cfg.company_dir, cfg.company.remote, cfg.company.token_env, git=spy)
    rt = Runtime(cfg, now=clock.now, sync=s)
    add_ticket(rt)
    import base64
    basic = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()
    assert seen and not any(TOKEN in " ".join(a) or basic in " ".join(a) for a, _ in seen)   # never on argv
    remote_calls = [env for a, env in seen if a[0] in ("fetch", "push")]
    assert remote_calls and all(env["GIT_CONFIG_VALUE_0"] == f"Authorization: Basic {basic}" for env in remote_calls)
    assert all("GIT_CONFIG_VALUE_0" not in env for a, env in seen if a[0] not in ("fetch", "push"))
    for f in list(tmp_path.rglob("*")):                               # nothing on disk: config, logs, db, remote
        if f.is_file():
            data = f.read_bytes()
            assert TOKEN.encode() not in data and basic.encode() not in data, f


def test_errors_are_scrubbed_of_the_token(tmp_path):
    from startup_os.sync import GitResult, SyncError
    def failing(args, cwd, env):
        return GitResult(128, "", f"fatal: auth failed for {TOKEN}")
    s = CompanySync(tmp_path, "https://github.com/x/y.git", TOKEN, git=failing, warn=lambda m: None)
    with pytest.raises(SyncError) as e:
        s.pull()
    assert TOKEN not in str(e.value) and "***" in str(e.value)


def test_config_rejects_credentials_in_the_remote_url(tmp_path):
    p = make_cfg(tmp_path, "a", f"https://x-access-token:{TOKEN}@github.com/nicolasthrax/c.git")
    with pytest.raises(ValueError, match="must not contain credentials"):
        load_config(p)


def test_claude_code_env_never_gets_the_company_token():
    assert "COMPANY_REPO_TOKEN" not in clean_env({"COMPANY_REPO_TOKEN": TOKEN, "PATH": "/usr/bin"})


# ------------------------------------------------------------------ CLI
def test_sync_command(tmp_path, clock, monkeypatch, remote, capsys):
    monkeypatch.setenv("COMPANY_REPO_TOKEN", TOKEN)
    p = make_cfg(tmp_path, "a", remote)
    assert cli.main(["--config", str(p), "sync"]) == 0
    out = capsys.readouterr().out
    assert "pull: remote is empty" in out and "push: pushed" in out and TOKEN not in out
    head = remote_head(remote)
    monkeypatch.setenv("GIT_AUTHOR_DATE", "2026-02-01T00:00:00Z")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2026-02-01T00:00:00Z")
    p2 = make_cfg(tmp_path, "b", remote)
    assert cli.main(["--config", str(p2), "sync"]) == 0
    assert "pull: adopted remote history" in capsys.readouterr().out
    assert remote_head(remote) == head


def test_sync_command_without_token_fails_loudly(tmp_path, monkeypatch, remote, capsys):
    monkeypatch.delenv("COMPANY_REPO_TOKEN", raising=False)
    p = make_cfg(tmp_path, "a", remote)
    assert cli.main(["--config", str(p), "sync"]) == 1
    assert "COMPANY_REPO_TOKEN" in capsys.readouterr().err
