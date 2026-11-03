"""Ship pipeline: gate order, session cap, 2-strike rescope, spend-gate stop, rollback, secrets.
`claude`, npm, Vercel, Supabase, smoke and health checks are all mocked."""
import json
from pathlib import Path

import pytest

from startup_os import frontmatter
from startup_os.agents.ceo import gather_context
from startup_os.config import load_config
from startup_os.runtime import Runtime
from startup_os.ship.claude import ClaudeResult, ClaudeRunner
from startup_os.ship.providers import PaidPlanRequired, StepFailed
from startup_os.ship.shell import CmdResult
from startup_os.ship.smoke import SmokeResult, derive_checks
from startup_os.tickets import Ticket

SECRETS = {"VENTURE_HELLO_VERCEL_TOKEN": "vercel-secret-token-123", "VENTURE_HELLO_VERCEL_ORG_ID": "team_abc",
           "VENTURE_HELLO_VERCEL_PROJECT_ID": "prj_hello", "VENTURE_HELLO_PROD_URL": "https://hello.vercel.app",
           "VENTURE_HELLO_SUPABASE_STAGING_DB_URL": "postgresql://staging:pw-staging@db/x",
           "VENTURE_HELLO_SUPABASE_PROD_DB_URL": "postgresql://prod:pw-prod@db/x",
           "OPENROUTER_API_KEY": "sk-or-v1-should-not-leak", "PATH": "/usr/bin"}
ORDER = ["Claude Code", "a) build", "b) tests", "c) preview deploy", "d) smoke test", "e) staging migrations",
         "e) prod migrations", "f) promote to prod", "g) prod health watch"]


class Calls(list):
    def names(self):
        return [c[0] for c in self]


class FakeClaude:
    def __init__(self, calls, ok=True, changes=True, timed_out=False, paid=None, migrations=False):
        self.calls, self.ok, self.changes, self.timed_out = calls, ok, changes, timed_out
        self.paid, self.migrations, self.prompts = paid, migrations, []

    def __call__(self, prompt, *, cwd, transcript, timeout_s):
        self.calls.append(("claude", timeout_s))
        self.prompts.append(prompt)
        cwd = Path(cwd)
        if self.changes:
            (cwd / "package.json").write_text(json.dumps({"scripts": {"build": "next build", "test": "vitest run"}}))
            (cwd / "app").mkdir(exist_ok=True)
            (cwd / "app" / f"page-{len(self.prompts)}.tsx").write_text("export default () => 'Hello'")
            if self.paid:
                (cwd / "ship.json").write_text(json.dumps({"paid_requirements": [self.paid]}))
            if self.migrations:
                (cwd / "supabase" / "migrations").mkdir(parents=True, exist_ok=True)
                (cwd / "supabase" / "migrations" / "001_init.sql").write_text("create table t(id int);")
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text('{"type":"result","result":"done"}\n')
        return ClaudeResult(ok=self.ok and not self.timed_out, exit_code=0 if self.ok else 1,
                            transcript=str(transcript), duration_s=1.5, summary="built it",
                            timed_out=self.timed_out)


class FakeShell:
    def __init__(self, calls, fail_on=()):
        self.calls, self.fail_on, self.envs = calls, set(fail_on), []

    def __call__(self, cmd, *, cwd, env, timeout_s, stdout_path=None):
        what = "test" if cmd[-1] == "test" else ("build" if cmd[-1] == "build" else "install")
        self.calls.append((f"npm {what}", None))
        self.envs.append(env)
        bad = what in self.fail_on
        return CmdResult(1 if bad else 0, "", f"{what} broke" if bad else "", 0.1)


class FakeVercel:
    def __init__(self, calls, fail=None, paid_on=None, rollback_ok=True):
        self.calls, self.fail, self.paid_on, self.rollback_ok = calls, fail, paid_on, rollback_ok
        self.secrets = []

    def __call__(self, secrets, cwd):
        self.secrets.append(secrets)
        secrets.get("VERCEL_TOKEN")      # like the real client: needs project-scoped creds
        return self

    def _maybe(self, name):
        self.calls.append((f"vercel {name}", None))
        if self.paid_on == name:
            raise PaidPlanRequired("Vercel Pro plan", "Error: this feature requires a Pro plan", 20)
        if self.fail == name:
            raise StepFailed(f"vercel {name} failed")

    def deploy_preview(self):
        self._maybe("deploy")
        return "https://hello-abc123.vercel.app"

    def promote(self, url):
        self._maybe("promote")
        return "https://hello.vercel.app"

    def rollback(self):
        self.calls.append(("vercel rollback", None))
        return CmdResult(0 if self.rollback_ok else 1, "rolled back to dpl_prev", "", 0.2)


class FakeSupabase:
    def __init__(self, calls, fail=None):
        self.calls, self.fail = calls, fail

    def __call__(self, secrets, cwd):
        return self

    def push(self, which):
        self.calls.append((f"supabase {which}", None))
        if self.fail == which:
            raise StepFailed(f"migration failed on {which}")
        return "applied"


def make_rt(cfg_path, clock, *, claude=None, shell_fail=(), vercel=None, supabase_fail=None, smoke_ok=True,
            health_seq=None, cap=3, environ=None):
    calls = Calls()
    health_seq = list(health_seq or [])
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["claude_code"] = {"sessions_per_day": cap, "ticket_timeout_min": 7}
    cfg["ship"] = {"health_minutes": 2, "health_interval_s": 30}
    cfg_path.write_text(yaml.safe_dump(cfg))
    fakes = dict(
        claude=claude or FakeClaude(calls),
        shell=FakeShell(calls, shell_fail),
        vercel=vercel or FakeVercel(calls),
        supabase=FakeSupabase(calls, supabase_fail),
        smoke=lambda url, checks, bypass: (calls.append(("smoke", url)) or
                                           SmokeResult(smoke_ok, [{"check": "GET /api/health", "ok": smoke_ok}])),
        health=lambda url, bypass: (calls.append(("health", url)) or
                                    (health_seq.pop(0) if health_seq else (True, "ok"))),
        sleep=lambda s: calls.append(("sleep", s)),
        environ=dict(SECRETS if environ is None else environ),
    )
    if isinstance(fakes["claude"], FakeClaude):
        fakes["claude"].calls = calls
    if isinstance(fakes["vercel"], FakeVercel):
        fakes["vercel"].calls = calls
    rt = Runtime(load_config(cfg_path), now=clock.now, ship_overrides=fakes)
    rt.calls, rt.fakes = calls, fakes
    return rt


def hello_ticket(rt, **kw):
    return rt.add_ticket(title=kw.pop("title", "Hello page and health route"), owner="claude-code", venture="hello",
                         criteria=kw.pop("criteria", ['The home page / shows "Hello, world"',
                                                      "GET /api/health returns {\"status\": \"ok\"}"]), **kw)


def steps(o):
    return [(s["step"], s["status"]) for s in o.steps]


# ------------------------------------------------------------------ gate order
def test_full_ship_runs_every_gate_in_order(cfg_path, clock):
    rt = make_rt(cfg_path, clock, claude=FakeClaude(None, migrations=True))
    tid = hello_ticket(rt)
    [o] = rt.build()
    assert o.status == "shipped"
    assert steps(o) == [(s, "pass") for s in ORDER]
    names = [n for n in rt.calls.names() if n != "sleep"]
    assert names[:5] == ["claude", "npm install", "npm build", "npm test", "vercel deploy"]
    assert names[5:9] == ["smoke", "supabase staging", "supabase prod", "vercel promote"]
    assert names.count("health") == 4                               # 2 min / 30 s
    t = rt.tickets.get(tid)
    assert t.status == "done" and t.prod_url == "https://hello.vercel.app"
    assert (rt.cfg.ventures_dir / "hello" / ".git").exists()


@pytest.mark.parametrize("fail_at,setup", [
    ("a) build", dict(shell_fail={"build"})),
    ("b) tests", dict(shell_fail={"test"})),
    ("c) preview deploy", dict(vercel="deploy")),
    ("d) smoke test", dict(smoke_ok=False)),
    ("e) staging migrations", dict(supabase_fail="staging")),
    ("e) prod migrations", dict(supabase_fail="prod")),
    ("f) promote to prod", dict(vercel="promote")),
])
def test_later_steps_never_run_after_a_failure(cfg_path, clock, fail_at, setup):
    if "vercel" in setup:
        setup["vercel"] = FakeVercel(None, fail=setup["vercel"])
    rt = make_rt(cfg_path, clock, claude=FakeClaude(None, migrations=True), **setup)
    hello_ticket(rt)
    [o] = rt.build()
    assert o.status == "failed"
    ran = [s for s, _ in steps(o)]
    assert ran == ORDER[:ORDER.index(fail_at) + 1]                  # nothing recorded after the failure
    assert steps(o)[-1] == (fail_at, "fail")
    later = ORDER[ORDER.index(fail_at) + 1:]
    called = rt.calls.names()
    if "e) staging migrations" in later:
        assert "supabase staging" not in called
    if "e) prod migrations" in later:
        assert "supabase prod" not in called
    if "f) promote to prod" in later:
        assert "vercel promote" not in called
    assert "health" not in called


def test_dry_run_stops_after_preview_and_smoke(cfg_path, clock):
    rt = make_rt(cfg_path, clock, claude=FakeClaude(None, migrations=True))
    tid = hello_ticket(rt)
    [o] = rt.build(dry_run=True)
    assert o.status == "dry_run_ok" and o.preview_url == "https://hello-abc123.vercel.app"
    assert steps(o)[:5] == [(s, "pass") for s in ORDER[:5]]
    assert steps(o)[5:] == [(s, "skip") for s in ORDER[5:]]
    called = rt.calls.names()
    assert not {"supabase staging", "supabase prod", "vercel promote", "health"} & set(called)
    t = rt.tickets.get(tid)
    assert t.status == "todo" and t.preview_url == o.preview_url and t.built_commit
    # a real build later reuses the dry-run commit: no second Claude Code session
    [o2] = rt.build()
    assert o2.status == "shipped" and steps(o2)[0] == ("Claude Code", "skip")
    assert rt.calls.names().count("claude") == 1


# ------------------------------------------------------------------ session cap
def test_claude_session_cap_per_day(cfg_path, clock):
    rt = make_rt(cfg_path, clock, cap=1)
    a, b = hello_ticket(rt), hello_ticket(rt, title="Second page")
    outs = rt.build(dry_run=True)
    assert [(o.ticket, o.status) for o in outs] == [(a, "dry_run_ok"), (b, "capped")]
    assert rt.calls.names().count("claude") == 1
    assert rt.tickets.get(b).attempts == 0                          # a cap is not a strike
    clock.advance(24 * 3600)
    [o] = rt.build(b, dry_run=True)
    assert o.status == "dry_run_ok"


def test_session_cap_is_separate_from_openrouter_budget(cfg_path, clock):
    rt = make_rt(cfg_path, clock, cap=1)
    hello_ticket(rt)
    rt.build(dry_run=True)
    assert all(r["requests"] == 0 for r in rt.scheduler.usage_summary())


# ------------------------------------------------------------------ 2 strikes
def test_two_failures_send_ticket_back_to_ceo(cfg_path, clock):
    rt = make_rt(cfg_path, clock, shell_fail={"test"})
    tid = hello_ticket(rt)
    [o1] = rt.build()
    t = rt.tickets.get(tid)
    assert o1.status == "failed" and t.status == "todo" and t.attempts == 1
    assert "b) tests" in t.last_error and t.built_commit is None
    [o2] = rt.build()
    assert "Previous attempt failed" in rt.fakes["claude"].prompts[-1] and "test broke" in rt.fakes["claude"].prompts[-1]
    t = rt.tickets.get(tid)
    assert o2.status == "needs_rescope" and t.status == "needs_rescope" and t.attempts == 2
    assert "## Failure summary (needs rescope)" in rt.memory.read(rt.tickets.path_of(tid))
    assert rt.build() == []                                         # never retried by build
    from startup_os.runtime import Runtime as R
    ctx = gather_context(rt.ceo_deps(__import__("startup_os.runtime", fromlist=["NullRunLog"]).NullRunLog()))
    assert "need rescoping" in ctx and tid in ctx and "b) tests" in ctx


@pytest.mark.parametrize("claude_kw,why", [(dict(timed_out=True), "timed out"), (dict(changes=False), "no changes"),
                                           (dict(ok=False), "session failed")])
def test_claude_failures_are_strikes(cfg_path, clock, claude_kw, why):
    rt = make_rt(cfg_path, clock, claude=FakeClaude(None, **claude_kw))
    tid = hello_ticket(rt)
    [o] = rt.build()
    assert o.status == "failed" and why in o.reason and steps(o)[-1][0] == "Claude Code"
    assert "npm build" not in rt.calls.names()
    assert rt.tickets.get(tid).attempts == 1
    assert rt.calls[0] == ("claude", 7 * 60)                        # per-ticket timeout


# ------------------------------------------------------------------ spend gate
def test_paid_plan_error_files_spend_request_and_stops(cfg_path, clock):
    rt = make_rt(cfg_path, clock, vercel=FakeVercel(None, paid_on="deploy"))
    tid = hello_ticket(rt)
    [o] = rt.build()
    assert o.status == "blocked_spend" and "spend request A-1" in o.reason
    assert steps(o)[-1] == ("c) preview deploy", "fail")
    assert "smoke" not in rt.calls.names()
    a = rt.approvals.get(1)
    assert a["kind"] == "spend_request" and a["requested_by"] == "build" and a["payload"]["blocks"] == [tid]
    t = rt.tickets.get(tid)
    assert t.status == "blocked" and "A-1" in t.depends_on and t.attempts == 0   # not a strike
    assert rt.build() == []                                          # blocked until a human approves
    clock.advance(1000 * 3600)
    Runtime(rt.cfg, now=clock.now, ship_overrides=rt.fakes)          # startup auto-approval pass
    assert rt.approvals.get(1)["status"] == "pending"                # never auto-approves
    rt.approve("A-1")
    assert [x.id for x in rt.builder().runnable()] == [tid]


def test_paid_requirement_declared_by_code_stops_before_build(cfg_path, clock):
    rt = make_rt(cfg_path, clock, claude=FakeClaude(None, paid={"item": "Custom domain hello.com", "amount_usd": 12,
                                                                "why": "branding"}))
    tid = hello_ticket(rt)
    [o] = rt.build()
    assert o.status == "blocked_spend"
    assert not any(n.startswith(("npm", "vercel")) for n in rt.calls.names())
    assert rt.approvals.get(1)["payload"]["amount_usd"] == 12
    assert rt.tickets.get(tid).status == "blocked"


# ------------------------------------------------------------------ rollback
def test_failed_prod_health_rolls_back_and_writes_incident(cfg_path, clock):
    rt = make_rt(cfg_path, clock, health_seq=[(True, "ok"), (False, "HTTP 500")])
    tid = hello_ticket(rt)
    [o] = rt.build()
    assert o.status == "failed" and steps(o)[-1] == ("g) prod health watch", "fail")
    names = rt.calls.names()
    assert names.index("vercel rollback") > names.index("vercel promote")
    assert names.count("health") == 2                                 # stopped at the first failure
    retros = list((rt.memory.root / "retros").glob("*incident*.md"))
    assert len(retros) == 1
    meta, body = frontmatter.parse(retros[0].read_text())
    assert meta["type"] == "incident" and meta["ticket"] == tid and meta["rolled_back"] is True
    assert "HTTP 500" in body and "## Timeline" in body
    t = rt.tickets.get(tid)
    assert t.status == "todo" and t.attempts == 1 and "rollback succeeded" in t.last_error
    assert rt.memory.log(5)[1].startswith("build: incident:") or any("incident" in m for m in rt.memory.log(5))


def test_healthy_prod_watch_passes(cfg_path, clock):
    rt = make_rt(cfg_path, clock)
    hello_ticket(rt)
    [o] = rt.build()
    assert o.status == "shipped" and "vercel rollback" not in rt.calls.names()
    assert rt.calls.names().count("sleep") == 3                     # between the 4 checks


# ------------------------------------------------------------------ selection, secrets, config
def test_only_unblocked_claude_code_tickets_run(cfg_path, clock):
    rt = make_rt(cfg_path, clock)
    research = rt.add_ticket(title="Some research", owner="research", venture="hello", criteria=["x"])
    dep = hello_ticket(rt, title="Needs research first", depends_on=[research])
    free = hello_ticket(rt, title="Independent page")
    assert [t.id for t in rt.builder().runnable()] == [free]
    assert rt.tickets.get(dep).status == "blocked"


def test_secrets_never_reach_claude_prompt_or_env(cfg_path, clock, monkeypatch):
    rt = make_rt(cfg_path, clock)
    hello_ticket(rt)
    rt.build(dry_run=True)
    prompt = rt.fakes["claude"].prompts[0]
    for v in SECRETS.values():
        if v != "/usr/bin":
            assert v not in prompt
    assert '"status": "ok"' in prompt and "Write tests for EVERY feature" in prompt
    # the real runner hands Claude Code an environment without project secrets
    seen = {}
    def shell(cmd, *, cwd, env, timeout_s, stdout_path=None):
        seen.update(env=env, cmd=cmd)
        Path(stdout_path).write_text('{"type":"result","subtype":"success","result":"ok"}\n')
        return CmdResult(0, "", "", 0.1)
    monkeypatch.setattr("os.environ", {**SECRETS, "ANTHROPIC_API_KEY": "claude-own-auth"})
    r = ClaudeRunner(rt.cfg.claude_code, shell=shell)(prompt, cwd=Path("."), transcript=rt.cfg.logs_dir / "t.jsonl",
                                                       timeout_s=10)
    assert r.ok and not any(k.startswith(("VENTURE_", "OPENROUTER")) for k in seen["env"])
    assert seen["env"]["ANTHROPIC_API_KEY"] == "claude-own-auth"
    assert "--disallowedTools" in seen["cmd"] and "Bash(vercel:*)" in seen["cmd"][seen["cmd"].index("--disallowedTools") + 1]
    # npm build/test steps also run without secrets
    assert all("VENTURE_HELLO_VERCEL_TOKEN" not in e and "VERCEL_TOKEN" not in e for e in rt.fakes["shell"].envs)


def test_prompt_refuses_secret_values(cfg_path, clock):
    rt = make_rt(cfg_path, clock)
    tid = rt.add_ticket(title="Leaky", owner="claude-code", venture="hello",
                        criteria=["use token vercel-secret-token-123"])
    [o] = rt.build(tid)
    assert o.status == "blocked_config" and "secret value" in o.reason
    assert "claude" not in rt.calls.names() and rt.tickets.get(tid).attempts == 0


def test_vercel_token_is_passed_by_env_not_argv(tmp_path):
    from startup_os.ship.providers import Vercel
    from startup_os.ship.secrets import VentureSecrets
    seen = {}
    def shell(cmd, *, cwd, env, timeout_s, stdout_path=None):
        seen.update(cmd=cmd, env=env)
        return CmdResult(0, "Preview: https://hello-xyz.vercel.app\n", "", 0.1)
    v = Vercel("vercel@64.7.0", VentureSecrets.from_env("hello", SECRETS), tmp_path, 60, shell=shell)
    assert v.deploy_preview() == "https://hello-xyz.vercel.app"
    assert seen["env"]["VERCEL_TOKEN"] == "vercel-secret-token-123"
    assert "vercel-secret-token-123" not in " ".join(seen["cmd"]) and "OPENROUTER_API_KEY" not in seen["env"]


def test_paid_plan_detection_from_cli_output(tmp_path):
    from startup_os.ship.providers import Supabase
    from startup_os.ship.secrets import VentureSecrets
    shell = lambda cmd, **kw: CmdResult(1, "", "Error: Upgrade to the Pro plan to enable this feature", 0.1)
    s = Supabase("supabase@2.121.0", VentureSecrets.from_env("hello", SECRETS), tmp_path, 60, shell=shell)
    with pytest.raises(PaidPlanRequired):
        s.push("staging")


def test_missing_credentials_block_without_strike(cfg_path, clock):
    rt = make_rt(cfg_path, clock, environ={"PATH": "/usr/bin"})
    tid = hello_ticket(rt)
    [o] = rt.build(dry_run=True)
    assert o.status == "blocked_config" and "VENTURE_HELLO_VERCEL_TOKEN" in o.reason
    assert steps(o)[-1] == ("c) preview deploy", "fail")
    assert rt.tickets.get(tid).attempts == 0


def test_run_is_logged_with_steps_durations_urls_and_transcript(cfg_path, clock):
    rt = make_rt(cfg_path, clock)
    hello_ticket(rt)
    [o] = rt.build(dry_run=True)
    ev = [json.loads(l) for l in open(o.log_path)]
    step_ev = [e for e in ev if e["type"] == "ship_step"]
    assert [e["step"] for e in step_ev] == ORDER
    assert all("duration_s" in e and e["status"] in ("pass", "fail", "skip") for e in step_ev)
    assert any(e["url"] == "https://hello-abc123.vercel.app" for e in step_ev)
    end = ev[-1]
    assert end["type"] == "run_end" and end["transcript"] == o.transcript and end["preview_url"] == o.preview_url
    run = rt.builder().d.store.runs()[0]
    assert run["status"] == "dry_run_ok" and len(run["steps"]) == len(ORDER) and run["transcript"]


def test_derive_smoke_checks_from_acceptance_criteria():
    t = Ticket(id="T-0001", title="Hello page", owner="claude-code", acceptance_criteria=[
        'The home page shows "Hello, world"', 'GET /api/health returns {"status": "ok"}',
        'The /about page contains "About us"', "It looks nice"])
    checks = derive_checks(t)
    assert [(c.path, c.expect_text) for c in checks] == [("/", "Hello, world"), ("/about", "About us")]
    t2 = t.model_copy(update={"acceptance_criteria": ["It looks nice"]})
    assert [(c.path, c.expect_text) for c in derive_checks(t2)] == [("/", None)]
