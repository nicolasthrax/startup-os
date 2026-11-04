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
           "VENTURE_HELLO_SUPABASE_STAGING_API_URL": "https://staging.supabase.co",
           "VENTURE_HELLO_SUPABASE_STAGING_ANON_KEY": "anon-staging-key",
           "VENTURE_HELLO_SUPABASE_PROD_API_URL": "https://prod.supabase.co",
           "VENTURE_HELLO_SUPABASE_PROD_ANON_KEY": "anon-prod-key",
           "OPENROUTER_API_KEY": "sk-or-v1-should-not-leak", "PATH": "/usr/bin"}
ORDER = ["Claude Code", "a) build", "b) tests", "c) staging migrations", "d) preview deploy", "e) smoke test",
         "f) prod migrations", "g) prod deploy", "h) prod smoke test", "i) prod health watch"]
PREVIEW, PROD = "https://hello-abc123.vercel.app", "https://hello.vercel.app"


class Calls(list):
    def names(self):
        return [c[0] for c in self]


class FakeClaude:
    def __init__(self, calls, ok=True, changes=True, timed_out=False, paid=None, migrations=False,
                 migration_sql="create table t(id int not null);"):
        self.calls, self.ok, self.changes, self.timed_out = calls, ok, changes, timed_out
        self.paid, self.migrations, self.prompts = paid, migrations, []
        self.migration_sql = migration_sql

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
                (cwd / "supabase" / "migrations" / f"{len(self.prompts):03d}_m.sql").write_text(self.migration_sql)
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

    def deploy_prod(self):
        self._maybe("deploy --prod")
        return "https://hello-prod789.vercel.app"

    def sync_env(self, target, values):
        self.calls.append((f"vercel env {target}", dict(values)))

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
            prod_smoke_ok=True, health_seq=None, cap=3, environ=None):
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
        smoke=lambda url, checks, bypass: _smoke(calls, url, checks, smoke_ok if url == PREVIEW else prod_smoke_ok),
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


def _smoke(calls, url, checks, ok):
    calls.append(("smoke", url))
    calls.smoke_checks = getattr(calls, "smoke_checks", []) + [(url, [(c.path, c.expect_text) for c in checks])]
    return SmokeResult(ok, [{"check": "GET /api/health", "ok": ok}])


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
    assert names[:12] == ["claude", "npm install", "npm build", "npm test", "supabase staging", "vercel env preview",
                          "vercel deploy", "smoke", "supabase prod", "vercel env production", "vercel deploy --prod",
                          "smoke"]
    assert "vercel promote" not in names                            # prod is a fresh build, never a promoted preview
    assert names.count("health") == 4                               # 2 min / 30 s
    assert [c for c in rt.calls if c[0] in ("smoke", "health")][:3] == [("smoke", PREVIEW), ("smoke", PROD),
                                                                         ("health", PROD)]
    # the prod smoke test runs the same Playwright checks as the preview one
    (u1, c1), (u2, c2) = rt.calls.smoke_checks
    assert (u1, u2) == (PREVIEW, PROD) and c1 == c2 == [("/", "Hello, world")]
    g = next(x for x in o.steps if x["step"] == "g) prod deploy")
    assert "https://hello-prod789.vercel.app" in g["detail"] and g["url"] == PROD
    assert rt.tickets.get(tid).built_commit[:10] in g["detail"]     # same commit as the preview
    envs = {c[0]: c[1] for c in rt.calls if c[0].startswith("vercel env")}
    assert envs["vercel env preview"]["NEXT_PUBLIC_SUPABASE_URL"] == "https://staging.supabase.co"   # preview → staging
    assert envs["vercel env production"]["NEXT_PUBLIC_SUPABASE_URL"] == "https://prod.supabase.co"
    t = rt.tickets.get(tid)
    assert t.status == "done" and t.prod_url == "https://hello.vercel.app"
    assert (rt.cfg.ventures_dir / "hello" / ".git").exists()


@pytest.mark.parametrize("fail_at,setup", [
    ("a) build", dict(shell_fail={"build"})),
    ("b) tests", dict(shell_fail={"test"})),
    ("c) staging migrations", dict(supabase_fail="staging")),
    ("d) preview deploy", dict(vercel="deploy")),
    ("e) smoke test", dict(smoke_ok=False)),
    ("f) prod migrations", dict(supabase_fail="prod")),
    ("g) prod deploy", dict(vercel="deploy --prod")),
    ("h) prod smoke test", dict(prod_smoke_ok=False)),
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
    called = [c if c[0] == "smoke" else c[0] for c in rt.calls]
    gate_calls = {"b) tests": "npm test", "c) staging migrations": "supabase staging", "d) preview deploy": "vercel deploy",
                  "e) smoke test": ("smoke", PREVIEW), "f) prod migrations": "supabase prod",
                  "g) prod deploy": "vercel deploy --prod", "h) prod smoke test": ("smoke", PROD),
                  "i) prod health watch": "health"}
    for step in later:
        assert gate_calls[step] not in called, f"{step} ran after {fail_at} failed"


def test_dry_run_lints_but_applies_no_migrations_and_stops_after_smoke(cfg_path, clock):
    rt = make_rt(cfg_path, clock, claude=FakeClaude(None, migrations=True))
    tid = hello_ticket(rt)
    [o] = rt.build(dry_run=True)
    assert o.status == "dry_run_ok" and o.preview_url == "https://hello-abc123.vercel.app"
    assert steps(o) == [("Claude Code", "pass"), ("a) build", "pass"), ("b) tests", "pass"),
                        ("c) staging migrations", "skip"), ("d) preview deploy", "pass"), ("e) smoke test", "pass"),
                        ("f) prod migrations", "skip"), ("g) prod deploy", "skip"), ("h) prod smoke test", "skip"),
                        ("i) prod health watch", "skip")]
    assert "dry run: not applied; lint ok (1 new migration file(s)" in o.steps[3]["detail"]
    called = rt.calls.names()
    assert not {"supabase staging", "supabase prod", "vercel deploy --prod", "vercel env production", "health"} & set(called)
    assert ("smoke", PROD) not in rt.calls
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
    assert steps(o)[-1] == ("d) preview deploy", "fail")
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
def test_three_consecutive_failed_health_checks_roll_back_and_write_incident(cfg_path, clock):
    rt = make_rt(cfg_path, clock, health_seq=[(True, "ok"), (False, "HTTP 500"), (False, "HTTP 500"),
                                              (False, "HTTP 502")])
    tid = hello_ticket(rt)
    [o] = rt.build()
    assert o.status == "failed" and steps(o)[-1] == ("i) prod health watch", "fail")
    assert "failed 3x in a row" in o.reason
    names = rt.calls.names()
    assert names.index("vercel rollback") > names.index("vercel deploy --prod")
    assert names.count("health") == 4                                 # stopped at the 3rd consecutive failure
    retros = list((rt.memory.root / "retros").glob("*incident*.md"))
    assert len(retros) == 1
    meta, body = frontmatter.parse(retros[0].read_text())
    assert meta["type"] == "incident" and meta["ticket"] == tid and meta["rolled_back"] is True
    assert "HTTP 502" in body and "## Timeline" in body and body.count("| FAIL |") == 3
    t = rt.tickets.get(tid)
    assert t.status == "todo" and t.attempts == 1 and "rollback succeeded" in t.last_error
    assert rt.memory.log(5)[1].startswith("build: incident:") or any("incident" in m for m in rt.memory.log(5))


@pytest.mark.parametrize("seq", [
    [(False, "HTTP 500"), (False, "timeout"), (True, "ok"), (True, "ok")],      # 2 in a row: tolerated
    [(False, "x"), (True, "ok"), (False, "x"), (False, "x")],                   # never 3 consecutive
])
def test_fewer_than_three_consecutive_failures_do_not_roll_back(cfg_path, clock, seq):
    rt = make_rt(cfg_path, clock, health_seq=seq)
    hello_ticket(rt)
    [o] = rt.build()
    assert o.status == "shipped" and "vercel rollback" not in rt.calls.names()
    assert not list((rt.memory.root / "retros").glob("*incident*.md"))


def test_rollback_threshold_is_configurable(cfg_path, clock):
    import yaml
    rt = make_rt(cfg_path, clock, health_seq=[(False, "x")])
    assert rt.cfg.ship.health_failures_before_rollback == 3            # default
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["ship"]["health_failures_before_rollback"] = 1
    cfg_path.write_text(yaml.safe_dump(cfg))
    rt2 = Runtime(load_config(cfg_path), now=clock.now, ship_overrides=rt.fakes)
    hello_ticket(rt2)
    [o] = rt2.build()
    assert o.status == "failed" and "vercel rollback" in rt.calls.names()


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
    assert steps(o)[-1] == ("d) preview deploy", "fail")
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


# ------------------------------------------------------------------ migration lint
@pytest.mark.parametrize("sql,ok", [
    ("create table t (id int not null, name text not null);", True),
    ("alter table t add column c text;", True),
    ("alter table t add column c int not null default 0;", True),
    ("alter table t alter column c drop not null;", True),
    ("create index i on t(c);", True),
    ("create function f() returns void as $$ begin drop table x; end $$ language plpgsql;", True),
    ("-- drop table t\ninsert into t values ('rename me');", True),
    ("drop table t;", False),
    ("alter table t drop column c;", False),
    ("drop index if exists i;", False),
    ("alter table t rename column a to b;", False),
    ("alter table t rename to u;", False),
    ("alter table t add column c int not null;", False),
    ("alter table t add column a int, add column b int not null;", False),
    ("alter table t alter column c set not null;", False),
    ("alter table t alter column c type bigint;", False),
    ("truncate t;", False),
])
def test_migration_lint_rules(sql, ok):
    from startup_os.ship.migration_lint import lint_sql
    assert (lint_sql(sql) == []) == ok


@pytest.mark.parametrize("dry_run", [False, True])
def test_non_additive_migration_fails_before_anything_is_applied(cfg_path, clock, dry_run):
    rt = make_rt(cfg_path, clock, claude=FakeClaude(None, migrations=True,
                                                     migration_sql="alter table users drop column email;"))
    tid = hello_ticket(rt)
    [o] = rt.build(dry_run=dry_run)
    assert o.status == "failed" and steps(o)[-1] == ("c) staging migrations", "fail")
    assert "not backward-compatible" in o.reason and "DROP" in o.reason and "001_m.sql" in o.reason
    assert not {"supabase staging", "vercel deploy", "vercel env preview"} & set(rt.calls.names())
    t = rt.tickets.get(tid)
    assert t.attempts == 1 and t.built_commit is None                 # code must change: new session next time


def test_lint_only_checks_migrations_added_since_last_ship(cfg_path, clock):
    rt = make_rt(cfg_path, clock, claude=FakeClaude(None, migrations=True))
    hello_ticket(rt)
    [o1] = rt.build()
    assert o1.status == "shipped"
    hello_ticket(rt, title="Second feature")
    [o2] = rt.build()
    c_step = next(s for s in o2.steps if s["step"] == "c) staging migrations")
    assert "lint ok (1 new migration file(s)" in c_step["detail"]     # 002 only; 001 already shipped


def test_supabase_ventures_need_staging_app_settings(cfg_path, clock):
    env = {k: v for k, v in SECRETS.items() if "STAGING_ANON" not in k}
    rt = make_rt(cfg_path, clock, claude=FakeClaude(None, migrations=True), environ=env)
    tid = hello_ticket(rt)
    [o] = rt.build(dry_run=True)
    assert o.status == "blocked_config" and "SUPABASE_STAGING_ANON_KEY" in o.reason
    assert "vercel deploy" not in rt.calls.names() and rt.tickets.get(tid).attempts == 0


def test_ventures_without_migrations_skip_supabase_wiring(cfg_path, clock):
    rt = make_rt(cfg_path, clock)
    hello_ticket(rt)
    [o] = rt.build()
    assert o.status == "shipped"
    assert not any(n.startswith(("vercel env", "supabase")) for n in rt.calls.names())


def test_prompt_states_additive_migration_rule(cfg_path, clock):
    rt = make_rt(cfg_path, clock)
    hello_ticket(rt)
    rt.build(dry_run=True)
    p = rt.fakes["claude"].prompts[0]
    assert "backward-compatible (additive only)" in p and "no RENAME" in p and "no DROP" in p


# ------------------------------------------------------------------ prod deploy
def test_failed_prod_smoke_rolls_back_before_health_watch(cfg_path, clock):
    rt = make_rt(cfg_path, clock, prod_smoke_ok=False)
    tid = hello_ticket(rt)
    [o] = rt.build()
    assert o.status == "failed" and steps(o)[-1] == ("h) prod smoke test", "fail")
    assert "prod smoke test failed" in o.reason and "rollback succeeded" in o.reason
    names = rt.calls.names()
    assert names.index("vercel rollback") > names.index("vercel deploy --prod") and "health" not in names
    [note] = list((rt.memory.root / "retros").glob("*incident*.md"))
    meta, body = frontmatter.parse(note.read_text())
    assert meta["rolled_back"] is True and "smoke test" in meta["title"] and PROD in body
    t = rt.tickets.get(tid)
    assert t.attempts == 1 and t.built_commit is None                 # code suspect: new session next time


def test_prod_deploy_refuses_a_commit_other_than_the_previews(cfg_path, clock):
    rt = make_rt(cfg_path, clock)
    hello_ticket(rt)
    repo_dir = rt.cfg.ventures_dir / "hello"
    orig = rt.fakes["supabase"].push
    def push(which):                                                  # something changes the repo after the preview
        (repo_dir / "late.txt").write_text("x")
        return orig(which)
    rt.fakes["supabase"].push = push
    rt.fakes["claude"].migrations = True
    [o] = rt.build()
    assert o.status == "failed" and steps(o)[-1] == ("g) prod deploy", "fail")
    assert "refusing to deploy to prod" in o.reason and "vercel deploy --prod" not in rt.calls.names()


def test_vercel_prod_deploy_uses_deploy_prod_not_promote(tmp_path):
    from startup_os.ship.providers import Vercel
    from startup_os.ship.secrets import VentureSecrets
    seen = {}
    def shell(cmd, *, cwd, env, timeout_s, stdout_path=None):
        seen.update(cmd=cmd, env=env)
        return CmdResult(0, "Production: https://hello-prod1.vercel.app [3s]\n", "", 0.1)
    v = Vercel("vercel@64.7.0", VentureSecrets.from_env("hello", SECRETS), tmp_path, 60, shell=shell)
    assert v.deploy_prod() == "https://hello-prod1.vercel.app"
    assert seen["cmd"][-3:] == ["deploy", "--prod", "--yes"] and "promote" not in seen["cmd"]
    assert seen["env"]["VERCEL_TOKEN"] == "vercel-secret-token-123"
    assert not hasattr(v, "promote")


# ------------------------------------------------------------------ auth/config errors are not strikes
class ConfigFailVercel(FakeVercel):
    def __init__(self, calls, on):
        super().__init__(calls)
        self.on = on

    def _maybe(self, name):
        from startup_os.ship.providers import ConfigError
        self.calls.append((f"vercel {name}", None))
        if name == self.on:
            raise ConfigError(f"vercel {name}: auth/config error (Project not found), not a code failure")


@pytest.mark.parametrize("on,step", [("deploy", "d) preview deploy"), ("deploy --prod", "g) prod deploy")])
def test_vercel_config_errors_block_without_a_strike(cfg_path, clock, on, step):
    rt = make_rt(cfg_path, clock, vercel=ConfigFailVercel(None, on))
    tid = hello_ticket(rt)
    [o] = rt.build()
    assert o.status == "blocked_config" and "Project not found" in o.reason
    assert steps(o)[-1] == (step, "fail")
    t = rt.tickets.get(tid)
    assert t.attempts == 0 and t.status == "todo" and t.built_commit     # no strike; commit kept for reuse
    assert "health" not in rt.calls.names()
    # the fixed config reruns without a new Claude Code session
    rt.fakes["vercel"].on = None
    [o2] = rt.build()
    assert o2.status == "shipped" and steps(o2)[0] == ("Claude Code", "skip")


@pytest.mark.parametrize("out,config", [
    ('Error: Project not found ({"VERCEL_PROJECT_ID":"prj_x","VERCEL_ORG_ID":"y"})', True),
    ("Error: The specified token is not valid. Use `vercel login` to generate a new token.", True),
    ("Error: The token provided via VERCEL_TOKEN environment variable is not valid. Please provide a valid token.", True),
    ("Error: Your token has expired", True),
    ("Error: Request failed with status 403 Forbidden", True),
    ("Error: 401 Unauthorized", True),
    ("Error: You do not have access to the specified account", True),
    ("Error: You specified VERCEL_ORG_ID but you forgot to specify VERCEL_PROJECT_ID", True),
    ("Error: Command \"npm run build\" exited with 1", False),               # a code failure: still a strike
    ("Error: this feature requires a Pro plan (402)", "paid"),
])
def test_vercel_cli_error_classification(tmp_path, out, config):
    from startup_os.ship.providers import ConfigError, Vercel
    from startup_os.ship.secrets import VentureSecrets
    v = Vercel("vercel@64.7.0", VentureSecrets.from_env("hello", SECRETS), tmp_path, 60,
               shell=lambda cmd, **kw: CmdResult(1, "", out, 0.1))
    for call in (v.deploy_preview, v.deploy_prod):
        exc = PaidPlanRequired if config == "paid" else (ConfigError if config else StepFailed)
        with pytest.raises(exc) as e:
            call()
        assert type(e.value) is exc
        if config is True:
            assert "VERCEL_ORG_ID" in str(e.value) and "vercel-secret-token-123" not in str(e.value)


@pytest.mark.parametrize("code", [401, 403, 404])
def test_vercel_env_api_auth_errors_are_config_errors(tmp_path, monkeypatch, code):
    import io
    import urllib.error
    from startup_os.ship.providers import ConfigError, Vercel
    from startup_os.ship.secrets import VentureSecrets
    def urlopen(req, timeout):
        raise urllib.error.HTTPError(req.full_url, code, "x", {}, io.BytesIO(b'{"error":{"code":"forbidden"}}'))
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    v = Vercel("vercel@64.7.0", VentureSecrets.from_env("hello", SECRETS), tmp_path, 60)
    with pytest.raises(ConfigError, match=f"HTTP {code}"):
        v.sync_env("preview", {"NEXT_PUBLIC_SUPABASE_URL": "https://s.supabase.co"})


@pytest.mark.parametrize("out,config", [
    ('FATAL: password authentication failed for user "postgres"', True),
    ("Error: Tenant or user not found", True),
    ('ERROR: syntax error at or near "tabel"', False),
])
def test_supabase_auth_errors_are_config_errors(tmp_path, out, config):
    from startup_os.ship.providers import ConfigError, Supabase
    from startup_os.ship.secrets import VentureSecrets
    s = Supabase("supabase@2.121.0", VentureSecrets.from_env("hello", SECRETS), tmp_path, 60,
                 shell=lambda cmd, **kw: CmdResult(1, "", out, 0.1))
    with pytest.raises(ConfigError if config else StepFailed) as e:
        s.push("staging")
    assert (type(e.value) is ConfigError) == config and "pw-staging" not in str(e.value)


# ------------------------------------------------------------------ venture project creation
def _api(monkeypatch, project, user_id="user_me"):
    import io
    seen = {"urls": []}
    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def urlopen(req, timeout):
        seen.update(url=req.full_url, auth=req.headers.get("Authorization"))
        seen["urls"].append(req.full_url)
        body = {"user": {"id": user_id}} if "/v2/user" in req.full_url else project
        return Resp(json.dumps(body).encode())
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return seen


def test_create_project_has_no_git_link_and_keeps_token_off_argv(tmp_path, monkeypatch):
    from startup_os.ship.providers import Vercel
    from startup_os.ship.secrets import VentureSecrets
    api = _api(monkeypatch, {"id": "prj_new123", "name": "startup-os-hello"})
    seen = {}
    def shell(cmd, *, cwd, env, timeout_s, stdout_path=None):
        seen.update(cmd=cmd, env=env)
        return CmdResult(0, "Success! Project startup-os-hello added", "", 0.1)
    v = Vercel("vercel@64.7.0", VentureSecrets.from_env("hello", SECRETS), tmp_path, 60, shell=shell)
    assert v.create_project("startup-os-hello") == {"id": "prj_new123", "name": "startup-os-hello", "git": None}
    assert seen["cmd"][-5:] == ["project", "add", "startup-os-hello", "--scope", "team_abc"]
    assert "vercel-secret-token-123" not in " ".join(seen["cmd"]) and "git" not in seen["cmd"]
    assert seen["env"]["VERCEL_TOKEN"] == "vercel-secret-token-123"
    assert "VERCEL_PROJECT_ID" not in seen["env"] and "VERCEL_ORG_ID" not in seen["env"]
    assert api["url"].endswith("/v9/projects/startup-os-hello?teamId=team_abc")


def test_create_project_refuses_a_git_connected_project(tmp_path, monkeypatch):
    from startup_os.ship.providers import Vercel
    from startup_os.ship.secrets import VentureSecrets
    _api(monkeypatch, {"id": "prj_new123", "name": "startup-os-hello",
                       "link": {"type": "github", "org": "nicolasthrax", "repo": "startup-os"}})
    v = Vercel("vercel@64.7.0", VentureSecrets.from_env("hello", SECRETS), tmp_path, 60,
               shell=lambda cmd, **kw: CmdResult(0, "added", "", 0.1))
    with pytest.raises(StepFailed, match="connected to Git"):
        v.create_project("startup-os-hello")


def test_vercel_project_command_prints_id_and_stores_nothing(cfg_path, monkeypatch, capsys):
    from startup_os import cli
    from startup_os.ship import providers
    monkeypatch.setattr(providers.Vercel, "create_project",
                        lambda self, name: {"id": "prj_new123", "name": name, "git": None})
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    assert cli.main(["--config", str(cfg_path), "vercel-project", "--venture", "hello",
                     "--name", "startup-os-hello"]) == 0
    out = capsys.readouterr().out
    assert "project id: prj_new123" in out and "VENTURE_HELLO_VERCEL_PROJECT_ID is set to a different project" in out
    assert not any(b"prj_new123" in f.read_bytes() for f in cfg_path.parent.rglob("*") if f.is_file())


@pytest.mark.parametrize("uid,ok", [("8l7yPersonal", True), ("someone_else", False)])
def test_create_project_in_a_personal_account(tmp_path, monkeypatch, uid, ok):
    from startup_os.ship.providers import ConfigError, Vercel
    from startup_os.ship.secrets import VentureSecrets
    api = _api(monkeypatch, {"id": "prj_new123", "name": "startup-os-hello", "accountId": "8l7yPersonal"}, uid)
    calls = []
    v = Vercel("vercel@64.7.0", VentureSecrets.from_env("hello", {**SECRETS, "VENTURE_HELLO_VERCEL_ORG_ID": "8l7yPersonal"}),
               tmp_path, 60, shell=lambda cmd, **kw: (calls.append(cmd) or CmdResult(0, "added", "", 0.1)))
    if ok:
        assert v.create_project("startup-os-hello")["id"] == "prj_new123"
        assert "--scope" not in calls[0]                               # --scope takes teams only
        assert "teamId" not in api["url"]
    else:
        with pytest.raises(ConfigError, match="not the token's own user"):
            v.create_project("startup-os-hello")
        assert calls == []                                              # nothing created
