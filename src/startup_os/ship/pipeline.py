"""`startup-os build`: run claude-code tickets through headless Claude Code and the ship bar.

Ship bar (run by Startup OS, in order, stopping at the first failure):
  a) build  b) tests  c) preview deploy  d) smoke test
  e1) staging migrations  e2) prod migrations  f) promote  g) prod health (rollback on failure)
--dry-run stops after d): no migrations and no prod."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from .. import frontmatter
from ..approvals import ApprovalQueue
from ..config import Config
from ..memory import MemoryRepo
from ..runlog import RunLog
from ..search_index import SearchIndex
from ..spend import SpendGate
from ..tickets import Ticket, TicketStore
from .claude import ClaudeResult, ClaudeRunner
from .prompt import build_prompt
from .providers import PaidPlanRequired, StepFailed, Supabase, Vercel
from .secrets import MissingCredential, VentureSecrets, clean_env, secret_values_in_env
from .shell import Shell, run
from .smoke import Check, SmokeResult, derive_checks, health, run_smoke
from .store import ShipStore
from .ventures import VentureRepo

AGENT = "build"
STEPS = [("build", "a) build"), ("test", "b) tests"), ("preview", "c) preview deploy"), ("smoke", "d) smoke test"),
         ("migrate_staging", "e) staging migrations"), ("migrate_prod", "e) prod migrations"),
         ("promote", "f) promote to prod"), ("prod_health", "g) prod health watch")]
DRY_RUN_STEPS = {"build", "test", "preview", "smoke"}
CODE_STEPS = {"claude_code", "build", "test", "smoke", "prod_health"}   # failures here need new code


class NotRunnable(ValueError):
    pass


class ShipFailure(Exception):
    def __init__(self, step: str, reason: str):
        super().__init__(f"{step}: {reason}")
        self.step, self.reason = step, reason


@dataclass
class ShipDeps:
    cfg: Config
    memory: MemoryRepo
    index: SearchIndex
    tickets: TicketStore
    approvals: ApprovalQueue
    spend_gate: SpendGate
    store: ShipStore
    now: Callable[[], datetime]
    claude: Callable[..., ClaudeResult] | None = None
    shell: Shell = run
    vercel: Callable[[VentureSecrets, Path], Vercel] | None = None
    supabase: Callable[[VentureSecrets, Path], Supabase] | None = None
    smoke: Callable[[str, list[Check], str | None], SmokeResult] = run_smoke
    health: Callable[[str, str | None], tuple[bool, str]] = health
    sleep: Callable[[float], None] = time.sleep
    environ: dict[str, str] | None = None

    def __post_init__(self):
        c = self.cfg
        self.claude = self.claude or ClaudeRunner(c.claude_code)
        step_s = c.ship.step_timeout_min * 60
        self.vercel = self.vercel or (lambda s, cwd: Vercel(c.ship.vercel_cli, s, cwd, step_s))
        self.supabase = self.supabase or (lambda s, cwd: Supabase(c.ship.supabase_cli, s, cwd, step_s))


@dataclass
class Outcome:
    ticket: str
    status: str            # shipped | dry_run_ok | failed | needs_rescope | blocked_spend | blocked_config | capped
    run_id: int | None = None
    reason: str | None = None
    preview_url: str | None = None
    prod_url: str | None = None
    log_path: str | None = None
    transcript: str | None = None
    steps: list[dict] = field(default_factory=list)


class Builder:
    def __init__(self, deps: ShipDeps):
        self.d = deps

    # ------------------------------------------------------------ selection --
    def runnable(self, ticket_id: str | None = None) -> list[Ticket]:
        """claude-code tickets that are todo, with every dependency done and no spend request pending."""
        d = self.d
        d.tickets.refresh(d.approvals.status)
        if ticket_id:
            t = d.tickets.get(ticket_id)
            if t.owner != "claude-code":
                raise NotRunnable(f"{t.id} is owned by {t.owner}, not claude-code")
            reasons = d.tickets.blocking(t, d.approvals.status)
            if t.status != "todo" or reasons:
                raise NotRunnable(f"{t.id} is not runnable: {t.status}" + (f" ({'; '.join(reasons)})" if reasons else ""))
            return [t]
        return [t for t in d.tickets.list(owner="claude-code", status="todo")
                if not d.tickets.blocking(t, d.approvals.status)]

    def build(self, ticket_id: str | None = None, dry_run: bool = False) -> list[Outcome]:
        out = []
        for t in self.runnable(ticket_id):
            o = self.build_ticket(t, dry_run=dry_run)
            out.append(o)
            if o.status in ("capped", "blocked_config"):
                break
        return out

    # --------------------------------------------------------------- ticket --
    def build_ticket(self, t: Ticket, *, dry_run: bool) -> Outcome:
        d, cfg = self.d, self.d.cfg
        if not t.venture:
            return Outcome(t.id, "blocked_config", reason=f"{t.id} has no venture")
        repo = VentureRepo(cfg.ventures_dir, t.venture)
        reuse = bool(t.built_commit and (repo.path / ".git").exists() and repo.head() == t.built_commit
                     and not repo.dirty())
        if not reuse and d.store.sessions_today() >= cfg.claude_code.sessions_per_day:
            return Outcome(t.id, "capped", reason=f"Claude Code session cap reached "
                           f"({cfg.claude_code.sessions_per_day}/day); try again tomorrow")
        repo.ensure()
        environ = d.environ if d.environ is not None else __import__("os").environ
        secrets = VentureSecrets.from_env(repo.slug, environ)
        log = RunLog(cfg.logs_dir, "build", ticket=t.id, venture=t.venture, slug=repo.slug, dry_run=dry_run,
                     repo=str(repo.path))
        run_id = d.store.start_run(t.id, t.venture, dry_run, str(log.path))
        o = Outcome(t.id, "failed", run_id=run_id, log_path=str(log.path))
        idx = [0]

        def record(step: str, label: str, status: str, dur: float, detail: str = "", url: str | None = None):
            idx[0] += 1
            d.store.add_step(run_id, idx[0], label, status, dur, detail, url)
            row = {"step": label, "status": status, "duration_s": round(dur, 1), "detail": detail[:300], "url": url}
            o.steps.append(row)
            log.event("ship_step", **row)

        state: dict = {"preview_url": None, "prod_url": None}
        try:
            # 0. Claude Code
            if reuse:
                record("claude_code", "Claude Code", "skip", 0, f"reusing built commit {t.built_commit[:10]}")
            else:
                t = self._claude(t, repo, secrets, environ, log, o, record)
            sha = repo.head()
            d.store.update_run(run_id, commit_sha=sha, transcript=o.transcript)
            # paid requirements declared by the code → spend gate
            manifest = repo.manifest()
            for req in manifest.get("paid_requirements") or []:
                raise PaidPlanRequired(req.get("item", "paid requirement"), req.get("why", "declared in ship.json"),
                                       float(req.get("amount_usd") or 1))
            # a–g
            for name, label in STEPS:
                if dry_run and name not in DRY_RUN_STEPS:
                    record(name, label, "skip", 0, "dry run")
                    continue
                t0 = time.monotonic()
                try:
                    detail, url = getattr(self, f"_step_{name}")(t, repo, secrets, state, log)
                except (StepFailed, ShipFailure) as e:
                    record(name, label, "fail", time.monotonic() - t0, str(e))
                    raise ShipFailure(label, str(e)) from e
                except (PaidPlanRequired, MissingCredential) as e:
                    record(name, label, "fail", time.monotonic() - t0, str(e))
                    raise
                record(name, label, "pass", time.monotonic() - t0, detail, url)
        except PaidPlanRequired as e:
            self._spend_stop(t, e, log)
            o.status, o.reason = self._last_status, self._last_reason
            return self._finish(t, o, log, None, state)
        except MissingCredential as e:
            o.status, o.reason = "blocked_config", str(e)
            return self._finish(t, o, log, t, state)
        except ShipFailure as e:
            self._strike(t, e, log)
            o.status, o.reason = self._last_status, str(e)
            return self._finish(t, o, log, None, state)
        # success
        if dry_run:
            o.status = "dry_run_ok"
            t = t.model_copy(update={"preview_url": state["preview_url"], "last_error": None})
            msg = f"{t.id} dry run passed (preview {state['preview_url']})"
        else:
            o.status = "shipped"
            t = t.model_copy(update={"status": "done", "preview_url": state["preview_url"], "prod_url": state["prod_url"],
                                     "last_error": None, "result": f"Shipped to {state['prod_url']} "
                                     f"(commit {repo.head()[:10]}, preview {state['preview_url']})"})
            msg = f"{t.id} shipped to {state['prod_url']}"
        d.tickets.save(t, agent=AGENT, message=msg)
        return self._finish(t, o, log, None, state)

    def _finish(self, t: Ticket, o: Outcome, log: RunLog, saved, state) -> Outcome:
        o.preview_url, o.prod_url = state.get("preview_url"), state.get("prod_url")
        self.d.store.update_run(o.run_id, status=o.status, finished_at=self.d.now().isoformat(),
                                preview_url=o.preview_url, prod_url=o.prod_url, error=o.reason)
        log.close(o.status, ticket=t.id, reason=o.reason, preview_url=o.preview_url, prod_url=o.prod_url,
                  transcript=o.transcript, steps=o.steps)
        if o.status in ("failed", "needs_rescope") or o.status.startswith("blocked"):
            o.reason = o.reason or "see run log"
        return o

    # ------------------------------------------------------------ claude --
    def _claude(self, t, repo, secrets, environ, log, o, record) -> Ticket:
        d, cfg = self.d, self.d.cfg
        prompt, files = build_prompt(t, memory=d.memory, index=d.index, slug=repo.slug,
                                     secrets=secrets.secret_values() + secret_values_in_env(environ),
                                     previous_error=t.last_error if t.attempts else None)
        stamp = d.now().strftime("%Y%m%dT%H%M%SZ")
        transcript = cfg.logs_dir / "transcripts" / repo.slug / f"{t.id}-{stamp}.jsonl"
        log.event("claude_prompt", files=files, prompt=prompt, transcript=str(transcript))
        sid = d.store.start_session(t.id, t.venture)
        res = d.claude(prompt, cwd=repo.path, transcript=transcript,
                       timeout_s=cfg.claude_code.ticket_timeout_min * 60)
        d.store.end_session(sid, "timeout" if res.timed_out else ("ok" if res.ok else "failed"),
                            res.duration_s, res.transcript)
        o.transcript = res.transcript
        log.event("claude_session", ok=res.ok, timed_out=res.timed_out, exit_code=res.exit_code,
                  duration_s=round(res.duration_s, 1), transcript=res.transcript, cost_usd=res.cost_usd,
                  summary=res.summary[:1000])
        record("claude_code", "Claude Code", "pass" if res.ok else "fail", res.duration_s,
               f"transcript {res.transcript}" + ("" if res.ok else f" — {res.summary[:300]}"), None)
        if not res.ok:
            raise ShipFailure("Claude Code", ("timed out" if res.timed_out else "session failed") + f": {res.summary[:600]}")
        sha = repo.commit_all(f"{t.id}: {t.title}\n\nImplemented by headless Claude Code.\nTranscript: {res.transcript}")
        if not sha:
            raise ShipFailure("Claude Code", "made no changes")
        t = t.model_copy(update={"built_commit": sha})
        d.tickets.save(t, agent=AGENT, message=f"{t.id}: Claude Code session committed {sha[:10]} in ventures/{repo.slug}")
        return t

    # --------------------------------------------------------------- steps --
    def _npm(self, repo, cmd: list[str], what: str) -> str:
        env = clean_env(CI="1", NEXT_TELEMETRY_DISABLED="1")
        res = self.d.shell(cmd, cwd=repo.path, env=env, timeout_s=self.d.cfg.ship.step_timeout_min * 60)
        if not res.ok:
            raise StepFailed(f"{what} failed ({'timeout' if res.timed_out else f'exit {res.code}'}): {res.tail(1200)}")
        return res.tail(300)

    def _step_build(self, t, repo, secrets, state, log):
        if not repo.package_json():
            raise StepFailed("no package.json in the venture repo")
        self._npm(repo, self.d.cfg.ship.install_cmd, "install")
        return self._npm(repo, self.d.cfg.ship.build_cmd, "build"), None

    def _step_test(self, t, repo, secrets, state, log):
        script = (repo.package_json().get("scripts") or {}).get("test", "")
        if not script or "no test specified" in script:
            raise StepFailed("no test suite: package.json has no real `test` script")
        return self._npm(repo, self.d.cfg.ship.test_cmd, "tests"), None

    def _step_preview(self, t, repo, secrets, state, log):
        url = self.d.vercel(secrets, repo.path).deploy_preview()
        state["preview_url"] = url
        return "preview deployed", url

    def _checks(self, t) -> list[Check]:
        return derive_checks(t, self.d.cfg.ship.max_smoke_checks)

    def _step_smoke(self, t, repo, secrets, state, log):
        checks = self._checks(t)
        res = self.d.smoke(state["preview_url"], checks, secrets.values.get("VERCEL_BYPASS_SECRET"))
        log.event("smoke", url=state["preview_url"], results=res.results)
        if not res.ok:
            raise StepFailed(f"smoke test failed: {res.summary()}")
        return res.summary(), state["preview_url"]

    def _migrate(self, repo, secrets, which: str):
        if not repo.migrations():
            return "no migrations", None
        out = self.d.supabase(secrets, repo.path).push(which)
        return f"{len(repo.migrations())} migration file(s) applied to {which}: {out[-150:]}", None

    def _step_migrate_staging(self, t, repo, secrets, state, log):
        return self._migrate(repo, secrets, "staging")

    def _step_migrate_prod(self, t, repo, secrets, state, log):
        return self._migrate(repo, secrets, "prod")

    def _step_promote(self, t, repo, secrets, state, log):
        state["prod_url"] = self.d.vercel(secrets, repo.path).promote(state["preview_url"])
        return f"promoted {state['preview_url']}", state["prod_url"]

    def _step_prod_health(self, t, repo, secrets, state, log):
        c = self.d.cfg.ship
        url = state["prod_url"]
        checks = max(1, int(c.health_minutes * 60 / c.health_interval_s))
        history, consecutive = [], 0
        for i in range(checks):
            ok, detail = self.d.health(url, None)
            history.append({"at": self.d.now().isoformat(timespec="seconds"), "ok": ok, "detail": detail})
            log.event("prod_health", n=i + 1, ok=ok, detail=detail)
            consecutive = 0 if ok else consecutive + 1
            if consecutive >= c.health_failures_before_rollback:
                rb = self.d.vercel(secrets, repo.path).rollback()
                log.event("rollback", ok=rb.ok, output=rb.tail(500))
                note = self._incident(t, repo, url, history, rb)
                raise StepFailed(f"prod health failed ({detail}); Vercel rollback "
                                 f"{'succeeded' if rb.ok else 'FAILED'}; incident note {note}")
            if i < checks - 1:
                self.d.sleep(c.health_interval_s)
        return f"{checks} health checks over {c.health_minutes:g} min all passed", url

    def _incident(self, t, repo, url, history, rb) -> str:
        now = self.d.now()
        rel = self.d.memory.unique_path(f"retros/{now.date().isoformat()}-incident-{t.id.lower()}.md")
        meta = {"title": f"Incident: {t.venture} prod health failed after {t.id}", "type": "incident",
                "date": now.isoformat(timespec="seconds"), "ticket": t.id, "venture": t.venture,
                "prod_url": url, "commit": repo.head(), "rolled_back": rb.ok}
        body = [f"# Incident: {t.venture} — prod health check failed", "",
                f"After promoting {t.id} ({t.title}), `GET {url}/api/health` failed. "
                f"Startup OS {'rolled back' if rb.ok else 'tried to roll back'} the Vercel production deployment.", "",
                "## Timeline (UTC)", "", "| Time | Health | Detail |", "|---|---|---|"]
        body += [f"| {h['at']} | {'ok' if h['ok'] else 'FAIL'} | {h['detail']} |" for h in history]
        body += ["", "## Rollback", "", f"Result: {'succeeded' if rb.ok else 'FAILED — manual action needed'}", "",
                 "```", rb.tail(800) or "(no output)", "```", "",
                 "## Follow-up", "", f"- {t.id} counts as a failed attempt; after {self.d.cfg.ship.max_attempts} "
                 "failures it goes back to the CEO (needs_rescope).",
                 "- Note: prod database migrations, if any, were applied before promotion and are NOT rolled back."]
        self.d.memory.write(rel, frontmatter.dump(meta, "\n".join(body) + "\n"), agent=AGENT,
                            message=f"incident: {t.venture} prod health failed after {t.id}; rollback "
                                    f"{'ok' if rb.ok else 'FAILED'}")
        return rel

    # ------------------------------------------------------------ outcomes --
    def _strike(self, t: Ticket, e: ShipFailure, log: RunLog) -> Ticket:
        d = self.d
        attempts = t.attempts + 1
        error = f"Attempt {attempts} failed at {e.step}: {e.reason}"[:1500]
        rescope = attempts >= d.cfg.ship.max_attempts
        upd = {"attempts": attempts, "last_error": error, "status": "needs_rescope" if rescope else "todo"}
        if e.step in {"Claude Code", "a) build", "b) tests", "d) smoke test", "g) prod health watch"}:
            upd["built_commit"] = None   # the code itself is suspect: next attempt starts a new session
        t = t.model_copy(update=upd)
        d.tickets.save(t, agent=AGENT, message=f"{t.id} attempt {attempts} failed at {e.step}"
                                               + ("; back to CEO (needs_rescope)" if rescope else ""))
        log.event("strike", attempts=attempts, rescope=rescope, error=error)
        self._last_status = "needs_rescope" if rescope else "failed"
        return t

    def _spend_stop(self, t: Ticket, e: PaidPlanRequired, log: RunLog) -> Ticket:
        a = self.d.spend_gate.request({"item": e.item, "amount_usd": e.amount_usd, "vendor": e.item.split()[0],
                                       "justification": f"{t.id} ({t.title}) cannot ship without it: {e.detail[:400]}",
                                       "blocks": [t.id]}, agent=AGENT)
        t = t.model_copy(update={"depends_on": t.depends_on + [a["ref"]], "status": "blocked",
                                 "blocked_reason": f"{a['ref']} is pending (spend request: {e.item})"})
        self.d.tickets.save(t, agent=AGENT, message=f"{t.id} blocked on spend request {a['ref']} ({e.item})")
        log.event("spend_stop", approval=a["ref"], item=e.item, amount_usd=e.amount_usd)
        self._last_status = "blocked_spend"
        self._last_reason = f"needs {e.item}: filed spend request {a['ref']} (never auto-approves)"
        return t
