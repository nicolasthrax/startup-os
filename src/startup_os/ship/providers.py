"""Vercel and Supabase via their pinned CLIs, with project-scoped credentials passed
only as env vars to these subprocesses. Paid-plan errors raise PaidPlanRequired."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .secrets import VentureSecrets, clean_env
from .shell import CmdResult, Shell, run

PAID_PATTERNS = re.compile(
    r"\b402\b|payment required|upgrade (?:your|to (?:a |the )?(?:pro|team|paid))|pro plan|team plan|paid plan|"
    r"requires? (?:a )?(?:pro|team|paid|enterprise)|billing|add a payment method|spend (?:cap|limit)|"
    r"exceeded .*(?:free|hobby) (?:tier|plan)|hobby plan (?:limit|does not)", re.IGNORECASE)
URL_RE = re.compile(r"https://[a-z0-9.-]+\.vercel\.app\S*", re.IGNORECASE)


class PaidPlanRequired(Exception):
    def __init__(self, item: str, detail: str, amount_usd: float = 20.0):
        super().__init__(f"{item} needs a paid plan: {detail[:300]}")
        self.item, self.detail, self.amount_usd = item, detail, amount_usd


class StepFailed(Exception):
    pass


def check_paid(res: CmdResult, item: str, amount_usd: float) -> None:
    if not res.ok and PAID_PATTERNS.search(res.tail(4000)):
        raise PaidPlanRequired(item, res.tail(600), amount_usd)


@dataclass
class Vercel:
    cli: str
    secrets: VentureSecrets
    cwd: Path
    timeout_s: float
    shell: Shell = run

    def _env(self) -> dict[str, str]:
        s = self.secrets
        return clean_env(VERCEL_TOKEN=s.get("VERCEL_TOKEN"), VERCEL_ORG_ID=s.get("VERCEL_ORG_ID"),
                         VERCEL_PROJECT_ID=s.get("VERCEL_PROJECT_ID"), VERCEL_TELEMETRY_DISABLED="1")

    def _run(self, *args: str) -> CmdResult:
        # Token goes in the env (VERCEL_TOKEN), never on the command line.
        return self.shell(["npx", "--yes", self.cli, *args], cwd=self.cwd, env=self._env(), timeout_s=self.timeout_s)

    def deploy_preview(self) -> str:
        res = self._run("deploy", "--yes")
        check_paid(res, "Vercel Pro plan", 20)
        if not res.ok:
            raise StepFailed(f"vercel deploy failed: {res.tail()}")
        urls = URL_RE.findall(res.stdout) or URL_RE.findall(res.stderr)
        if not urls:
            raise StepFailed(f"vercel deploy printed no URL: {res.tail()}")
        return urls[-1].rstrip(".,)")

    def promote(self, url: str) -> str:
        res = self._run("promote", url, "--yes")
        check_paid(res, "Vercel Pro plan", 20)
        if not res.ok:
            raise StepFailed(f"vercel promote failed: {res.tail()}")
        return self.secrets.get("PROD_URL")

    def rollback(self) -> CmdResult:
        return self._run("rollback", "--yes")

    def sync_env(self, target: str, values: dict[str, str]) -> None:
        """Upsert project env vars for one target ("preview" or "production") via the Vercel API."""
        s = self.secrets
        q = {"upsert": "true"}
        if s.get("VERCEL_ORG_ID").startswith("team_"):
            q["teamId"] = s.get("VERCEL_ORG_ID")
        url = f"https://api.vercel.com/v10/projects/{s.get('VERCEL_PROJECT_ID')}/env?{urllib.parse.urlencode(q)}"
        for key, value in values.items():
            body = json.dumps({"key": key, "value": value, "type": "encrypted", "target": [target]}).encode()
            req = urllib.request.Request(url, data=body, method="POST", headers={
                "Authorization": f"Bearer {s.get('VERCEL_TOKEN')}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=30):
                    pass
            except urllib.error.HTTPError as e:
                detail = e.read(500).decode("utf-8", "replace")
                if e.code == 402 or PAID_PATTERNS.search(detail):
                    raise PaidPlanRequired("Vercel Pro plan", detail, 20) from e
                raise StepFailed(f"setting {key} for {target} failed: HTTP {e.code} {detail[:200]}") from e


@dataclass
class Supabase:
    cli: str
    secrets: VentureSecrets
    cwd: Path
    timeout_s: float
    shell: Shell = run

    def push(self, which: str) -> str:
        """Apply supabase/migrations to the `staging` or `prod` project."""
        url = self.secrets.get(f"SUPABASE_{which.upper()}_DB_URL")
        env = clean_env(SUPABASE_DB_URL=url)
        # The URL reaches the CLI through the environment, expanded by the shell, not via our argv.
        res = self.shell(["sh", "-c", f'npx --yes {self.cli} db push --db-url "$SUPABASE_DB_URL" --include-all'],
                         cwd=self.cwd, env=env, timeout_s=self.timeout_s)
        check_paid(res, f"Supabase paid plan ({which})", 25)
        if not res.ok:
            raise StepFailed(f"supabase db push ({which}) failed: {res.tail()}")
        return res.tail(300)
