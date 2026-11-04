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

from .secrets import MissingCredential, VentureSecrets, clean_env
from .shell import CmdResult, Shell, run

PAID_PATTERNS = re.compile(
    r"\b402\b|payment required|upgrade (?:your|to (?:a |the )?(?:pro|team|paid))|pro plan|team plan|paid plan|"
    r"requires? (?:a )?(?:pro|team|paid|enterprise)|billing|add a payment method|spend (?:cap|limit)|"
    r"exceeded .*(?:free|hobby) (?:tier|plan)|hobby plan (?:limit|does not)", re.IGNORECASE)
# Auth/config errors: the credentials or project settings are wrong, not the code. These stop the run
# as blocked_config without a strike (like a missing credential).
VERCEL_CONFIG_PATTERNS = re.compile(
    r"project not found|\b40[13]\b|forbidden|not authori[sz]ed|unauthori[sz]ed|"
    r"(?:invalid|expired|missing) (?:token|credentials)|token (?:is )?(?:not valid|invalid|expired)|"
    r"you (?:do not|don't) have (?:permission|access)|no access to|team not found|scope not found|"
    r"could not retrieve project settings|specified VERCEL_(?:ORG|PROJECT)_ID|forgot to specify VERCEL_",
    re.IGNORECASE)
SUPABASE_CONFIG_PATTERNS = re.compile(
    r"password authentication failed|invalid api key|\b40[13]\b|unauthori[sz]ed|forbidden|"
    r"no pg_hba\.conf entry|tenant or user not found|role .* does not exist|database .* does not exist|"
    r"invalid (?:connection string|database url)", re.IGNORECASE)


class ConfigError(MissingCredential):
    """A provider rejected our credentials or project settings: blocked_config, not a strike."""


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


def check_config(res: CmdResult, what: str, patterns: re.Pattern, hint: str) -> None:
    if not res.ok:
        m = patterns.search(res.tail(4000))
        if m:
            raise ConfigError(f"{what}: auth/config error ({m.group(0)}), not a code failure: {hint}. "
                              f"Output: {res.tail(400)}")


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

    HINT = ("check VENTURE_<SLUG>_VERCEL_TOKEN, _VERCEL_ORG_ID (the team id, team_…) and "
            "_VERCEL_PROJECT_ID (the venture's own project)")

    def _check(self, res: CmdResult, what: str) -> None:
        check_paid(res, "Vercel Pro plan", 20)
        check_config(res, what, VERCEL_CONFIG_PATTERNS, self.HINT)

    def deploy_preview(self) -> str:
        res = self._run("deploy", "--yes")
        self._check(res, "vercel deploy")
        if not res.ok:
            raise StepFailed(f"vercel deploy failed: {res.tail()}")
        urls = URL_RE.findall(res.stdout) or URL_RE.findall(res.stderr)
        if not urls:
            raise StepFailed(f"vercel deploy printed no URL: {res.tail()}")
        return urls[-1].rstrip(".,)")

    def deploy_prod(self) -> str:
        """Fresh production build and deploy of the working tree (`vercel deploy --prod`); returns its URL."""
        res = self._run("deploy", "--prod", "--yes")
        self._check(res, "vercel deploy --prod")
        if not res.ok:
            raise StepFailed(f"vercel deploy --prod failed: {res.tail()}")
        urls = URL_RE.findall(res.stdout) or URL_RE.findall(res.stderr)
        if not urls:
            raise StepFailed(f"vercel deploy --prod printed no URL: {res.tail()}")
        return urls[-1].rstrip(".,)")

    def _api_get(self, path: str) -> dict:
        s = self.secrets
        q = f"?teamId={urllib.parse.quote(s.get('VERCEL_ORG_ID'))}" if s.get("VERCEL_ORG_ID").startswith("team_") else ""
        req = urllib.request.Request(f"https://api.vercel.com{path}{q}",
                                     headers={"Authorization": f"Bearer {s.get('VERCEL_TOKEN')}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read(500).decode("utf-8", "replace")
            raise ConfigError(f"Vercel API GET {path}: HTTP {e.code}: {self.HINT}. {detail[:200]}") from e

    def create_project(self, name: str) -> dict:
        """Create a project with the CLI, in the venture's team, with NO Git connection: ventures deploy
        only through our pipeline. Returns {"id", "name", "git"}. Nothing is stored."""
        s = self.secrets
        org = s.get("VERCEL_ORG_ID")
        # Only the token in the env: VERCEL_ORG_ID/VERCEL_PROJECT_ID would point the CLI at an existing project.
        env = clean_env(VERCEL_TOKEN=s.get("VERCEL_TOKEN"), VERCEL_TELEMETRY_DISABLED="1")
        if org.startswith("team_"):
            scope = ["--scope", org]
        else:   # a personal account: ORG_ID is the user id, which --scope does not take; verify it first
            uid = (self._api_get("/v2/user").get("user") or {}).get("id")
            if uid != org:
                raise ConfigError("VERCEL_ORG_ID is not a team id (team_…) and is not the token's own user "
                                  "account either; refusing to create the project in some other account")
            scope = []
        res = self.shell(["npx", "--yes", self.cli, "project", "add", name, *scope],
                         cwd=self.cwd, env=env, timeout_s=self.timeout_s)
        check_paid(res, "Vercel Pro plan", 20)
        check_config(res, f"vercel project add {name}", VERCEL_CONFIG_PATTERNS, self.HINT)
        if not res.ok:
            raise StepFailed(f"vercel project add {name} failed: {res.tail()}")
        p = self._api_get(f"/v9/projects/{urllib.parse.quote(name)}")
        git = p.get("link")
        if git:
            raise StepFailed(f"project {name} ({p.get('id')}) is connected to Git ({git.get('type')}: "
                             f"{git.get('org') or ''}/{git.get('repo') or ''}); ventures must deploy only "
                             "through Startup OS — disconnect it in Vercel")
        if p.get("accountId") and p["accountId"] != org:
            raise StepFailed(f"project {name} ({p['id']}) was created in account {p['accountId']}, not {org}")
        return {"id": p["id"], "name": p["name"], "git": None}

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
                if e.code in (401, 403, 404) or VERCEL_CONFIG_PATTERNS.search(detail):
                    raise ConfigError(f"setting {key} for {target}: Vercel API HTTP {e.code} (auth/config "
                                      f"error, not a code failure): {self.HINT}. {detail[:200]}") from e
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
        check_config(res, f"supabase db push ({which})", SUPABASE_CONFIG_PATTERNS,
                     f"check VENTURE_<SLUG>_SUPABASE_{which.upper()}_DB_URL")
        if not res.ok:
            raise StepFailed(f"supabase db push ({which}) failed: {res.tail()}")
        return res.tail(300)
