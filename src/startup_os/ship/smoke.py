"""Smoke tests against a deployment: GET /api/health plus 1-3 Playwright checks
derived from the ticket's acceptance criteria. Run by Startup OS, not Claude Code."""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..tickets import Ticket


@dataclass
class Check:
    path: str
    expect_text: str | None = None
    expect_status: int = 200
    source: str = ""          # the acceptance criterion it came from


@dataclass
class SmokeResult:
    ok: bool
    results: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return "; ".join(f"{r['check']}: {'ok' if r['ok'] else 'FAIL ' + r.get('error', '')}" for r in self.results)


_PATH = re.compile(r"(?<![\w.])(/[A-Za-z0-9_\-./]*)")
_QUOTED = re.compile(r"[\"“'‘`]([^\"”'’`]{2,80})[\"”'’`]")


def derive_checks(ticket: Ticket, limit: int = 3) -> list[Check]:
    """Page checks from acceptance criteria that name a path and a quoted text, e.g.
    'The home page / shows "Hello, world"'. /api/health is always checked separately."""
    explicit = [Check(**c) for c in (ticket.smoke_checks or [])]
    checks = explicit[:limit]
    for crit in ticket.acceptance_criteria:
        if len(checks) >= limit:
            break
        text = _QUOTED.search(crit)
        paths = [p for p in _PATH.findall(crit) if not p.startswith("/api/health")]
        if re.search(r"home ?page|landing page|root page", crit, re.I) and "/" not in paths:
            paths.insert(0, "/")
        if paths and text and "api/health" not in crit:
            c = Check(path=paths[0], expect_text=text.group(1), source=crit)
            if all((c.path, c.expect_text) != (x.path, x.expect_text) for x in checks):
                checks.append(c)
    return checks or [Check(path="/", source="fallback: home page responds")]


def _headers(bypass: str | None) -> dict[str, str]:
    return {"x-vercel-protection-bypass": bypass, "x-vercel-set-bypass-cookie": "true"} if bypass else {}


def health(base_url: str, bypass: str | None = None, timeout_s: float = 20) -> tuple[bool, str]:
    url = base_url.rstrip("/") + "/api/health"
    req = urllib.request.Request(url, headers={"User-Agent": "startup-os-smoke", **_headers(bypass)})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            body = r.read(2000).decode("utf-8", "replace")
            if r.status != 200:
                return False, f"HTTP {r.status}"
            try:
                ok = json.loads(body).get("status") == "ok"
            except (json.JSONDecodeError, AttributeError):
                ok = False
            return ok, "ok" if ok else f"unexpected body: {body[:200]}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}" + (" (deployment protection? set VENTURE_<SLUG>_VERCEL_BYPASS_SECRET)" if e.code in (401, 403) else "")
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


PLAYWRIGHT_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const spec = JSON.parse(process.argv[2]);
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ extraHTTPHeaders: spec.headers });
  const page = await ctx.newPage();
  const out = [];
  for (const c of spec.checks) {
    const r = { check: `${c.path}` + (c.expect_text ? ` contains "${c.expect_text}"` : ` → ${c.expect_status}`), ok: false };
    try {
      const resp = await page.goto(spec.base + c.path, { waitUntil: 'networkidle', timeout: 30000 });
      const status = resp ? resp.status() : 0;
      if (status !== c.expect_status) { r.error = `HTTP ${status}`; }
      else if (c.expect_text) {
        const body = await page.textContent('body');
        r.ok = (body || '').includes(c.expect_text);
        if (!r.ok) r.error = 'text not found';
      } else { r.ok = true; }
    } catch (e) { r.error = String(e).slice(0, 200); }
    out.push(r);
  }
  await browser.close();
  console.log(JSON.stringify(out));
})().catch(e => { console.log(JSON.stringify([{ check: 'playwright', ok: false, error: String(e).slice(0, 300) }])); });
"""


def playwright_checks(base_url: str, checks: list[Check], bypass: str | None = None,
                      timeout_s: float = 180) -> list[dict]:
    spec = {"base": base_url.rstrip("/"), "headers": _headers(bypass),
            "checks": [{"path": c.path, "expect_text": c.expect_text, "expect_status": c.expect_status} for c in checks]}
    with tempfile.TemporaryDirectory() as d:
        script = Path(d) / "smoke.js"
        script.write_text(PLAYWRIGHT_JS)
        r = subprocess.run(["node", str(script), json.dumps(spec)], capture_output=True, text=True, timeout=timeout_s)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return [{"check": "playwright", "ok": False, "error": (r.stderr or r.stdout)[-300:]}]


class SmokeRunner(Protocol):
    def __call__(self, base_url: str, checks: list[Check], bypass: str | None) -> SmokeResult: ...


def run_smoke(base_url: str, checks: list[Check], bypass: str | None = None) -> SmokeResult:
    ok, detail = health(base_url, bypass)
    results = [{"check": "GET /api/health → {status: ok}", "ok": ok, **({} if ok else {"error": detail})}]
    if ok:
        results += playwright_checks(base_url, checks, bypass)
    return SmokeResult(all(r["ok"] for r in results), results)
