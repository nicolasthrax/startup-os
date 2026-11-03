"""Build the Claude Code prompt from a ticket and the relevant company/ files."""
from __future__ import annotations

from ..memory import MemoryRepo
from ..paths import check_path
from ..search_index import SearchIndex
from ..tickets import Ticket
from .secrets import assert_no_secrets

MAX_FILE_CHARS = 3500
MAX_FILES = 4


def relevant_files(ticket: Ticket, memory: MemoryRepo, index: SearchIndex) -> list[str]:
    """Ticket inputs first (validated), then top search hits for the ticket's title and description."""
    out: list[str] = []
    for p in ticket.inputs:
        rel, problem = check_path(memory, p)
        if rel and rel not in out:
            out.append(rel)
    query = f"{ticket.title} {ticket.description} {ticket.venture or ''}"
    for hit in index.search(query, limit=8):
        p = hit["path"]
        if p.startswith(("tickets/", "plans/")) or p in out or p == "venture.md":
            continue
        out.append(p)
        if len(out) >= MAX_FILES:
            break
    return out[:MAX_FILES]


def build_prompt(ticket: Ticket, *, memory: MemoryRepo, index: SearchIndex, slug: str,
                 secrets: list[str], previous_error: str | None = None) -> tuple[str, list[str]]:
    files = relevant_files(ticket, memory, index)
    context = "\n\n".join(f"--- company/{p} ---\n{memory.read(p)[:MAX_FILE_CHARS]}" for p in files)
    env_names = "NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY"
    retry = (f"\n## Previous attempt failed\n\n{previous_error}\n\nFix the cause; do not weaken tests.\n"
             if previous_error else "")
    prompt = f"""You are implementing ticket {ticket.id} for the venture "{ticket.venture}" in this repository
(the current directory is the venture repo; work only inside it).

## Ticket {ticket.id}: {ticket.title}

{ticket.description or '(no description)'}

## Acceptance criteria

{chr(10).join(f'- {c}' for c in ticket.acceptance_criteria)}
{retry}
## Engineering rules (mandatory)

1. Stack: Next.js (App Router) + TypeScript, deployable on Vercel's free Hobby plan. If the repo is
   empty, scaffold it by hand (package.json, tsconfig, app/); do not use interactive generators.
2. `npm run build` must succeed and `npm test` must run a real test suite (Vitest) that passes.
   Write tests for EVERY feature you add or change; every acceptance criterion needs a test.
3. Provide `GET /api/health` (app/api/health/route.ts) returning HTTP 200 with JSON {{"status": "ok"}}.
   It is used for smoke tests and production health checks; keep it dependency-free.
4. Do NOT deploy, push, or run vercel/supabase CLIs. Startup OS builds, tests, deploys and migrates
   after you finish. Leave changes uncommitted; Startup OS commits them.
5. Database changes go in `supabase/migrations/<timestamp>_<name>.sql` (plain SQL, idempotent where possible).
   Migrations MUST be backward-compatible (additive only), because the previous version of the app
   keeps running against the same database during and after the deploy:
   - no DROP (tables, columns, indexes, constraints, ...) and no TRUNCATE;
   - no RENAME (add a new column/table and backfill instead);
   - no NOT NULL without a DEFAULT on an existing table (no `ADD COLUMN ... NOT NULL` without
     `DEFAULT`, no `SET NOT NULL`), and no column TYPE changes.
   New tables may use NOT NULL freely. Startup OS rejects non-additive migrations before they run.
6. Never hardcode secrets or URLs of external services. Read configuration from environment variables
   (e.g. {env_names}); document any you need in README.md. You do not have, and must not ask for, credentials.
7. Money: do not add anything that needs a paid plan, a custom domain or a paid API. If the ticket truly
   cannot be done without one, write `ship.json` with
   {{"paid_requirements": [{{"item": "...", "amount_usd": 0, "why": "..."}}]}} and stop.
8. Keep it minimal: only what the ticket needs. Finish with a short summary of what you built and how
   it is tested.

## Relevant company context (read-only)

{context or '(none found)'}
"""
    assert_no_secrets(prompt, secrets)
    return prompt, files
