# Startup OS — notes for Claude Code

## Hard rules

- **Never commit or push unless the full test suite passed in the same step.** Chain them in one
  command so a failure stops everything, e.g.

  ```bash
  uv run pytest && git add -A && git commit -m "..." && git push
  ```

  A test run from an earlier step does not count, and neither does a partial run (`-k`, a single
  file). If the suite fails, fix it first; never commit "fix tests later".
- **The test gate must not be piped.** In `pytest | tail && git commit`, the `&&` sees `tail`'s exit
  status, so a failing suite still commits. Run `uv run pytest` unpiped before `&&`, or set
  `pipefail` if you must trim the output:

  ```bash
  # OK: unpiped gate
  uv run pytest -q && git add -A && git commit -m "..." && git push
  # OK: piped, but pipefail makes a pytest failure stop the chain
  set -o pipefail; uv run pytest -q 2>&1 | tail -5 && git add -A && git commit -m "..." && git push
  # WRONG: tail's exit status (0) is what `&&` checks
  uv run pytest -q | tail -5 && git commit -m "..."
  ```
- Never skip, disable or weaken a test to get green.
- Nothing may spend money. Payment-capable code goes through `SpendGate` (`src/startup_os/spend.py`).
- Idea picks are the human's. Agents may only recommend; `startup-os pick` needs an explicit file.
- Ship pipeline: steps a–g are run by Startup OS, never by Claude Code. Never pass secrets into a
  Claude Code prompt or its environment; credentials reach only our deploy/migrate subprocesses as env vars.
- The OpenRouter call budget has exactly three agents: `ceo`, `research`, `reserve`. Claude Code uses
  a separate subscription and never takes OpenRouter quota.

## Project

- Python 3.12, managed with uv. Run everything with `uv run ...`.
- Tests: `uv run pytest` (LLM calls are mocked; no network needed).
- Runtime state is git-ignored: `company/` (its own Git repo), `data/`, `logs/`, `.env`.
- `company/` is synced to the private remote `company.remote` (config.yaml) with `COMPANY_REPO_TOKEN`:
  pulled at the start of every command, pushed after every commit. Never force-push it, and never put
  the token in a URL, file, log or prompt. `data/` and `logs/` are not synced.
- Architecture and commands: see `README.md`.
