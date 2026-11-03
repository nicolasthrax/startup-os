# Startup OS — notes for Claude Code

## Hard rules

- **Never commit or push unless the full test suite passed in the same step.** Chain them in one
  command so a failure stops everything, e.g.

  ```bash
  uv run pytest && git add -A && git commit -m "..." && git push
  ```

  A test run from an earlier step does not count, and neither does a partial run (`-k`, a single
  file). If the suite fails, fix it first; never commit "fix tests later".
- Never skip, disable or weaken a test to get green.
- Nothing may spend money. Payment-capable code goes through `SpendGate` (`src/startup_os/spend.py`).
- Idea picks are the human's. Agents may only recommend; `startup-os pick` needs an explicit file.
- The OpenRouter call budget has exactly three agents: `ceo`, `research`, `reserve`. Claude Code uses
  a separate subscription and never takes OpenRouter quota.

## Project

- Python 3.12, managed with uv. Run everything with `uv run ...`.
- Tests: `uv run pytest` (LLM calls are mocked; no network needed).
- Runtime state is git-ignored: `company/` (its own Git repo), `data/`, `logs/`, `.env`.
- Architecture and commands: see `README.md`.
