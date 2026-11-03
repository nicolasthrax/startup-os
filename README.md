# Startup OS

A reusable multi-agent system that generates startup ideas, validates one, and ships an MVP.

**This slice:** the foundation (memory repo, search, call budget, model fallback, logs, loop
guards) and one agent, **Research**, which writes idea briefs. It runs locally and only uses
OpenRouter `:free` models, so it spends nothing. Not built yet: CEO agent, dashboard,
Claude Code handoff, deploys, outbound messages.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env        # then set OPENROUTER_API_KEY
```

`.env` is git-ignored. Settings live in `config.yaml` (models, budget, guards, web search).
Free models on OpenRouter change often; `uv run startup-os models` lists the ones currently
offered and marks the configured ones (a read-only request; no spend).

## Run

```bash
uv run startup-os ideas --n 5     # Research agent → company/research/ideas/*.md
uv run startup-os status          # today's usage per agent + recent memory commits
uv run startup-os queue [--all]   # deferred tasks
uv run startup-os run-due         # resume deferred tasks whose time has come
uv run startup-os search "invoices agencies"   # FTS5 search over company memory
uv run pytest                     # tests (LLM calls are mocked)
```

Each brief has: problem, ICP, why now, competitors, kill criteria, and a 1–10 demand-signal
score with reasoning. Front matter records the models used, run id and `evidence: web` or
`evidence: reasoning-only`; reasoning-only briefs also carry a visible warning.

## Architecture

```
cli.py ──► runtime.py ──► agents/research.py (LangGraph, SqliteSaver checkpoints)
                │                 │ uses
                │        ┌────────┼──────────────┬───────────────┐
                │        ▼        ▼              ▼               ▼
                │   llm.ModelRouter   tools.Toolbox     guards.LoopGuard   memory.MemoryRepo
                │        │            search / web_search                    │ git commit per write
                │        ▼                 │                                 ▼
                └─► budget.Scheduler   search_index (FTS5) ◄── rebuilt after every write
```

| Piece | File | What it does |
|---|---|---|
| Memory repo | `memory.py` | `./company/` is its own Git repo (`venture.md`, `plans/`, `decisions/`, `research/`, `tickets/`, `retros/`, `approvals.log`), created on first run. Every write is a commit: `research: add idea brief "X" (demand 6/10, web)`. Paths can't escape the repo. |
| Search | `search_index.py` | SQLite FTS5 (`data/index.db`, porter stemming, BM25) over `company/**/*.md`, rebuilt after each write. Exposed to agents as the `search(query)` tool. |
| Budget | `budget.py` | Daily request quota (default 1000/day) split by agent percentages in `config.yaml`; a token bucket enforces the per-minute rate (default 20/min) by waiting. Every attempt, fallbacks included, is reserved then logged in `data/state.db` (`usage` table: model, status, tokens, latency). An agent over its daily share raises `BudgetExceeded`; the task is queued (`tasks` table) for 00:00 UTC next day, never dropped. |
| Fallback | `llm.py` | OpenRouter via the OpenAI SDK (`max_retries=0`). Tries `llm.models` in order; a 429, error or empty reply moves to the next model. If all fail, the task is deferred by `defer_hours_on_failure`. |
| Research agent | `agents/research.py` | Graph: `seed` (N idea seeds, avoiding existing briefs) → per idea `start_idea → agent ⇄ tool → write`. The agent replies in JSON (tool call or final brief), validated with Pydantic. State is checkpointed in `data/checkpoints.db`, so a deferred task resumes at the exact step. |
| Loop guards | `guards.py` | Max LLM steps per brief (`max_steps_per_task`) and stop on repeated identical tool calls (same tool + args, case-insensitive). When tripped, the agent gets one forced "final answer, no tools" step; if that fails the idea is skipped and logged. |
| Web search | `tools.py` | Free DuckDuckGo search via `ddgs` (no key). If disabled or failing, briefs are flagged reasoning-only. |
| Logs | `runlog.py` | `logs/runs/<time>_<cmd>_<run_id>.jsonl`: every LLM call (full prompt, model, response, tokens, latency, fallbacks), LLM error, tool call, guard trip and brief written; plus a one-line summary per run in `logs/runs.jsonl`. |

Runtime state (`company/`, `data/`, `logs/`, `.env`) is git-ignored in this repo. Moving to a
VPS later means copying those directories and running the same commands (e.g. `run-due` from cron).
