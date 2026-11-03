# Startup OS

A reusable multi-agent system that generates startup ideas, validates one, and ships an MVP.

**Built so far:** the foundation (memory repo, search, call budget, model fallback, logs, loop
guards), the **Research** agent (idea briefs and validation reports), the **CEO** agent (venture
stage and weekly plans), tickets, and a human approval queue with a spend gate. It runs locally and
only uses OpenRouter `:free` models, so it spends nothing. Not built yet: dashboard, Claude Code
execution of tickets, deploys, outbound messages.

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

uv run startup-os pick research/ideas/<brief>.md   # pick the venture → stage Validation + validation ticket
uv run startup-os validate        # Research: company/research/validation/<date>-<venture>.md
uv run startup-os plan            # CEO: company/plans/<ISO-week>.md + tickets + approval
uv run startup-os inbox [--all]   # pending approvals (and recent decisions)
uv run startup-os approve A-3     # or: reject A-3 --reason "..."
uv run startup-os tickets [--all] # tickets and why they are blocked
uv run startup-os venture         # stage and stage history
uv run pytest                     # tests (LLM calls are mocked)
```

Each brief has: problem, ICP, why now, competitors, kill criteria, and a 1–10 demand-signal
score with reasoning. Front matter records the models used, run id and `evidence: web` or
`evidence: reasoning-only`; reasoning-only briefs also carry a visible warning.

## The venture loop

```
ideas ──► pick ──► validate ──► plan ──► inbox / approve
          (human)  (Research)   (CEO)    (human)
```

- **Stages** (`company/venture.md`, owned by the CEO): Ideation → Validation → Build → Launch → Iterate.
  Iterate can loop back to Build, and any stage can return to Ideation. Every stage change writes a
  decision record in `company/decisions/` (what, why, alternatives) in the same commit as `venture.md`.
  Policy enforced in code: Validation → Build only with a validation report scoring at least
  `validation.go_threshold` (default 7); dropping a GO idea back to Ideation needs a human.
- **pick** records an `idea_pick` approval (approved by you), moves the stage to Validation, and the
  CEO writes a validation ticket for Research.
- **validate** runs Research in validation mode: competitor teardown, pricing scan, demand signals,
  kill-criteria check, go/no-go score and evidence links. Links the model cites but never saw in a
  tool result are listed separately as unverified.
- **plan** has the CEO read `venture.md`, the latest research (validation report, or the idea briefs
  in Ideation) and the last retro, then write `plans/<ISO-week>.md`: goals, tickets (owner `research`
  or `claude-code`), the proposed call-budget split, spend requests, and any stage change. Re-planning
  in the same week supersedes the earlier plan.
- **Approvals** (`approvals` table in `data/state.db`; each decision also appended to `company/approvals.log`):
  - `plan_approval` auto-approves after 48h. Approval makes the plan's budget split live;
    rejection cancels its tickets.
  - `spend_request` never auto-approves and never expires into approval. Tickets that depend on
    it stay blocked until you approve it. Approval only unblocks tickets; nothing is ever spent.
  - `idea_pick` is created by `pick`, or proposed by the CEO in Ideation.
- **Tickets** are `company/tickets/T-0001-<slug>.md` with YAML front matter (`id`, `owner`, `status`,
  `depends_on`, `acceptance_criteria`, …). `depends_on` takes ticket ids (`T-0001`) and approval ids
  (`A-3`). `claude-code` tickets are created but not executed yet.
- **Spend gate** (`spend.py`): agents never hold anything that can spend money. A payment-capable
  function (marked with `@payment_capable`) can only be registered through the gate, which replaces
  it with a stand-in that files a `spend_request`; registering one without a gate raises. Agents also
  get a `request_spend` tool. Neither path ever calls the real function.

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
| CEO agent | `agents/ceo.py` | Plan graph: `gather → draft (validated JSON, repaired up to ceo.max_steps) → write_plan`. Also writes the validation ticket after a pick (template if the LLM is unavailable). |
| Validation | `agents/validation.py`, `agents/loop.py` | Research in validation mode, built on a shared guarded JSON tool loop. |
| Venture, tickets, approvals | `venture.py`, `tickets.py`, `approvals.py`, `spend.py` | See *The venture loop* above. |
| Research agent | `agents/research.py` | Graph: `seed` (N idea seeds, avoiding existing briefs) → per idea `start_idea → agent ⇄ tool → write`. The agent replies in JSON (tool call or final brief), validated with Pydantic. State is checkpointed in `data/checkpoints.db`, so a deferred task resumes at the exact step. |
| Loop guards | `guards.py` | Max LLM steps per brief (`max_steps_per_task`), max tool calls per brief (`max_tool_calls_per_task`; the agent sees its remaining budget after each call), and stop on repeated identical tool calls (same tool + args, case-insensitive). When tripped, the agent gets one forced "final answer, no tools" step; if that fails the idea is skipped and logged. |
| Web search | `tools.py` | Free search via `ddgs` (no key), trying `web_search.backends` in order (DuckDuckGo, Google, Brave, Bing, Mojeek) until one returns results. If disabled or all fail, briefs are flagged reasoning-only. |
| Crash recovery | `budget.py` | Tasks record their process ID. On startup, a task left `running` by a dead process is requeued (Ctrl-C does the same immediately) and resumes from its checkpoint with `run-due`; a request reserved but never answered is marked `interrupted` and still counts against the quota. |
| Logs | `runlog.py` | `logs/runs/<time>_<cmd>_<run_id>.jsonl`: every LLM call (full prompt, model, response, tokens, latency, fallbacks), LLM error, tool call, guard trip and brief written; plus a one-line summary per run in `logs/runs.jsonl`. |

Runtime state (`company/`, `data/`, `logs/`, `.env`) is git-ignored in this repo. Moving to a
VPS later means copying those directories and running the same commands (e.g. `run-due` from cron).
