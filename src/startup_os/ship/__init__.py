"""Ship pipeline: run claude-code tickets with headless Claude Code, then a ship bar
(build → tests → preview → smoke → staging/prod migrations → prod deploy → prod smoke → prod health)."""
