"""Per-venture credentials and clean environments.

Credentials are read from env vars (loaded from .env) named VENTURE_<SLUG>_<NAME>,
so each venture has its own project-scoped tokens. They are only ever handed to
our own deploy/migrate subprocesses, as env vars. Claude Code gets an environment
with every project secret removed, and prompts are checked for secret values."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

# Secret env vars that must never reach Claude Code (its own auth vars are kept).
SECRET_PREFIXES = ("VENTURE_", "VERCEL_", "SUPABASE_", "OPENROUTER_")
SECRET_NAMES = {"GH_TOKEN", "GITHUB_TOKEN", "CLOUDSDK_AUTH_ACCESS_TOKEN", "DATABASE_URL", "NPM_TOKEN"}
NAMES = ("VERCEL_TOKEN", "VERCEL_ORG_ID", "VERCEL_PROJECT_ID", "VERCEL_BYPASS_SECRET", "PROD_URL",
         "SUPABASE_STAGING_DB_URL", "SUPABASE_PROD_DB_URL",
         "SUPABASE_STAGING_API_URL", "SUPABASE_STAGING_ANON_KEY", "SUPABASE_PROD_API_URL", "SUPABASE_PROD_ANON_KEY")


class MissingCredential(Exception):
    """Configuration problem (not the ticket's fault): stops the run without a strike."""


class SecretInPrompt(MissingCredential):
    """A secret value appears in text bound for Claude Code (e.g. pasted into a ticket)."""


def env_prefix(slug: str) -> str:
    return "VENTURE_" + re.sub(r"[^A-Z0-9]", "_", slug.upper()) + "_"


@dataclass
class VentureSecrets:
    slug: str
    values: dict[str, str]

    @classmethod
    def from_env(cls, slug: str, environ: dict[str, str] | None = None) -> "VentureSecrets":
        environ = os.environ if environ is None else environ
        pre = env_prefix(slug)
        return cls(slug, {n: environ[pre + n] for n in NAMES if environ.get(pre + n)})

    def get(self, name: str) -> str:
        v = self.values.get(name)
        if not v:
            raise MissingCredential(f"{env_prefix(self.slug)}{name} is not set (add it to .env)")
        return v

    def has(self, name: str) -> bool:
        return bool(self.values.get(name))

    def secret_values(self) -> list[str]:
        return [v for k, v in self.values.items() if k != "PROD_URL" and len(v) >= 6]


def is_secret_name(name: str) -> bool:
    return name.startswith(SECRET_PREFIXES) or name in SECRET_NAMES


def clean_env(environ: dict[str, str] | None = None, **extra: str) -> dict[str, str]:
    """Copy of the environment without project secrets, plus `extra`."""
    environ = os.environ if environ is None else environ
    env = {k: v for k, v in environ.items() if not is_secret_name(k)}
    env.update(extra)
    return env


def secret_values_in_env(environ: dict[str, str] | None = None) -> list[str]:
    environ = os.environ if environ is None else environ
    return [v for k, v in environ.items() if is_secret_name(k) and len(v) >= 8]


def assert_no_secrets(text: str, secrets: list[str]) -> None:
    for s in secrets:
        if s and s in text:
            raise SecretInPrompt("refusing to send a secret value into a Claude Code prompt; "
                                 "remove it from the ticket and use an env var instead")
