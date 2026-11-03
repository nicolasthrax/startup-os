"""Configuration loading (config.yaml + .env)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator


class PathsConfig(BaseModel):
    company: Path = Path("company")
    data: Path = Path("data")
    logs: Path = Path("logs")


class LLMConfig(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout_s: float = 90
    temperature: float = 0.7
    max_tokens: int = 2500
    models: list[str] = Field(min_length=1)
    defer_hours_on_failure: float = 1


class BudgetConfig(BaseModel):
    daily_requests: int = 1000
    per_minute: int = 20
    agents: dict[str, float] = Field(default_factory=lambda: {"research": 100})

    @model_validator(mode="after")
    def _shares_sum(self) -> "BudgetConfig":
        total = sum(self.agents.values())
        if total > 100 + 1e-9:
            raise ValueError(f"budget.agents percentages sum to {total} (> 100)")
        return self

    def agent_limit(self, agent: str) -> int:
        return int(self.daily_requests * self.agents.get(agent, 0) / 100)


class GuardsConfig(BaseModel):
    max_steps_per_task: int = 8
    max_identical_tool_calls: int = 2
    max_tool_calls_per_task: int = 4


class WebSearchConfig(BaseModel):
    enabled: bool = True
    provider: str = "ddgs"
    backends: list[str] = Field(default_factory=lambda: ["duckduckgo", "google", "brave", "bing", "mojeek"])
    max_results: int = 5


class ValidationConfig(BaseModel):
    go_threshold: int = Field(default=7, ge=1, le=10)
    max_steps: int = 14
    max_tool_calls: int = 8


class CEOConfig(BaseModel):
    max_steps: int = 3          # draft attempts (initial + repairs) per plan


class Config(BaseModel):
    root: Path = Path(".")
    paths: PathsConfig = PathsConfig()
    llm: LLMConfig
    budget: BudgetConfig = BudgetConfig()
    guards: GuardsConfig = GuardsConfig()
    web_search: WebSearchConfig = WebSearchConfig()
    validation: ValidationConfig = ValidationConfig()
    ceo: CEOConfig = CEOConfig()

    @property
    def company_dir(self) -> Path:
        return (self.root / self.paths.company).resolve()

    @property
    def data_dir(self) -> Path:
        return (self.root / self.paths.data).resolve()

    @property
    def logs_dir(self) -> Path:
        return (self.root / self.paths.logs).resolve()

    def api_key(self) -> str | None:
        return os.environ.get(self.llm.api_key_env) or None


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or os.environ.get("STARTUP_OS_CONFIG", "config.yaml")).resolve()
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    load_dotenv(path.parent / ".env")
    raw = yaml.safe_load(path.read_text()) or {}
    return Config(root=path.parent, **raw)
