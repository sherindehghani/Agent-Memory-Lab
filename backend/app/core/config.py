from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_prefix="LAB_", extra="ignore"
    )

    database_url: str = f"sqlite:///{PROJECT_ROOT / 'data' / 'lab.db'}"
    workspace_dir: Path = PROJECT_ROOT / "workspace"

    random_seed: int = 42

    acceptance_threshold: float = Field(0.001, ge=0.0, le=0.5)

    max_experiments: int = Field(5, ge=1, le=50)
    max_run_seconds: int = Field(600, ge=10, le=3600)
    max_retries: int = Field(1, ge=0, le=3)
    experiment_timeout_seconds: int = Field(120, ge=1, le=1800)

    execution_mode: Literal["subprocess", "inprocess"] = "subprocess"

    llm_provider: Literal["none", "anthropic"] = "none"
    llm_model: str = "claude-opus-5-5"
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    llm_timeout_seconds: int = 60

    autoresearch_run_timeout_seconds: int = Field(120, ge=10, le=900)

    log_level: str = "INFO"
    log_json: bool = True
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
