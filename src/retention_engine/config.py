"""Central configuration.

Every tunable lives here and can be overridden with an environment variable prefixed
``RE_`` (e.g. ``RE_N_USERS=10000``) or a ``.env`` file. Nested models use ``__``
(e.g. ``RE_CHURN__HORIZON_DAYS=45``). Provider API keys keep their standard names.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
# Source checkout → keep data next to the repo; installed wheel → use the working directory.
ROOT_DIR = _SOURCE_ROOT if (_SOURCE_ROOT / "pyproject.toml").exists() else Path.cwd()


class SimulationConfig(BaseModel):
    """Knobs for the synthetic quick-commerce marketplace."""

    seed: int = 42
    n_users: int = Field(6_000, ge=100)
    start_date: date = date(2025, 1, 1)
    end_date: date = date(2026, 6, 30)
    # Fraction of the window during which new users can sign up (rest is observation tail).
    signup_window_frac: float = Field(1.0, gt=0, le=1)
    late_delivery_minutes: int = 15  # the "10-minute delivery" promise, plus grace

    @field_validator("end_date")
    @classmethod
    def _end_after_start(cls, v: date, info: ValidationInfo) -> date:
        start = info.data.get("start_date")
        if start and v <= start:
            raise ValueError("end_date must be after start_date")
        return v


class ChurnConfig(BaseModel):
    """Churn definition and modelling choices."""

    horizon_days: int = Field(30, ge=7)  # churn = no order in the next N days
    active_window_days: int = Field(30, ge=14)  # population = ordered in the last N days
    snapshot_freq: str = "MS"  # monthly snapshots for building training data
    min_tenure_days: int = 7
    validation_snapshots: int = 2  # most-recent fully-labelled snapshots held out
    high_risk_threshold: float = Field(0.60, gt=0, lt=1)
    medium_risk_threshold: float = Field(0.35, gt=0, lt=1)


class LLMConfig(BaseModel):
    provider: Literal["auto", "openai", "anthropic", "template"] = "auto"
    openai_model: str = "gpt-5.4-mini"
    anthropic_model: str = "claude-haiku-5-5"
    max_output_tokens: int = 800
    timeout_s: float = 30.0
    max_retries: int = 1  # retries after a guardrail violation, before template fallback
    cache: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RE_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    data_dir: Path = ROOT_DIR / "data"
    artifacts_dir: Path = ROOT_DIR / "artifacts"
    currency_symbol: str = "₹"
    brand_name: str = "Zippy"  # fictional quick-commerce brand used in CRM copy

    simulation: SimulationConfig = SimulationConfig()
    churn: ChurnConfig = ChurnConfig()
    llm: LLMConfig = LLMConfig()

    # --- derived paths -------------------------------------------------------------------
    @property
    def users_path(self) -> Path:
        return self.data_dir / "users.parquet"

    @property
    def orders_path(self) -> Path:
        return self.data_dir / "orders.parquet"

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "churn_model.joblib"

    @property
    def model_card_path(self) -> Path:
        return self.artifacts_dir / "churn_model_card.json"

    @property
    def scores_path(self) -> Path:
        return self.artifacts_dir / "user_scores.parquet"

    @property
    def campaigns_path(self) -> Path:
        return self.artifacts_dir / "campaigns.json"

    @property
    def llm_cache_dir(self) -> Path:
        return self.artifacts_dir / "llm_cache"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
