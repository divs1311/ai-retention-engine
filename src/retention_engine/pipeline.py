"""End-to-end orchestration shared by the CLI, API and dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from retention_engine.churn import model as churn_model
from retention_engine.config import Settings, get_settings
from retention_engine.crm.generator import CampaignGenerator
from retention_engine.crm.playbook import assign_stages, build_audiences
from retention_engine.crm.providers import get_provider
from retention_engine.crm.schemas import AudienceBrief, Campaign
from retention_engine.data.io import generate_and_save, load_dataset, read_table, write_table
from retention_engine.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class Artifacts:
    users: pd.DataFrame
    orders: pd.DataFrame
    pipeline: Any
    card: dict[str, Any]
    scores: pd.DataFrame


def run_generate(settings: Settings | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate_and_save(settings or get_settings())


def run_train(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    users, orders = load_dataset(settings)
    result = churn_model.train(orders, users, settings.churn, seed=settings.simulation.seed)
    churn_model.save(result, settings)
    return result.card


def run_score(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    users, orders = load_dataset(settings)
    pipe, _ = churn_model.load(settings)
    scores = assign_stages(churn_model.score(pipe, orders, users, settings.churn))
    write_table(scores, settings.scores_path)
    log.info("Scored %s active users → %s", f"{len(scores):,}", settings.scores_path)
    return scores


def audiences(
    scores: pd.DataFrame, orders: pd.DataFrame, settings: Settings | None = None
) -> list[AudienceBrief]:
    settings = settings or get_settings()
    return build_audiences(
        scores,
        orders,
        brand_name=settings.brand_name,
        currency_symbol=settings.currency_symbol,
    )


def run_campaigns(
    settings: Settings | None = None, provider: str = "auto", limit: int | None = None
) -> list[Campaign]:
    settings = settings or get_settings()
    _, orders = load_dataset(settings)
    scores = (
        read_table(settings.scores_path) if settings.scores_path.exists() else run_score(settings)
    )
    briefs = audiences(scores, orders, settings)[:limit]
    gen = CampaignGenerator(
        get_provider(provider, settings.llm), settings.llm, cache_dir=settings.llm_cache_dir
    )
    campaigns = gen.generate_all(briefs)
    settings.campaigns_path.write_text(
        json.dumps([c.model_dump(mode="json") for c in campaigns], indent=2, ensure_ascii=False)
    )
    log.info(
        "Generated %d campaigns with %s → %s",
        len(campaigns),
        gen.provider.name,
        settings.campaigns_path,
    )
    return campaigns


def run_all(settings: Settings | None = None, provider: str = "auto") -> list[Campaign]:
    settings = settings or get_settings()
    run_generate(settings)
    run_train(settings)
    run_score(settings)
    return run_campaigns(settings, provider)


def ensure_artifacts(settings: Settings | None = None) -> Artifacts:
    """Load everything the app/API needs, building whatever is missing or stale."""
    settings = settings or get_settings()
    users, orders = load_dataset(settings, generate_if_missing=True)

    needs_train = not settings.model_path.exists()
    if not needs_train:
        _, card = churn_model.load(settings)
        needs_train = card.get("data_fingerprint") != churn_model._data_fingerprint(orders)
    if needs_train:
        log.info("Model missing or stale — training")
        run_train(settings)
    pipe, card = churn_model.load(settings)

    stale_scores = (
        not settings.scores_path.exists()
        or settings.scores_path.stat().st_mtime < settings.model_path.stat().st_mtime
    )
    scores = run_score(settings) if stale_scores else read_table(settings.scores_path)
    return Artifacts(users=users, orders=orders, pipeline=pipe, card=card, scores=scores)
