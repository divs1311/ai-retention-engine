"""REST API.

Run with ``retention-engine api`` (or ``uvicorn retention_engine.api.main:app``).
Interactive docs at ``/docs``. If ``RE_API_KEY`` is set, every ``/v1`` route requires an
``X-API-Key`` header.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated, Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request

from retention_engine import __version__
from retention_engine.analytics.kpis import kpi_summary
from retention_engine.analytics.rfm import compute_rfm, segment_summary
from retention_engine.api.schemas import (
    CampaignRequest,
    ScoreRequest,
    ScoreResponse,
    ScoreResult,
    UserRisk,
)
from retention_engine.churn.explain import reasons_for
from retention_engine.churn.features import FEATURES
from retention_engine.churn.model import risk_tier
from retention_engine.config import get_settings
from retention_engine.crm.generator import CampaignGenerator
from retention_engine.crm.providers import available_providers, get_provider
from retention_engine.crm.schemas import AudienceBrief, Campaign
from retention_engine.pipeline import audiences, ensure_artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    settings = get_settings()
    art = ensure_artifacts(settings)
    app.state.settings = settings
    app.state.art = art
    app.state.briefs = {b.audience_id: b for b in audiences(art.scores, art.orders, settings)}
    yield


app = FastAPI(
    title="Retention Engine API",
    version=__version__,
    description="Churn scoring, risk queues and LLM lifecycle campaigns for quick-commerce.",
    lifespan=lifespan,
)


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = os.getenv("RE_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


Auth = Depends(require_api_key)


@app.get("/health", tags=["meta"])
def health(request: Request) -> dict[str, Any]:
    art = request.app.state.art
    return {
        "status": "ok",
        "version": __version__,
        "model": art.card["selected_model"],
        "trained_at": art.card["trained_at"],
        "scored_users": len(art.scores),
    }


@app.get("/v1/model", tags=["model"], dependencies=[Auth])
def model_card(request: Request) -> dict[str, Any]:
    card = request.app.state.art.card
    return {k: v for k, v in card.items() if k != "curves"}


@app.get("/v1/kpis", tags=["analytics"], dependencies=[Auth])
def kpis(request: Request) -> dict[str, Any]:
    return kpi_summary(request.app.state.art.orders).to_dict()


@app.get("/v1/segments", tags=["analytics"], dependencies=[Auth])
def segments(request: Request) -> list[dict[str, Any]]:
    rfm = compute_rfm(request.app.state.art.orders)
    return segment_summary(rfm).to_dict(orient="records")


def _risk_rows(df: pd.DataFrame) -> list[UserRisk]:
    cols = list(UserRisk.model_fields)
    return [UserRisk.model_validate(r) for r in df[cols].to_dict(orient="records")]


@app.get("/v1/users/at-risk", tags=["churn"], dependencies=[Auth], response_model=list[UserRisk])
def at_risk(
    request: Request,
    tier: Annotated[str | None, Query(pattern="^(low|medium|high)$")] = None,
    stage: str | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 50,
) -> list[UserRisk]:
    """Users ranked by value at risk (churn probability x expected 90-day spend)."""
    df = request.app.state.art.scores
    if tier:
        df = df[df["risk_tier"] == tier]
    if stage:
        df = df[df["lifecycle_stage"] == stage]
    return _risk_rows(df.head(limit))


@app.get("/v1/users/{user_id}/risk", tags=["churn"], dependencies=[Auth], response_model=UserRisk)
def user_risk(request: Request, user_id: int) -> UserRisk:
    df = request.app.state.art.scores
    row = df[df["user_id"] == user_id]
    if row.empty:
        raise HTTPException(404, f"User {user_id} is not in the active, scored population")
    return _risk_rows(row)[0]


@app.post("/v1/score", tags=["churn"], dependencies=[Auth], response_model=ScoreResponse)
def score(request: Request, body: ScoreRequest) -> ScoreResponse:
    """Score arbitrary feature vectors (e.g. from an upstream feature store)."""
    art = request.app.state.art
    X = pd.DataFrame([u.model_dump() for u in body.users])[FEATURES]
    p = pd.Series(art.pipeline.predict_proba(X)[:, 1])
    tiers = risk_tier(p, request.app.state.settings.churn)
    results = []
    for i, row in X.iterrows():
        rs = reasons_for(row)
        results.append(
            ScoreResult(
                churn_probability=round(float(p[i]), 4),
                risk_tier=tiers[i],
                reason_codes=[r.code for r in rs[:3]],
                reasons=[r.text for r in rs[:3]],
            )
        )
    return ScoreResponse(model=art.card["selected_model"], results=results)


@app.get("/v1/audiences", tags=["crm"], dependencies=[Auth], response_model=list[AudienceBrief])
def list_audiences(request: Request) -> list[AudienceBrief]:
    return list(request.app.state.briefs.values())


@app.get("/v1/providers", tags=["crm"], dependencies=[Auth])
def providers() -> dict[str, list[str]]:
    return {"available": available_providers()}


@app.post("/v1/campaigns/generate", tags=["crm"], dependencies=[Auth], response_model=Campaign)
def generate_campaign(request: Request, body: CampaignRequest) -> Campaign:
    brief = request.app.state.briefs.get(body.audience_id)
    if brief is None:
        raise HTTPException(404, f"Unknown audience_id {body.audience_id!r}")
    settings = request.app.state.settings
    try:
        provider = get_provider(body.provider, settings.llm)
    except Exception as exc:
        raise HTTPException(400, f"Provider unavailable: {exc}") from exc
    gen = CampaignGenerator(provider, settings.llm, cache_dir=settings.llm_cache_dir)
    return gen.generate(brief)
