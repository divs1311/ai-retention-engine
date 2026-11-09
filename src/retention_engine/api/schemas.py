"""Request/response models for the REST API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class UserFeatures(BaseModel):
    """Raw features for ad-hoc scoring (e.g. from a feature store)."""

    recency_days: float = Field(ge=0)
    tenure_days: float = Field(ge=0)
    orders_total: int = Field(ge=1)
    orders_30d: int = Field(ge=0)
    orders_90d: int = Field(ge=0)
    gmv_30d: float = Field(ge=0)
    gmv_90d: float = Field(ge=0)
    aov_90d: float = Field(ge=0)
    freq_trend: float = Field(ge=0)
    mean_gap_days: float = Field(gt=0)
    std_gap_days: float = Field(ge=0)
    overdue_ratio: float = Field(ge=0)
    late_rate_90d: float = Field(ge=0, le=1)
    late_orders_30d: int = Field(ge=0)
    avg_delivery_min_90d: float = Field(ge=0)
    last_order_late: float = Field(ge=0, le=1)
    discount_share_90d: float = Field(ge=0, le=1)
    avg_category_width_90d: float = Field(ge=1)
    distinct_categories_90d: int = Field(ge=0)
    weekend_share: float = Field(ge=0, le=1)
    signup_promo: float = Field(ge=0, le=1)
    city: str
    acquisition_channel: str
    device_os: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "recency_days": 12,
                    "tenure_days": 140,
                    "orders_total": 18,
                    "orders_30d": 1,
                    "orders_90d": 7,
                    "gmv_30d": 420,
                    "gmv_90d": 3150,
                    "aov_90d": 450,
                    "freq_trend": 0.5,
                    "mean_gap_days": 7.5,
                    "std_gap_days": 4.1,
                    "overdue_ratio": 1.6,
                    "late_rate_90d": 0.29,
                    "late_orders_30d": 1,
                    "avg_delivery_min_90d": 13.9,
                    "last_order_late": 1,
                    "discount_share_90d": 0.04,
                    "avg_category_width_90d": 2.4,
                    "distinct_categories_90d": 3,
                    "weekend_share": 0.33,
                    "signup_promo": 0,
                    "city": "Mumbai",
                    "acquisition_channel": "paid_social",
                    "device_os": "android",
                }
            ]
        }
    }


class ScoreRequest(BaseModel):
    users: list[UserFeatures] = Field(min_length=1, max_length=1000)


class ScoreResult(BaseModel):
    churn_probability: float
    risk_tier: Literal["low", "medium", "high"]
    reason_codes: list[str]
    reasons: list[str]


class ScoreResponse(BaseModel):
    model: str
    results: list[ScoreResult]


class UserRisk(BaseModel):
    user_id: int
    churn_probability: float
    risk_tier: str
    value_at_risk: float
    rfm_segment: str | None
    lifecycle_stage: str
    reason_codes: str
    reason_text: str
    recency_days: float
    orders_90d: int
    gmv_90d: float


class CampaignRequest(BaseModel):
    audience_id: str
    provider: Literal["auto", "anthropic", "openai", "template"] = "auto"
