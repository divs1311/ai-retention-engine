"""Typed contracts for the CRM layer."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class LifecycleStage(str, Enum):
    ONBOARDING = "onboarding"
    HABIT_FORMATION = "habit_formation"
    CROSS_SELL = "cross_sell"
    CHURN_MITIGATION = "churn_mitigation"
    VIP_RETENTION = "vip_retention"
    WIN_BACK = "win_back"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title().replace("Vip", "VIP")


class Incentive(BaseModel):
    """Decided by policy, never by the LLM. The model may only *describe* this offer."""

    type: Literal["none", "free_delivery", "percent_off", "flat_off"] = "none"
    value: float = 0.0  # percent for percent_off, currency amount for flat_off
    max_cap: float = 0.0  # currency cap for percent_off

    def describe(self, currency: str = "₹") -> str:
        if self.type == "none":
            return "No incentive — rely on relevance and convenience."
        if self.type == "free_delivery":
            return "Free delivery on the next order."
        if self.type == "percent_off":
            return f"{self.value:.0f}% off the next order (max {currency}{self.max_cap:.0f})."
        return f"{currency}{self.value:.0f} off the next order."


class AudienceBrief(BaseModel):
    audience_id: str
    stage: LifecycleStage
    primary_reason: str
    reason_summary: str
    size: int
    avg_churn_probability: float
    value_at_risk: float
    avg_order_value: float
    avg_category_width: float
    top_categories: list[str] = Field(default_factory=list)
    dominant_rfm_segments: list[str] = Field(default_factory=list)
    incentive: Incentive = Field(default_factory=Incentive)
    brand_name: str = "Zippy"
    currency_symbol: str = "₹"


class CampaignCopy(BaseModel):
    push_title: str
    push_body: str
    whatsapp_message: str
    email_subject: str
    email_preheader: str
    cta: str
    rationale: str = ""


class GuardrailReport(BaseModel):
    passed: bool
    violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Campaign(BaseModel):
    brief: AudienceBrief
    content: CampaignCopy
    provider: str
    model: str | None = None
    guardrails: GuardrailReport
    attempts: int = 1
    fallback_reason: str | None = None
    cached: bool = False
    latency_ms: int = 0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
