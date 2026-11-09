"""Lifecycle playbook: map scored users → stages → audiences → briefs.

Two design choices worth calling out:

* **Audiences, not individuals.** Copy is generated per (stage × primary reason) audience
  and personalised at send time with merge tokens. That keeps LLM cost flat as the user
  base grows and makes every message reviewable before it ships.
* **Incentives are policy.** The playbook — not the model — decides whether an audience
  gets an offer. Low-risk users never get discounts (discount addiction is a real cost),
  VIPs get service recovery rather than coupons, and only lapsing users get money off.
"""

from __future__ import annotations

import pandas as pd

from retention_engine.crm.schemas import AudienceBrief, Incentive, LifecycleStage

HIGH_VALUE_SEGMENTS = {"Champions", "Loyal", "Can't Lose"}


def assign_stage(row: pd.Series) -> LifecycleStage:
    risky = row["risk_tier"] in {"high", "medium"}
    if row["tenure_days"] < 45 and risky:
        return LifecycleStage.ONBOARDING
    if risky and row["rfm_segment"] in HIGH_VALUE_SEGMENTS:
        return LifecycleStage.VIP_RETENTION
    if row["risk_tier"] == "high" and row["recency_days"] > 21:
        return LifecycleStage.WIN_BACK
    if risky:
        return LifecycleStage.CHURN_MITIGATION
    if row["avg_category_width_90d"] <= 3:
        return LifecycleStage.CROSS_SELL
    return LifecycleStage.HABIT_FORMATION


def incentive_policy(stage: LifecycleStage, primary_reason: str) -> Incentive:
    if primary_reason == "LATE_DELIVERY" and stage in {
        LifecycleStage.VIP_RETENTION,
        LifecycleStage.CHURN_MITIGATION,
        LifecycleStage.WIN_BACK,
    }:
        return Incentive(type="flat_off", value=75)  # service-recovery credit
    if stage == LifecycleStage.WIN_BACK:
        if primary_reason == "DISCOUNT_DEPENDENT":
            return Incentive(type="free_delivery")
        return Incentive(type="percent_off", value=15, max_cap=100)
    if stage in {LifecycleStage.ONBOARDING, LifecycleStage.VIP_RETENTION}:
        return Incentive(type="free_delivery")
    if stage == LifecycleStage.CHURN_MITIGATION and primary_reason != "DISCOUNT_DEPENDENT":
        return Incentive(type="free_delivery")
    return Incentive(type="none")


def assign_stages(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["lifecycle_stage"] = out.apply(lambda r: assign_stage(r).value, axis=1)
    return out


def build_audiences(
    scored: pd.DataFrame,
    orders: pd.DataFrame | None = None,
    *,
    min_size: int = 15,
    brand_name: str = "Zippy",
    currency_symbol: str = "₹",
) -> list[AudienceBrief]:
    """Group scored users into audiences and produce one brief per audience."""
    df = scored if "lifecycle_stage" in scored else assign_stages(scored)
    briefs: list[AudienceBrief] = []

    recent = None
    if orders is not None:
        cutoff = orders["order_ts"].max() - pd.Timedelta(days=90)
        recent = orders[orders["order_ts"] >= cutoff]

    for (stage_value, reason), grp in df.groupby(["lifecycle_stage", "primary_reason"]):
        if len(grp) < min_size:
            continue
        stage = LifecycleStage(stage_value)
        top_categories: list[str] = []
        if recent is not None:
            top_categories = (
                recent[recent["user_id"].isin(grp["user_id"])]["primary_category"]
                .value_counts()
                .head(3)
                .index.str.replace("_", " ")
                .tolist()
            )
        reason_texts = grp["reason_text"].value_counts().head(2).index.tolist()
        briefs.append(
            AudienceBrief(
                audience_id=f"{stage.value}__{str(reason).lower()}",
                stage=stage,
                primary_reason=str(reason),
                reason_summary=" / ".join(reason_texts)[:400],
                size=len(grp),
                avg_churn_probability=round(float(grp["churn_probability"].mean()), 3),
                value_at_risk=round(float(grp["value_at_risk"].sum()), 2),
                avg_order_value=round(float(grp["aov_90d"].mean()), 2),
                avg_category_width=round(float(grp["avg_category_width_90d"].mean()), 2),
                top_categories=top_categories,
                dominant_rfm_segments=grp["rfm_segment"].value_counts().head(2).index.tolist(),
                incentive=incentive_policy(stage, str(reason)),
                brand_name=brand_name,
                currency_symbol=currency_symbol,
            )
        )
    return sorted(briefs, key=lambda b: b.value_at_risk, reverse=True)
