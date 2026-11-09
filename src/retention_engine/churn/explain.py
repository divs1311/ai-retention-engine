"""Human-readable reason codes for each scored user.

A probability alone isn't actionable for a CRM team. Reason codes turn the feature vector
into the *why* ("3 late deliveries this month", "ordering 60% less often"), which drives
both the lifecycle playbook and the LLM prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Reason:
    code: str
    severity: float  # higher = more important; used for ranking
    text: str


def _fmt_days(d: float) -> str:
    return f"{d:.1f}-day" if d < 3 else f"{d:.0f}-day"


def reasons_for(row: pd.Series) -> list[Reason]:
    out: list[Reason] = []
    if row["overdue_ratio"] >= 1.5 and row["recency_days"] >= 7:
        out.append(
            Reason(
                "OVERDUE",
                min(row["overdue_ratio"], 6),
                f"{row['recency_days']:.0f} days since last order vs a usual "
                f"{_fmt_days(row['mean_gap_days'])} cadence",
            )
        )
    if row["freq_trend"] <= 0.6 and row["orders_90d"] >= 2:
        drop = (1 - row["freq_trend"]) * 100
        out.append(Reason("FREQ_DECLINE", 1 + drop / 25, f"Order frequency down ~{drop:.0f}%"))
    if row["late_orders_30d"] >= 2 or row["late_rate_90d"] >= 0.25:
        n = int(row["late_orders_30d"])
        txt = (
            f"{n} late deliver{'y' if n == 1 else 'ies'} in the last 30 days"
            if n
            else f"{row['late_rate_90d']:.0%} of recent orders arrived late"
        )
        out.append(Reason("LATE_DELIVERY", 1 + 0.75 * n + 2 * row["late_rate_90d"], txt))
    if row["avg_category_width_90d"] <= 2.0:
        out.append(
            Reason(
                "NARROW_BASKET",
                1.5,
                f"Narrow basket (~{row['avg_category_width_90d']:.1f} categories per order)",
            )
        )
    if row["tenure_days"] < 45:
        out.append(Reason("EARLY_TENURE", 2.0, "New user — ordering habit not yet formed"))
    if row["signup_promo"] >= 1 and row["discount_share_90d"] >= 0.08:
        out.append(
            Reason(
                "DISCOUNT_DEPENDENT",
                1.2,
                f"Promo-acquired; {row['discount_share_90d']:.0%} of recent spend discounted",
            )
        )
    return sorted(out, key=lambda r: r.severity, reverse=True)


def attach_reasons(scored: pd.DataFrame, max_reasons: int = 3) -> pd.DataFrame:
    reasons = scored.apply(reasons_for, axis=1)
    scored = scored.copy()
    scored["reason_codes"] = reasons.map(lambda rs: "|".join(r.code for r in rs[:max_reasons]))
    scored["primary_reason"] = reasons.map(lambda rs: rs[0].code if rs else "NONE")
    scored["reason_text"] = reasons.map(
        lambda rs: "; ".join(r.text for r in rs[:max_reasons]) or "No strong risk signals"
    )
    return scored
