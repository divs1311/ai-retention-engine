"""RFM (Recency, Frequency, Monetary) segmentation.

Scores are quintiles computed on percentile ranks (robust to the heavy ties you get in
frequency). Segments follow the widely used R×F segment map so they're legible to any
CRM team.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# (regex on "<R><F>" score string, segment) — first match wins.
SEGMENT_MAP: list[tuple[str, str]] = [
    (r"5[4-5]", "Champions"),
    (r"[3-4][4-5]", "Loyal"),
    (r"[1-2]5", "Can't Lose"),
    (r"[1-2][3-4]", "At Risk"),
    (r"[4-5][2-3]", "Potential Loyalist"),
    (r"51", "New Customer"),
    (r"41", "Promising"),
    (r"33", "Needs Attention"),
    (r"3[1-2]", "About to Sleep"),
    (r"[1-2][1-2]", "Hibernating"),
]

SEGMENT_ORDER = [s for _, s in SEGMENT_MAP]

SEGMENT_DESCRIPTIONS = {
    "Champions": "Ordered recently, order often and spend the most.",
    "Loyal": "Order regularly; responsive to recommendations.",
    "Can't Lose": "Used to be top customers but haven't returned in a while.",
    "At Risk": "Solid history, but recency is slipping.",
    "Potential Loyalist": "Recent customers with growing frequency.",
    "New Customer": "Just placed their first order(s).",
    "Promising": "Recent but low frequency so far.",
    "Needs Attention": "Middle-of-the-road on every dimension.",
    "About to Sleep": "Below-average recency and frequency.",
    "Hibernating": "Last order long ago, few orders overall.",
}


def _quintile(values: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    pct = values.rank(method="average", pct=True)
    if not higher_is_better:
        pct = 1 - pct + 1 / len(values)
    return np.ceil(pct * 5).clip(1, 5).astype(int)


def compute_rfm(orders: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per user with raw R/F/M, 1–5 scores and a named segment."""
    as_of = as_of or orders["order_ts"].max()
    hist = orders[orders["order_ts"] <= as_of]
    rfm = hist.groupby("user_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "count"),
        monetary=("order_amount", "sum"),
    )
    rfm["recency_days"] = (as_of - rfm["last_order"]).dt.days
    rfm["r_score"] = _quintile(rfm["recency_days"], higher_is_better=False)
    rfm["f_score"] = _quintile(rfm["frequency"])
    rfm["m_score"] = _quintile(rfm["monetary"])
    rfm["rfm_score"] = (
        rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
    )
    rf = rfm["r_score"].astype(str) + rfm["f_score"].astype(str)
    rfm["segment"] = rf.map(_segment_for)
    return rfm.drop(columns="last_order").reset_index()


def _segment_for(rf: str) -> str:
    for pattern, name in SEGMENT_MAP:
        if re.fullmatch(pattern, rf):
            return name
    return "Needs Attention"  # unreachable with the full map, kept as a safe default


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    total_rev = rfm["monetary"].sum()
    out = (
        rfm.groupby("segment")
        .agg(
            users=("user_id", "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_orders=("frequency", "mean"),
            revenue=("monetary", "sum"),
        )
        .reindex(SEGMENT_ORDER)
        .dropna(subset=["users"])
    )
    out["user_share_pct"] = out["users"] / out["users"].sum() * 100
    out["revenue_share_pct"] = out["revenue"] / total_rev * 100
    out["description"] = out.index.map(SEGMENT_DESCRIPTIONS)
    return out.round(1).reset_index()
