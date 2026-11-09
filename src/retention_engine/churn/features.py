"""Point-in-time feature engineering for churn prediction.

Leakage rules (enforced, and covered by tests):

* features at snapshot ``as_of`` only use orders with ``order_ts < as_of``;
* the label uses only orders in ``[as_of, as_of + horizon)``;
* the population is users who are *currently active* (ordered within
  ``active_window_days``) and have at least ``min_tenure_days`` of history — predicting
  churn for someone who left months ago is not a useful product question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retention_engine.config import ChurnConfig

NUMERIC_FEATURES: list[str] = [
    "recency_days",
    "tenure_days",
    "orders_total",
    "orders_30d",
    "orders_90d",
    "gmv_30d",
    "gmv_90d",
    "aov_90d",
    "freq_trend",
    "mean_gap_days",
    "std_gap_days",
    "overdue_ratio",
    "late_rate_90d",
    "late_orders_30d",
    "avg_delivery_min_90d",
    "last_order_late",
    "discount_share_90d",
    "avg_category_width_90d",
    "distinct_categories_90d",
    "weekend_share",
    "signup_promo",
]
CATEGORICAL_FEATURES: list[str] = ["city", "acquisition_channel", "device_os"]
FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _window(hist: pd.DataFrame, as_of: pd.Timestamp, days: int) -> pd.DataFrame:
    return hist[hist["order_ts"] >= as_of - pd.Timedelta(days=days)]


def build_features(
    orders: pd.DataFrame,
    users: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: ChurnConfig | None = None,
    *,
    restrict_population: bool = True,
) -> pd.DataFrame:
    """Feature matrix (index = user_id) as seen at ``as_of``."""
    cfg = cfg or ChurnConfig()
    as_of = pd.Timestamp(as_of)
    hist = orders[orders["order_ts"] < as_of]
    if hist.empty:
        return pd.DataFrame(columns=FEATURES).rename_axis("user_id")

    g = hist.groupby("user_id")
    f = pd.DataFrame(
        {
            "first_order": g["order_ts"].min(),
            "last_order": g["order_ts"].max(),
            "orders_total": g.size(),
        }
    )
    f["recency_days"] = (as_of - f["last_order"]).dt.total_seconds() / 86400
    f["tenure_days"] = (as_of - f["first_order"]).dt.total_seconds() / 86400

    if restrict_population:
        f = f[
            (f["recency_days"] <= cfg.active_window_days)
            & (f["tenure_days"] >= cfg.min_tenure_days)
        ]
    hist = hist[hist["user_id"].isin(f.index)]

    w30 = _window(hist, as_of, 30).groupby("user_id")
    w90 = _window(hist, as_of, 90)
    g90 = w90.groupby("user_id")

    f["orders_30d"] = w30.size()
    f["orders_90d"] = g90.size()
    f["gmv_30d"] = w30["order_amount"].sum()
    f["gmv_90d"] = g90["order_amount"].sum()
    f[["orders_30d", "orders_90d", "gmv_30d", "gmv_90d"]] = f[
        ["orders_30d", "orders_90d", "gmv_30d", "gmv_90d"]
    ].fillna(0)
    f["aov_90d"] = (f["gmv_90d"] / f["orders_90d"].replace(0, np.nan)).fillna(
        g["order_amount"].mean()
    )

    # Momentum: last-30d order rate vs the preceding 60 days (per 30d), Laplace-smoothed.
    prior_rate = (f["orders_90d"] - f["orders_30d"]) / 2
    f["freq_trend"] = (f["orders_30d"] + 1) / (prior_rate + 1)

    # Cadence: inter-order gaps across the full history.
    gaps = hist.sort_values("order_ts").groupby("user_id")["order_ts"].diff().dt.total_seconds() / 86400
    gap_stats = gaps.groupby(hist.sort_values("order_ts")["user_id"]).agg(["mean", "std"])
    f["mean_gap_days"] = gap_stats["mean"]
    f["std_gap_days"] = gap_stats["std"]
    # Users with a single order have no cadence yet; fall back to tenure (conservative).
    f["mean_gap_days"] = f["mean_gap_days"].fillna(f["tenure_days"]).clip(lower=1)
    f["std_gap_days"] = f["std_gap_days"].fillna(0)
    f["overdue_ratio"] = f["recency_days"] / f["mean_gap_days"]

    # Experience quality.
    f["late_rate_90d"] = g90["is_late"].mean()
    f["late_orders_30d"] = w30["is_late"].sum()
    f["avg_delivery_min_90d"] = g90["delivery_minutes"].mean()
    last = hist.sort_values("order_ts").groupby("user_id").tail(1).set_index("user_id")
    f["last_order_late"] = last["is_late"].astype(float)

    # Price sensitivity & basket breadth.
    f["discount_share_90d"] = g90["discount_amount"].sum() / g90["gross_amount"].sum()
    f["avg_category_width_90d"] = g90["category_width"].mean()
    f["distinct_categories_90d"] = g90["primary_category"].nunique()
    f["weekend_share"] = g["order_ts"].apply(lambda s: (s.dt.dayofweek >= 5).mean())

    # Users with no orders in the 90d window (possible when active_window > 90).
    fill_zero = ["late_rate_90d", "late_orders_30d", "discount_share_90d", "distinct_categories_90d"]
    f[fill_zero] = f[fill_zero].fillna(0)
    f["avg_delivery_min_90d"] = f["avg_delivery_min_90d"].fillna(g["delivery_minutes"].mean())
    f["avg_category_width_90d"] = f["avg_category_width_90d"].fillna(g["category_width"].mean())

    u = users.set_index("user_id")
    for col in CATEGORICAL_FEATURES:
        f[col] = u[col].reindex(f.index).astype(str)
    f["signup_promo"] = u["signup_promo"].reindex(f.index).astype(float)

    f.index.name = "user_id"
    return f[FEATURES]


def build_labels(
    orders: pd.DataFrame, user_ids: pd.Index, as_of: pd.Timestamp, horizon_days: int
) -> pd.Series:
    """1 = churned (no order in the horizon), 0 = retained."""
    as_of = pd.Timestamp(as_of)
    future = orders[
        (orders["order_ts"] >= as_of)
        & (orders["order_ts"] < as_of + pd.Timedelta(days=horizon_days))
    ]
    returned = pd.Index(future["user_id"].unique())
    return pd.Series((~user_ids.isin(returned)).astype(int), index=user_ids, name="churned")


def snapshot_dates(orders: pd.DataFrame, cfg: ChurnConfig) -> list[pd.Timestamp]:
    """Snapshot dates whose label window is fully observed in the data."""
    first = orders["order_ts"].min().normalize() + pd.Timedelta(days=90)
    last_labelled = orders["order_ts"].max() - pd.Timedelta(days=cfg.horizon_days)
    return list(pd.date_range(first, last_labelled, freq=cfg.snapshot_freq))


def build_training_set(
    orders: pd.DataFrame, users: pd.DataFrame, cfg: ChurnConfig | None = None
) -> pd.DataFrame:
    """Stacked (snapshot, user) rows with features and label."""
    cfg = cfg or ChurnConfig()
    frames = []
    for as_of in snapshot_dates(orders, cfg):
        X = build_features(orders, users, as_of, cfg)
        if X.empty:
            continue
        y = build_labels(orders, X.index, as_of, cfg.horizon_days)
        frames.append(X.assign(churned=y, snapshot=as_of).reset_index())
    if not frames:
        raise ValueError("Not enough history to build any labelled snapshot.")
    return pd.concat(frames, ignore_index=True)
