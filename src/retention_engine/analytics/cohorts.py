"""Cohort retention analytics.

Fixes versus the original notebook-style script:

* **Right-censoring** – cells for calendar periods that haven't fully happened yet are
  masked (NaN) instead of being shown as artificially low retention.
* **Multiple metrics** – user retention, order retention and *revenue* retention (the
  latter can exceed 100% when surviving users expand their baskets).
* **Size-weighted average curve** – the headline M1/M3/M6 numbers weight each cohort by
  its size and only use observed cells.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Metric = Literal["users", "orders", "revenue"]
Freq = Literal["M", "W"]


def assign_cohorts(orders: pd.DataFrame, freq: Freq = "M") -> pd.DataFrame:
    """Add ``cohort`` (first-order period), ``period`` and ``cohort_index`` columns."""
    out = orders.copy()
    out["period"] = out["order_ts"].dt.to_period(freq)
    out["cohort"] = out.groupby("user_id")["period"].transform("min")
    out["cohort_index"] = _period_diff(out["period"], out["cohort"])
    return out


def _period_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.astype("int64") - b.astype("int64")).astype(int)


def cohort_counts(
    orders: pd.DataFrame,
    metric: Metric = "users",
    freq: Freq = "M",
    *,
    include_partial: bool = False,
) -> pd.DataFrame:
    """Absolute cohort matrix: rows = acquisition cohort, columns = periods since."""
    df = assign_cohorts(orders, freq)
    grouped = df.groupby(["cohort", "cohort_index"])
    if metric == "users":
        values = grouped["user_id"].nunique()
    elif metric == "orders":
        values = grouped.size()
    elif metric == "revenue":
        values = grouped["order_amount"].sum()
    else:  # pragma: no cover - guarded by Literal
        raise ValueError(f"unknown metric {metric!r}")

    matrix = values.unstack("cohort_index").sort_index()
    matrix = matrix.reindex(columns=range(int(matrix.columns.max()) + 1))

    # Mask cells that are in the future (NaN) or in the still-running last period.
    last_ts = orders["order_ts"].max()
    last_period = last_ts.to_period(freq)
    period_complete = last_ts >= (last_period + 1).start_time - pd.Timedelta(minutes=1)
    last_observed = last_period if (include_partial or period_complete) else last_period - 1
    cohort_ord = matrix.index.astype("int64").to_numpy()[:, None]
    cell_period = cohort_ord + matrix.columns.to_numpy()[None, :]
    observable = cell_period <= last_observed.ordinal
    # Observable cells with no activity are genuine zeros; unobservable ones stay NaN.
    matrix = matrix.fillna(0).where(observable)
    # Cohorts acquired in a still-running period are dropped entirely.
    matrix = matrix[matrix[0].notna()]
    matrix.index = matrix.index.astype(str)
    matrix.index.name = "cohort"
    matrix.columns.name = "periods_since_acquisition"
    return matrix


def retention_matrix(
    orders: pd.DataFrame,
    metric: Metric = "users",
    freq: Freq = "M",
    *,
    include_partial: bool = False,
) -> pd.DataFrame:
    """Cohort matrix normalised by period 0, in percent."""
    counts = cohort_counts(orders, metric, freq, include_partial=include_partial)
    return counts.div(counts[0], axis=0) * 100


def cohort_sizes(orders: pd.DataFrame, freq: Freq = "M") -> pd.Series:
    return cohort_counts(orders, "users", freq)[0].rename("cohort_size")


def average_retention_curve(
    orders: pd.DataFrame,
    metric: Metric = "users",
    freq: Freq = "M",
    *,
    min_cohorts: int = 2,
) -> pd.Series:
    """Cohort-size-weighted retention per period, using observed cells only."""
    counts = cohort_counts(orders, metric, freq)
    base = counts[0]
    curve = {}
    for k in counts.columns:
        observed = counts[k].notna()
        if observed.sum() < min_cohorts:
            break
        curve[k] = counts.loc[observed, k].sum() / base[observed].sum() * 100
    return pd.Series(curve, name=f"{metric}_retention_pct")


def category_breadth_retention(
    orders: pd.DataFrame,
    *,
    onboarding_days: int = 30,
    outcome_start_day: int = 60,
    outcome_end_day: int = 90,
) -> pd.DataFrame:
    """Does basket breadth in a user's first month predict whether they're still around?

    For users with at least ``outcome_end_day`` days of observable history, bucket their
    average category width over the first ``onboarding_days`` and measure the share that
    ordered between days ``outcome_start_day`` and ``outcome_end_day``.
    """
    df = orders[["user_id", "order_ts", "category_width"]].copy()
    first = df.groupby("user_id")["order_ts"].transform("min")
    df["day"] = (df["order_ts"] - first).dt.days
    horizon_ok = first <= orders["order_ts"].max() - pd.Timedelta(days=outcome_end_day)
    df = df[horizon_ok]

    onboarding = df[df["day"] < onboarding_days].groupby("user_id")["category_width"].mean()
    retained = (
        df[(df["day"] >= outcome_start_day) & (df["day"] < outcome_end_day)]
        .groupby("user_id")
        .size()
        .gt(0)
    )
    users = onboarding.to_frame("avg_category_width")
    users["retained"] = retained.reindex(users.index, fill_value=False)
    users["breadth_bucket"] = pd.cut(
        users["avg_category_width"],
        bins=[0, 2, 4, 6, np.inf],
        labels=["1–2", "3–4", "5–6", "7+"],
        right=True,
    )
    out = (
        users.groupby("breadth_bucket", observed=True)
        .agg(users=("retained", "size"), retention_pct=("retained", "mean"))
        .reset_index()
    )
    out["retention_pct"] = (out["retention_pct"] * 100).round(1)
    baseline = users["retained"].mean() * 100
    out["lift_vs_avg"] = (out["retention_pct"] / baseline).round(2)
    return out
