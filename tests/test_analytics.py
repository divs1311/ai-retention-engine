from __future__ import annotations

import numpy as np
import pytest

from conftest import make_orders
from retention_engine.analytics.cohorts import (
    average_retention_curve,
    category_breadth_retention,
    cohort_counts,
    retention_matrix,
)
from retention_engine.analytics.kpis import kpi_summary, weekly_trends
from retention_engine.analytics.rfm import SEGMENT_ORDER, _segment_for, compute_rfm, segment_summary


@pytest.fixture
def tiny():
    # Jan cohort: users 1-4. Feb cohort: users 5-6. Data ends mid-April (April is partial).
    return make_orders(
        [
            (1, "2025-01-03", 100), (2, "2025-01-05", 100), (3, "2025-01-09", 100), (4, "2025-01-20", 100),
            (1, "2025-02-02", 200), (2, "2025-02-10", 50),
            (1, "2025-03-01", 100),
            (5, "2025-02-04", 80), (6, "2025-02-07", 80),
            (5, "2025-03-15", 80),
            (1, "2025-04-10", 100), (6, "2025-04-12", 80),
        ]
    )  # fmt: skip


def test_user_retention_matrix_exact(tiny):
    m = retention_matrix(tiny)
    assert m.loc["2025-01"].tolist()[:3] == [100.0, 50.0, 25.0]
    assert m.loc["2025-02"].tolist()[:2] == [100.0, 50.0]


def test_partial_period_is_censored(tiny):
    m = retention_matrix(tiny)
    # April is incomplete → Jan M3 and Feb M2 are unknown, not 0%.
    assert np.isnan(m.loc["2025-01", 3])
    assert np.isnan(m.loc["2025-02", 2])
    with_partial = retention_matrix(tiny, include_partial=True)
    assert with_partial.loc["2025-01", 3] == 25.0


def test_observed_zero_is_zero_not_nan():
    o = make_orders(
        [
            (1, "2025-01-01", 10),
            (2, "2025-01-02", 10),
            (2, "2025-03-05", 10),
            (3, "2025-04-30 23:59", 1),
        ]
    )
    counts = cohort_counts(o)
    assert counts.loc["2025-01", 1] == 0  # nobody ordered in Feb, but Feb is observed


def test_revenue_retention(tiny):
    m = retention_matrix(tiny, "revenue")
    assert m.loc["2025-01", 1] == pytest.approx(250 / 400 * 100)


def test_average_curve_is_size_weighted(tiny):
    curve = average_retention_curve(tiny, min_cohorts=1)
    # M1: (2 + 1) retained / (4 + 2) users
    assert curve[1] == pytest.approx(50.0)


def test_breadth_buckets(dataset):
    _, orders = dataset
    out = category_breadth_retention(orders)
    assert out["users"].sum() > 0
    assert out["retention_pct"].between(0, 100).all()
    # Simulator encodes "broader baskets retain better" — analytics should recover it.
    assert out["retention_pct"].iloc[-1] > out["retention_pct"].iloc[0]


def test_rfm_segment_map():
    assert _segment_for("55") == "Champions"
    assert _segment_for("15") == "Can't Lose"
    assert _segment_for("11") == "Hibernating"
    assert _segment_for("51") == "New Customer"
    assert _segment_for("33") == "Needs Attention"
    every = {_segment_for(f"{r}{f}") for r in range(1, 6) for f in range(1, 6)}
    assert every <= set(SEGMENT_ORDER)


def test_rfm_scores_and_summary(dataset):
    _, orders = dataset
    rfm = compute_rfm(orders)
    for col in ("r_score", "f_score", "m_score"):
        assert rfm[col].between(1, 5).all()
    # Lower recency → better R score.
    assert rfm.groupby("r_score")["recency_days"].mean().is_monotonic_decreasing
    s = segment_summary(rfm)
    assert s["users"].sum() == len(rfm)
    assert s["revenue_share_pct"].sum() == pytest.approx(100, abs=0.5)


def test_kpis_and_trends(dataset):
    _, orders = dataset
    k = kpi_summary(orders)
    assert k.orders == len(orders)
    assert 0 < k.m3_retention_pct < k.m1_retention_pct < 100
    w = weekly_trends(orders)
    assert (w["new_users"] + w["returning_users"] == w["active_users"]).all()
