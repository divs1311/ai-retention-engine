from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_orders, make_users
from retention_engine.churn import model as churn_model
from retention_engine.churn.evaluate import expected_calibration_error, gains_table, metrics
from retention_engine.churn.explain import reasons_for
from retention_engine.churn.features import FEATURES, build_features, build_labels
from retention_engine.config import ChurnConfig


def test_features_ignore_the_future(dataset):
    """Changing or deleting anything at/after the snapshot must not change features."""
    users, orders = dataset
    as_of = pd.Timestamp("2025-10-01")
    base = build_features(orders, users, as_of)

    future = orders["order_ts"] >= as_of
    tampered = orders.copy()
    tampered.loc[future, "order_amount"] *= 10
    tampered.loc[future, "is_late"] = True
    pd.testing.assert_frame_equal(base, build_features(tampered, users, as_of))
    pd.testing.assert_frame_equal(base, build_features(orders[~future], users, as_of))


def test_population_is_currently_active_with_min_tenure(dataset):
    users, orders = dataset
    cfg = ChurnConfig()
    X = build_features(orders, users, pd.Timestamp("2025-10-01"), cfg)
    assert (X["recency_days"] <= cfg.active_window_days).all()
    assert (X["tenure_days"] >= cfg.min_tenure_days).all()
    assert list(X.columns) == FEATURES
    assert not X[FEATURES[:-3]].isna().any().any()


def test_labels_on_hand_built_data():
    o = make_orders(
        [
            (1, "2025-03-01", 100),
            (1, "2025-03-20", 100),
            (2, "2025-03-05", 100),
            (2, "2025-04-25", 100),
        ]
    )
    as_of = pd.Timestamp("2025-04-01")
    y = build_labels(o, pd.Index([1, 2]), as_of, horizon_days=30)
    assert y.to_dict() == {1: 1, 2: 0}  # user 1 never returns, user 2 returns on day 24


def test_hand_built_features():
    o = make_orders([(1, "2025-03-01", 100), (1, "2025-03-11", 200), (1, "2025-03-21", 300)])
    X = build_features(o, make_users([1]), pd.Timestamp("2025-03-31"))
    row = X.loc[1]
    assert row["orders_30d"] == 3 and row["gmv_30d"] == 600
    assert row["mean_gap_days"] == pytest.approx(10)
    assert row["recency_days"] == pytest.approx(10)
    assert row["overdue_ratio"] == pytest.approx(1.0)


def test_metrics_helpers():
    rng = np.random.default_rng(0)
    p = rng.random(2000)
    y = (rng.random(2000) < p).astype(int)  # perfectly calibrated by construction
    m = metrics(y, p)
    assert m["roc_auc"] > 0.7
    assert m["ece"] < 0.05
    assert expected_calibration_error(y, np.clip(p + 0.3, 0, 1)) > 0.15
    g = gains_table(y, p)
    assert g["churn_rate"].iloc[0] > g["churn_rate"].iloc[-1]
    assert g["cum_capture"].iloc[-1] == pytest.approx(1.0)


def test_model_learns_signal_out_of_time(trained):
    card = trained.card
    best = card["metrics"][card["selected_model"]]
    assert card["selected_model"] in churn_model.CANDIDATES
    assert best["roc_auc"] > 0.75
    assert best["lift_at_top10pct"] > 1.5
    assert best["ece"] < 0.08
    # The model should not be worse than ranking by recency alone.
    assert best["pr_auc"] >= card["baselines"]["recency_ranking"]["pr_auc"] - 0.01
    # Validation snapshots must be strictly later than training snapshots.
    assert card["n_train_rows"] + card["n_validation_rows"] == card["n_final_rows"]


def test_save_load_and_score(trained, dataset, settings):
    users, orders = dataset
    churn_model.save(trained, settings)
    pipe, card = churn_model.load(settings)
    assert card["selected_model"] == trained.card["selected_model"]
    scores = churn_model.score(pipe, orders, users, settings.churn)
    assert scores["churn_probability"].between(0, 1).all()
    assert scores["value_at_risk"].is_monotonic_decreasing
    assert set(scores["risk_tier"]) <= {"low", "medium", "high"}
    assert scores["reason_text"].str.len().gt(0).all()
    high = scores[scores["risk_tier"] == "high"]["churn_probability"]
    assert (high >= settings.churn.high_risk_threshold).all()


def test_reason_codes_are_ranked_and_readable():
    row = pd.Series(
        {
            "overdue_ratio": 3.0, "recency_days": 21, "mean_gap_days": 7, "freq_trend": 0.3,
            "orders_90d": 6, "late_orders_30d": 2, "late_rate_90d": 0.4,
            "avg_category_width_90d": 1.5, "tenure_days": 200, "signup_promo": 0,
            "discount_share_90d": 0.0,
        }
    )  # fmt: skip
    codes = [r.code for r in reasons_for(row)]
    assert set(codes) == {"OVERDUE", "FREQ_DECLINE", "LATE_DELIVERY", "NARROW_BASKET"}
    sev = [r.severity for r in reasons_for(row)]
    assert sev == sorted(sev, reverse=True)
