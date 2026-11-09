"""Train, select, persist and score the churn model.

Protocol
--------
1. Build stacked monthly snapshots (see :mod:`features`).
2. **Out-of-time validation**: the most recent ``validation_snapshots`` snapshots are held
   out; candidates train on strictly earlier snapshots. Random K-fold would leak the
   future and inflate metrics.
3. Compare a regularised logistic-regression baseline against gradient boosting; pick the
   winner on PR-AUC (the business cares about ranking the churners at the top).
4. Refit the winner on *all* labelled snapshots and score the current active base.
5. Persist the pipeline + a JSON model card (metrics, curves, importances, config).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from retention_engine import __version__
from retention_engine.analytics.rfm import compute_rfm
from retention_engine.churn.evaluate import curves, metrics
from retention_engine.churn.explain import attach_reasons
from retention_engine.churn.features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    build_features,
    build_training_set,
)
from retention_engine.config import ChurnConfig, Settings, get_settings
from retention_engine.logging_utils import get_logger

log = get_logger(__name__)

CANDIDATES = ("logistic_regression", "gradient_boosting")


def make_pipeline(kind: str, seed: int = 42) -> Pipeline:
    if kind == "logistic_regression":
        pre = ColumnTransformer(
            [
                ("num", StandardScaler(), NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ]
        )
        est = LogisticRegression(C=0.5, max_iter=3000)
    elif kind == "gradient_boosting":
        pre = ColumnTransformer(
            [
                ("num", "passthrough", NUMERIC_FEATURES),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )
        est = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return Pipeline([("pre", pre), ("model", est)])


@dataclass
class TrainResult:
    pipeline: Pipeline
    card: dict[str, Any]


def _data_fingerprint(orders: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(str(len(orders)).encode())
    h.update(str(orders["order_ts"].min()).encode())
    h.update(str(orders["order_ts"].max()).encode())
    h.update(f"{orders['order_amount'].sum():.2f}".encode())
    return h.hexdigest()[:12]


def train(
    orders: pd.DataFrame,
    users: pd.DataFrame,
    cfg: ChurnConfig | None = None,
    seed: int = 42,
) -> TrainResult:
    cfg = cfg or ChurnConfig()
    data = build_training_set(orders, users, cfg)
    snaps = sorted(data["snapshot"].unique())
    if len(snaps) <= cfg.validation_snapshots:
        raise ValueError(f"Need more than {cfg.validation_snapshots} snapshots, got {len(snaps)}")
    cutoff = snaps[-cfg.validation_snapshots]
    tr, va = data[data["snapshot"] < cutoff], data[data["snapshot"] >= cutoff]
    log.info(
        "Training rows: %s (%s snapshots) | validation rows: %s (%s snapshots)",
        f"{len(tr):,}",
        tr["snapshot"].nunique(),
        f"{len(va):,}",
        va["snapshot"].nunique(),
    )

    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    preds: dict[str, np.ndarray] = {}
    for kind in CANDIDATES:
        pipe = make_pipeline(kind, seed).fit(tr[FEATURES], tr["churned"])
        p = pipe.predict_proba(va[FEATURES])[:, 1]
        results[kind] = metrics(va["churned"].to_numpy(), p)
        fitted[kind], preds[kind] = pipe, p
        log.info(
            "%-20s ROC-AUC %.3f | PR-AUC %.3f | Brier %.4f | lift@10%% %.2fx",
            kind,
            results[kind]["roc_auc"],
            results[kind]["pr_auc"],
            results[kind]["brier"],
            results[kind]["lift_at_top10pct"],
        )

    best = max(results, key=lambda k: results[k]["pr_auc"])
    baselines = _heuristic_baselines(va, preds[best])
    log.info("Selected %s; computing permutation importance", best)
    imp = permutation_importance(
        fitted[best],
        va[FEATURES],
        va["churned"],
        scoring="average_precision",
        n_repeats=5,
        random_state=seed,
        n_jobs=1,
    )
    importance = (
        pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False).round(5)
    )

    final = make_pipeline(best, seed).fit(data[FEATURES], data["churned"])
    card = {
        "model_name": "churn_classifier",
        "package_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_model": best,
        "churn_definition": (
            f"No order in the {cfg.horizon_days} days after the snapshot date, among users "
            f"who ordered in the prior {cfg.active_window_days} days "
            f"(min tenure {cfg.min_tenure_days} days)."
        ),
        "validation": "out-of-time: last "
        f"{cfg.validation_snapshots} monthly snapshots held out (train < {pd.Timestamp(cutoff).date()})",
        "data_fingerprint": _data_fingerprint(orders),
        "n_train_rows": len(tr),
        "n_validation_rows": len(va),
        "n_final_rows": len(data),
        "snapshots": [str(pd.Timestamp(s).date()) for s in snaps],
        "features": FEATURES,
        "metrics": {k: {m: round(v, 4) for m, v in r.items()} for k, r in results.items()},
        "baselines": baselines,
        "curves": curves(va["churned"].to_numpy(), preds[best]),
        "permutation_importance": importance.to_dict(),
        "thresholds": {"high": cfg.high_risk_threshold, "medium": cfg.medium_risk_threshold},
    }
    return TrainResult(pipeline=final, card=card)


def _heuristic_baselines(va: pd.DataFrame, model_p: np.ndarray) -> dict[str, dict[str, float]]:
    """What a rules-only approach achieves on the same validation rows.

    * ``recency_ranking`` – rank users purely by days since last order.
    * ``inactive_14d_rule`` – the original v0 rule: flag anyone inactive > 14 days.
    * ``model_at_same_budget`` – the model flagging the same number of users as that rule.
    """
    y = va["churned"].to_numpy()
    rec = va["recency_days"].to_numpy()
    ranking = {
        k: round(v, 4)
        for k, v in metrics(y, rec / (rec.max() + 1)).items()
        if k in {"roc_auc", "pr_auc", "lift_at_top10pct", "recall_at_top20pct"}
    }
    flagged = rec > 14
    rule = {
        "flagged_share": round(float(flagged.mean()), 4),
        "precision": round(float(y[flagged].mean()) if flagged.any() else 0.0, 4),
        "recall": round(float(y[flagged].sum()) / float(y.sum()), 4),
    }
    k = int(flagged.sum())
    top = np.argsort(-model_p)[:k]
    same_budget = {
        "flagged_share": rule["flagged_share"],
        "precision": round(float(y[top].mean()) if k else 0.0, 4),
        "recall": round(float(y[top].sum()) / float(y.sum()), 4),
    }
    return {
        "recency_ranking": ranking,
        "inactive_14d_rule": rule,
        "model_at_same_budget": same_budget,
    }


def save(result: TrainResult, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    settings.ensure_dirs()
    joblib.dump({"pipeline": result.pipeline, "features": FEATURES}, settings.model_path)
    settings.model_card_path.write_text(json.dumps(result.card, indent=2, default=str))
    log.info("Saved model → %s", settings.model_path)


def load(settings: Settings | None = None) -> tuple[Pipeline, dict[str, Any]]:
    settings = settings or get_settings()
    if not settings.model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {settings.model_path}. Run `retention-engine train` first."
        )
    bundle = joblib.load(settings.model_path)
    card = json.loads(Path(settings.model_card_path).read_text())
    return bundle["pipeline"], card


def risk_tier(p: pd.Series, cfg: ChurnConfig) -> pd.Series:
    return pd.Series(
        np.select(
            [p >= cfg.high_risk_threshold, p >= cfg.medium_risk_threshold],
            ["high", "medium"],
            default="low",
        ),
        index=p.index,
    )


def score(
    pipeline: Pipeline,
    orders: pd.DataFrame,
    users: pd.DataFrame,
    cfg: ChurnConfig | None = None,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Score the currently-active base. One row per user, sorted by revenue at risk."""
    cfg = cfg or ChurnConfig()
    as_of = as_of or (orders["order_ts"].max().normalize() + pd.Timedelta(days=1))
    X = build_features(orders, users, as_of, cfg)
    out = X.copy()
    out["churn_probability"] = pipeline.predict_proba(X[FEATURES])[:, 1]
    out["risk_tier"] = risk_tier(out["churn_probability"], cfg)
    # Expected next-90-day spend if retained ≈ trailing-90-day spend. Value at risk weights
    # it by churn probability, which is how the CRM queue is prioritised.
    out["expected_90d_value"] = out["gmv_90d"].round(2)
    out["value_at_risk"] = (out["churn_probability"] * out["expected_90d_value"]).round(2)

    rfm = compute_rfm(orders[orders["order_ts"] < as_of], as_of=as_of).set_index("user_id")
    out["rfm_segment"] = rfm["segment"].reindex(out.index)
    out["rfm_score"] = rfm["rfm_score"].reindex(out.index)
    out = attach_reasons(out)
    out["scored_at"] = as_of
    return out.sort_values("value_at_risk", ascending=False).reset_index()
