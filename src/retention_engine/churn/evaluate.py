"""Model evaluation: ranking quality, calibration and business lift."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def gains_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile table: decile 1 = highest predicted risk."""
    df = pd.DataFrame({"y": y, "p": p}).sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, n_bins, labels=range(1, n_bins + 1)).astype(int)
    t = df.groupby("decile").agg(users=("y", "size"), churners=("y", "sum"), avg_p=("p", "mean"))
    t["churn_rate"] = t["churners"] / t["users"]
    t["lift"] = t["churn_rate"] / y.mean()
    t["cum_capture"] = t["churners"].cumsum() / y.sum()
    return t.reset_index()


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y)
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    order = np.argsort(-p)
    top10 = order[: max(1, len(p) // 10)]
    top20 = order[: max(1, len(p) // 5)]
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "base_rate": float(y.mean()),
        "precision_at_top10pct": float(y[top10].mean()),
        "lift_at_top10pct": float(y[top10].mean() / y.mean()),
        "recall_at_top20pct": float(y[top20].sum()) / float(y.sum()),
        "n": len(y),
    }


def curves(y: np.ndarray, p: np.ndarray, n_points: int = 60) -> dict[str, dict[str, list[Any]]]:
    """Downsampled curves for plotting in the dashboard (kept small for the model card)."""

    def _thin(*arrays: np.ndarray) -> list[list[float]]:
        idx = np.unique(np.linspace(0, len(arrays[0]) - 1, n_points).astype(int))
        return [np.round(a[idx], 4).tolist() for a in arrays]

    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    fpr_t, tpr_t = _thin(fpr, tpr)
    rec_t, prec_t = _thin(rec[::-1], prec[::-1])

    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    calib_pred, calib_obs, calib_n = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum() >= 20:
            calib_pred.append(round(float(p[m].mean()), 4))
            calib_obs.append(round(float(y[m].mean()), 4))
            calib_n.append(int(m.sum()))

    g = gains_table(y, p)
    return {
        "roc": {"fpr": fpr_t, "tpr": tpr_t},
        "pr": {"recall": rec_t, "precision": prec_t},
        "calibration": {"predicted": calib_pred, "observed": calib_obs, "n": calib_n},
        "gains": {
            "decile": g["decile"].tolist(),
            "churn_rate": g["churn_rate"].round(4).tolist(),
            "lift": g["lift"].round(3).tolist(),
            "cum_capture": g["cum_capture"].round(4).tolist(),
        },
    }
