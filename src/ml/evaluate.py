"""Evaluation for a risk score on heavily imbalanced data.

Accuracy is useless here (predicting "never fraud" scores 99.8%), so everything
is expressed through precision, recall, and the alert volume an analyst team
would actually have to review.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_THRESHOLDS = (0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999)
DEFAULT_BUDGETS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02)


def _validate(y_true: ArrayLike, risk: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).ravel()
    r = np.asarray(risk, dtype=float).ravel()
    if y.shape != r.shape:
        raise ValueError(f"y_true has {y.size} rows but risk has {r.size}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 (normal) or 1 (fraud)")
    y = y.astype(int)
    if y.sum() in (0, y.size):
        raise ValueError("evaluation needs both normal and fraud rows")
    return y, r


def summary_metrics(y_true: ArrayLike, risk: ArrayLike) -> dict[str, float]:
    """Threshold-free metrics.

    pr_auc is average precision. Its baseline is the fraud rate, which is what a
    model that ranks at random would score, so read it relative to that.
    """
    y, r = _validate(y_true, risk)
    return {
        "pr_auc": float(average_precision_score(y, r)),
        "pr_auc_baseline": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, r)),
        "n": int(y.size),
        "n_fraud": int(y.sum()),
    }


def confusion_at(y_true: ArrayLike, risk: ArrayLike, threshold: float) -> dict[str, int]:
    """Confusion counts when every row with risk >= threshold is flagged."""
    y, r = _validate(y_true, risk)
    flagged = r >= threshold
    is_fraud = y == 1
    return {
        "tp": int((flagged & is_fraud).sum()),
        "fp": int((flagged & ~is_fraud).sum()),
        "fn": int((~flagged & is_fraud).sum()),
        "tn": int((~flagged & ~is_fraud).sum()),
    }


def threshold_table(
    y_true: ArrayLike,
    risk: ArrayLike,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    amounts: ArrayLike | None = None,
) -> pd.DataFrame:
    """Precision, recall and alert volume at each candidate threshold.

    Pass transaction amounts to also report the share of fraud *money* caught,
    which is what the business loses, not just the share of fraud *cases*.
    """
    y, r = _validate(y_true, risk)
    amt = None if amounts is None else np.asarray(amounts, dtype=float).ravel()
    if amt is not None and amt.shape != y.shape:
        raise ValueError(f"amounts has {amt.size} rows but y_true has {y.size}")

    rows = []
    for threshold in thresholds:
        c = confusion_at(y, r, threshold)
        flagged = c["tp"] + c["fp"]
        precision = c["tp"] / flagged if flagged else np.nan
        recall = c["tp"] / (c["tp"] + c["fn"])
        row = {
            "threshold": threshold,
            "flagged": flagged,
            "flagged_%": 100 * flagged / y.size,
            "tp": c["tp"],
            "fp": c["fp"],
            "fn": c["fn"],
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if c["tp"] else 0.0,
            "normal_flagged_%": 100 * c["fp"] / (c["fp"] + c["tn"]),
            "alerts_per_fraud": flagged / c["tp"] if c["tp"] else np.inf,
        }
        if amt is not None:
            fraud_total = amt[y == 1].sum()
            caught = amt[(r >= threshold) & (y == 1)].sum()
            row["fraud_amount_caught_%"] = 100 * caught / fraud_total if fraud_total else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_pr_auc(
    y_true: ArrayLike,
    risk: ArrayLike,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Confidence interval for PR-AUC by resampling the evaluation rows.

    Fraud and normal rows are resampled separately, so every resample keeps the
    same number of fraud cases. With under a hundred of them, one PR-AUC number
    hides a lot of noise; the interval shows how much.
    """
    y, r = _validate(y_true, risk)
    rng = np.random.default_rng(seed)
    fraud_rows = np.flatnonzero(y == 1)
    normal_rows = np.flatnonzero(y == 0)

    scores = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = np.concatenate(
            [
                rng.choice(fraud_rows, fraud_rows.size),
                rng.choice(normal_rows, normal_rows.size),
            ]
        )
        scores[i] = average_precision_score(y[sample], r[sample])

    tail = (1 - confidence) / 2
    low, high = np.quantile(scores, [tail, 1 - tail])
    return float(low), float(high)


def budget_table(
    y_true: ArrayLike, scores: ArrayLike, budgets: Iterable[float] = DEFAULT_BUDGETS
) -> pd.DataFrame:
    """Precision and recall when only the top `budget` share of rows is alerted.

    Comparing models at the same alert volume is fair even when their scores
    live on different scales (a percentile risk vs a predicted probability).
    Ties are broken by row order.
    """
    y, s = _validate(y_true, scores)
    order = np.argsort(-s, kind="stable")
    caught_within_top = np.cumsum(y[order])
    total_fraud = y.sum()

    rows = []
    for budget in budgets:
        alerts = int(round(budget * y.size))
        tp = int(caught_within_top[alerts - 1]) if alerts else 0
        rows.append(
            {
                "budget_%": 100 * budget,
                "alerts": alerts,
                "tp": tp,
                "precision": tp / alerts if alerts else np.nan,
                "recall": tp / total_fraud,
                "alerts_per_fraud": alerts / tp if tp else np.inf,
            }
        )
    return pd.DataFrame(rows)
