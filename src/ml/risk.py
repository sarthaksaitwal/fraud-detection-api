"""Turn raw Isolation Forest scores into a 0-1 risk score.

risk(x) = share of normal reference transactions that look *more normal* than x.

- Higher means more suspicious, and the value is always within [0, 1].
- It never changes the ranking, so precision, recall and PR-AUC are unaffected.
- It has a direct operational meaning: flagging every transaction with
  risk >= t flags about (1 - t) of normal traffic. A threshold becomes an alert
  budget ("review 0.5% of transactions") instead of a magic number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.pipeline import Pipeline


def _as_finite_array(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


@dataclass(frozen=True, eq=False)
class RiskCalibrator:
    """Maps raw scores to risk using the sorted scores of normal transactions."""

    reference_scores: np.ndarray

    @classmethod
    def fit(cls, normal_scores: ArrayLike) -> RiskCalibrator:
        scores = np.sort(_as_finite_array(normal_scores, "normal_scores").ravel())
        if scores.size == 0:
            raise ValueError("need at least one reference score")
        return cls(reference_scores=scores)

    def transform(self, raw_scores: ArrayLike) -> np.ndarray:
        # A NaN would sort past every reference score and read as risk 0, i.e.
        # "safe". For a fraud model that failure mode is unacceptable, so refuse.
        scores = _as_finite_array(raw_scores, "raw_scores")
        n = self.reference_scores.size
        at_or_below = np.searchsorted(self.reference_scores, scores, side="right")
        return (n - at_or_below) / n


@dataclass(frozen=True, eq=False)
class RiskModel:
    """The fitted pipeline plus its calibrator: raw transactions in, risk out."""

    pipeline: Pipeline
    calibrator: RiskCalibrator

    def raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Isolation Forest score_samples: lower means more anomalous."""
        return self.pipeline.score_samples(X)

    def risk(self, X: pd.DataFrame) -> np.ndarray:
        """0-1 risk score: higher means more suspicious."""
        return self.calibrator.transform(self.raw_scores(X))
