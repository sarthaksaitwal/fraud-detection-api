"""Turn transactions into risk scores and decisions (Step 2.2).

This is the only code that applies the model and the review/block thresholds.
The API (Phase 2) and the Kafka consumer (Phase 4) both call it, so a
transaction gets the same decision whichever way it arrives.

    risk >= block_threshold                 -> block
    review_threshold <= risk < block        -> review
    risk < review_threshold                 -> approve

The rule matches src/ml/thresholds.py, which priced these thresholds.

From Phase 5 an approved transaction can still be sent for review by the
velocity rules in src/features/rules.py, on what the card has been doing rather
than on this transaction. The result keeps both answers: `model_decision` is
what the model said, `decision` is what the service did.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.api.schemas import Decision, RiskResult, Transaction, VelocityFeatures
from src.config import settings
from src.features.rules import escalate, reasons_for
from src.ml.artifact import load_model
from src.ml.model import fraud_probability
from src.ml.preprocess import RAW_FEATURES

log = logging.getLogger("src.scoring")

# Indexed by how many thresholds a score reaches: none, review only, review and block.
DECISIONS_BY_LEVEL = (Decision.APPROVE, Decision.REVIEW, Decision.BLOCK)


def decide(
    risk: np.ndarray, review_threshold: float, block_threshold: float | None
) -> list[Decision]:
    """The decision for each risk score. Assumes block_threshold >= review_threshold."""
    # float64, as in src/ml/thresholds.py, so a score exactly at a threshold is
    # decided the same way here as when the threshold was chosen.
    risk = np.asarray(risk, dtype=float)
    block = np.inf if block_threshold is None else block_threshold
    levels = (risk >= review_threshold).astype(int) + (risk >= block)
    return [DECISIONS_BY_LEVEL[level] for level in levels.tolist()]


@dataclass(frozen=True)
class Scorer:
    """A loaded model plus the thresholds that turn its scores into decisions."""

    pipeline: Any
    model_version: str
    review_threshold: float
    block_threshold: float | None

    def __post_init__(self) -> None:
        if not 0 <= self.review_threshold <= 1:
            raise ValueError(f"review_threshold must be in [0, 1], got {self.review_threshold}")
        if self.block_threshold is not None and not (
            self.review_threshold <= self.block_threshold <= 1
        ):
            raise ValueError(
                f"block_threshold must be in [review_threshold, 1] or None, "
                f"got {self.block_threshold}"
            )

    @classmethod
    def load(
        cls,
        model_path: Path | None = None,
        metadata_path: Path | None = None,
        review_threshold: float | None = None,
        block_threshold: float | None = None,
    ) -> Scorer:
        """Load the saved model. Thresholds default to the ones in settings (.env)."""
        pipeline, metadata = load_model(model_path, metadata_path)
        review = settings.review_threshold if review_threshold is None else review_threshold
        block = settings.block_threshold if block_threshold is None else block_threshold
        _warn_if_thresholds_differ(metadata, review, block)
        scorer = cls(pipeline, metadata["model_version"], review, block)
        log.info(
            "loaded model %s: review >= %s, block >= %s",
            scorer.model_version,
            scorer.review_threshold,
            scorer.block_threshold,
        )
        return scorer

    def score(
        self,
        transactions: Sequence[Transaction],
        velocity: Sequence[VelocityFeatures | None] | None = None,
    ) -> list[RiskResult]:
        """Score many transactions with one model call. Results keep the input order.

        velocity: what each card had done recently (Step 5.4), measured before
        scoring. The model never sees it -- it was trained without it -- but the
        velocity rules (Step 5.5) can send an approved transaction for review on
        the strength of it, and the result says so.
        """
        if not transactions:
            return []
        X = pd.DataFrame([t.features() for t in transactions], columns=RAW_FEATURES)
        risk = fraud_probability(self.pipeline, X).astype(float)
        decisions = decide(risk, self.review_threshold, self.block_threshold)
        features = [None] * len(transactions) if velocity is None else list(velocity)
        results = []
        for transaction, score, decision, measured in zip(
            transactions, risk.tolist(), decisions, features, strict=True
        ):
            reasons = reasons_for(measured)
            results.append(
                RiskResult(
                    transaction_id=transaction.transaction_id,
                    risk_score=score,
                    decision=escalate(decision, reasons),
                    review_threshold=self.review_threshold,
                    block_threshold=self.block_threshold,
                    model_version=self.model_version,
                    card_id=transaction.card_id,
                    velocity=measured,
                    model_decision=decision,
                    reasons=reasons,
                )
            )
        return results

    def score_one(
        self, transaction: Transaction, velocity: VelocityFeatures | None = None
    ) -> RiskResult:
        return self.score([transaction], [velocity])[0]


def _warn_if_thresholds_differ(
    metadata: dict[str, Any], review: float, block: float | None
) -> None:
    trained = metadata.get("decision_thresholds", {})
    if (trained.get("review"), trained.get("block")) != (review, block):
        log.warning(
            "serving with review=%s, block=%s but model %s was trained with review=%s, "
            "block=%s; its thresholds were priced for the trained values",
            review,
            block,
            metadata.get("model_version"),
            trained.get("review"),
            trained.get("block"),
        )
