"""Choose the review and block thresholds by expected cost.

Every transaction gets one of three decisions:

    risk >= block_threshold                 -> block
    review_threshold <= risk < block        -> review by an analyst
    risk < review_threshold                 -> approve

Each decision has a price:

    review   costs `review_cost` per transaction, fraud or not
    block    costs `false_block_cost` for every legitimate customer blocked
    approve  costs the amount + `chargeback_fee` for every fraud let through

Reviewed and blocked fraud are assumed stopped. The chosen thresholds are the
cheapest pair that stays within analyst capacity. A block threshold of None
means nothing is ever blocked automatically.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from src.config import settings

# Grid for the Isolation Forest's percentile risk (Step 1.6).
CANDIDATE_THRESHOLDS = tuple(float(t) for t in np.round(np.arange(0.900, 0.9995, 0.001), 3))
# Grid for XGBoost's fraud probability: 0.001 steps below 0.01, where review
# thresholds tend to land, then 0.01 steps up to 0.99.
PROBABILITY_THRESHOLDS = tuple(
    float(t)
    for t in np.unique(np.round(np.r_[np.arange(1, 10) / 1000, np.arange(1, 100) / 100], 3))
)


@dataclass(frozen=True)
class Costs:
    review_cost: float
    false_block_cost: float
    chargeback_fee: float


@dataclass(frozen=True)
class ThresholdChoice:
    review_threshold: float
    block_threshold: float | None
    cost_per_100k: float
    review_rate: float
    block_rate: float


def costs_from_settings() -> Costs:
    return Costs(
        review_cost=settings.review_cost,
        false_block_cost=settings.false_block_cost,
        chargeback_fee=settings.chargeback_fee,
    )


def _arrays(
    y_true: ArrayLike, risk: ArrayLike, amounts: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true).ravel()
    r = np.asarray(risk, dtype=float).ravel()
    a = np.asarray(amounts, dtype=float).ravel()
    if not y.shape == r.shape == a.shape:
        raise ValueError("y_true, risk and amounts must have the same length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 (normal) or 1 (fraud)")
    if not (y == 1).any():
        raise ValueError("need at least one fraud row to price missed fraud")
    return y.astype(int), r, a


def approve_all_cost_per_100k(y_true: ArrayLike, amounts: ArrayLike, costs: Costs) -> float:
    """What fraud costs with no model at all: every transaction approved."""
    y, _, a = _arrays(y_true, np.zeros(np.size(y_true)), amounts)
    return float((a[y == 1] + costs.chargeback_fee).sum() * 100_000 / y.size)


def _count_at_or_above(sorted_values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return sorted_values.size - np.searchsorted(sorted_values, thresholds, side="left")


def policy_costs(
    y_true: ArrayLike,
    risk: ArrayLike,
    amounts: ArrayLike,
    costs: Costs,
    review_candidates: Sequence[float] = CANDIDATE_THRESHOLDS,
    block_candidates: Sequence[float | None] = (*CANDIDATE_THRESHOLDS, None),
) -> pd.DataFrame:
    """Cost of every (review, block) threshold pair, per 100k transactions.

    Works from sorted scores and running totals, so all ~10,000 pairs are
    priced in one vectorised pass instead of re-masking the data 10,000 times.
    """
    y, r, a = _arrays(y_true, risk, amounts)
    is_fraud = y == 1

    normal_sorted = np.sort(r[~is_fraud])
    fraud_order = np.argsort(r[is_fraud])
    fraud_sorted = r[is_fraud][fraud_order]
    fraud_loss = a[is_fraud][fraud_order] + costs.chargeback_fee
    # loss_from[i] = total loss of the fraud rows at sorted position i and above.
    loss_from = np.append(np.cumsum(fraud_loss[::-1])[::-1], 0.0)

    review = np.asarray(review_candidates, dtype=float)
    block = np.array([np.inf if t is None else t for t in block_candidates], dtype=float)
    review_grid, block_grid = np.meshgrid(review, block, indexing="ij")
    valid = block_grid >= review_grid
    review_grid, block_grid = review_grid[valid], block_grid[valid]

    normal_at_review = _count_at_or_above(normal_sorted, review_grid)
    fraud_at_review = _count_at_or_above(fraud_sorted, review_grid)
    normal_at_block = _count_at_or_above(normal_sorted, block_grid)
    fraud_at_block = _count_at_or_above(fraud_sorted, block_grid)

    reviewed = (normal_at_review + fraud_at_review) - (normal_at_block + fraud_at_block)
    stopped_loss = loss_from[fraud_sorted.size - fraud_at_review]
    missed_loss = loss_from[0] - stopped_loss

    n = y.size
    per_100k = 100_000 / n
    table = pd.DataFrame(
        {
            "review_threshold": review_grid,
            "block_threshold": np.where(np.isinf(block_grid), np.nan, block_grid),
            "review_rate": reviewed / n,
            "block_rate": (normal_at_block + fraud_at_block) / n,
            "fraud_stopped": fraud_at_review / fraud_sorted.size,
            "fraud_loss_stopped": stopped_loss / loss_from[0],
            "review_cost": costs.review_cost * reviewed * per_100k,
            "block_cost": costs.false_block_cost * normal_at_block * per_100k,
            "missed_fraud_cost": missed_loss * per_100k,
        }
    )
    table["cost_per_100k"] = table["review_cost"] + table["block_cost"] + table["missed_fraud_cost"]
    return table


def choose_thresholds(
    costs_table: pd.DataFrame, max_review_rate: float, tolerance: float = 0.02
) -> ThresholdChoice:
    """Cheapest policy within analyst capacity, breaking near-ties toward less friction.

    Costs within `tolerance` of the minimum count as a tie: with a few hundred
    fraud cases, differences that small are noise. Among tied policies, the one
    that blocks the fewest customers wins, then the one that reviews the fewest.
    """
    feasible = costs_table[costs_table["review_rate"] <= max_review_rate]
    if feasible.empty:
        raise ValueError(f"no policy reviews at most {max_review_rate:.2%} of transactions")
    cheapest = feasible["cost_per_100k"].min()
    tied = feasible[feasible["cost_per_100k"] <= cheapest * (1 + tolerance)]
    best = tied.sort_values(["block_rate", "review_rate", "cost_per_100k"]).iloc[0]
    return ThresholdChoice(
        review_threshold=float(best["review_threshold"]),
        block_threshold=(
            None if np.isnan(best["block_threshold"]) else float(best["block_threshold"])
        ),
        cost_per_100k=float(best["cost_per_100k"]),
        review_rate=float(best["review_rate"]),
        block_rate=float(best["block_rate"]),
    )
