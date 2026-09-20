"""POST /score and POST /score/batch: risk scores and decisions."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import get_scorer, get_store, get_velocity
from src.api.schemas import BatchResult, RiskResult, Transaction, TransactionBatch, VelocityFeatures
from src.features.velocity import VelocityStore
from src.scoring import Scorer
from src.storage.store import DecisionStore

router = APIRouter(tags=["scoring"])

Store = Annotated[DecisionStore | None, Depends(get_store)]
Velocity = Annotated[VelocityStore | None, Depends(get_velocity)]


def measure(
    velocity: VelocityStore | None, transactions: list[Transaction]
) -> list[VelocityFeatures | None]:
    """What each card has done recently, or no features at all when velocity is off."""
    if velocity is None:
        return [None] * len(transactions)
    return velocity.measure_or_none(transactions)


# Plain `def`, not `async def`: scoring is CPU work, so FastAPI runs it in a
# worker thread and the server keeps accepting requests meanwhile.
@router.post("/score", response_model=RiskResult)
def score(
    transaction: Transaction,
    scorer: Annotated[Scorer, Depends(get_scorer)],
    store: Store,
    velocity: Velocity,
) -> RiskResult:
    # Velocity is measured first: it describes the card as of this transaction,
    # and recording it in Redis is part of measuring it.
    (features,) = measure(velocity, [transaction])
    result = scorer.score_one(transaction, features)
    if store is not None:
        store.record([result])
    return result


@router.post("/score/batch", response_model=BatchResult)
def score_batch(
    batch: TransactionBatch,
    scorer: Annotated[Scorer, Depends(get_scorer)],
    store: Store,
    velocity: Velocity,
) -> BatchResult:
    """Score up to 1,000 transactions in one model call. Results keep the request order.

    All or nothing: if any transaction is invalid, the whole batch is a 422.
    """
    results = scorer.score(batch.transactions, measure(velocity, batch.transactions))
    if store is not None:
        store.record(results)
    return BatchResult(results=results)
