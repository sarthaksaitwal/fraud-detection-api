"""POST /score and POST /score/batch: risk scores and decisions."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import get_scorer
from src.api.schemas import BatchResult, RiskResult, Transaction, TransactionBatch
from src.scoring import Scorer

router = APIRouter(tags=["scoring"])


# Plain `def`, not `async def`: scoring is CPU work, so FastAPI runs it in a
# worker thread and the server keeps accepting requests meanwhile.
@router.post("/score", response_model=RiskResult)
def score(transaction: Transaction, scorer: Annotated[Scorer, Depends(get_scorer)]) -> RiskResult:
    return scorer.score_one(transaction)


@router.post("/score/batch", response_model=BatchResult)
def score_batch(
    batch: TransactionBatch, scorer: Annotated[Scorer, Depends(get_scorer)]
) -> BatchResult:
    """Score up to 1,000 transactions in one model call. Results keep the request order.

    All or nothing: if any transaction is invalid, the whole batch is a 422.
    """
    return BatchResult(results=scorer.score(batch.transactions))
