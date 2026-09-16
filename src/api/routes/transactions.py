"""GET /transactions and GET /transactions/{id}: decisions already made (Step 3.2)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from src.api.dependencies import require_store
from src.api.schemas import Decision, DecisionList, StoredDecision
from src.storage.store import DecisionStore

MAX_LIST_LIMIT = 500

router = APIRouter(prefix="/transactions", tags=["transactions"])

Store = Annotated[DecisionStore, Depends(require_store)]


@router.get("", response_model=DecisionList)
def list_transactions(
    store: Store,
    decision: Decision | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LIST_LIMIT)] = 50,
) -> DecisionList:
    """The most recent decisions, newest first. `?decision=block` for blocks only."""
    records = store.recent(decision, limit)
    return DecisionList(decisions=[StoredDecision.model_validate(r) for r in records])


@router.get("/{transaction_id}", response_model=StoredDecision)
def get_transaction(
    transaction_id: Annotated[str, Path(min_length=1, max_length=64)], store: Store
) -> StoredDecision:
    """The recorded decision for one transaction."""
    record = store.get(transaction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="no decision recorded for this transaction")
    return StoredDecision.model_validate(record)
