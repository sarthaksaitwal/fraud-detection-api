"""Request and response bodies for the scoring API (Step 2.1).

A request carries one transaction in exactly the columns the model was trained
on: Time, V1-V28 and Amount. Pydantic validates it before any scoring code
runs, so the model never sees a missing or misspelled column, text, NaN or
infinity. Invalid requests get a 422 response that names the bad field.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.ml.preprocess import RAW_FEATURES

MAX_BATCH_SIZE = 1000

# Card ids, merchants and countries: letters, digits and a few separators, up to
# 64 characters. Anything else is a mistake, and these strings become key names.
ENTITY_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$"


def new_transaction_id() -> str:
    return uuid4().hex


class Decision(str, Enum):
    APPROVE = "approve"
    REVIEW = "review"
    BLOCK = "block"


class Transaction(BaseModel):
    """One card transaction, in the columns of the Kaggle dataset."""

    # extra="forbid": an unknown field (a typo like "v14", or the "Class" label) is an error.
    # allow_inf_nan=False: NaN and infinity are rejected instead of reaching the model.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    transaction_id: str = Field(
        default_factory=new_transaction_id,
        min_length=1,
        max_length=64,
        description="your id for the transaction; generated if omitted",
    )
    Time: float = Field(ge=0, description="seconds since the first transaction in the dataset")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0, description="transaction amount")

    # ------------------------------------------------ who and where (Phase 5)
    # Optional: the dataset's own columns are still a complete request, and a
    # transaction without a card simply gets no velocity features. The pattern
    # keeps these short and printable, because card_id becomes part of a Redis
    # key and all three end up in logs.
    card_id: str | None = Field(
        None, pattern=ENTITY_PATTERN, description="the card this transaction was made with"
    )
    merchant: str | None = Field(None, pattern=ENTITY_PATTERN, description="where it was spent")
    country: str | None = Field(
        None, pattern=r"^[A-Z]{2}$", description="ISO 3166-1 alpha-2 country code"
    )

    def features(self) -> dict[str, float]:
        """The model inputs, in the column order the model was trained on."""
        return {name: getattr(self, name) for name in RAW_FEATURES}


class VelocityFeatures(BaseModel):
    """What the card had done recently when this transaction was scored (Step 5.3).

    Counted in Redis, per card, over the windows in src/features/velocity.py.
    The transaction being scored is included in the counts.
    """

    count_1m: int = Field(ge=0, description="transactions on this card in the last minute")
    count_5m: int = Field(ge=0)
    count_1h: int = Field(ge=0)
    amount_1h: float = Field(ge=0, description="total amount on this card in the last hour")
    countries_1h: int = Field(ge=0, description="distinct countries used in the last hour")
    seconds_since_previous: float | None = Field(
        description="gap to this card's previous transaction; null if there was none"
    )


class RiskResult(BaseModel):
    transaction_id: str
    risk_score: float = Field(
        ge=0, le=1, description="predicted probability that the transaction is fraud"
    )
    decision: Decision
    review_threshold: float
    block_threshold: float | None = Field(description="null means the service never blocks")
    model_version: str


class TransactionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[Transaction] = Field(min_length=1, max_length=MAX_BATCH_SIZE)

    @model_validator(mode="after")
    def _transaction_ids_are_unique(self) -> TransactionBatch:
        counts = Counter(transaction.transaction_id for transaction in self.transactions)
        duplicates = sorted(tid for tid, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate transaction_id in batch: {duplicates[:10]}")
        return self


class BatchResult(BaseModel):
    results: list[RiskResult]


class StoredDecision(RiskResult):
    """A decision as it was recorded, including when it was made."""

    # Built straight from a database row (DecisionRecord) by model_validate.
    model_config = ConfigDict(from_attributes=True)

    scored_at: datetime = Field(description="when the decision was made (UTC)")


class DecisionList(BaseModel):
    decisions: list[StoredDecision]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    review_threshold: float
    block_threshold: float | None
    decision_store: Literal["ok", "unavailable", "disabled"] = Field(
        description="unavailable: scoring still works, but decisions are not being recorded"
    )
    velocity: Literal["ok", "unavailable", "disabled"] = Field(
        description="unavailable: scoring still works, but without recent-activity features"
    )
