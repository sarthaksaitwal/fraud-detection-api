"""Step 2.1 guard rails: request and response validation."""

import math

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    MAX_BATCH_SIZE,
    Decision,
    RiskResult,
    Transaction,
    TransactionBatch,
)
from src.ml.preprocess import RAW_FEATURES


@pytest.fixture
def payload(raw_df):
    """One valid request body, taken from the fake dataset."""
    return raw_df[RAW_FEATURES].iloc[0].to_dict()


def test_a_dataset_row_is_a_valid_transaction(payload):
    features = Transaction(**payload).features()
    assert list(features) == RAW_FEATURES
    assert features == payload


def test_transaction_fields_match_the_model_inputs():
    assert [name for name in Transaction.model_fields if name != "transaction_id"] == RAW_FEATURES


def test_transaction_id_is_generated_when_omitted(payload):
    first, second = Transaction(**payload), Transaction(**payload)
    assert first.transaction_id and first.transaction_id != second.transaction_id
    assert Transaction(**payload, transaction_id="tx-1").transaction_id == "tx-1"


@pytest.mark.parametrize("column", ["Time", "V14", "Amount"])
def test_missing_column_is_rejected(payload, column):
    del payload[column]
    with pytest.raises(ValidationError, match=column):
        Transaction(**payload)


@pytest.mark.parametrize("extra", ["v14", "Class"])
def test_unknown_field_is_rejected(payload, extra):
    with pytest.raises(ValidationError, match=extra):
        Transaction(**payload, **{extra: 1})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_rejected(payload, value):
    payload["V3"] = value
    with pytest.raises(ValidationError, match="V3"):
        Transaction(**payload)


@pytest.mark.parametrize("column", ["Time", "Amount"])
def test_negative_time_or_amount_is_rejected(payload, column):
    payload[column] = -1
    with pytest.raises(ValidationError, match=column):
        Transaction(**payload)


def test_text_is_rejected(payload):
    payload["V1"] = "abc"
    with pytest.raises(ValidationError, match="V1"):
        Transaction(**payload)


def test_batch_size_limits(payload):
    with pytest.raises(ValidationError):
        TransactionBatch(transactions=[])
    with pytest.raises(ValidationError):
        TransactionBatch(transactions=[payload] * (MAX_BATCH_SIZE + 1))
    assert len(TransactionBatch(transactions=[payload] * 3).transactions) == 3


def test_result_serialises_decision_as_lowercase_text():
    result = RiskResult(
        transaction_id="tx-1",
        risk_score=0.5,
        decision=Decision.REVIEW,
        review_threshold=0.24,
        block_threshold=None,
        model_version="v1",
    )
    body = result.model_dump(mode="json")
    assert body["decision"] == "review"
    assert body["block_threshold"] is None


def test_risk_score_outside_0_1_is_rejected():
    with pytest.raises(ValidationError):
        RiskResult(
            transaction_id="tx-1",
            risk_score=1.5,
            decision=Decision.BLOCK,
            review_threshold=0.24,
            block_threshold=0.95,
            model_version="v1",
        )


def test_duplicate_transaction_ids_in_a_batch_are_rejected(payload):
    with pytest.raises(ValidationError, match="tx-1"):
        TransactionBatch(transactions=[{**payload, "transaction_id": "tx-1"}] * 2)
