"""Step 2.5 guard rails: the sample request script."""

import json

import numpy as np
import pytest

from scripts import sample_request
from src.api.schemas import Decision, Transaction, TransactionBatch
from src.features.entities import entities_for
from src.ml.preprocess import RAW_FEATURES, TARGET
from src.scoring import Scorer


class AmountAsProbability:
    """Stands in for the model: fraud probability = Amount / 100, capped at 1."""

    def predict_proba(self, X):
        p = np.clip(X["Amount"].to_numpy() / 100, 0, 1)
        return np.column_stack([1 - p, p])


SCORER = Scorer(AmountAsProbability(), "v-test", 0.24, 0.95)


@pytest.fixture
def X(raw_df):
    """Ten rows the fake model decides as: A A R B A R B B A A."""
    rows = raw_df[RAW_FEATURES].head(10).copy()
    rows["Amount"] = [10, 10, 50, 95, 10, 50, 95, 99, 10, 10]
    return rows


def test_selects_rows_with_the_requested_decision(X):
    rows = sample_request.select_rows(X, SCORER, Decision.BLOCK, n=2)
    assert list(rows.index) == [3, 6]


def test_the_body_carries_the_same_identity_the_stream_uses(X):
    """Step 5.2: a sample request looks like a real card's transaction, not an unknown one."""
    body = sample_request.request_body(X.head(1), batch=False)
    assert {"card_id", "merchant", "country"} <= body.keys()
    assert body["card_id"] == entities_for(body["transaction_id"])["card_id"]
    assert Transaction(**body).card_id == body["card_id"]


def test_without_a_decision_takes_the_first_rows(X):
    assert list(sample_request.select_rows(X, SCORER, n=3).index) == [0, 1, 2]


def test_asking_for_more_rows_than_match_raises(X):
    with pytest.raises(ValueError, match="only 2"):
        sample_request.select_rows(X, SCORER, Decision.REVIEW, n=3)


def test_bodies_are_valid_requests(X):
    single = sample_request.request_body(X.head(1), batch=False)
    batch = sample_request.request_body(X.head(3), batch=True)
    assert Transaction(**single).transaction_id == "test-row-0"
    assert len(TransactionBatch(**batch).transactions) == 3


def test_cli_writes_utf8_json_and_reports_labels(X, raw_df, tmp_path, monkeypatch, capsys):
    y = raw_df[TARGET].head(10)
    monkeypatch.setattr(sample_request, "load_test_data", lambda: (X, y, SCORER))
    out = tmp_path / "samples" / "block.json"
    sample_request.main(["--decision", "block", "--batch", "3", "--out", str(out)])

    body = json.loads(out.read_text(encoding="utf-8"))
    assert [t["transaction_id"] for t in body["transactions"]] == [
        "test-row-3",
        "test-row-6",
        "test-row-7",
    ]
    # conftest makes the first 50 rows fraud, so all three are labelled fraud.
    assert "3 labelled fraud" in capsys.readouterr().err


@pytest.mark.parametrize("size", ["0", "1001"])
def test_cli_rejects_batch_sizes_the_api_would_reject(size):
    with pytest.raises(SystemExit):
        sample_request.main(["--batch", size])
