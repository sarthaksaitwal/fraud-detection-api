"""Write real test-set transactions as JSON request bodies (Step 2.5).

    python -m scripts.sample_request --decision block --out data/samples/block.json
    python -m scripts.sample_request --batch 100 --out data/samples/batch.json

    curl -X POST http://127.0.0.1:8000/score \\
         -H "Content-Type: application/json" -d @data/samples/block.json

Rows come from data/processed/test.parquet, which the model never trained on.
They are scored with the saved model first, so you can ask for a transaction
the model approves, reviews or blocks. Use --out rather than ">": in Windows
PowerShell ">" writes UTF-16, which the API cannot read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.api.schemas import MAX_BATCH_SIZE, Decision
from src.config import settings
from src.ml.model import fraud_probability
from src.ml.preprocess import RAW_FEATURES, TARGET
from src.scoring import Scorer, decide


def select_rows(
    X: pd.DataFrame, scorer: Scorer, decision: Decision | None = None, n: int = 1
) -> pd.DataFrame:
    """The first `n` rows of X, keeping only those given `decision` if one is asked for."""
    if decision is not None:
        risk = fraud_probability(scorer.pipeline, X)
        decisions = decide(risk, scorer.review_threshold, scorer.block_threshold)
        X = X[[d is decision for d in decisions]]
    if len(X) < n:
        raise ValueError(f"only {len(X)} test transactions match; asked for {n}")
    return X.head(n)


def request_body(rows: pd.DataFrame, batch: bool) -> dict[str, Any]:
    """A /score body for one row, or a /score/batch body for several."""
    transactions = [
        {"transaction_id": f"test-row-{index}", **values}
        for index, values in zip(rows.index, rows.to_dict(orient="records"), strict=True)
    ]
    return {"transactions": transactions} if batch else transactions[0]


def load_test_data() -> tuple[pd.DataFrame, pd.Series, Scorer]:
    """The test split written by `python -m src.ml.train`, and the saved model."""
    test = pd.read_parquet(settings.test_split_path)
    return test[RAW_FEATURES], test[TARGET], Scorer.load()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.sample_request",
        description="Write real test-set transactions as JSON request bodies.",
    )
    parser.add_argument("--decision", choices=[d.value for d in Decision])
    parser.add_argument(
        "--batch",
        type=int,
        metavar="N",
        help=f"write a /score/batch body with N transactions (1-{MAX_BATCH_SIZE})",
    )
    parser.add_argument("--out", type=Path, help="file to write (default: print)")
    args = parser.parse_args(argv)
    if args.batch is not None and not 1 <= args.batch <= MAX_BATCH_SIZE:
        parser.error(f"--batch must be between 1 and {MAX_BATCH_SIZE}")

    X_test, y_test, scorer = load_test_data()
    decision = Decision(args.decision) if args.decision else None
    rows = select_rows(X_test, scorer, decision, n=args.batch or 1)
    text = json.dumps(request_body(rows, batch=args.batch is not None), indent=2) + "\n"

    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    fraud = int(y_test.loc[rows.index].sum())
    where = args.out or "stdout"
    print(f"wrote {len(rows)} transaction(s) to {where}; {fraud} labelled fraud", file=sys.stderr)


if __name__ == "__main__":
    main()
