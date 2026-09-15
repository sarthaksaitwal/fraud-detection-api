"""Train, evaluate and save the fraud model in one command (Step 1.9).

    python -m src.ml.train

Rebuilds everything from data/raw/creditcard.csv:

1. Deduplicate, make the stratified train/test split, save it to data/processed/.
2. Score the training data out-of-fold and choose review/block thresholds by cost.
3. Fit the final XGBoost model on the training data.
4. Measure it once on the test set.
5. Save models/fraud_model.joblib and models/model_metadata.json.

It runs the same code as notebooks 002, 008 and 009, so the results match them.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import settings
from src.ml.artifact import save_model
from src.ml.evaluate import bootstrap_pr_auc, summary_metrics
from src.ml.model import (
    VALIDATION_SIZE,
    XGB_PARAMS,
    fit_xgboost,
    fraud_probability,
    out_of_fold_probability,
)
from src.ml.preprocess import deduplicate, load_raw, save_splits, split
from src.ml.thresholds import (
    PROBABILITY_THRESHOLDS,
    Costs,
    ThresholdChoice,
    approve_all_cost_per_100k,
    choose_thresholds,
    costs_from_settings,
    policy_costs,
)

log = logging.getLogger("src.ml.train")


def choose_model_thresholds(
    y: pd.Series, proba: Any, amounts: pd.Series, costs: Costs
) -> ThresholdChoice:
    """Cheapest (review, block) pair on the probability grid, within analyst capacity."""
    table = policy_costs(
        y,
        proba,
        amounts,
        costs,
        review_candidates=PROBABILITY_THRESHOLDS,
        block_candidates=(*PROBABILITY_THRESHOLDS, None),
    )
    return choose_thresholds(table, max_review_rate=settings.max_review_rate)


def policy_summary(
    y: pd.Series,
    proba: Any,
    amounts: pd.Series,
    costs: Costs,
    review: float,
    block: float | None,
) -> dict[str, float]:
    """What one (review, block) pair does on a dataset."""
    row = policy_costs(
        y, proba, amounts, costs, review_candidates=[review], block_candidates=[block]
    ).iloc[0]
    no_model = approve_all_cost_per_100k(y, amounts, costs)
    return {
        "reviewed_rate": row["review_rate"],
        "blocked_rate": row["block_rate"],
        "fraud_stopped": row["fraud_stopped"],
        "fraud_loss_stopped": row["fraud_loss_stopped"],
        "cost_per_100k": row["cost_per_100k"],
        "saving_vs_no_model": 1 - row["cost_per_100k"] / no_model,
    }


def train(
    raw: pd.DataFrame | None = None,
    model_path: Path | None = None,
    metadata_path: Path | None = None,
    n_splits: int = 5,
    bootstrap_resamples: int = 1000,
    save_split_files: bool = True,
) -> dict[str, Any]:
    """Run the whole pipeline and return the metadata that was saved."""
    started = time.perf_counter()

    raw = load_raw() if raw is None else raw
    df = deduplicate(raw)
    X_train, X_test, y_train, y_test = split(df)
    log.info(
        "data: %d rows, %d duplicates removed; train %d rows (%d fraud), test %d rows (%d fraud)",
        len(raw),
        len(raw) - len(df),
        len(X_train),
        y_train.sum(),
        len(X_test),
        y_test.sum(),
    )
    if save_split_files:
        save_splits(X_train, X_test, y_train, y_test)
        log.info("saved train/test splits to %s", settings.processed_data_dir)

    costs = costs_from_settings()
    log.info("scoring the training data out-of-fold (%d folds)", n_splits)
    oof_proba = out_of_fold_probability(X_train, y_train, n_splits=n_splits)
    choice = choose_model_thresholds(y_train, oof_proba, X_train["Amount"], costs)
    log.info(
        "thresholds: review >= %s, block >= %s", choice.review_threshold, choice.block_threshold
    )

    log.info("fitting the final model")
    model = fit_xgboost(X_train, y_train)
    xgb = model.named_steps["model"]
    test_proba = fraud_probability(model, X_test)
    test_summary = summary_metrics(y_test, test_proba)
    ci_low, ci_high = bootstrap_pr_auc(
        y_test, test_proba, n_resamples=bootstrap_resamples, seed=settings.random_seed
    )
    thresholds = (choice.review_threshold, choice.block_threshold)

    metadata = {
        "model_type": "xgboost",
        "risk_score": "predicted probability that the transaction is fraud",
        "decision_thresholds": {
            "review": choice.review_threshold,
            "block": choice.block_threshold,
            "chosen_by": (
                f"expected-cost minimisation on {n_splits}-fold out-of-fold probabilities "
                "(src/ml/train.py)"
            ),
        },
        "cost_assumptions": {
            "review_cost": costs.review_cost,
            "false_block_cost": costs.false_block_cost,
            "chargeback_fee": costs.chargeback_fee,
            "max_review_rate": settings.max_review_rate,
        },
        "training": {
            "data": settings.raw_data_filename,
            "raw_rows": len(raw),
            "duplicates_removed": len(raw) - len(df),
            "train_rows": len(X_train),
            "train_fraud": int(y_train.sum()),
            "test_size": settings.test_size,
            "random_seed": settings.random_seed,
            "early_stopping_validation_size": VALIDATION_SIZE,
            "trees": int(xgb.best_iteration + 1),
            "xgboost_params": XGB_PARAMS,
        },
        "out_of_fold_metrics": {
            "pr_auc": summary_metrics(y_train, oof_proba)["pr_auc"],
            "at_decision_thresholds": policy_summary(
                y_train, oof_proba, X_train["Amount"], costs, *thresholds
            ),
        },
        "test_metrics": {
            "rows": test_summary["n"],
            "fraud": test_summary["n_fraud"],
            "pr_auc": test_summary["pr_auc"],
            "pr_auc_95ci": [ci_low, ci_high],
            "pr_auc_random_baseline": test_summary["pr_auc_baseline"],
            "roc_auc": test_summary["roc_auc"],
            "at_decision_thresholds": policy_summary(
                y_test, test_proba, X_test["Amount"], costs, *thresholds
            ),
        },
    }

    saved = save_model(model, metadata, model_path, metadata_path)
    at_thresholds = saved["test_metrics"]["at_decision_thresholds"]
    log.info(
        "test set: PR-AUC %.3f (95%% CI %.3f-%.3f), reviewed %.2f%%, blocked %.2f%%, "
        "fraud stopped %.0f%%, saving vs no model %.0f%%",
        saved["test_metrics"]["pr_auc"],
        ci_low,
        ci_high,
        100 * at_thresholds["reviewed_rate"],
        100 * at_thresholds["blocked_rate"],
        100 * at_thresholds["fraud_stopped"],
        100 * at_thresholds["saving_vs_no_model"],
    )
    log.info("saved model version %s", saved["model_version"])
    _warn_if_config_differs(choice)
    log.info("done in %.0fs", time.perf_counter() - started)
    return saved


def _warn_if_config_differs(choice: ThresholdChoice) -> None:
    configured = (settings.review_threshold, settings.block_threshold)
    chosen = (choice.review_threshold, choice.block_threshold)
    if configured != chosen:
        block = "none" if choice.block_threshold is None else choice.block_threshold
        log.warning(
            "thresholds in .env (review=%s, block=%s) differ from the ones this model was "
            "trained with. Set REVIEW_THRESHOLD=%s and BLOCK_THRESHOLD=%s before serving it.",
            *configured,
            choice.review_threshold,
            block,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.ml.train", description="Train, evaluate and save the fraud model."
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="folds for out-of-fold threshold selection (default: 5)",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        help="resamples for the PR-AUC confidence interval (default: 1000)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    train(n_splits=args.n_splits, bootstrap_resamples=args.bootstrap_resamples)


if __name__ == "__main__":
    main()
