# fraud-detection-api

Real-time transaction risk scoring in the spirit of Stripe Radar: an XGBoost fraud
model that returns **approve / review / block** for each transaction, to be served
behind FastAPI and fed by a Kafka stream.

```
Producer ──► Kafka ──► Consumer ──► Postgres ──► Dashboard
                          │
                     scoring core  ◄── FastAPI  POST /score
                          │
               models/fraud_model.joblib
```

The plan is for one scoring function to serve both paths: a synchronous HTTP call
and an asynchronous stream consumer. The model work (Phase 1) is nearly finished;
the service around it starts in Phase 2.

## Results so far

Measured on a held-out test set of 56,746 transactions containing 95 fraud cases,
which no model or threshold choice ever saw.

**Model: supervised XGBoost, chosen over an unsupervised Isolation Forest**

| | PR-AUC (95% CI) | ROC-AUC |
|---|---|---|
| Random guessing | 0.0017 | 0.50 |
| Isolation Forest (no labels) | 0.085 (0.061–0.124) | 0.938 |
| **XGBoost (labels)** | **0.821** (0.745–0.891) | **0.970** |

The gap holds on a time-based split (train on the first 40 hours, test on the last
8): XGBoost 0.802, Isolation Forest 0.041. ROC-AUC makes the two models look close;
PR-AUC, which matters when only 0.17% of transactions are fraud, shows a 10x gap.

**Decisions: thresholds chosen by expected cost**

| | Isolation Forest policy | **XGBoost policy** |
|---|---|---|
| Rule | review if risk ≥ 0.983 | **review if p ≥ 0.24, block if p ≥ 0.95** |
| Transactions reviewed | 1.88% | **0.08%** |
| Transactions blocked | 0% | 0.06% (32 blocks, all fraud) |
| Fraud cases stopped | 66% | **76%** |
| Fraud cost saved vs no model | 34% | **73%** |

Thresholds minimise total cost: $5 per analyst review, $50 per legitimate customer
wrongly blocked, and the transaction amount plus a $15 chargeback fee for every
fraud let through, with analyst capacity capped at 2% of traffic. They were chosen
on out-of-fold predictions from the training data, then checked once on the test
set. **These costs are illustrative, not industry figures;** they live in `.env`.

## Status

- [x] **Phase 0**: project skeleton, configuration, dependencies
- [ ] **Phase 1**: the model
  - [x] 1.1 Explore the data
  - [x] 1.2 Deduplicate, stratified split, preprocessing pipeline
  - [x] 1.3 Isolation Forest trained on normal transactions
  - [x] 1.4 0–1 risk score
  - [x] 1.5 Evaluation: PR-AUC, threshold table, confusion matrix
  - [x] 1.6 Cost-based review/block thresholds
  - [x] 1.7 XGBoost comparison, adopted as the product model; thresholds redone
  - [x] 1.8 Save the model with versioned metadata
  - [ ] 1.9 One-command training script
- [ ] **Phase 2**: FastAPI scoring service
- [ ] **Phase 3**: Docker Compose + Postgres persistence
- [ ] **Phase 4**: Kafka producer/consumer (the "real time" part)
- [ ] **Phase 5**: Redis velocity features
- [ ] **Phase 6**: Streamlit dashboard, Prometheus metrics, load test

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows (Git Bash); use bin/activate on macOS/Linux
pip install -r requirements-dev.txt

cp .env.example .env
bash scripts/download_data.sh        # needs a Kaggle API token
pytest
```

Until the training script lands (Step 1.9), build the model by running these
notebooks in order: `002_preprocess` (creates the train/test split), then
`009_export_model` (trains and saves `models/fraud_model.joblib`).

## How it works

1. **Preprocessing** (`src/ml/preprocess.py`). Drops 1,081 duplicate rows *before*
   splitting, so no transaction lands in both train and test. `Amount` is
   log-transformed and robust-scaled; `Time` becomes a time-of-day position on a
   24-hour circle; `V1`–`V28` pass through. The fitted preprocessing is saved
   inside the model file, so training and serving can never transform data differently.
2. **Model** (`src/ml/model.py`). XGBoost with early stopping on a slice of the
   training data. Its output, the predicted fraud probability, is the risk score.
3. **Thresholds** (`src/ml/thresholds.py`). Prices every (review, block) threshold
   pair and picks the cheapest within analyst capacity. Blocking only beats reviewing
   when precision exceeds 1 − $5/$50 = 90%.
4. **Artifact** (`src/ml/artifact.py`). Saves the pipeline plus
   `model_metadata.json`: version, SHA-256, input feature order, thresholds, test
   metrics and library versions. Loading checks the SHA-256 before unpickling and
   warns if library versions differ.

## Layout

| Path | What lives there |
|---|---|
| `src/config.py` | every path, threshold, cost assumption and connection string |
| `src/ml/preprocess.py` | loading, deduplication, split, preprocessing pipeline |
| `src/ml/model.py` | the XGBoost fraud model and out-of-fold scoring |
| `src/ml/thresholds.py` | cost model and threshold selection |
| `src/ml/evaluate.py` | PR-AUC, threshold and alert-budget tables, bootstrap intervals |
| `src/ml/artifact.py` | saving and loading the model with metadata |
| `src/ml/isolation_forest.py`, `src/ml/risk.py` | the unsupervised model from Steps 1.3–1.6, kept for the research notebooks |
| `src/api/`, `src/streaming/`, `src/storage/`, `src/features/` | Phases 2–5 (not built yet) |
| `notebooks/` | one notebook per step, each ending in a findings cell; nothing imports from here |
| `tests/` | 75 tests; run with `pytest` |

Notebooks: `001_explore` · `002_preprocess` · `003_isolation_forest` · `004_risk_score` ·
`005_evaluation` · `006_thresholds` · `007_xgboost_baseline` · `008_xgboost_thresholds` ·
`009_export_model`

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):
284,807 European card transactions over two days, 492 of them fraud (0.172%).
Features `V1`–`V28` are PCA-transformed for privacy, plus `Time` and `Amount`. Because
of the imbalance, models are judged on **PR-AUC and precision/recall**, never accuracy.

## Limitations

- **Two days of data from one source.** Fraud patterns are very similar across train
  and test, which favours a supervised model. The dataset cannot show how XGBoost
  degrades when fraudsters change tactics, or handle labels arriving weeks late
  through chargebacks. Those are the situations anomaly detection exists for.
- **Anonymised features.** `V1`–`V28` cannot be interpreted or extended. Phase 5 adds
  velocity features using synthetic transactions with real fields.
- **Small fraud counts.** 95 test fraud cases means wide confidence intervals; treat
  differences of a few percentage points as noise.
- **Thresholds belong to one model.** The probability scale shifts between retrains,
  so thresholds must be re-checked each time the model is retrained.
- **Pickled models run code when loaded.** Only load model files you trained yourself.
