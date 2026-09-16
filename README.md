# fraud-detection-api

Real-time transaction risk scoring in the spirit of Stripe Radar: an XGBoost fraud
model that returns **approve / review / block** for each transaction, served
behind FastAPI, recorded in Postgres and, from Phase 4, fed by a Kafka stream.

```
Producer ──► Kafka ──► Consumer ──┐                    (Phase 4)
                                  ▼
             FastAPI ──► scoring core ──► decision store ──► Postgres ──► Dashboard
   POST /score, /score/batch      │                              ▲
   GET /transactions              │                              │
                       models/fraud_model.joblib      GET /transactions
```

One scoring function and one decision store serve both paths: the synchronous HTTP
API and, from Phase 4, the asynchronous stream consumer. The model (Phase 1), the
scoring API (Phase 2) and decision storage with Docker Compose (Phase 3) are
finished; streaming comes next.

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

**Serving: measured on a Windows laptop, one uvicorn process**

| | Without storage (Phase 2, local) | Recording to Postgres (Docker Compose) |
|---|---|---|
| `POST /score`, time inside the server (median) | 10–20 ms | 16 ms |
| `POST /score`, HTTP round trip (median) | about 20 ms | 36 ms |
| `POST /score/batch`, 1,000 transactions, inside the server (median) | about 200 ms | 323 ms |
| `POST /score/batch`, 1,000 transactions, HTTP round trip (median) | about 220 ms | 413 ms |

Recording a 1,000-row batch adds about 100 ms: it is one `INSERT ... ON CONFLICT`
statement. Most of the remaining HTTP overhead in the Docker column is Docker
Desktop's port forwarding on Windows.

Replaying the whole test set through `/score/batch` gives exactly the decisions
above, 47 reviews and 32 blocks, and Postgres ends up holding the same counts.
Replaying it a second time leaves the row count unchanged.

**Database outage: Postgres stopped for 60 seconds under steady traffic**

| | |
|---|---|
| `/score` requests answered | 117 of 117, all `200` |
| Median latency during the outage | 44 ms |
| Requests that paid to find the database still down | 5, about 4 s each (one per 10 s retry window) |
| Container health | stayed `healthy` |
| After Postgres came back | recording resumed within 10 s; the log reported 142 decisions not recorded |

## Status

- [x] **Phase 0**: project skeleton, configuration, dependencies
- [x] **Phase 1**: the model
  - [x] 1.1 Explore the data
  - [x] 1.2 Deduplicate, stratified split, preprocessing pipeline
  - [x] 1.3 Isolation Forest trained on normal transactions
  - [x] 1.4 0–1 risk score
  - [x] 1.5 Evaluation: PR-AUC, threshold table, confusion matrix
  - [x] 1.6 Cost-based review/block thresholds
  - [x] 1.7 XGBoost comparison, adopted as the product model; thresholds redone
  - [x] 1.8 Save the model with versioned metadata
  - [x] 1.9 One-command training script
- [x] **Phase 2**: FastAPI scoring service
  - [x] 2.1 Request and response schemas with strict validation
  - [x] 2.2 Scoring core shared by the API and the future stream consumer
  - [x] 2.3 FastAPI app: model loaded at startup, `GET /health`, `POST /score`
  - [x] 2.4 `POST /score/batch`, JSON 422 and 500 errors
  - [x] 2.5 Request logging with latency and request ids, sample request script
- [x] **Phase 3**: Docker Compose + Postgres persistence
  - [x] 3.1 Storage layer: `decisions` table, one-statement batch upsert, UTC timestamps
  - [x] 3.2 Record every decision from the API; `GET /transactions`, `GET /transactions/{id}`
  - [x] 3.3 Docker image: pinned libraries matching the model, non-root, health check
  - [x] 3.4 Docker Compose with Postgres; retry window so a database outage stays fast
  - [x] 3.5 Storage tests on real Postgres, schema script
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

Build the model (about 45 seconds):

```bash
python -m src.ml.train
```

This deduplicates and splits the data, chooses review/block thresholds on out-of-fold
predictions, trains XGBoost, evaluates it on the test set, and writes
`models/fraud_model.joblib` and `models/model_metadata.json`. If the thresholds it
chooses differ from `.env`, it logs the values to set. Training is reproducible:
rebuilding produces a byte-identical model file.

Serve it with its database, in Docker (needs Docker Desktop and the trained model):

```bash
docker compose up -d --build         # or: make up
docker compose ps                    # api and postgres both "(healthy)"
```

Or run the API directly for development, against the Compose database or none:

```bash
docker compose up -d postgres        # optional: or set PERSIST_DECISIONS=false in .env
uvicorn src.api.main:app --reload --no-access-log
```

Interactive docs are at http://127.0.0.1:8000/docs. To send a real transaction, write
one from the test set and post it:

```bash
python -m scripts.sample_request --decision block --out data/samples/block.json
curl -X POST http://127.0.0.1:8000/score \
     -H "Content-Type: application/json" -d @data/samples/block.json
```

```json
{
  "transaction_id": "test-row-95238",
  "risk_score": 0.9863,
  "decision": "block",
  "review_threshold": 0.24,
  "block_threshold": 0.95,
  "model_version": "20260915-071627-19429cbb"
}
```

`--decision` accepts `approve`, `review` or `block`; `--batch N` writes a
`/score/batch` body. Use `--out` rather than `>`: in Windows PowerShell, `>` writes
UTF-16, which the API rejects.

Every decision is recorded, so it can be looked up afterwards:

```bash
curl http://127.0.0.1:8000/transactions/test-row-95238      # the decision above, plus scored_at
curl "http://127.0.0.1:8000/transactions?decision=block&limit=10"
python -m scripts.init_db                                    # row counts by decision
docker compose exec postgres psql -U fraud -d fraud          # or: make psql
docker compose down                                          # decisions are kept; down -v deletes them
```

The Compose Postgres listens on `127.0.0.1:5433`, not 5432, which a locally
installed Postgres often already uses. Use `127.0.0.1` rather than `localhost` in
`DATABASE_URL`: on Windows `localhost` tries IPv6 first, which Compose does not
listen on, and each new connection then waits about 5 seconds.

## API

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | | `status`, the model version and thresholds being served, `decision_store`: `ok`, `unavailable` or `disabled` |
| `POST /score` | one transaction: `Time`, `V1`–`V28`, `Amount`, optional `transaction_id` | risk score, decision, thresholds, model version |
| `POST /score/batch` | `{"transactions": [...]}`, 1–1,000 with unique ids | `{"results": [...]}` in request order |
| `GET /transactions/{transaction_id}` | | the recorded decision plus `scored_at`, or 404 |
| `GET /transactions?decision=block&limit=50` | | `{"decisions": [...]}`, newest first; `limit` 1–500 |

- **Decisions:** `block` if risk ≥ `BLOCK_THRESHOLD`, otherwise `review` if risk ≥
  `REVIEW_THRESHOLD`, otherwise `approve`. Thresholds come from `.env`; the service
  logs a warning at startup if they differ from the ones stored with the model.
- **422:** a missing, misspelled or extra field, text, NaN, infinity, a negative
  `Time` or `Amount`, or a bad batch. The body lists each bad field's location and
  the reason, without echoing the values sent. A batch is all or nothing.
- **500:** `{"detail": "internal server error"}`. The traceback goes to the server
  log only.
- **Logging:** one line per request with method, path, status, latency and an
  `X-Request-ID`, which is also returned as a response header. A client's own
  `X-Request-ID` is kept if it is 1–64 letters, digits, `.`, `_` or `-`. Transaction
  values are never logged.
- **Startup:** the model is loaded and checked once. If it is missing or does not
  match its metadata, the server refuses to start. A misconfigured `DATABASE_URL`
  (wrong driver) also stops startup; a database that is merely down does not.
- **Recording:** `/score` and `/score/batch` save every decision with the thresholds
  and model version that produced it. Scoring the same `transaction_id` again
  replaces its row. Set `PERSIST_DECISIONS=false` to score without a database.
- **Database outages:** recording is best effort. If Postgres is down, scoring still
  returns `200` and the failure is logged; after a failure the database is left
  alone for 10 seconds, so only one request per window waits to find out whether it
  is back. `/transactions` returns `503`. `/health` stays `200` and reports
  `"decision_store": "unavailable"`, so Docker does not restart a working API over a
  database problem. When the database returns, the log says how many decisions
  were not recorded.

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
5. **Scoring core** (`src/scoring.py`). The only code that turns transactions into
   decisions: one model call per list of transactions, the same `>=` rule the
   thresholds were priced with. A single transaction is a list of one, so single
   and batch scoring always agree.
6. **API** (`src/api/`). Pydantic schemas validate every request before it reaches
   the model; routes are thin wrappers around the scoring core. Scoring routes are
   plain `def`, so CPU work runs in a worker thread instead of blocking the server.
7. **Decision store** (`src/storage/`). One row per transaction in a `decisions`
   table, keyed by `transaction_id`: risk score, decision, both thresholds, model
   version and `scored_at` in UTC. The features are not stored. A batch is saved
   with one `INSERT ... ON CONFLICT DO UPDATE`, so replaying a transaction, as
   Kafka's at-least-once delivery will in Phase 4, updates its row instead of
   duplicating it. Check constraints keep decisions and scores valid in the
   database itself. The schema is created on first use; there are no migrations
   yet.
8. **Container** (`Dockerfile`, `docker-compose.yml`). A two-stage image on
   `python:3.10-slim` running as a non-root user, with the model copied in, so an
   image tag identifies exactly one model. It installs `requirements-api.txt`,
   whose exact versions a test keeps equal to the training environment: the model
   is a pickle and must load under the libraries it was trained with (`xgboost-cpu`
   leaves out ~200 MB of GPU libraries). Compose adds Postgres 16 with a health
   check and a named volume, and publishes both ports on `127.0.0.1` only.

## Layout

| Path | What lives there |
|---|---|
| `src/config.py` | every path, threshold, cost assumption and connection string |
| `src/ml/train.py` | the one-command training pipeline |
| `src/ml/preprocess.py` | loading, deduplication, split, preprocessing pipeline |
| `src/ml/model.py` | the XGBoost fraud model and out-of-fold scoring |
| `src/ml/thresholds.py` | cost model and threshold selection |
| `src/ml/evaluate.py` | PR-AUC, threshold and alert-budget tables, bootstrap intervals |
| `src/ml/artifact.py` | saving and loading the model with metadata |
| `src/ml/isolation_forest.py`, `src/ml/risk.py` | the unsupervised model from Steps 1.3–1.6, kept for the research notebooks |
| `src/scoring.py` | the scoring core: model + thresholds → decisions |
| `src/api/main.py` | the FastAPI app: startup, request logging, error handlers |
| `src/api/schemas.py` | request and response bodies |
| `src/api/routes/` | `/health`, `/score`, `/score/batch` |
| `src/api/routes/transactions.py` | `GET /transactions`, `GET /transactions/{id}` |
| `src/storage/models.py` | the `decisions` table and its UTC timestamp type |
| `src/storage/repository.py` | batch upsert, lookup, listing and counts |
| `src/storage/store.py` | best-effort recording, the retry window, startup |
| `src/storage/db.py` | connection pool, sessions, timeouts, schema creation |
| `scripts/sample_request.py` | writes real test-set transactions as request bodies |
| `scripts/init_db.py` | creates the table if missing and reports what it holds |
| `Dockerfile`, `requirements-api.txt`, `.dockerignore` | the API image and its pinned libraries |
| `docker-compose.yml` | the API and Postgres together |
| `src/streaming/`, `src/features/` | Phases 4–5 (not built yet) |
| `notebooks/` | one notebook per step, each ending in a findings cell; nothing imports from here |
| `tests/` | 214 tests; run with `pytest`. Storage tests run on SQLite and, when `docker compose up` is running, on real Postgres too (`pytest -m postgres`); without it those are skipped |

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
- **The API is a local demo.** It has no authentication, rate limiting or TLS, and
  the latency figures come from one process on a laptop, not a load test (Phase 6).
  The Compose database password is a demo value.
- **Recording can lose decisions.** Recording is best effort, so decisions made
  while Postgres is down, or in the 10 seconds after it returns, are logged as a
  count but not stored. Writing each decision to a stream first (Phase 4) is the
  real fix.
- **No schema migrations.** The table is created if it is missing but never
  altered. Changing a column needs a migration tool such as Alembic.
