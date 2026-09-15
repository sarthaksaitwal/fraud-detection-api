# fraud-detection-api

Real-time transaction risk scoring — an Isolation Forest anomaly detector served
behind a FastAPI endpoint and fed by a Kafka stream, in the spirit of Stripe Radar.

```
Producer ──► Kafka ──► Consumer ──► Postgres ──► Dashboard
                          │
                     scoring core  ◄── FastAPI  POST /score
                          │
                   models/*.joblib
```

The same `score(transaction) -> RiskResult` function serves both transports:
a synchronous HTTP call and an asynchronous stream consumer.

## Status

Built in phases — see `docs/` / the issue tracker for the full plan.

- [x] **Phase 0** — project skeleton, configuration, dependencies
- [ ] **Phase 1** — train + evaluate the Isolation Forest
- [ ] **Phase 2** — FastAPI scoring service
- [ ] **Phase 3** — Docker Compose + Postgres persistence
- [ ] **Phase 4** — Kafka producer/consumer (the "real time" part)
- [ ] **Phase 5** — Redis velocity features
- [ ] **Phase 6** — Streamlit dashboard, Prometheus metrics, load test

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows (Git Bash); use bin/activate on macOS/Linux
pip install -r requirements-dev.txt

cp .env.example .env
bash scripts/download_data.sh        # needs a Kaggle API token
pytest
```

## Layout

| Path | What lives there |
|---|---|
| `src/config.py` | every path, threshold and connection string |
| `src/schemas.py` | the `Transaction` / `RiskResult` contract, shared by API and stream |
| `src/ml/` | training script, preprocessing, model loading, **the scoring core** |
| `src/api/` | FastAPI app and routes |
| `src/streaming/` | synthetic generator, Kafka producer and consumer |
| `src/storage/` | SQLAlchemy models and the repository (the only place with queries) |
| `src/features/` | Redis-backed velocity features |
| `dashboard/` | Streamlit live view |
| `notebooks/` | exploration only — nothing imports from here |

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
— 284,807 transactions, 492 fraudulent (0.172%). Features are PCA-transformed
(`V1`–`V28`) plus `Time` and `Amount`. Because of the class imbalance, the model
is judged on **PR-AUC and precision/recall**, never accuracy.
