# fraud-detection-api

Real-time transaction risk scoring in the spirit of Stripe Radar: an XGBoost fraud
model that returns **approve / review / block** for each transaction, served behind
FastAPI and a Kafka stream, with velocity rules on top and every decision recorded
in Postgres.

```
                   ┌──────────── transactions.dlq  (messages that are not transactions)
                   │
Producer ──► Kafka "transactions" ──► Consumer ──┐
                                                 ▼
    client ──► FastAPI ──────────────────► scoring core ──► decision store ──► Postgres
       POST /score, /score/batch                 │  ▲                             │
       GET /transactions ◄───────────────────────┼──┼─────────────────────────────┤
                                                 │  └── velocity ──► Redis        │
                                   models/fraud_model.joblib   (recent activity)   │
                                                                                   │
    analyst ──► Streamlit dashboard ───────────── reads decisions ─────────────────┘
       GET /metrics on the API and the consumer ──► Prometheus
```

Two ways in, one set of rules. A client that needs an answer now calls the API and
waits about 24 ms. A stream of transactions goes through Kafka, where the consumer
scores them in batches of 500 with nobody waiting. Both paths use the same schema,
the same scoring code and the same store, so a transaction gets the same decision
whichever way it arrives. The model scores each transaction alone; the velocity
rules then look at what the card has been doing and can send an approved
transaction for review. Both paths export Prometheus metrics, and an analyst sees the
result in a Streamlit dashboard that only ever reads. The model (Phase 1), the API
(Phase 2), storage and Docker Compose (Phase 3), streaming (Phase 4), velocity
features (Phase 5) and the dashboard, metrics and load test (Phase 6) are finished:
**the project is complete.**

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

**Streaming: the whole test set through Kafka**

| | |
|---|---|
| Producer, full speed | 56,746 transactions in 8.2 s (about 6,900/s) |
| Consumer, real model and Postgres | about 1,850 transactions/s, in batches of 500 |
| Decisions recorded | 56,667 approve, 47 review, 32 block: the same as the API and the test-set evaluation |
| Committed offsets afterwards | equal to the end of each of the 3 partitions (lag 0) |
| Replayed under a new consumer group | the same 56,746 rows, none duplicated |

**Database outage on the stream: Postgres stopped for 30 seconds mid-run**

| | |
|---|---|
| What the consumer did | paused its partitions, held its batch of 500, retried 5 times, resumed after 33 s |
| Transactions recorded | **56,746 of 56,746**: nothing lost |
| Committed offsets | equal to the end of every partition |

The API answers a waiting caller, so it can lose decisions during an outage. The
consumer has nobody waiting, so it waits for the database instead.

**Velocity rules: what they cost**

The model scores one transaction in isolation. The velocity rules add what the card
has been doing: more than 5 transactions in a minute, 10 in an hour, 2 countries in
an hour, or two payments within 2 seconds sends an approved transaction for review.

| Thresholds | Share of traffic escalated |
|---|---|
| 2 countries / 20 an hour (first guess) | 2.88%, nearly all the country rule |
| **3 countries / 10 an hour (shipped)** | **0.79%** |
| 4 countries / 20 an hour | 0.05%: barely fires |

Measured over the whole test set at **the pace the transactions really happened**,
using the dataset's own `Time` column. With the model's own 0.14% review rate, the
shipped thresholds keep the review queue near 0.9% of traffic, inside the 2% analyst
capacity the cost model assumes.

The same rules escalate **14.5%** of traffic when the producer replays at 100/s,
because that compresses about 8.5 hours of card activity into 100 seconds. Velocity
measures arrival time, so a fast replay makes every card look frantic; the
compressed figure says nothing about a real deployment.

**These rules cannot be shown to catch more fraud, and the project does not claim
it.** Card ids here are synthetic and independent of the fraud label by
construction, so any lift would be an artifact of the generator. What this measures
is the cost: extra reviews, and about 1 ms of extra latency per transaction.

| | |
|---|---|
| Velocity lookup, one transaction | 1 round trip, p50 1.8 ms, p95 3.8 ms |
| Velocity lookup, batch of 500 | 1 round trip, 88 ms (0.18 ms each) |
| `POST /score` with a card vs without | 22.8 ms vs 21.9 ms (median HTTP round trip, measured back to back) |
| Redis memory | 2.2 MB per 5,000 events; about 20 MB for the whole test set |

**Redis outage: Redis stopped under steady traffic**

| | |
|---|---|
| `/score` requests answered | all `200`, with `"velocity": null` |
| First request after the outage began | 4.0 s: the container name stops resolving, which no socket timeout covers |
| The next requests | 20–38 ms each: Redis is left alone for 10 s after a failure |
| `/health` | `200` in 8 ms, `"velocity": "unavailable"`, `"status": "ok"` |
| After Redis came back | features resumed; the log reported 6 transactions scored without them |

**Load: one process saturates at about one concurrent caller**

`python -m scripts.load_test` keeps a fixed number of callers busy: each sends a
request, waits for the answer, and sends the next. Closed loop, so it cannot pretend
the server is keeping up while a queue grows behind it. `server` is what the API says
it spent, from its own histogram; the gap is queueing.

| Concurrent callers | Throughput | p50 | p95 | p99 | server | errors |
|---|---|---|---|---|---|---|
| 1 | 36.7/s | 23 ms | 32 ms | 219 ms | 22 ms | 0 |
| 2 | 12.4/s | 157 ms | 452 ms | 562 ms | 126 ms | 0 |
| 4 | 4.9/s | 782 ms | 1,217 ms | 1,431 ms | 685 ms | 0 |
| 6 | 34.8/s | 117 ms | 546 ms | 1,360 ms | 143 ms | 0 |
| 8 | 23.8/s | 167 ms | 1,930 ms | 3,300 ms | 281 ms | 0 |

One uvicorn process scoring on CPU has no headroom: throughput peaks at roughly one
busy caller and extra callers buy latency, not work. Client and server times move
together, so the queue is inside the service rather than on the network. **The rows
are one run on a laptop** (`data/load-test.json`); rows 2, 4 and 6 show how much
Docker Desktop and everything else on the machine move the numbers, and repeat runs
gave 31/s and 41/s where this one gave 12/s and 4.9/s. The shape reproduces; the
absolute figures are not a benchmark. Batching is the answer to volume, not
concurrency: `POST /score/batch` sustains 300–420 transactions/s, and the Kafka
consumer about 1,850/s.

**The load test found a real bug: XGBoost fighting itself**

The first runs showed 223 ms at two callers against 26 ms at one. The model is saved
with `n_jobs=-1`, so every prediction started one OpenMP thread per core, and two
concurrent requests put sixteen threads on eight cores.

| Threads per prediction | One transaction | Batch of 500 |
|---|---|---|
| `OMP_NUM_THREADS=1` | 5–6 ms, every run | 6 ms |
| `OMP_NUM_THREADS=8` | 7 ms to 182 ms | 6 ms to 107 ms |

Eight threads are not slower on average so much as unpredictable, and a single-row
prediction has nothing to parallelise in the first place. The image now pins
`OMP_NUM_THREADS=1`. Scale with processes, not threads.

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

Run the whole pipeline in Docker (needs Docker Desktop and the trained model):

```bash
docker compose up -d --build         # or: make up; the first build takes a few minutes
docker compose ps                    # api, consumer, postgres, kafka, redis, dashboard "(healthy)"
```

That starts six containers: the API on http://127.0.0.1:8000, the Kafka consumer,
Postgres, Kafka, Redis, and the dashboard on http://127.0.0.1:8501. Kafka holds no
transactions until you run the producer, so the dashboard starts empty.

Or run the API directly for development, against the Compose database or none:

```bash
docker compose up -d postgres redis  # optional: or set PERSIST_DECISIONS=false and
uvicorn src.api.main:app --reload --no-access-log   # VELOCITY_FEATURES=false in .env
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
  "model_version": "20260915-071627-19429cbb",
  "card_id": "card-04512",
  "velocity": {
    "count_1m": 2, "count_5m": 2, "count_1h": 2,
    "amount_1h": 1389.2, "countries_1h": 1, "seconds_since_previous": 48.996
  },
  "model_decision": "block",
  "reasons": []
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
`DATABASE_URL` and `KAFKA_BOOTSTRAP_SERVERS`: on Windows `localhost` tries IPv6
first, which Compose does not listen on, and each new connection then waits about
5 seconds.

## Streaming

Send test-set transactions into Kafka and watch the consumer score them:

```bash
docker compose run --rm producer --limit 1000 --rate 100    # or: make stream
docker compose logs -f consumer
curl "http://127.0.0.1:8000/transactions?decision=block&limit=5"
```

```
INFO src.streaming.consumer consumed 3000 message(s) in 8 batch(es), 20.5s: scored 3000
     (2990 approve, 9 review, 1 block), 0 dead-lettered, 0 duplicate(s) skipped; lag 0
```

The producer and consumer also run on your machine, against the Compose broker:

```bash
python -m scripts.check_kafka                          # topics and partitions
python -m scripts.produce --rate 0                     # the whole test set, full speed
python -m scripts.produce --rate 20 --start 1000       # resume from position 1000
python -m scripts.consume --until-idle 10              # stop once the topic is drained
python -m scripts.consume --group candidate-model      # replay the topic separately
```

- **Messages:** the key is `transaction_id`; the value is the same JSON body
  `POST /score` takes, checked with the same schema. The stream is stricter in one
  way: `transaction_id` is required, because a generated id would give a redelivered
  message a new id and store it twice.
- **Producer:** sends the test set in time order (`Time`), with ids
  `test-row-<row>`, at `--rate` per second on a fixed schedule. Kafka confirms every
  message (`acks=all`, idempotent). Ctrl+C stops cleanly and prints the `--start`
  that resumes after the last confirmed message.
- **Consumer:** for each batch it decodes every message, sends failures to
  `transactions.dlq` (the original bytes, plus the reason and source partition and
  offset in headers), drops repeated ids, scores with one model call, records with
  one insert, and **commits the offsets only after recording**. A crash before the
  commit means the batch is read again; recording is an upsert, so that is harmless.
- **Database outages:** the consumer pauses its partitions, keeps polling so Kafka
  does not hand its partitions to another consumer, and retries with growing delays
  (1, 2, 4, 8, then 15 s). Nothing is committed until the decisions are recorded.
- **Stopping:** Ctrl+C or `docker compose stop consumer` finishes and commits the
  batch in hand, in about 2 s. A restart continues after the last commit.
- **Replay:** a new `--group` reads the whole topic from the start, which is how a
  candidate model could be compared with the live one. Messages are kept for 7 days.

## Velocity features

What a card did in the last minute is one of the oldest fraud signals there is, and
the model cannot see it: it was trained on `V1`–`V28` and `Amount`, one transaction
at a time. Phase 5 adds that memory in Redis and lets it escalate a decision.

```bash
docker compose up -d redis                             # or: make redis
python -m scripts.check_redis                          # version, keys, memory, eviction policy
docker compose run --rm producer --limit 2000 --rate 100
curl "http://127.0.0.1:8000/transactions?decision=review&limit=5"
```

```json
{
  "transaction_id": "test-row-36125", "risk_score": 0.00008,
  "card_id": "card-00000", "model_decision": "approve", "decision": "review",
  "reasons": ["many_in_a_minute", "many_in_an_hour", "back_to_back"],
  "velocity": {"count_1m": 40, "count_1h": 44, "countries_1h": 1, "…": "…"}
}
```

A transaction the model scored at 0.00008 — as innocent as it gets — sent for review
purely on its card's behaviour.

- **Identities.** The Kaggle columns are PCA output, so no card, merchant or country
  survived anonymisation and velocity would have nothing to count. The producer
  invents them (`src/features/entities.py`) by hashing the transaction id: stable
  across restarts and redeliveries, heavy-tailed so some cards are busy enough to
  watch, and **never derived from the fraud label**, so the rules cannot be rigged to
  look like they catch fraud. They are optional fields: a transaction without a card
  is scored by the model alone.
- **Storage.** One sorted set per card, `vel:card:<card_id>`, scored by arrival time,
  with the amount and country inside the member so one read answers every feature.
  Each measurement is five commands — trim, add, cap, read, expire — sent as one
  pipeline, and a whole batch goes in one round trip.
- **Features.** Transactions in the last 1 m / 5 m / 1 h, amount in the last hour,
  distinct countries in the last hour, and seconds since that card's previous
  transaction. They are returned with the score and stored with the decision.
- **Rules** (`src/features/rules.py`). Thresholds live in `.env`. They escalate
  `approve` to `review` and nothing else: a wrong block costs a customer their
  payment ($50 in the cost model) against $5 for an analyst, so velocity can ask for
  a human but never refuse a payment, and never soften a decision the model made.
  Every rule that fires is named in the response and the stored row.
- **Time is arrival time**, not the dataset's `Time` column. The producer replays two
  days in minutes, so the dataset's clock would put every transaction in one window.
- **Nothing in Redis is durable.** It runs with persistence off and a 256 MB ceiling
  with `allkeys-lru`; every key is derived from transactions Kafka still holds, and a
  card that goes quiet expires by itself within the hour.
- **Redis outages.** The transaction is scored without its features, and the response
  says `"velocity": null`. After a failure Redis is left alone for 10 seconds, so
  only one transaction per window pays the ~4 s a vanished container name takes to
  fail. `/health` reports `"velocity": "unavailable"` and stays `200`; when Redis
  returns, the log says how many transactions went without.
- **Switching it off.** `VELOCITY_FEATURES=false` scores on the transaction alone and
  never opens a connection.

## Watching it run

Phase 6 is about what you can see from outside: what the service says about itself,
what an analyst sees, and what happens when you push it.

```bash
docker compose up -d --build                      # the dashboard comes up with everything else
docker compose run --rm producer --limit 20000 --rate 500
start http://127.0.0.1:8501                       # the dashboard; on macOS/Linux: open
curl -s http://127.0.0.1:8000/metrics | grep ^fraud_ | head
python -m scripts.load_test --concurrency 1,2,4 --seconds 10 --out data/load-test.json
```

- **Metrics** (`src/observability/metrics.py`). The API serves `/metrics` for
  Prometheus: requests by route, method and status, request duration, decisions by
  outcome and source, escalations, velocity duration and failures, and whether a
  model is loaded. Every label is **bounded** — the route comes from the route
  template (`/transactions/{transaction_id}`), never the URL, so a million
  transaction ids cannot become a million time series, which is the classic way to
  kill a Prometheus server. Histogram buckets straddle the latencies this service
  actually produces, because a p95 read off the default buckets would land between
  5 ms and 10 ms and tell you nothing. Scrapes of `/metrics` are not counted as
  requests.
- **The consumer has no web server**, so it starts a small one of its own on 8001 for
  `/metrics` alone: messages scored, batch sizes and durations, dead letters, and
  **consumer lag per partition**. Compose's health check scrapes that endpoint:
  answering proves the loop is not wedged. Lag is deliberately *not* part of the
  health check — lag usually means the producer sped up, and restarting a healthy
  consumer would only make the backlog worse. It is an alert for a person, not a
  trigger for Docker. `CONSUMER_METRICS_PORT=0` switches the server off.
- **Dashboard** (`dashboard/`). Decisions over time, the risk distribution with the
  two thresholds drawn on it, which rules fired, the review queue an analyst would
  work from, and any card's own history. It is a **reader**: it opens its own pool,
  runs SELECTs, and is never in the path of a decision, so a slow dashboard cannot
  slow a payment — and it has no model in its image, so it could not score one even
  by accident. Every query lives in `dashboard/data.py` and is tested against real
  Postgres without a browser; `dashboard/app.py` is layout alone. Reads are capped at
  200,000 rows and say so when they truncate, and results are cached for ten seconds.
  It has its own image and its own pinned requirements, which is not tidiness: the
  dashboard image is 1.26 GB against the API's 805 MB, and all of that difference is
  front end that would otherwise ship with every scoring container.
- **Load test** (`scripts/load_test.py`). Closed loop, rising concurrency, client and
  server latency side by side, and it reports where the knee is. Percentiles are
  nearest rank, so a reported p99 is a request that someone really waited for rather
  than an interpolation between two of them. What it found is in
  [Results](#results-so-far): one process saturates at about one concurrent caller,
  and the OpenMP threads were costing 30x.

## API

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | | `status`, the model version and thresholds being served, `decision_store` and `velocity`: `ok`, `unavailable` or `disabled` |
| `POST /score` | one transaction: `Time`, `V1`–`V28`, `Amount`, optional `transaction_id`, `card_id`, `merchant`, `country` | risk score, decision, thresholds, model version, velocity features and the rules that fired |
| `POST /score/batch` | `{"transactions": [...]}`, 1–1,000 with unique ids | `{"results": [...]}` in request order |
| `GET /transactions/{transaction_id}` | | the recorded decision plus `scored_at`, or 404 |
| `GET /transactions?decision=block&limit=50` | | `{"decisions": [...]}`, newest first; `limit` 1–500 |

- **Decisions:** `block` if risk ≥ `BLOCK_THRESHOLD`, otherwise `review` if risk ≥
  `REVIEW_THRESHOLD`, otherwise `approve`. Thresholds come from `.env`; the service
  logs a warning at startup if they differ from the ones stored with the model. A
  velocity rule can then turn `approve` into `review`; `model_decision` keeps what
  the model said and `reasons` lists the rules that changed it.
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
   version, `scored_at` in UTC, and from Phase 5 the card, the velocity counts, the
   model's own decision and the rules that changed it. The transaction's own
   features are not stored. A batch is saved
   with one `INSERT ... ON CONFLICT DO UPDATE`, so replaying a transaction, as
   Kafka's at-least-once delivery will in Phase 4, updates its row instead of
   duplicating it. Check constraints keep decisions and scores valid in the
   database itself. The schema is created on first use, and missing **nullable**
   columns are added to a table that already holds rows — which is how Phase 5's
   columns reached 56,746 existing decisions. Anything else (a type change, a NOT
   NULL column, a drop) is refused and needs a migration tool such as Alembic.
8. **Container** (`Dockerfile`, `docker-compose.yml`). A two-stage image on
   `python:3.10-slim` running as a non-root user, with the model copied in, so an
   image tag identifies exactly one model. It installs `requirements-api.txt`,
   whose exact versions a test keeps equal to the training environment: the model
   is a pickle and must load under the libraries it was trained with (`xgboost-cpu`
   leaves out ~200 MB of GPU libraries). Compose adds Postgres 16 with a health
   check and a named volume, and publishes both ports on `127.0.0.1` only.
9. **Kafka** (`docker-compose.yml`, `src/streaming/kafka.py`). One Kafka 4.3 broker
   in KRaft mode with two listeners: `kafka:29092` for containers and
   `127.0.0.1:9092` for your machine. A broker tells each client which address to
   use next, so each listener advertises one that works from where its clients run.
   `kafka-init` creates `transactions` (3 partitions) and `transactions.dlq` (1).
   Automatic topic creation is off, so a misspelled topic name fails at startup
   instead of creating an empty topic that a consumer would wait on forever.
10. **Stream processing** (`src/streaming/`). `messages.py` defines the message
    format and dead letters, `producer.py` paces and confirms sends, and
    `consumer.py` runs the decode → dead-letter → deduplicate → score → record →
    commit cycle, with scoring and database work in a worker thread so the Kafka
    session stays alive. The consumer runs the API's image with a different command;
    the producer has its own build target, which adds a parquet reader the API does
    not need.
11. **Velocity** (`src/features/`, `docker-compose.yml`). Redis 8 with persistence
    off, a memory ceiling and `allkeys-lru`, because everything in it is derived from
    transactions Kafka still holds. `velocity.py` keeps one sorted set per card and
    measures a whole batch in one pipeline; `entities.py` invents the card, merchant
    and country by hashing the transaction id. The client is synchronous, like the
    decision store, and the consumer calls both from a worker thread, so there is one
    implementation rather than a sync and an async copy of every query.
12. **Rules** (`src/features/rules.py`). Run after the model, on the measured
    features. They escalate `approve` to `review`, never block and never soften, and
    each rule that fires is named in the response and the row. Thresholds are set
    from plausible cardholder behaviour and then checked against analyst capacity,
    never fitted to the fraud labels: the card ids are synthetic, so fitting them to
    labels would fit noise.
13. **Metrics** (`src/observability/metrics.py`). One module defines every metric, so
    the API and the consumer cannot drift into measuring the same thing differently.
    Labels are bounded by construction; anything unbounded, such as a transaction id,
    belongs in a log line, not a label. `record_results` is called by both paths and
    counts an escalation only when the rules actually changed the model's decision.
14. **Dashboard** (`dashboard/`). Queries in `data.py`, layout in `app.py`, which is
    what makes the queries testable without a browser, and a build target of its own
    in the same Dockerfile so the scoring image never grows a front end.

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
| `src/streaming/kafka.py` | connecting to Kafka and checking its topics |
| `src/streaming/messages.py` | message format, validation and dead letters |
| `src/streaming/producer.py`, `scripts/produce.py` | streams the test set into Kafka |
| `src/streaming/consumer.py`, `scripts/consume.py` | scores the stream and records the decisions |
| `scripts/check_kafka.py` | checks that Kafka answers and the topics exist |
| `src/features/entities.py` | the synthetic card, merchant and country |
| `src/features/redis_client.py` | the Redis connection: fails fast, never retries ten times |
| `src/features/velocity.py` | the velocity store: sorted sets, windows, the retry window |
| `src/features/rules.py` | the velocity rules and what each one means |
| `scripts/check_redis.py` | checks that Redis answers and shows what it holds |
| `src/observability/metrics.py` | every Prometheus metric, and what its labels may contain |
| `src/api/routes/metrics.py` | `GET /metrics` for the API |
| `dashboard/data.py` | every query the dashboard runs, tested without a browser |
| `dashboard/app.py` | the Streamlit layout, and nothing else |
| `scripts/load_test.py` | the closed-loop load test and its knee detection |
| `Dockerfile`, `requirements-api.txt`, `requirements-producer.txt`, `requirements-dashboard.txt`, `.dockerignore` | the API/consumer, producer and dashboard images and their pinned libraries |
| `docker-compose.yml` | the pipeline: API, consumer, Postgres, Kafka, Redis, dashboard, and the producer on demand |
| `notebooks/` | one notebook per step, each ending in a findings cell; nothing imports from here |
| `tests/` | 503 tests; run with `pytest`. Storage tests run on SQLite and, with `docker compose up`, on real Postgres too (`pytest -m postgres`). Kafka tests (`pytest -m kafka`) use throwaway topics on the Compose broker, and Redis tests (`pytest -m redis`) a database of their own. Without the services running, those tests are skipped |

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
- **Anonymised features, and invented identities.** `V1`–`V28` cannot be interpreted
  or extended, and no card, merchant or country survived the PCA. The velocity
  features therefore run on synthetic identities, which are deliberately independent
  of the fraud label: Phase 5 can show what the rules cost, and **cannot show that
  they catch more fraud**. On real identities that is exactly what they would be
  judged on.
- **Small fraud counts.** 95 test fraud cases means wide confidence intervals; treat
  differences of a few percentage points as noise.
- **Thresholds belong to one model.** The probability scale shifts between retrains,
  so thresholds must be re-checked each time the model is retrained.
- **Pickled models run code when loaded.** Only load model files you trained yourself.
- **The API is a local demo.** It has no authentication, rate limiting or TLS, and
  the Compose database password is a demo value. The dashboard is open to anyone who
  can reach port 8501, which is why Compose publishes it on `127.0.0.1` only.
- **The load-test numbers are from a laptop.** One uvicorn process, Docker Desktop,
  and everything else running on a Windows machine at the time. Run to run, the same
  stage varied between 5/s and 41/s. The shape — flat throughput, latency rising with
  concurrency — reproduces; treat the absolute numbers as an order of magnitude, not
  a benchmark, and re-measure on the hardware you would actually deploy on.
- **Metrics are exported, not collected.** The service exposes `/metrics` in
  Prometheus format; there is no Prometheus server, Grafana or alerting in this
  repo, so nothing retains the series or pages anyone.
- **The API can lose decisions; the stream cannot.** `POST /score` records best
  effort, so decisions made while Postgres is down, or in the 10 seconds after it
  returns, are logged as a count but not stored. The Kafka consumer waits for the
  database instead and loses nothing. A caller that needs both an immediate answer
  and a guaranteed record would also write the transaction to Kafka.
- **One Kafka broker.** Every topic has a single copy, so losing the broker's volume
  loses unconsumed messages. A production cluster runs three or more brokers with
  replicated topics.
- **Consumer lag is reported, not acted on.** The consumer exports lag per partition
  and Compose checks that it answers a scrape, but nothing reacts to a growing
  backlog: there is no autoscaling and no alert, because lag needs a judgement
  (has the producer sped up, or has the consumer stalled?) that a restart cannot make.
- **Velocity depends on how fast the stream runs.** The rules measure arrival time,
  and the producer compresses two days into minutes, so a fast replay makes every
  card look frantic (14.5% escalated at 100/s against 0.79% at the data's own pace).
  Thresholds for a real deployment have to be set against real traffic.
- **Redis keeps nothing.** It runs without persistence, so a restart empties every
  card's history and the first transactions afterwards look like a card's first.
  Nothing is lost that matters — decisions are in Postgres, transactions in Kafka —
  but the features are briefly wrong, and there is one Redis, not a replicated pair.
- **Only additive schema changes.** Missing nullable columns are added automatically;
  a type change, a NOT NULL column or a drop is refused and needs a migration tool
  such as Alembic.

## What I would do next

The project is finished as a demonstration. Turning it into something a payments team
could run means the parts a public dataset cannot teach:

1. **Real identities, and then a feature store.** Velocity currently runs on cards
   invented by hashing a transaction id, so it can show what the rules cost and never
   that they catch fraud. With real cards, the same sorted sets would need to survive
   a Redis restart and be shared by every scorer — a feature store, with the training
   pipeline reading the same definitions the API does, or the features drift apart.
2. **Monitoring the model, not just the service.** `/metrics` says how fast the
   service answers and what it decided. It says nothing about whether the score
   distribution has moved, whether precision at the review threshold still holds, or
   how many reviewed transactions turned out to be fraud. Labels arrive weeks later
   through chargebacks, so that loop is a scheduled job over the decisions table, not
   a live metric.
3. **Scale with processes.** One uvicorn worker saturates at about one concurrent
   caller. Several workers behind a load balancer, sized from the load test rather
   than guessed, and the Kafka consumer group grown to its three partitions.
4. **The operational floor.** Authentication and rate limiting on the API, TLS,
   secrets that are not demo values, Alembic for schema changes that are not additive,
   a replicated Kafka cluster, and a Prometheus and Grafana that actually retain the
   series this service exports.

What the project does show is the shape of the thing: one scoring core behind two
entry points so a transaction cannot get two different answers, decisions recorded
with the thresholds and model version that produced them, rules that can ask for a
human but never refuse a payment, every dependency allowed to fail without taking
scoring down, and numbers measured rather than assumed — including the ones that were
embarrassing, like a model library quietly starting eight threads per prediction.
