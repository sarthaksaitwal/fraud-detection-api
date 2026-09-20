.DEFAULT_GOAL := help
PY := ./.venv/Scripts/python.exe   # macOS/Linux: ./.venv/bin/python

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

install:  ## create the venv and install dev dependencies
	python -m venv .venv && $(PY) -m pip install -r requirements-dev.txt

data:     ## download the Kaggle dataset into data/raw/
	bash scripts/download_data.sh

train:    ## train the model and write models/*.joblib   (Phase 1)
	$(PY) -m src.ml.train

api:      ## run the API with autoreload                 (Phase 2)
	$(PY) -m uvicorn src.api.main:app --reload

test:     ## run the test suite
	$(PY) -m pytest

lint:     ## ruff + black check
	$(PY) -m ruff check . && $(PY) -m black --check .

fmt:      ## autoformat
	$(PY) -m ruff check --fix . && $(PY) -m black .

image:    ## build the API image                         (Phase 3)
	docker build -t fraud-detection-api .

up:       ## start the pipeline in Docker: API, consumer, Postgres, Kafka
	docker compose up -d --build

down:     ## stop the containers; decisions are kept
	docker compose down

logs:     ## follow the API and consumer logs
	docker compose logs -f api consumer

stream:   ## stream 1,000 test transactions in through the producer container (Phase 4)
	docker compose run --rm producer --limit 1000 --rate 100

psql:     ## open a SQL shell on the Compose database
	docker compose exec postgres psql -U fraud -d fraud

init-db:  ## create the decisions table in DATABASE_URL  (Phase 3)
	$(PY) -m scripts.init_db

test-postgres:  ## run only the tests that need Postgres (make up first)
	$(PY) -m pytest -m postgres -rs

kafka:    ## start Kafka and create its topics          (Phase 4)
	docker compose up -d kafka kafka-init

check-kafka:  ## check Kafka answers and the topics exist
	$(PY) -m scripts.check_kafka

test-kafka:  ## run only the tests that need Kafka (make kafka first)
	$(PY) -m pytest -m kafka -rs

redis:    ## start Redis, where recent card activity lives  (Phase 5)
	docker compose up -d redis

check-redis:  ## check Redis answers and show what it holds
	$(PY) -m scripts.check_redis

test-redis:  ## run only the tests that need Redis (make redis first)
	$(PY) -m pytest -m redis -rs

produce:  ## stream the test set into Kafka at PRODUCER_RATE_PER_SEC
	$(PY) -m scripts.produce

consume:  ## score transactions from Kafka into Postgres (Ctrl+C to stop)
	$(PY) -m scripts.consume

.PHONY: help install data train api test lint fmt image up down logs psql init-db test-postgres kafka check-kafka test-kafka redis check-redis test-redis produce consume stream

