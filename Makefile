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

up:       ## start the API and Postgres in Docker        (Phase 3)
	docker compose up -d --build

down:     ## stop the containers; decisions are kept
	docker compose down

logs:     ## follow the API logs
	docker compose logs -f api

psql:     ## open a SQL shell on the Compose database
	docker compose exec postgres psql -U fraud -d fraud

init-db:  ## create the decisions table in DATABASE_URL  (Phase 3)
	$(PY) -m scripts.init_db

test-postgres:  ## run only the tests that need Postgres (make up first)
	$(PY) -m pytest -m postgres -rs

.PHONY: help install data train api test lint fmt image up down logs psql init-db test-postgres

