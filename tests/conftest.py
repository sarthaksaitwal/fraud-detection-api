"""Fixtures shared across test modules."""

import os
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.config import settings
from src.ml.preprocess import TARGET, V_COLUMNS
from src.storage.db import create_db_engine
from src.storage.store import failure_reason

# The columns that separate fraud most strongly in the real data (Step 1.1, Cell 9).
FRAUD_SIGNAL_COLUMNS = ["V3", "V10", "V12", "V14", "V17"]


@pytest.fixture(autouse=True)
def no_configured_database(monkeypatch):
    """No test writes to the database in .env. Tests that need one open their own."""
    monkeypatch.setattr(settings, "persist_decisions", False)


@pytest.fixture(scope="session")
def postgres_engine():
    """An engine on a throwaway schema in a real Postgres, or a skip if none answers.

    Uses TEST_DATABASE_URL, else DATABASE_URL: normally the Compose database
    from `make up`. Tests get their own schema, dropped afterwards, so they
    never see or touch the decisions already stored there.
    """
    url = os.environ.get("TEST_DATABASE_URL", settings.database_url)
    if not url.startswith("postgresql"):
        pytest.skip(f"not a Postgres URL: {url.split(':', 1)[0]}")

    schema = f"test_{uuid4().hex[:12]}"
    fast_fail = {"connect_timeout": 2}
    admin = create_db_engine(url, connect_args=fast_fail)
    try:
        with admin.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    except SQLAlchemyError as exc:
        admin.dispose()
        pytest.skip(f"no Postgres answering at {admin.url!r}: {failure_reason(exc)}")

    # Every connection from this engine looks in the throwaway schema first.
    engine = create_db_engine(url, connect_args=fast_fail | {"options": f"-csearch_path={schema}"})
    yield engine
    engine.dispose()
    with admin.begin() as connection:
        connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
    admin.dispose()


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """2,000 fake transactions shaped like the Kaggle data, 2.5% fraud.

    Fraud rows are shifted by -7 on the columns that separate fraud in the real
    data, so an anomaly detector has a real signal to find.
    """
    rng = np.random.default_rng(0)
    n, n_fraud = 2000, 50
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=V_COLUMNS)
    df.insert(0, "Time", rng.uniform(0, 172_800, n))
    df["Amount"] = rng.exponential(80, n)
    df[TARGET] = 0
    df.loc[: n_fraud - 1, TARGET] = 1
    df.loc[: n_fraud - 1, FRAUD_SIGNAL_COLUMNS] -= 7
    return df
