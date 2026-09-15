"""Fixtures shared across test modules."""

import numpy as np
import pandas as pd
import pytest

from src.ml.preprocess import TARGET, V_COLUMNS

# The columns that separate fraud most strongly in the real data (Step 1.1, Cell 9).
FRAUD_SIGNAL_COLUMNS = ["V3", "V10", "V12", "V14", "V17"]


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
