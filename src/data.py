"""Loading and *row-wise, statistics-free* cleaning of the raw datasets.

Everything in this module is deterministic and uses no information from other
rows or from the target, so it is safe to apply before splitting. Anything that
learns statistics (imputation values, scaling, encodings, feature selection)
lives in ``src/preprocessing.py`` and is fitted on training data only.
"""
import numpy as np
import pandas as pd

from src.config import BANK_CSV, HOUSING_CSV

TARGET = "y"

# Feature groups of the Bank Marketing data (after ``clean_bank``).
BANK_NUMERIC = [
    "age", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]
BANK_BINARY_DERIVED = ["contacted_before"]  # derived from the pdays sentinel
BANK_CATEGORICAL = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
LEAKY_FEATURE = "duration"  # known only after the call -> target leakage


def load_bank_raw() -> pd.DataFrame:
    """Raw file exactly as distributed by UCI (separator ';')."""
    return pd.read_csv(BANK_CSV, sep=";")


def clean_bank(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Row-wise cleaning only (no fitted statistics).

    * exact duplicate rows removed (they could land in both train and test);
    * the string ``"unknown"`` becomes a real missing value (NaN);
    * ``pdays == 999`` ("never contacted") is a sentinel, not a number of days:
      it becomes NaN and an explicit indicator ``contacted_before`` is added;
    * the target is encoded as 1 = "yes", 0 = "no".
    """
    df = df.copy()
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    df = df.replace("unknown", np.nan)
    df["contacted_before"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].where(df["pdays"] != 999, np.nan)
    df[TARGET] = (df[TARGET] == "yes").astype(int)
    return df


def bank_xy(df: pd.DataFrame, include_duration: bool = False):
    """Split a cleaned frame into X, y. ``duration`` is excluded by default."""
    cols = BANK_NUMERIC + BANK_BINARY_DERIVED + BANK_CATEGORICAL
    if include_duration:
        cols = cols + [LEAKY_FEATURE]
    return df[cols].copy(), df[TARGET].copy()


def load_housing_raw() -> pd.DataFrame:
    return pd.read_csv(HOUSING_CSV)
