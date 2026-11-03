"""Fixed train/validation/test partitioning shared by Q1, Q2, Q4 and Q5."""
from sklearn.model_selection import train_test_split

from src.config import SEED

TEST_SIZE = 0.20
VAL_SIZE = 0.20  # fraction of the *whole* dataset


def train_val_test_split(X, y, stratify: bool = True, seed: int = SEED):
    """60 / 20 / 20 split. Returns X_train, X_val, X_test, y_train, y_val, y_test.

    The test part is split off first and must not be touched until the final
    evaluation in Q5(e).
    """
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed,
        stratify=y if stratify else None,
    )
    val_rel = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, random_state=seed,
        stratify=y_tmp if stratify else None,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def bank_split(include_duration: bool = False, stratify: bool = True, seed: int = SEED):
    """Convenience wrapper: load, clean and split the Bank Marketing data."""
    from src.data import bank_xy, clean_bank, load_bank_raw

    X, y = bank_xy(clean_bank(load_bank_raw()), include_duration=include_duration)
    return train_val_test_split(X, y, stratify=stratify, seed=seed)
