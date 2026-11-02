"""Shared constants for every experiment. Import from here; never hard-code paths or seeds."""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"

BANK_CSV = DATA_RAW / "bank-additional-full.csv"
HOUSING_CSV = DATA_RAW / "california_housing.csv"
