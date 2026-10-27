"""Download, load and split the IEEE-CIS Fraud Detection data."""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from upi import config

COMPETITION = "ieee-fraud-detection"
TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"
TARGET = "isFraud"
TIME_COL = "TransactionDT"     # seconds from an undisclosed reference point
CACHE_FILE = "ieee_train_merged.parquet"


def download(raw_dir: Path = config.RAW_DIR) -> None:
    """Fetch the two training files with the Kaggle API.

    Needs ``~/.kaggle/kaggle.json`` (Kaggle -> Settings -> API -> Create New Token)
    and the competition rules accepted on
    https://www.kaggle.com/competitions/ieee-fraud-detection/rules
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    kaggle = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    for name in (TRANSACTION_FILE, IDENTITY_FILE):
        if (raw_dir / name).exists():
            print(f"{name} already present")
            continue
        subprocess.run([kaggle, "competitions", "download", "-c", COMPETITION, "-f", name, "-p", str(raw_dir)],
                       check=True)
        archive = raw_dir / f"{name}.zip"
        if archive.exists():
            with zipfile.ZipFile(archive) as z:
                z.extractall(raw_dir)
            archive.unlink()
    print(f"data ready in {raw_dir}")


def _read_dtypes() -> dict:
    """float32 for the wide numeric blocks so the full file fits in ~1.5 GB of RAM."""
    dtypes = {f"V{i}": "float32" for i in range(1, 340)}
    dtypes.update({f"C{i}": "float32" for i in range(1, 15)})
    dtypes.update({f"D{i}": "float32" for i in range(1, 16)})
    dtypes.update({"TransactionAmt": "float32", "dist1": "float32", "dist2": "float32"})
    return dtypes


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == "float64":
            df[col] = df[col].astype("float32")
        elif dtype == "int64" and col not in (TIME_COL, "TransactionID"):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif dtype == "object":
            df[col] = df[col].astype("category")
    return df


def load(raw_dir: Path = config.RAW_DIR, nrows: int | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Transactions left-joined to identity info (only ~24% of rows have identity data)."""
    cache = raw_dir / CACHE_FILE
    if use_cache and nrows is None and cache.exists():
        return pd.read_parquet(cache)
    tx_path = raw_dir / TRANSACTION_FILE
    if not tx_path.exists():
        raise FileNotFoundError(f"{tx_path} not found -- run `make data` (see README for Kaggle setup).")
    tx = pd.read_csv(tx_path, nrows=nrows, dtype=_read_dtypes())
    identity = pd.read_csv(raw_dir / IDENTITY_FILE)
    identity.columns = [c.replace("-", "_") for c in identity.columns]   # id-01 vs id_01 across files
    df = reduce_memory(tx.merge(identity, on="TransactionID", how="left"))
    df = df.sort_values(TIME_COL, kind="stable").reset_index(drop=True)
    if use_cache and nrows is None:
        df.to_parquet(cache, index=False)
    return df


def time_split(df: pd.DataFrame, valid_frac: float = 0.2, test_frac: float = 0.2
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / validation / test split.

    Fraud patterns drift and fraudsters reuse cards over time, so a random split
    leaks future information and inflates scores. We train on the past and
    evaluate on the future, exactly as the model would be used.
    """
    df = df.sort_values(TIME_COL, kind="stable")
    n = len(df)
    i_valid = int(n * (1 - valid_frac - test_frac))
    i_test = int(n * (1 - test_frac))
    return df.iloc[:i_valid], df.iloc[i_valid:i_test], df.iloc[i_test:]


def undersample(df: pd.DataFrame, ratio: int = 5, seed: int = 42) -> pd.DataFrame:
    """Keep every fraud and ``ratio`` random genuine transactions per fraud."""
    rng = np.random.default_rng(seed)
    fraud = df[df[TARGET] == 1]
    genuine = df[df[TARGET] == 0]
    keep = rng.choice(len(genuine), size=min(len(genuine), ratio * len(fraud)), replace=False)
    return pd.concat([fraud, genuine.iloc[np.sort(keep)]]).sort_values(TIME_COL, kind="stable")
