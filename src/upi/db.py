"""SQLite access: build the analytics marts and run the SQL files in ``sql/``."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from upi import config

# Columns a query may be grouped by -- keeps {segment} substitution injection-safe.
SEGMENTS = {"channel", "city_tier", "age_group", "device", "activated_d7", "cohort_month"}


def connect(db_path: Path = config.DB_PATH) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found -- run `make simulate` first.")
    return sqlite3.connect(db_path)


def load_sql(name: str, **params) -> str:
    """Read ``sql/<name>`` and fill ``{placeholders}``."""
    if "segment" in params and params["segment"] not in SEGMENTS:
        raise ValueError(f"segment must be one of {sorted(SEGMENTS)}")
    text = (config.SQL_DIR / name).read_text()
    return text.format(**params) if params else text


def query(con: sqlite3.Connection, name: str, **params) -> pd.DataFrame:
    return pd.read_sql_query(load_sql(name, **params), con)


def build_marts(con: sqlite3.Connection) -> None:
    con.executescript(load_sql("00_build_marts.sql"))
    con.commit()


def marts_exist(con: sqlite3.Connection) -> bool:
    row = con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='user_metrics'"
    ).fetchone()
    return bool(row[0])
