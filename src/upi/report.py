"""Tiny helpers for writing Markdown reports and JSON metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from upi import config


def md_table(df: pd.DataFrame, floatfmt: str = "{:,.3f}", index: bool = False) -> str:
    """Render a DataFrame as a GitHub Markdown table without extra dependencies."""
    if index:
        df = df.reset_index()

    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def write_report(name: str, text: str) -> Path:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / name
    path.write_text(text.strip() + "\n")
    return path


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    raise TypeError(type(o))


def write_metrics(name: str, metrics: dict) -> Path:
    out_dir = config.REPORTS_DIR / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(metrics, indent=2, default=_jsonable))
    return path
