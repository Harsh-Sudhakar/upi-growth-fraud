"""Feature engineering for IEEE-CIS, as a fit/transform object.

Everything learnt from data (frequency maps, group statistics, category codes)
is fitted on the training period only and then applied to later periods, so no
information flows backwards in time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from upi.fraud.data import TARGET, TIME_COL

SECONDS_PER_DAY = 86_400

# Columns whose frequency (how common the value is) is predictive
FREQ_COLS = ["card1", "card2", "card3", "card5", "addr1", "P_emaildomain", "R_emaildomain",
             "DeviceInfo", "uid"]
# Group keys for "is this amount unusual for this card / user?"
AGG_KEYS = ["card1", "uid"]
DROP_COLS = ["TransactionID", TIME_COL, TARGET, "uid", "_day", "_d1_origin"]


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level features that need no fitting."""
    out = pd.DataFrame(index=df.index)
    t = df[TIME_COL]
    day = t // SECONDS_PER_DAY
    out["hour"] = ((t // 3600) % 24).astype("float32")
    out["weekday"] = (day % 7).astype("float32")
    amt = df["TransactionAmt"].astype("float64")
    out["amt_log"] = np.log1p(amt).astype("float32")
    # Non-round cents often mean a currency conversion, i.e. a foreign card
    out["amt_cents"] = (amt - np.floor(amt)).round(3).astype("float32")

    if {"P_emaildomain", "R_emaildomain"} <= set(df.columns):
        p = df["P_emaildomain"].astype("string")
        r = df["R_emaildomain"].astype("string")
        out["email_match"] = (p == r).astype("float32").where(p.notna() & r.notna())
        out["P_email_provider"] = p.str.split(".").str[0]
        out["R_email_provider"] = r.str.split(".").str[0]
    if "DeviceInfo" in df.columns:
        out["device_brand"] = df["DeviceInfo"].astype("string").str.split(r"[ /_-]").str[0].str.lower()

    # Approximate user id: card + billing address + the day the card was first seen
    # (D1 = days since the card's first transaction, so day - D1 is constant per card)
    out["_day"] = day
    if {"card1", "addr1", "D1"} <= set(df.columns):
        out["_d1_origin"] = (day - df["D1"]).round()
        out["uid"] = (df["card1"].astype("string") + "_" + df["addr1"].astype("string").fillna("na")
                      + "_" + out["_d1_origin"].astype("string").fillna("na"))
    return out


class FraudFeatureBuilder:
    def fit(self, df: pd.DataFrame) -> "FraudFeatureBuilder":
        full = self._combine(df)
        self.freq_maps_ = {
            c: full[c].astype("string").value_counts(normalize=True, dropna=False)
            for c in FREQ_COLS if c in full.columns
        }
        self.amt_stats_ = {
            key: full.groupby(full[key].astype("string"), dropna=False)["TransactionAmt"]
                     .agg(["mean", "std"]).astype("float32")
            for key in AGG_KEYS if key in full.columns
        }
        self.categorical_ = [c for c in full.columns
                             if c not in DROP_COLS
                             and (isinstance(full[c].dtype, pd.CategoricalDtype)
                                  or full[c].dtype == object or pd.api.types.is_string_dtype(full[c]))]
        self.cat_codes_ = {
            c: {v: i for i, v in enumerate(full[c].astype("string").dropna().unique())}
            for c in self.categorical_
        }
        self.feature_names_ = list(self._encode(full).columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._encode(self._combine(df))[self.feature_names_]

    # ---------------------------------------------------------------- internals
    @staticmethod
    def _combine(df: pd.DataFrame) -> pd.DataFrame:
        base = add_base_features(df)
        return pd.concat([df, base], axis=1)

    def _encode(self, full: pd.DataFrame) -> pd.DataFrame:
        new = {}
        for c, freq in self.freq_maps_.items():
            new[f"{c}_freq"] = full[c].astype("string").map(freq).astype("float32").fillna(0.0)
        for key, st in self.amt_stats_.items():
            k = full[key].astype("string")
            mean = k.map(st["mean"]).astype("float32")
            std = k.map(st["std"]).astype("float32")
            new[f"amt_to_mean_{key}"] = full["TransactionAmt"] / mean
            new[f"amt_z_{key}"] = (full["TransactionAmt"] - mean) / std.replace(0, np.nan)
        for c in self.categorical_:
            new[c] = full[c].astype("string").map(self.cat_codes_[c]).astype("float32").fillna(-1.0)
        numeric = full.drop(columns=[c for c in DROP_COLS + self.categorical_ if c in full.columns])
        numeric = numeric.drop(columns=[c for c in new if c in numeric.columns])
        out = pd.concat([numeric.astype("float32"), pd.DataFrame(new, index=full.index).astype("float32")], axis=1)
        return out.replace([np.inf, -np.inf], np.nan)
