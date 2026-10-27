from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from upi import db, simulate
from upi.config import SimConfig


@pytest.fixture(scope="session")
def sim_tables():
    return simulate.simulate(SimConfig(n_users=8_000, seed=7))


@pytest.fixture(scope="session")
def sim_con(sim_tables, tmp_path_factory):
    out = tmp_path_factory.mktemp("sim")
    con = db.connect(simulate.save(sim_tables, out))
    db.build_marts(con)
    yield con
    con.close()


def make_fake_ieee(n: int = 30_000, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame with the IEEE-CIS schema (a subset of columns) and a learnable fraud signal.

    Only for testing the pipeline mechanics -- never for reporting results.
    """
    rng = np.random.default_rng(seed)
    product = rng.choice(list("WCRHS"), n, p=[0.74, 0.12, 0.06, 0.05, 0.03])
    card1 = rng.integers(1000, 4000, n)
    c1 = rng.poisson(2, n).astype(float)
    amt = np.round(rng.lognormal(4, 1, n), 2)
    foreign = rng.random(n) < 0.2
    amt = np.where(foreign, amt + rng.random(n).round(3), np.floor(amt))
    logit = -4.2 + 1.5 * (product == "C") + 0.25 * c1 + 0.9 * foreign + rng.normal(0, 0.5, n)
    fraud = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "TransactionID": np.arange(2_987_000, 2_987_000 + n),
        "isFraud": fraud,
        "TransactionDT": np.sort(rng.integers(86_400, 86_400 * 180, n)),
        "TransactionAmt": amt.astype("float32"),
        "ProductCD": pd.Categorical(product),
        "card1": card1,
        "card2": rng.choice([np.nan, 111.0, 321.0, 555.0], n),
        "card4": pd.Categorical(rng.choice(["visa", "mastercard", "discover", "american express"], n,
                                           p=[0.65, 0.3, 0.03, 0.02])),
        "card5": rng.choice([226.0, 224.0, 166.0], n),
        "card6": pd.Categorical(rng.choice(["debit", "credit"], n)),
        "addr1": rng.choice([np.nan, 299.0, 325.0, 204.0], n),
        "P_emaildomain": pd.Categorical(rng.choice(["gmail.com", "yahoo.com", "hotmail.com", None], n)),
        "R_emaildomain": pd.Categorical(rng.choice(["gmail.com", "anonymous.com", None], n)),
        "C1": c1.astype("float32"),
        "D1": rng.integers(0, 300, n).astype("float32"),
        "M4": pd.Categorical(rng.choice(["M0", "M1", "M2", None], n)),
        "V1": rng.normal(0, 1, n).astype("float32"),
        "DeviceType": pd.Categorical(rng.choice(["desktop", "mobile", None], n)),
        "DeviceInfo": pd.Categorical(rng.choice(["Windows", "iOS Device", "SM-G960U Build/R16NW", None], n)),
    })
    return df


@pytest.fixture(scope="session")
def fake_ieee():
    return make_fake_ieee()
