"""Simulate a UPI payments platform: 100K users over 18 months.

Pipeline: acquisition -> experiment assignment -> KYC -> activation ->
lifetime (churn) -> monthly transactions.

A hidden per-user ``engagement`` score drives behaviour and is never written
out, so the analysis has to infer it from behaviour -- as with real data.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from upi import config
from upi.config import SimConfig

CHANNELS = ["organic", "referral", "paid"]
CHANNEL_P = [0.50, 0.25, 0.25]
CHANNEL_ENGAGEMENT = {"organic": 1.0, "referral": 1.3, "paid": 0.8}
TIER_ENGAGEMENT = {1: 1.2, 2: 1.0, 3: 0.85}
AGE_GROUPS = ["18-24", "25-34", "35-44", "45+"]

# Lognormal (mu, sigma) per category; median amount ~ e^mu rupees
AMOUNT_PARAMS = {
    "p2p": (6.2, 1.0),        # ~Rs 490
    "merchant": (5.3, 0.9),   # ~Rs 200
    "bill_pay": (6.8, 0.6),   # ~Rs 900
    "recharge": (5.5, 0.4),   # ~Rs 245
}
NON_MERCHANT = ["p2p", "bill_pay", "recharge"]
NON_MERCHANT_P = [0.60, 0.25, 0.15]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    first_month: int   # users signing up in these months (inclusive) are enrolled
    last_month: int
    hypothesis: str


EXPERIMENTS = (
    ExperimentSpec("referral_nudge_banner", 9, 11,
                   "A home-screen banner nudging users to refer friends increases early usage."),
    ExperimentSpec("first_merchant_cashback", 12, 14,
                   "Cashback on the first merchant (P2M) payment increases 7-day activation."),
    ExperimentSpec("simplified_kyc_flow", 15, 17,
                   "A shorter KYC flow increases KYC completion."),
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _draw_amounts(rng: np.random.Generator, category: np.ndarray) -> np.ndarray:
    out = np.empty(len(category))
    for cat, (mu, sigma) in AMOUNT_PARAMS.items():
        mask = category == cat
        out[mask] = rng.lognormal(mu, sigma, mask.sum())
    return out.round(2)


def simulate(cfg: SimConfig = SimConfig()) -> dict[str, pd.DataFrame | dict]:
    """Return ``users``, ``transactions``, ``experiments`` tables and ``ground_truth``."""
    rng = np.random.default_rng(cfg.seed)
    n, n_months = cfg.n_users, cfg.n_months
    month_starts = pd.date_range(cfg.start_date, periods=n_months, freq="MS")
    end_date = month_starts[-1] + pd.offsets.MonthEnd(0)

    # ---- 1. Acquisition: growing platform with two campaign spikes ----
    weights = np.linspace(1, 3, n_months)
    weights[list(cfg.campaign_months)] *= cfg.campaign_boost
    signup_month = rng.choice(n_months, n, p=weights / weights.sum())
    signup_day = rng.integers(0, 26, n)
    order = np.lexsort((signup_day, signup_month))   # user_id grows with signup time
    signup_month, signup_day = signup_month[order], signup_day[order]
    signup_date = month_starts[signup_month] + pd.to_timedelta(signup_day, unit="D")

    channel = rng.choice(CHANNELS, n, p=CHANNEL_P)
    city_tier = rng.choice([1, 2, 3], n, p=[0.35, 0.40, 0.25])
    age_group = rng.choice(AGE_GROUPS, n, p=[0.30, 0.40, 0.20, 0.10])
    device = rng.choice(["android", "ios"], n, p=[0.85, 0.15])

    engagement = rng.lognormal(0.0, 0.5, n)
    engagement *= pd.Series(channel).map(CHANNEL_ENGAGEMENT).to_numpy()
    engagement *= pd.Series(city_tier).map(TIER_ENGAGEMENT).to_numpy()

    # ---- 2. Experiment assignment: 50/50 within each enrolment window ----
    experiment = np.full(n, None, dtype=object)
    treated = np.zeros(n, dtype=bool)
    for spec in EXPERIMENTS:
        enrolled = (signup_month >= spec.first_month) & (signup_month <= spec.last_month)
        experiment[enrolled] = spec.name
        treated[enrolled] = rng.random(enrolled.sum()) < 0.5
    t_banner = treated & (experiment == "referral_nudge_banner")
    t_cashback = treated & (experiment == "first_merchant_cashback")
    t_kyc = treated & (experiment == "simplified_kyc_flow")

    # ---- 3. KYC ----
    p_kyc = cfg.base_kyc_rate - cfg.tier3_kyc_penalty * (city_tier == 3) + cfg.kyc_flow_lift * t_kyc
    kyc = rng.random(n) < p_kyc
    kyc_delay = rng.integers(0, 3, n)

    # ---- 4. Activation: merchant payment within 7 days ----
    logit = (cfg.activation_intercept
             + cfg.activation_engagement_coef * np.log(engagement)
             + cfg.cashback_logit_lift * t_cashback)
    p_activate = _sigmoid(logit)
    activated = kyc & (rng.random(n) < p_activate)

    # ---- 5. Lifetime: monthly churn hazard, decaying with tenure ----
    lifetime = kyc.astype(int)          # months active, counting the signup month
    alive = kyc.copy()
    for k in range(1, n_months):
        hazard = cfg.base_churn * cfg.churn_decay ** k
        hazard = hazard * np.where(activated, cfg.activated_churn_mult, 1.0)
        hazard = hazard * np.where(channel == "referral", cfg.referral_churn_mult, 1.0)
        hazard = np.clip(hazard / engagement ** 0.4, 0.01, 0.95)
        alive &= rng.random(n) > hazard
        lifetime += alive

    # ---- 6. Transactions, month by month ----
    usage_mult = 1.0 + cfg.referral_banner_effect * t_banner
    frames = []
    for t in range(n_months):
        days_in_month = month_starts[t].days_in_month
        ids = np.flatnonzero((signup_month <= t) & (t < signup_month + lifetime))
        start_day = np.where(signup_month[ids] == t, signup_day[ids] + kyc_delay[ids], 0)
        month_fraction = (days_in_month - start_day) / days_in_month   # partial first month
        rate = (cfg.monthly_txn_rate * engagement[ids] * usage_mult[ids] * month_fraction
                * np.where(activated[ids], cfg.activated_rate_mult, 1.0))
        counts = rng.poisson(rate)
        uid = np.repeat(ids, counts)
        day = rng.integers(np.repeat(start_day, counts), days_in_month)
        p_merchant = np.where(activated[uid], 0.40, 0.15)
        other = rng.choice(NON_MERCHANT, len(uid), p=NON_MERCHANT_P)
        category = np.where(rng.random(len(uid)) < p_merchant, "merchant", other)
        frames.append(pd.DataFrame({
            "user_id": uid,
            "txn_date": month_starts[t] + pd.to_timedelta(day, unit="D"),
            "category": category,
            "guaranteed": False,
        }))
    tx = pd.concat(frames, ignore_index=True)

    # Make the activation flag observable and exact in the data:
    # non-activated users never pay a merchant in week 1; activated users always do.
    uid_arr = tx["user_id"].to_numpy()
    days_since_signup = (tx["txn_date"].to_numpy() - signup_date.to_numpy()[uid_arr]) / np.timedelta64(1, "D")
    early_merchant = (days_since_signup < 7) & (tx["category"].to_numpy() == "merchant")
    tx = tx[~(early_merchant & ~activated[uid_arr])]

    act = np.flatnonzero(activated)
    first_payment = pd.DataFrame({
        "user_id": act,
        "txn_date": (pd.Series(signup_date[act])
                     + pd.to_timedelta(rng.integers(kyc_delay[act], 7), unit="D")).clip(upper=end_date).to_numpy(),
        "category": "merchant",
        "guaranteed": True,
    })
    tx = pd.concat([tx, first_payment], ignore_index=True)

    category = tx["category"].to_numpy()
    tx["amount"] = _draw_amounts(rng, category)
    failed = (rng.random(len(tx)) < cfg.failure_rate) & ~tx["guaranteed"].to_numpy()
    tx["status"] = np.where(failed, "failed", "success")
    revenue = np.select(
        [category == "merchant", category == "bill_pay", category == "recharge"],
        [tx["amount"].to_numpy() * cfg.merchant_revenue_rate, cfg.bill_pay_fee, cfg.recharge_fee],
        0.0,
    )
    tx["revenue"] = np.where(failed, 0.0, revenue).round(2)
    tx = tx.drop(columns="guaranteed").sort_values(["txn_date", "user_id"], kind="stable").reset_index(drop=True)
    tx["user_id"] += 1
    tx.insert(0, "txn_id", np.arange(1, len(tx) + 1))

    # ---- 7. Assemble tables (ids are 1-based) ----
    user_id = np.arange(1, n + 1)
    kyc_date = pd.Series(signup_date + pd.to_timedelta(kyc_delay, unit="D")).where(kyc)
    users = pd.DataFrame({
        "user_id": user_id,
        "signup_date": signup_date,
        "channel": channel,
        "city_tier": city_tier,
        "age_group": age_group,
        "device": device,
        "kyc_completed_at": kyc_date.to_numpy(),
    })
    enrolled = experiment != None  # noqa: E711 -- elementwise on an object array
    experiments = pd.DataFrame({
        "user_id": user_id[enrolled],
        "experiment_name": experiment[enrolled],
        "variant": np.where(treated[enrolled], "treatment", "control"),
        "assigned_at": signup_date[enrolled],
    })

    ground_truth = {
        "config": asdict(cfg),
        "experiments": {
            spec.name: {
                "hypothesis": spec.hypothesis,
                "enrolled_signup_months": [spec.first_month, spec.last_month],
                "true_kyc_rate": {
                    v: float(p_kyc[(experiment == spec.name) & (treated == (v == "treatment"))].mean())
                    for v in ("control", "treatment")
                },
                "true_activation_rate": {     # P(KYC) * P(activate | KYC), averaged over the arm
                    v: float((p_kyc * p_activate)[(experiment == spec.name) & (treated == (v == "treatment"))].mean())
                    for v in ("control", "treatment")
                },
            }
            for spec in EXPERIMENTS
        },
        "realised": {
            "kyc_rate": float(kyc.mean()),
            "activation_rate": float(activated.mean()),
            "activation_rate_by_channel": {
                c: float(activated[channel == c].mean()) for c in CHANNELS
            },
        },
    }
    return {"users": users, "transactions": tx, "experiments": experiments, "ground_truth": ground_truth}


def save(tables: dict, out_dir: Path = config.SIM_DIR) -> Path:
    """Write Parquet files plus a SQLite database with indexes; return the DB path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "upi.db"
    db_path.unlink(missing_ok=True)

    frames = {k: v for k, v in tables.items() if isinstance(v, pd.DataFrame)}
    with sqlite3.connect(db_path) as con:
        for name, df in frames.items():
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
            out = df.copy()
            for col in out.select_dtypes("datetime").columns:
                out[col] = out[col].dt.strftime("%Y-%m-%d")      # ISO dates for SQLite date functions
            out.to_sql(name, con, index=False)
        con.executescript("""
            CREATE UNIQUE INDEX ix_users_id ON users(user_id);
            CREATE INDEX ix_tx_user ON transactions(user_id);
            CREATE INDEX ix_tx_date ON transactions(txn_date);
            CREATE INDEX ix_exp_user ON experiments(user_id);
        """)
    (out_dir / "ground_truth.json").write_text(json.dumps(tables["ground_truth"], indent=2))
    return db_path


def run(cfg: SimConfig = SimConfig()) -> Path:
    tables = simulate(cfg)
    db_path = save(tables)
    users, tx, exp = tables["users"], tables["transactions"], tables["experiments"]
    print(f"users: {len(users):,} | transactions: {len(tx):,} | experiment users: {len(exp):,}")
    print(f"date range: {tx['txn_date'].min().date()} -> {tx['txn_date'].max().date()}")
    print(f"KYC rate: {users['kyc_completed_at'].notna().mean():.1%} | "
          f"activation rate: {tables['ground_truth']['realised']['activation_rate']:.1%}")
    print(f"saved -> {db_path}")
    return db_path
