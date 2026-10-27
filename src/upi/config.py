"""Project paths and every tunable parameter in one place."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # Kaggle IEEE-CIS files land here
SIM_DIR = DATA_DIR / "simulated"        # simulator output
DB_PATH = SIM_DIR / "upi.db"
SQL_DIR = ROOT / "sql"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"


@dataclass(frozen=True)
class SimConfig:
    """Parameters of the simulated platform.

    Each effect below is *planted* in the data. The analysis never reads this
    config: it has to rediscover the effects from the raw tables, which is how
    we validate the SQL and statistics.
    """

    n_users: int = 100_000
    n_months: int = 18
    start_date: str = "2024-01-01"
    seed: int = 42

    # Acquisition
    campaign_months: tuple[int, ...] = (5, 11)   # 0-indexed months with marketing pushes
    campaign_boost: float = 1.6

    # Onboarding
    base_kyc_rate: float = 0.85
    tier3_kyc_penalty: float = 0.06

    # Activation = merchant payment within 7 days of signup (logistic model)
    activation_intercept: float = -0.8
    activation_engagement_coef: float = 1.0

    # Retention: monthly churn hazard that decays with tenure
    base_churn: float = 0.40
    churn_decay: float = 0.88
    activated_churn_mult: float = 0.55
    referral_churn_mult: float = 0.80

    # Usage
    monthly_txn_rate: float = 3.0
    activated_rate_mult: float = 1.5
    failure_rate: float = 0.02

    # Revenue model (UPI P2P/P2M is zero-MDR in India, so revenue comes from
    # merchant services and biller/recharge commissions)
    merchant_revenue_rate: float = 0.003
    bill_pay_fee: float = 2.0
    recharge_fee: float = 1.5

    # Experiments: true effects
    referral_banner_effect: float = 0.0     # true null: should NOT ship
    cashback_logit_lift: float = 0.35       # ~ +7pp activation
    kyc_flow_lift: float = 0.025            # +2.5pp KYC completion


@dataclass(frozen=True)
class FraudConfig:
    valid_frac: float = 0.2
    test_frac: float = 0.2
    seed: int = 42
    # Business costs used to pick the decision threshold
    false_positive_cost: float = 10.0       # support + friction + churn risk per blocked genuine txn (USD)
    # A fraud miss costs the transaction amount
    shap_sample: int = 5_000
