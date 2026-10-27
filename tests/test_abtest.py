import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

from upi import abtest


def test_sample_size_matches_closed_form():
    p, mde = 0.10, 0.02
    h = 2 * np.arcsin(np.sqrt(p + mde)) - 2 * np.arcsin(np.sqrt(p))
    # Var(2·asin(sqrt(p_hat))) ~ 1/n per arm, so the difference has variance 2/n
    expected = 2 * (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) ** 2 / h ** 2
    assert abs(abtest.sample_size_proportion(p, mde) - expected) <= 1


def test_srm_detects_broken_split():
    assert abtest.srm_test(5000, 5050) > 0.05
    assert abtest.srm_test(5000, 5600) < 0.001


def test_two_proportion_matches_statsmodels():
    res = abtest.two_proportion_test(300, 1000, 345, 1000)
    _, p = proportions_ztest([345, 300], [1000, 1000])
    assert np.isclose(res["p_value"], p)
    assert res["ci_low"] < res["diff"] < res["ci_high"]


def test_false_positive_rate_is_controlled():
    """Under a true null, ~5% of tests should come out significant."""
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(600):
        a, b = rng.binomial(1, 0.3, 2000), rng.binomial(1, 0.3, 2000)
        hits += abtest.two_proportion_test(a.sum(), 2000, b.sum(), 2000)["p_value"] < 0.05
    assert 0.03 < hits / 600 < 0.075


def test_welch_ci_covers_truth():
    rng = np.random.default_rng(2)
    covered = 0
    for _ in range(300):
        c, t = rng.exponential(3.0, 1500), rng.exponential(3.3, 1500)
        r = abtest.welch_test(c, t)
        covered += r["ci_low"] <= 0.3 <= r["ci_high"]
    assert covered / 300 > 0.92


def _experiment(n, p_c, p_t, seed):
    rng = np.random.default_rng(seed)
    variant = np.where(rng.random(n) < 0.5, "treatment", "control")
    p = np.where(variant == "treatment", p_t, p_c)
    return pd.DataFrame({"variant": variant, "days_observed": 100,
                         "activated_d7": rng.binomial(1, p), "active_d30_60": rng.binomial(1, 0.5, n),
                         "txns_first_30d": rng.poisson(3, n)})


def test_decision_rules():
    plan = abtest.PLANS["first_merchant_cashback"]
    _, s = abtest.analyze_experiment(_experiment(20_000, 0.27, 0.33, 3), "x", plan)
    assert s["decision"] == "SHIP"
    _, s = abtest.analyze_experiment(_experiment(20_000, 0.27, 0.27, 4), "x", plan)
    assert s["decision"].startswith("DON'T SHIP")
    _, s = abtest.analyze_experiment(_experiment(600, 0.27, 0.27, 5), "x", plan)
    assert s["decision"].startswith("INCONCLUSIVE")


def test_experiments_recover_planted_effects(sim_con):
    outcomes = pd.read_sql_query(open(abtest.config.SQL_DIR / "07_experiment_outcomes.sql").read(), sim_con)
    res, summary = abtest.analyze_experiment(
        outcomes[outcomes["experiment_name"] == "first_merchant_cashback"],
        "first_merchant_cashback", abtest.PLANS["first_merchant_cashback"])
    primary = res[res["role"] == "primary"].iloc[0]
    assert primary["diff"] > 0
