import numpy as np
import pandas as pd
import pytest

from upi import db, growth


def test_funnel_is_monotonic(sim_con):
    overall, by_channel = growth.funnel(sim_con)
    assert overall["users"].is_monotonic_decreasing
    assert overall["pct_of_signups"].iloc[0] == 1.0
    assert set(by_channel["channel"]) == {"organic", "paid", "referral"}


def test_cohort_retention_bounds(sim_con):
    m = growth.cohort_matrix(sim_con)
    values = m.to_numpy(dtype=float)
    assert np.nanmax(values) <= 1.0 and np.nanmin(values) >= 0.0
    # triangle: the latest cohort can only be observed in its signup month
    assert m.iloc[-1].notna().sum() == 1


def test_retention_curve_censoring(sim_con):
    curves = growth.retention_curves(sim_con, "activated_d7", max_month=6)
    assert list(curves.columns) == [0, 1]
    assert (curves[1] > curves[0]).all(), "activated users retain better at every month"


def test_segment_whitelist_blocks_injection():
    with pytest.raises(ValueError):
        db.load_sql("06_segment_value.sql", segment="channel; DROP TABLE users")


def test_project_retention():
    curve = np.array([1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768])
    out = growth.project_retention(curve, horizon=10)
    assert len(out) == 10
    np.testing.assert_allclose(out[:6], curve)
    np.testing.assert_allclose(out[6], curve[-1] * 0.8, rtol=1e-6)


def test_clv_ranks_referral_first(sim_con):
    clv = growth.clv_by_segment(sim_con, "channel")
    assert clv.iloc[0]["segment"] == "referral"
    assert (clv["clv_24m"] > 0).all()


def test_rfm_and_pareto(sim_con):
    um = pd.read_sql_query("SELECT * FROM user_metrics", sim_con)
    users, summary = growth.rfm_segments(um)
    assert summary["users"].sum() == (um["total_txns"] > 0).sum()
    assert np.isclose(summary["share_of_revenue"].sum(), 1.0)
    _, top20 = growth.pareto(um)
    assert 0.2 < top20 < 1.0


def test_activation_impact_odds_ratio(sim_con):
    um = pd.read_sql_query("SELECT * FROM user_metrics", sim_con)
    res = growth.activation_impact(um)
    assert res["odds_ratios"].loc["activated_d7", "odds_ratio"] > 1.5
