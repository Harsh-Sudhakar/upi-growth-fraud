import numpy as np
import pandas as pd


def test_table_shapes_and_keys(sim_tables):
    users, tx, exp = sim_tables["users"], sim_tables["transactions"], sim_tables["experiments"]
    assert len(users) == 8_000
    assert users["user_id"].is_unique and tx["txn_id"].is_unique
    assert set(tx["user_id"]) <= set(users["user_id"])
    assert set(exp["user_id"]) <= set(users["user_id"])
    assert set(exp["variant"]) == {"control", "treatment"}


def test_transactions_are_consistent(sim_tables):
    users, tx = sim_tables["users"].set_index("user_id"), sim_tables["transactions"]
    signup = users.loc[tx["user_id"], "signup_date"].to_numpy()
    assert (tx["txn_date"].to_numpy() >= signup).all(), "no transaction before signup"
    assert users.loc[tx["user_id"].unique(), "kyc_completed_at"].notna().all(), "only KYC users transact"
    assert (tx.loc[tx["status"] == "failed", "revenue"] == 0).all()
    assert (tx["amount"] > 0).all()
    assert tx["txn_date"].max() <= pd.Timestamp("2025-06-30")


def test_activation_flag_is_exactly_observable(sim_con, sim_tables):
    """The SQL definition of activation must recover the simulator's latent flag exactly."""
    sql_rate = sim_con.execute("SELECT AVG(activated_d7) FROM user_metrics").fetchone()[0]
    assert np.isclose(sql_rate, sim_tables["ground_truth"]["realised"]["activation_rate"])


def test_planted_effects_have_expected_direction(sim_con):
    by_channel = dict(sim_con.execute(
        "SELECT channel, AVG(active_d30_60) FROM user_metrics WHERE days_observed >= 60 GROUP BY 1").fetchall())
    assert by_channel["referral"] > by_channel["organic"] > by_channel["paid"]


def test_experiment_split_is_balanced(sim_tables):
    counts = sim_tables["experiments"].groupby("experiment_name")["variant"].value_counts(normalize=True)
    assert counts.between(0.45, 0.55).all()
