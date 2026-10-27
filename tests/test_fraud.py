import numpy as np
import pandas as pd
import pytest

from upi.config import FraudConfig
from upi.fraud import data, model
from upi.fraud.features import FraudFeatureBuilder


def test_time_split_is_chronological(fake_ieee):
    train, valid, test = data.time_split(fake_ieee)
    assert train["TransactionDT"].max() <= valid["TransactionDT"].min()
    assert valid["TransactionDT"].max() <= test["TransactionDT"].min()
    assert len(train) + len(valid) + len(test) == len(fake_ieee)


def test_features_fit_on_train_only(fake_ieee):
    train, _, test = data.time_split(fake_ieee)
    builder = FraudFeatureBuilder().fit(train)
    X_tr, X_te = builder.transform(train), builder.transform(test)
    assert list(X_tr.columns) == list(X_te.columns)
    assert "isFraud" not in X_tr.columns and "TransactionDT" not in X_tr.columns
    assert np.isfinite(X_tr.fillna(0).to_numpy()).all()
    # frequency maps come from train: a value never seen in train gets frequency 0
    unseen = test.head(5).copy()
    unseen["card1"] = 999_999
    assert (builder.transform(unseen)["card1_freq"] == 0).all()


def test_unseen_category_encoded_as_minus_one(fake_ieee):
    train, _, test = data.time_split(fake_ieee)
    builder = FraudFeatureBuilder().fit(train)
    row = test.head(1).copy()
    row["ProductCD"] = pd.Categorical(["Z"])
    assert builder.transform(row)["ProductCD"].iloc[0] == -1


def test_undersample_keeps_all_fraud(fake_ieee):
    under = data.undersample(fake_ieee, ratio=5)
    assert under["isFraud"].sum() == fake_ieee["isFraud"].sum()
    assert (under["isFraud"] == 0).sum() == 5 * fake_ieee["isFraud"].sum()


def test_threshold_costs():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    amt = np.array([10.0, 10.0, 100.0, 200.0])
    costs = model.threshold_costs(y, p, amt, fp_cost=5.0, grid=np.array([0.05, 0.5, 0.95]))
    assert costs["fraud_loss"].tolist() == [0.0, 100.0, 300.0]
    assert costs["friction_cost"].tolist() == [10.0, 5.0, 0.0]


def test_end_to_end_pipeline_beats_random(fake_ieee, monkeypatch):
    pytest.importorskip("xgboost")
    monkeypatch.setitem(model.XGB_PARAMS, "n_estimators", 200)
    res = __import__("upi.fraud.pipeline", fromlist=["x"]).train_and_evaluate(
        fake_ieee, FraudConfig(shap_sample=500), verbose=False)
    assert res["test_metrics"]["pr_auc"] > 2 * res["base_rate_test"]
    assert 0 < res["threshold"] < 1
    assert len(res["comparison"]) == 4
