"""Models, imbalance strategies, evaluation and cost-based thresholding."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

XGB_PARAMS = dict(
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.5,
    tree_method="hist",
    eval_metric="aucpr",
    early_stopping_rounds=100,
    n_jobs=-1,
)


def train_logreg(X: pd.DataFrame, y: np.ndarray, seed: int = 42):
    """Interpretable baseline: median-impute, scale, class-weighted logistic regression."""
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=300, random_state=seed),
    )
    return model.fit(X.clip(-1e6, 1e6), y)


def train_xgb(X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray,
              scale_pos_weight: float = 1.0, seed: int = 42, **overrides):
    from xgboost import XGBClassifier

    params = {**XGB_PARAMS, **overrides}
    model = XGBClassifier(**params, scale_pos_weight=scale_pos_weight, random_state=seed)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Fraud probability. XGBoost uses its early-stopped best iteration automatically."""
    if isinstance(model, Pipeline):
        X = X.clip(-1e6, 1e6)
    return model.predict_proba(X)[:, 1]


# ------------------------------------------------------------------ evaluation
def recall_at_precision(y: np.ndarray, p: np.ndarray, min_precision: float) -> float:
    precision, recall, _ = precision_recall_curve(y, p)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "alert_rate": float(pred.mean()),
        "false_positive_rate": float(fp / max((y == 0).sum(), 1)),
        "genuine_blocked_per_100_frauds_caught": float(100 * fp / tp) if tp else float("nan"),
        "recall_at_precision_50": recall_at_precision(y, p, 0.5),
        "recall_at_precision_80": recall_at_precision(y, p, 0.8),
    }


def threshold_costs(y: np.ndarray, p: np.ndarray, amount: np.ndarray, fp_cost: float,
                    grid: np.ndarray | None = None) -> pd.DataFrame:
    """Expected cost at each threshold.

    Missed fraud (FN) costs the transaction amount; blocking a genuine payment (FP)
    costs ``fp_cost`` (support contact, failed purchase, risk of the customer leaving).
    """
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2) if grid is None else grid
    is_fraud = y == 1
    rows = []
    for t in grid:
        flagged = p >= t
        fraud_loss = float(amount[is_fraud & ~flagged].sum())
        fp = int((flagged & ~is_fraud).sum())
        rows.append({"threshold": t, "fraud_loss": fraud_loss, "friction_cost": fp * fp_cost,
                     "total_cost": fraud_loss + fp * fp_cost,
                     "caught": int((flagged & is_fraud).sum()), "false_positives": fp})
    return pd.DataFrame(rows)


def business_summary(y: np.ndarray, p: np.ndarray, amount: np.ndarray, threshold: float, fp_cost: float) -> dict:
    is_fraud = y == 1
    flagged = p >= threshold
    fraud_amount = float(amount[is_fraud].sum())
    prevented = float(amount[is_fraud & flagged].sum())
    fp = int((flagged & ~is_fraud).sum())
    cost_with_model = fraud_amount - prevented + fp * fp_cost
    return {
        "fraud_amount_total": fraud_amount,
        "fraud_amount_prevented": prevented,
        "share_of_fraud_amount_prevented": prevented / fraud_amount if fraud_amount else float("nan"),
        "false_positives": fp,
        "share_of_genuine_txns_blocked": fp / max((~is_fraud).sum(), 1),
        "cost_without_model": fraud_amount,
        "cost_with_model": cost_with_model,
        "net_savings": fraud_amount - cost_with_model,
    }


def pr_curve(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    precision, recall, _ = precision_recall_curve(y, p)
    return precision, recall, float(average_precision_score(y, p))
