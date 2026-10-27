"""End-to-end fraud pipeline: load -> time split -> features -> models -> threshold -> report."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from upi import config, plots
from upi.config import FraudConfig
from upi.fraud import data, model
from upi.fraud.data import TARGET
from upi.fraud.features import FraudFeatureBuilder
from upi.report import md_table, write_metrics, write_report


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def train_and_evaluate(df: pd.DataFrame, cfg: FraudConfig = FraudConfig(), verbose: bool = True) -> dict:
    """Pure modelling step (no file output) so it can be tested on any IEEE-shaped frame."""
    t0 = time.time()
    log = (lambda m: _log(m, t0)) if verbose else (lambda m: None)

    train, valid, test = data.time_split(df, cfg.valid_frac, cfg.test_frac)
    split_info = pd.DataFrame([
        {"split": name, "rows": len(d), "fraud_rate": d[TARGET].mean(),
         "first_day": int(d[data.TIME_COL].min() // 86400), "last_day": int(d[data.TIME_COL].max() // 86400)}
        for name, d in (("train", train), ("valid", valid), ("test", test))
    ])
    log(f"split: train {len(train):,} / valid {len(valid):,} / test {len(test):,}")

    builder = FraudFeatureBuilder().fit(train)
    X_tr, X_va, X_te = builder.transform(train), builder.transform(valid), builder.transform(test)
    y_tr, y_va, y_te = (d[TARGET].to_numpy() for d in (train, valid, test))
    amt_va, amt_te = valid["TransactionAmt"].to_numpy(), test["TransactionAmt"].to_numpy()
    log(f"features: {X_tr.shape[1]} columns")

    # ---- 1. Baseline
    logreg = model.train_logreg(X_tr, y_tr, cfg.seed)
    log("logistic regression trained")

    # ---- 2. Imbalance strategies for XGBoost, compared on the validation period
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    strategies = {}
    strategies["XGBoost - no rebalancing"] = model.train_xgb(X_tr, y_tr, X_va, y_va, 1.0, cfg.seed)
    log("xgb (no rebalancing) trained")
    strategies[f"XGBoost - class weight (spw={spw:.0f})"] = model.train_xgb(X_tr, y_tr, X_va, y_va, spw, cfg.seed)
    log("xgb (class weight) trained")
    under = data.undersample(train, ratio=5, seed=cfg.seed)
    strategies["XGBoost - undersample 1:5"] = model.train_xgb(
        builder.transform(under), under[TARGET].to_numpy(), X_va, y_va, 1.0, cfg.seed)
    log("xgb (undersampled) trained")

    comparison = []
    for name, m in {"Logistic regression (balanced)": logreg, **strategies}.items():
        p_va = model.predict(m, X_va)
        comparison.append({"model": name, "valid_pr_auc": model.pr_curve(y_va, p_va)[2],
                           "valid_roc_auc": model.evaluate(y_va, p_va, 0.5)["roc_auc"],
                           "best_iteration": getattr(m, "best_iteration", None)})
    comparison = pd.DataFrame(comparison).sort_values("valid_pr_auc", ascending=False).reset_index(drop=True)
    best_name = comparison.loc[comparison["model"].str.startswith("XGBoost"), "model"].iloc[0]
    best = strategies[best_name]
    log(f"best on validation: {best_name}")

    # ---- 3. Threshold chosen on validation by business cost, then frozen
    p_va = model.predict(best, X_va)
    costs = model.threshold_costs(y_va, p_va, amt_va, cfg.false_positive_cost)
    threshold = float(costs.loc[costs["total_cost"].idxmin(), "threshold"])

    # ---- 4. One look at the untouched test period
    p_te = model.predict(best, X_te)
    p_te_lr = model.predict(logreg, X_te)
    test_metrics = model.evaluate(y_te, p_te, threshold)
    test_metrics_f1 = model.evaluate(y_te, p_te, 0.5)
    baseline_metrics = model.evaluate(y_te, p_te_lr, 0.5)
    business = model.business_summary(y_te, p_te, amt_te, threshold, cfg.false_positive_cost)
    log(f"test PR-AUC {test_metrics['pr_auc']:.4f} | ROC-AUC {test_metrics['roc_auc']:.4f}")

    # ---- 5. Explainability
    importance = pd.Series(best.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    sample = X_te.sample(min(cfg.shap_sample, len(X_te)), random_state=cfg.seed)
    shap_values = None
    try:
        import shap
        shap_values = shap.TreeExplainer(best).shap_values(sample)
    except Exception as exc:  # SHAP/XGBoost version mismatches shouldn't kill the run
        log(f"SHAP skipped: {exc}")

    del X_tr, under
    gc.collect()
    return {
        "split_info": split_info,
        "comparison": comparison,
        "best_name": best_name,
        "threshold": threshold,
        "threshold_costs": costs,
        "test_metrics": test_metrics,
        "test_metrics_at_0_5": test_metrics_f1,
        "baseline_test_metrics": baseline_metrics,
        "business": business,
        "importance": importance,
        "shap_sample": sample,
        "shap_values": shap_values,
        "pr_curves": {
            "Logistic regression": model.pr_curve(y_te, p_te_lr),
            best_name: model.pr_curve(y_te, p_te),
        },
        "base_rate_test": float(y_te.mean()),
        "model": best,
        "feature_names": builder.feature_names_,
        "n_rows": len(df),
    }


def _save_figures(df: pd.DataFrame, res: dict, fig_dir: Path) -> None:
    eda = df[["isFraud", "ProductCD", "card4", "TransactionDT"]].copy()
    eda["hour"] = (eda["TransactionDT"] // 3600) % 24
    plots.save(plots.fraud_rate_by(eda, "ProductCD", "Fraud rate by product code"), "fraud_rate_by_product", fig_dir)
    plots.save(plots.fraud_rate_by(eda, "card4", "Fraud rate by card network"), "fraud_rate_by_network", fig_dir)
    plots.save(plots.pr_curves(res["pr_curves"], res["base_rate_test"]), "fraud_pr_curves", fig_dir)
    plots.save(plots.threshold_cost(res["threshold_costs"], res["threshold"]), "fraud_threshold_cost", fig_dir)
    plots.save(plots.feature_importance(res["importance"].head(20), "Top 20 features (XGBoost gain)"),
               "fraud_feature_importance", fig_dir)
    if res["shap_values"] is not None:
        import matplotlib.pyplot as plt
        import shap
        shap.summary_plot(res["shap_values"], res["shap_sample"], max_display=20, show=False)
        fig = plt.gcf()
        fig.axes[0].set_title("SHAP: how each feature pushes the fraud score", loc="left")
        plots.save(fig, "fraud_shap_summary", fig_dir)


def run(raw_dir: Path = config.RAW_DIR, nrows: int | None = None, cfg: FraudConfig = FraudConfig()) -> dict:
    t0 = time.time()
    df = data.load(raw_dir, nrows=nrows)
    _log(f"loaded {len(df):,} rows x {df.shape[1]} cols, fraud rate {df[TARGET].mean():.2%}", t0)
    res = train_and_evaluate(df, cfg)
    _save_figures(df, res, config.FIG_DIR)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    res["model"].save_model(config.MODELS_DIR / "fraud_xgb.json")

    tm, bm, biz = res["test_metrics"], res["baseline_test_metrics"], res["business"]
    write_metrics("fraud", {
        "rows": res["n_rows"], "best_model": res["best_name"], "threshold": res["threshold"],
        "test": tm, "test_at_0_5": res["test_metrics_at_0_5"], "baseline_test": bm, "business": biz,
        "comparison": res["comparison"], "split": res["split_info"],
        "top_features": res["importance"].head(20).round(2).to_dict(),
    })

    metrics_table = pd.DataFrame([
        {"model": "Logistic regression (threshold 0.5)", **{k: bm[k] for k in ("pr_auc", "roc_auc", "precision", "recall", "f1")}},
        {"model": f"{res['best_name']} (threshold 0.5)", **{k: res["test_metrics_at_0_5"][k] for k in ("pr_auc", "roc_auc", "precision", "recall", "f1")}},
        {"model": f"{res['best_name']} (cost-optimal {res['threshold']:.2f})", **{k: tm[k] for k in ("pr_auc", "roc_auc", "precision", "recall", "f1")}},
    ])
    sample_note = "" if nrows is None else f"\n> **Sample run on the first {nrows:,} rows** — rerun `make fraud` for full-data numbers.\n"
    report = f"""
# Fraud detection report

*IEEE-CIS Fraud Detection (Vesta): {res['n_rows']:,} real e-commerce card transactions,
fraud rate {df[TARGET].mean():.2%}. Card-not-present payments, not UPI, but the same problem:
rare, adaptive fraud hidden among legitimate digital payments.*
{sample_note}
## Headline (held-out test period, never used for training or tuning)

- **PR-AUC {tm['pr_auc']:.3f}** vs {res['base_rate_test']:.3f} for a random model
  ({tm['pr_auc'] / res['base_rate_test']:.0f}x lift); logistic-regression baseline {bm['pr_auc']:.3f}.
- At the cost-optimal threshold ({res['threshold']:.2f}): **recall {tm['recall']:.1%}**, precision {tm['precision']:.1%},
  {tm['alert_rate']:.2%} of transactions flagged.
- **{biz['share_of_fraud_amount_prevented']:.1%} of fraud dollars stopped** while blocking
  {biz['share_of_genuine_txns_blocked']:.2%} of genuine transactions
  ({tm['genuine_blocked_per_100_frauds_caught']:.0f} genuine customers inconvenienced per 100 frauds caught).
- Net saving on the test period: **${biz['net_savings']:,.0f}** (fraud losses ${biz['cost_without_model']:,.0f} ->
  ${biz['cost_with_model']:,.0f} including friction cost).

## Method

1. **Time-based split.** 60% train / 20% validation / 20% test, in time order. A random split would let the model
   see the same cards in the future and overstate performance.

{md_table(res['split_info'], "{:.4f}")}

2. **Features** ({len(res['feature_names'])}), all fitted on the training period only: time of day and weekday, log amount,
   amount cents (foreign-currency signal), email-domain match, frequency encodings, an approximate user id
   (card1 + addr1 + first-seen day) with "amount vs this user's usual amount" ratios, plus the raw C/D/M/V and identity
   columns.
3. **Class imbalance.** Compared no rebalancing, class weights (`scale_pos_weight`) and random undersampling (1:5) on
   the validation period. SMOTE is not used: synthetic minority rows interpolated across heavily missing, categorical
   and anonymised columns are not realistic transactions, and tree models handle imbalance well with weighting.

{md_table(res['comparison'], "{:.4f}")}

4. **Metric.** PR-AUC, not accuracy or ROC-AUC: with {res['base_rate_test']:.1%} fraud, a model that flags nothing is
   {1 - res['base_rate_test']:.1%} accurate, and ROC-AUC is dominated by the easy true negatives.
5. **Threshold by cost, not by 0.5.** A missed fraud costs the transaction amount; a blocked genuine payment is
   costed at ${cfg.false_positive_cost:.0f} (support contact, failed purchase, churn risk). The threshold minimising total
   cost on validation is frozen and applied once to test.

## Results on the test period

{md_table(metrics_table, "{:.4f}")}

Recall at fixed precision: {tm['recall_at_precision_50']:.1%} of fraud caught at 50% precision,
{tm['recall_at_precision_80']:.1%} at 80% precision.

![PR curves](figures/fraud_pr_curves.png)

![Threshold cost](figures/fraud_threshold_cost.png)

## What drives the score

![Feature importance](figures/fraud_feature_importance.png)

{"![SHAP](figures/fraud_shap_summary.png)" if res['shap_values'] is not None else ""}

## Fraud patterns

![By product](figures/fraud_rate_by_product.png)

![By network](figures/fraud_rate_by_network.png)

## Trade-off: fraud prevention vs user experience

The threshold is a business dial. Lowering it catches more fraud but blocks more genuine payments; for a payments app
each false decline risks a user moving to a competitor. A practical deployment uses **three bands**: auto-approve low
scores, step-up authentication (OTP / UPI PIN re-entry) for the middle band, and block only the highest scores. This
recovers most of the fraud protection while keeping friction for genuine users low.

## Limitations

- Labels are from chargebacks/reports, so some fraud is unlabelled and labels arrive with delay.
- Features are anonymised (C, D, M, V), which limits business interpretation of some top features.
- One time split; production would use rolling backtests and monitor drift (PSI) and precision weekly.
"""
    write_report("fraud_report.md", report)
    _log("report written", t0)
    return res
