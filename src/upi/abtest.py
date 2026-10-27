"""A/B testing: design (power / sample size), validity (SRM), analysis and decisions.

Each experiment declares one primary metric, secondary metrics and guardrails
*before* looking at results. p-values are Holm-corrected across all metrics of
an experiment, and the ship decision is made on the primary metric only.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from upi import config, db, plots
from upi.report import md_table, write_metrics, write_report
from upi.simulate import EXPERIMENTS

ALPHA = 0.05
POWER = 0.80
SRM_ALPHA = 0.001


@dataclass(frozen=True)
class Metric:
    name: str
    kind: str              # "proportion" or "mean"
    role: str              # "primary", "secondary" or "guardrail"
    min_days: int          # users need this much history for the metric to be observable


@dataclass(frozen=True)
class Plan:
    metrics: tuple[Metric, ...]
    mde: float             # minimum detectable effect on the primary metric (absolute)


PLANS = {
    "referral_nudge_banner": Plan((
        Metric("txns_first_30d", "mean", "primary", 30),
        Metric("activated_d7", "proportion", "secondary", 7),
        Metric("active_d30_60", "proportion", "guardrail", 60),
    ), mde=0.25),          # +0.25 transactions in the first 30 days
    "first_merchant_cashback": Plan((
        Metric("activated_d7", "proportion", "primary", 7),
        Metric("active_d30_60", "proportion", "secondary", 60),
        Metric("txns_first_30d", "mean", "secondary", 30),
    ), mde=0.03),          # +3pp activation
    "simplified_kyc_flow": Plan((
        Metric("kyc_completed", "proportion", "primary", 3),
        Metric("activated_d7", "proportion", "secondary", 7),
    ), mde=0.02),          # +2pp KYC completion
}


# ------------------------------------------------------------------ design
def sample_size_proportion(p_base: float, mde: float, alpha: float = ALPHA, power: float = POWER) -> int:
    """Users per arm to detect an absolute lift ``mde`` on a baseline rate ``p_base``."""
    effect = proportion_effectsize(p_base + mde, p_base)
    return int(np.ceil(NormalIndPower().solve_power(effect_size=effect, alpha=alpha, power=power, ratio=1)))


def sample_size_mean(sd: float, mde: float, alpha: float = ALPHA, power: float = POWER) -> int:
    """Users per arm to detect an absolute difference ``mde`` in means with std ``sd``."""
    return int(np.ceil(NormalIndPower().solve_power(effect_size=mde / sd, alpha=alpha, power=power, ratio=1)))


def achieved_power(effect_size: float, n_per_arm: float, alpha: float = ALPHA) -> float:
    return float(NormalIndPower().power(effect_size=effect_size, nobs1=n_per_arm, alpha=alpha, ratio=1))


# ------------------------------------------------------------------ validity
def srm_test(n_control: int, n_treatment: int, expected_treatment_share: float = 0.5) -> float:
    """Chi-square p-value for sample-ratio mismatch. p < 0.001 means the split is broken."""
    total = n_control + n_treatment
    expected = [total * (1 - expected_treatment_share), total * expected_treatment_share]
    return float(stats.chisquare([n_control, n_treatment], expected).pvalue)


# ------------------------------------------------------------------ tests
def two_proportion_test(x_c: int, n_c: int, x_t: int, n_t: int, alpha: float = ALPHA) -> dict:
    """Pooled z-test for the p-value, unpooled Wald interval for the difference."""
    p_c, p_t = x_c / n_c, x_t / n_t
    p_pool = (x_c + x_t) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z = (p_t - p_c) / se_pool
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    zc = stats.norm.ppf(1 - alpha / 2)
    diff = p_t - p_c
    return {
        "control": p_c, "treatment": p_t, "diff": diff,
        "ci_low": diff - zc * se, "ci_high": diff + zc * se,
        "p_value": float(2 * stats.norm.sf(abs(z))),
    }


def welch_test(control: np.ndarray, treatment: np.ndarray, alpha: float = ALPHA) -> dict:
    """Welch t-test (unequal variances) plus a Mann-Whitney U robustness check."""
    m_c, m_t = control.mean(), treatment.mean()
    v_c, v_t = control.var(ddof=1) / len(control), treatment.var(ddof=1) / len(treatment)
    se = np.sqrt(v_c + v_t)
    dof = (v_c + v_t) ** 2 / (v_c ** 2 / (len(control) - 1) + v_t ** 2 / (len(treatment) - 1))
    tc = stats.t.ppf(1 - alpha / 2, dof)
    diff = m_t - m_c
    return {
        "control": m_c, "treatment": m_t, "diff": diff,
        "ci_low": diff - tc * se, "ci_high": diff + tc * se,
        "p_value": float(stats.ttest_ind(treatment, control, equal_var=False).pvalue),
        "mann_whitney_p": float(stats.mannwhitneyu(treatment, control, alternative="two-sided").pvalue),
    }


# ------------------------------------------------------------------ analysis
def analyze_experiment(df: pd.DataFrame, name: str, plan: Plan) -> tuple[pd.DataFrame, dict]:
    """Analyse one experiment. ``df``: one row per participant with ``variant`` + metric columns."""
    n_c = int((df["variant"] == "control").sum())
    n_t = int((df["variant"] == "treatment").sum())
    srm_p = srm_test(n_c, n_t)

    rows = []
    for m in plan.metrics:
        d = df[df["days_observed"] >= m.min_days]
        c = d.loc[d["variant"] == "control", m.name].to_numpy(dtype=float)
        t = d.loc[d["variant"] == "treatment", m.name].to_numpy(dtype=float)
        if m.kind == "proportion":
            res = two_proportion_test(int(c.sum()), len(c), int(t.sum()), len(t))
        else:
            res = welch_test(c, t)
        rows.append({"experiment": name, "metric": m.name, "role": m.role, "kind": m.kind,
                     "n_control": len(c), "n_treatment": len(t), **res})
    res = pd.DataFrame(rows)
    res["p_holm"] = multipletests(res["p_value"], method="holm")[1]
    res["significant"] = res["p_holm"] < ALPHA
    res["rel_lift"] = res["diff"] / res["control"]
    # relative CI treats the control mean as fixed (fine at these sample sizes)
    res["rel_ci_low"] = res["ci_low"] / res["control"]
    res["rel_ci_high"] = res["ci_high"] / res["control"]

    primary = res[res["role"] == "primary"].iloc[0]
    pm = next(m for m in plan.metrics if m.role == "primary")
    if pm.kind == "proportion":
        es = proportion_effectsize(primary["control"] + plan.mde, primary["control"])
    else:
        sd = df.loc[df["days_observed"] >= pm.min_days, pm.name].std()
        es = plan.mde / sd
    power = achieved_power(es, min(primary["n_control"], primary["n_treatment"]))

    guardrail_hurt = res[(res["role"] == "guardrail") & res["significant"] & (res["diff"] < 0)]
    if srm_p < SRM_ALPHA:
        decision = "INVALID - sample ratio mismatch; fix assignment and rerun"
    elif primary["significant"] and primary["diff"] > 0 and guardrail_hurt.empty:
        decision = "SHIP"
    elif primary["significant"] and primary["diff"] > 0:
        decision = "DON'T SHIP - primary up but a guardrail is significantly worse"
    elif primary["significant"]:
        decision = "DON'T SHIP - primary metric significantly worse"
    elif power >= POWER:
        decision = "DON'T SHIP - well-powered test found no meaningful effect"
    else:
        decision = "INCONCLUSIVE - underpowered; extend the test"

    summary = {"experiment": name, "n_control": n_c, "n_treatment": n_t, "srm_p": srm_p,
               "primary_metric": pm.name, "mde": plan.mde, "achieved_power": power, "decision": decision}
    return res, summary


def design_table(um: pd.DataFrame) -> pd.DataFrame:
    """Pre-test design: baseline from the 3 cohorts before each test, required n per arm."""
    start = pd.Period(config.SimConfig().start_date, freq="M")
    rows = []
    for spec in EXPERIMENTS:
        plan = PLANS[spec.name]
        pm = next(m for m in plan.metrics if m.role == "primary")
        pre = [str(start + k) for k in range(spec.first_month - 3, spec.first_month)]
        base = um[um["cohort_month"].isin(pre) & (um["days_observed"] >= pm.min_days)][pm.name]
        if pm.kind == "proportion":
            n = sample_size_proportion(base.mean(), plan.mde)
            mde_txt = f"+{plan.mde * 100:.0f} pts"
        else:
            n = sample_size_mean(base.std(), plan.mde)
            mde_txt = f"+{plan.mde} txns"
        rows.append({"experiment": spec.name, "primary_metric": pm.name, "baseline": base.mean(),
                     "baseline_sd": base.std(), "mde": mde_txt, "required_n_per_arm": n})
    return pd.DataFrame(rows)


def segment_effects(df: pd.DataFrame, metric: str, segment: str = "channel") -> pd.DataFrame:
    """Treatment effect within each segment (heterogeneity check, exploratory)."""
    rows = []
    for seg, d in df.groupby(segment):
        c, t = d[d["variant"] == "control"][metric], d[d["variant"] == "treatment"][metric]
        r = two_proportion_test(int(c.sum()), len(c), int(t.sum()), len(t))
        rows.append({segment: seg, "n": len(d), "control": r["control"], "treatment": r["treatment"],
                     "diff": r["diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_value": r["p_value"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ pipeline
def run(con: sqlite3.Connection | None = None) -> dict:
    con = con or db.connect()
    if not db.marts_exist(con):
        db.build_marts(con)

    outcomes = db.query(con, "07_experiment_outcomes.sql")
    um = pd.read_sql_query("SELECT cohort_month, days_observed, kyc_completed, activated_d7, "
                           "txns_first_30d, active_d30_60 FROM user_metrics", con)
    design = design_table(um)

    all_results, summaries = [], []
    for spec in EXPERIMENTS:
        res, summary = analyze_experiment(outcomes[outcomes["experiment_name"] == spec.name],
                                          spec.name, PLANS[spec.name])
        summary["hypothesis"] = spec.hypothesis
        all_results.append(res)
        summaries.append(summary)
    results = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    cashback = outcomes[(outcomes["experiment_name"] == "first_merchant_cashback")
                        & (outcomes["days_observed"] >= 7)]
    hetero = segment_effects(cashback, "activated_d7")

    plots.save(plots.ab_forest(results), "ab_test_results")

    # Validation against the simulator's planted truth
    truth_path = config.SIM_DIR / "ground_truth.json"
    truth = json.loads(truth_path.read_text())["experiments"] if truth_path.exists() else {}
    validation = []
    if truth:
        cb = results[(results["experiment"] == "first_merchant_cashback") & (results["metric"] == "activated_d7")].iloc[0]
        t_act = truth["first_merchant_cashback"]["true_activation_rate"]
        kyc = results[(results["experiment"] == "simplified_kyc_flow") & (results["metric"] == "kyc_completed")].iloc[0]
        t_kyc = truth["simplified_kyc_flow"]["true_kyc_rate"]
        for label, r, true_diff in [
            ("cashback -> activation", cb, t_act["treatment"] - t_act["control"]),
            ("KYC flow -> KYC completion", kyc, t_kyc["treatment"] - t_kyc["control"]),
        ]:
            validation.append({"effect": label, "planted_diff": true_diff, "estimated_diff": r["diff"],
                               "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                               "truth_in_ci": bool(r["ci_low"] <= true_diff <= r["ci_high"])})
        validation.append({"effect": "referral banner -> txns (true null)", "planted_diff": 0.0,
                           **{k: results.query("experiment=='referral_nudge_banner' and role=='primary'").iloc[0][k]
                              for k in ("ci_low", "ci_high")},
                           "estimated_diff": results.query("experiment=='referral_nudge_banner' and role=='primary'").iloc[0]["diff"]})
        validation[-1]["truth_in_ci"] = bool(validation[-1]["ci_low"] <= 0 <= validation[-1]["ci_high"])
    validation_df = pd.DataFrame(validation)

    # Non-primary metrics that reached significance in a test whose primary metric did not
    false_alarms = results[(results["role"] != "primary") & results["significant"]
                           & results["experiment"].isin(summary_df.loc[summary_df["decision"] != "SHIP", "experiment"])]
    false_alarm_note = "".join(
        f"\n- `{r.experiment}` / `{r.metric}` reached significance (Holm p = {r.p_holm:.3f}) although the "
        f"primary metric showed nothing{' and the planted effect is zero' if r.experiment == 'referral_nudge_banner' else ''}. "
        "This is the kind of false positive the primary-metric decision rule protects against."
        for r in false_alarms.itertuples()
    )

    write_metrics("abtest", {"summary": summary_df, "results": results, "design": design,
                             "validation": validation_df})

    show = results[["experiment", "metric", "role", "n_control", "n_treatment", "control", "treatment",
                    "diff", "ci_low", "ci_high", "rel_lift", "p_value", "p_holm", "significant"]]
    sections = []
    for s in summaries:
        r = results[(results["experiment"] == s["experiment"]) & (results["role"] == "primary")].iloc[0]
        unit = "pts" if r["kind"] == "proportion" else "txns"
        scale = 100 if r["kind"] == "proportion" else 1
        sections.append(f"""
### {s['experiment']}

*Hypothesis:* {s['hypothesis']}

- Sample: {s['n_control']:,} control / {s['n_treatment']:,} treatment; SRM p = {s['srm_p']:.3f}
  ({'OK' if s['srm_p'] >= SRM_ALPHA else 'FAILED'})
- Primary metric `{r['metric']}`: {r['control']:.3f} -> {r['treatment']:.3f}
  ({r['diff'] * scale:+.2f} {unit}, 95% CI {r['ci_low'] * scale:+.2f} to {r['ci_high'] * scale:+.2f};
  relative {r['rel_lift']:+.1%}; Holm-adjusted p = {r['p_holm']:.4f})
- Power to detect the planned MDE: {s['achieved_power']:.0%}
- **Decision: {s['decision']}**
""")

    report = f"""
# A/B testing report

Three growth experiments ran on consecutive signup cohorts with 50/50 random assignment.
Each one declares its primary metric, secondary metrics and guardrails up front. Tests are two-sided at
alpha = {ALPHA} with Holm correction across each experiment's metrics, and users only enter a metric once
they have enough history for it to be observable (no right-censoring bias).

## 1. Design (before launch)

Baselines come from the three signup cohorts before each test; sample sizes target {POWER:.0%} power.

{md_table(design, "{:,.3f}")}

## 2. Results

{''.join(sections)}

![A/B results](figures/ab_test_results.png)

Full results (all metrics):

{md_table(show, "{:,.4f}")}

## 3. Heterogeneity: cashback effect on activation by channel (exploratory)

{md_table(hetero, "{:,.4f}")}

Segment cuts are exploratory: they are not powered individually and are only used to generate hypotheses.

## 4. Validation against the planted ground truth

Because the platform is simulated, we know the true effects. A correct analysis should bracket them:

{md_table(validation_df, "{:,.4f}") if not validation_df.empty else "_ground_truth.json not found_"}
{false_alarm_note}

## Practical notes

- **Statistical vs practical significance.** The KYC flow lift is real but small; weigh it against
  engineering and compliance cost. The cashback lift should be weighed against its cost per activated user
  using the CLV estimates in the growth report.
- **Novelty effects.** A short window can overstate the effect of a banner; the retention guardrail covers
  this partly.
- **Peeking.** Results are read once at the planned sample size; continuous monitoring would need a
  sequential test (e.g. mSPRT or alpha spending).
"""
    write_report("abtest_report.md", report)
    return {"summary": summary_df, "results": results, "design": design, "validation": validation_df,
            "heterogeneity": hetero}
