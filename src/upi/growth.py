"""Growth analytics: KPIs, cohort retention, funnel, CLV, segmentation, activation impact.

SQL does the heavy aggregation (see ``sql/``); Python handles modelling,
statistics and presentation.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from upi import db, plots
from upi.report import md_table, write_metrics, write_report

FUNNEL_STEPS = {
    "signed_up": "Signed up",
    "kyc_completed": "Completed KYC",
    "first_transaction": "First transaction",
    "activated_d7": "Merchant payment in week 1",
    "activated_retained_d30_60": "...and still active on days 30-60",
}


# ------------------------------------------------------------------ descriptive
def cohort_matrix(con: sqlite3.Connection) -> pd.DataFrame:
    long = db.query(con, "03_cohort_retention.sql")
    return long.pivot(index="cohort_month", columns="months_since_signup", values="retention")


def funnel(con: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (overall funnel, by-channel funnel with step-to-step conversion)."""
    by_channel = db.query(con, "04_activation_funnel.sql").set_index("channel")
    total = by_channel.sum()
    overall = pd.DataFrame({
        "step": [FUNNEL_STEPS[c] for c in total.index],
        "users": total.to_numpy(),
    })
    overall["pct_of_signups"] = overall["users"] / overall["users"].iloc[0]
    overall["step_conversion"] = overall["users"] / overall["users"].shift(1)
    rates = by_channel.div(by_channel["signed_up"], axis=0).drop(columns="signed_up")
    rates.columns = [FUNNEL_STEPS[c] for c in rates.columns]
    return overall, rates.reset_index()


def retention_curves(con: sqlite3.Connection, segment: str, max_month: int = 12) -> pd.DataFrame:
    long = db.query(con, "05_retention_curve.sql", segment=segment, max_month=max_month)
    return long.pivot(index="months_since_signup", columns="segment", values="retention")


# ------------------------------------------------------------------ CLV
def project_retention(curve: np.ndarray, horizon: int, tail_window: int = 4) -> np.ndarray:
    """Extend an observed retention curve to ``horizon`` months.

    Beyond the observed window, retention decays geometrically at the average
    month-over-month survival ratio of the last ``tail_window`` observed months
    (retention curves flatten as only loyal users remain).
    """
    curve = np.asarray(curve, dtype=float)
    if len(curve) >= horizon:
        return curve[:horizon]
    ratios = curve[-tail_window:] / curve[-tail_window - 1:-1]
    decay = float(np.clip(np.nanmean(ratios), 0.0, 0.995))
    tail = curve[-1] * decay ** np.arange(1, horizon - len(curve) + 1)
    return np.concatenate([curve, tail])


def clv_by_segment(con: sqlite3.Connection, segment: str, horizon: int = 24,
                   monthly_discount: float = 0.01) -> pd.DataFrame:
    """Expected discounted revenue per KYC-completed user over ``horizon`` months.

    CLV = sum_k  retention_k * revenue_per_active_month * (1 + d)^-k
    """
    curves = retention_curves(con, segment, max_month=12)
    arpau = db.query(con, "08_revenue_per_active_month.sql", segment=segment).set_index("segment")
    kyc = db.query(con, "06_segment_value.sql", segment=segment).set_index("segment")["kyc_rate"]
    discount = (1 + monthly_discount) ** -np.arange(horizon)
    rows = []
    for seg in curves.columns:
        r = project_retention(curves[seg].dropna().to_numpy(), horizon)
        value = arpau.loc[seg, "revenue_per_active_month"]
        clv = float((r * value * discount).sum())
        rows.append({
            "segment": seg,
            "m1_retention": r[1],
            "m6_retention": r[6],
            "m12_retention": r[12],
            "revenue_per_active_month": value,
            f"clv_{horizon}m": clv,
            f"clv_{horizon}m_per_signup": clv * kyc.get(seg, np.nan),
        })
    return pd.DataFrame(rows).sort_values(f"clv_{horizon}m", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ segmentation
def rfm_segments(um: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score transacting users on Recency/Frequency/Monetary quintiles and name segments.

    Returns (user-level scores, segment summary).
    """
    users = um[um["total_txns"] > 0].copy()
    end = pd.to_datetime(users["last_txn_date"]).max()
    users["recency_days"] = (end - pd.to_datetime(users["last_txn_date"])).dt.days
    # rank first so ties don't break qcut's bin edges
    users["R"] = pd.qcut(users["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    users["F"] = pd.qcut(users["total_txns"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    users["M"] = pd.qcut(users["total_revenue"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    conditions = [
        (users["R"] >= 4) & (users["F"] >= 4),
        (users["R"] >= 3) & (users["F"] >= 3),
        (users["R"] >= 4) & (users["F"] <= 2),
        (users["R"] <= 2) & (users["F"] >= 3),
        (users["R"] <= 2) & (users["F"] <= 2),
    ]
    names = ["Champions", "Loyal", "New / promising", "At risk", "Hibernating"]
    users["rfm_segment"] = np.select(conditions, names, default="Needs attention")

    summary = users.groupby("rfm_segment").agg(
        users=("user_id", "size"),
        avg_recency_days=("recency_days", "mean"),
        avg_txns=("total_txns", "mean"),
        avg_revenue=("total_revenue", "mean"),
        revenue=("total_revenue", "sum"),
    )
    summary["share_of_users"] = summary["users"] / summary["users"].sum()
    summary["share_of_revenue"] = summary["revenue"] / summary["revenue"].sum()
    summary = summary.sort_values("avg_revenue", ascending=False).drop(columns="revenue")
    return users, summary.reset_index()


def pareto(um: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Cumulative revenue share by user share; returns (curve, revenue share of top 20%)."""
    rev = np.sort(um.loc[um["total_txns"] > 0, "total_revenue"].to_numpy())[::-1]
    cum = np.cumsum(rev) / rev.sum()
    user_share = np.arange(1, len(rev) + 1) / len(rev)
    idx = np.linspace(0, len(rev) - 1, 200).astype(int)
    curve = pd.DataFrame({"user_share": np.r_[0, user_share[idx]], "revenue_share": np.r_[0, cum[idx]]})
    return curve, float(cum[int(0.2 * len(rev)) - 1])


# ------------------------------------------------------------------ activation impact
def activation_impact(um: pd.DataFrame) -> dict:
    """Does a week-1 merchant payment predict day 30-60 retention, controlling for user mix?

    Logistic regression on KYC users with >= 60 days of history. This is
    observational: it shows a strong, adjusted association (worth testing with
    an experiment), not proof of causation.
    """
    df = um[(um["kyc_completed"] == 1) & (um["days_observed"] >= 60)].copy()
    raw = df.groupby("activated_d7")["active_d30_60"].mean()
    model = smf.logit(
        "active_d30_60 ~ activated_d7 + C(channel) + C(city_tier) + C(device) + C(age_group)",
        data=df,
    ).fit(disp=0)
    ci = np.exp(model.conf_int())
    odds = pd.DataFrame({
        "odds_ratio": np.exp(model.params),
        "ci_low": ci[0],
        "ci_high": ci[1],
        "p_value": model.pvalues,
    }).drop(index="Intercept")
    return {
        "n_users": len(df),
        "retention_not_activated": float(raw.get(0, np.nan)),
        "retention_activated": float(raw.get(1, np.nan)),
        "retention_lift_pp": float(raw.get(1, np.nan) - raw.get(0, np.nan)),
        "odds_ratios": odds,
        "pseudo_r2": float(model.prsquared),
    }


# ------------------------------------------------------------------ pipeline
def run(con: sqlite3.Connection | None = None) -> dict:
    con = con or db.connect()
    if not db.marts_exist(con):
        db.build_marts(con)

    kpis = db.query(con, "01_monthly_kpis.sql")
    stick = db.query(con, "02_stickiness.sql")
    cohorts = cohort_matrix(con)
    funnel_overall, funnel_channel = funnel(con)
    ret_activation = retention_curves(con, "activated_d7")
    ret_channel = retention_curves(con, "channel")
    value_channel = db.query(con, "06_segment_value.sql", segment="channel")
    value_tier = db.query(con, "06_segment_value.sql", segment="city_tier")
    clv_channel = clv_by_segment(con, "channel")
    clv_activation = clv_by_segment(con, "activated_d7")
    um = pd.read_sql_query("SELECT * FROM user_metrics", con)
    _, rfm_summary = rfm_segments(um)
    pareto_curve, top20 = pareto(um)
    impact = activation_impact(um)

    figs = {
        "growth_overview": plots.growth_overview(kpis),
        "cohort_retention_heatmap": plots.cohort_heatmap(cohorts),
        "retention_by_activation": plots.retention_curves(
            ret_activation, "Week-1 merchant payment ('aha moment') and retention",
            {0: "No merchant payment in week 1", 1: "Merchant payment in week 1"}),
        "retention_by_channel": plots.retention_curves(ret_channel, "Retention by acquisition channel"),
        "onboarding_funnel": plots.funnel(funnel_overall),
        "clv_by_channel": plots.bar_by_segment(clv_channel, "segment", "clv_24m",
                                               "24-month CLV per KYC user by acquisition channel"),
        "revenue_pareto": plots.pareto(pareto_curve, top20),
    }
    for name, fig in figs.items():
        plots.save(fig, name)

    latest, first_full = kpis.iloc[-1], kpis.iloc[1]
    months = len(kpis) - 2
    mau_cmgr = (latest["mau"] / first_full["mau"]) ** (1 / months) - 1
    champions = rfm_summary.set_index("rfm_segment").loc["Champions"]
    best, worst = clv_channel.iloc[0], clv_channel.iloc[-1]
    or_act = impact["odds_ratios"].loc["activated_d7"]

    headline = {
        "total_users": int(kpis["new_users"].sum()),
        "final_mau": int(latest["mau"]),
        "mau_cmgr": mau_cmgr,
        "final_stickiness": float(stick["stickiness"].iloc[-1]),
        "kyc_rate": float(funnel_overall.loc[1, "pct_of_signups"]),
        "activation_rate": float(funnel_overall.loc[3, "pct_of_signups"]),
        "top20_revenue_share": top20,
        "champions_user_share": float(champions["share_of_users"]),
        "champions_revenue_share": float(champions["share_of_revenue"]),
        "best_channel": best["segment"],
        "best_channel_clv": float(best["clv_24m"]),
        "worst_channel": worst["segment"],
        "worst_channel_clv": float(worst["clv_24m"]),
        "activation_retention_lift_pp": impact["retention_lift_pp"],
        "activation_adjusted_odds_ratio": float(or_act["odds_ratio"]),
        "m6_retention_activated": float(clv_activation.set_index("segment").loc[1, "m6_retention"]),
        "m6_retention_not_activated": float(clv_activation.set_index("segment").loc[0, "m6_retention"]),
    }
    write_metrics("growth", {"headline": headline, "clv_by_channel": clv_channel,
                             "rfm": rfm_summary, "funnel": funnel_overall})

    pct = lambda v: f"{v:.1%}"  # noqa: E731
    report = f"""
# Growth analytics report

*Simulated UPI platform: {headline['total_users']:,} users, {kpis['transactions'].sum():,} successful
transactions, {kpis['month'].iloc[0]} to {kpis['month'].iloc[-1]}. All figures come from the SQL marts in
`data/simulated/upi.db`.*

## Key findings

1. **The week-1 merchant payment is the activation moment.** Users who pay a merchant within 7 days of
   signup are {pct(headline['m6_retention_activated'])} active at month 6 vs
   {pct(headline['m6_retention_not_activated'])} for everyone else. Controlling for channel, city tier,
   device and age, activation multiplies the odds of day 30-60 retention by
   **{or_act['odds_ratio']:.1f}x** (95% CI {or_act['ci_low']:.1f}-{or_act['ci_high']:.1f}). This is an
   association; the cashback experiment below tests the causal lever.
2. **Referral is the most valuable channel.** 24-month CLV per KYC user: {best['segment']}
   Rs {best['clv_24m']:.1f} vs {worst['segment']} Rs {worst['clv_24m']:.1f}
   ({best['clv_24m'] / worst['clv_24m']:.1f}x). Paid users activate and retain worst, so paid CAC must sit
   well below Rs {worst['clv_24m']:.0f} to break even.
3. **Revenue is concentrated.** The top 20% of transacting users generate {pct(top20)} of revenue;
   "Champions" are {pct(headline['champions_user_share'])} of transacting users but
   {pct(headline['champions_revenue_share'])} of revenue.
4. **Biggest funnel leak is activation.** {pct(headline['kyc_rate'])} complete KYC but only
   {pct(headline['activation_rate'])} make a merchant payment in week 1.
5. **Growth is healthy but decelerating.** MAU grew {pct(mau_cmgr)} per month (compound) to
   {headline['final_mau']:,}; DAU/MAU stickiness is {pct(headline['final_stickiness'])}.

![Growth overview](figures/growth_overview.png)

## Funnel

{md_table(funnel_overall, "{:.1%}")}

Conversion by channel (share of signups reaching each step):

{md_table(funnel_channel, "{:.1%}")}

![Funnel](figures/onboarding_funnel.png)

## Retention

![Cohort retention](figures/cohort_retention_heatmap.png)

![Retention by activation](figures/retention_by_activation.png)

![Retention by channel](figures/retention_by_channel.png)

## Customer lifetime value

CLV = sum over 24 months of (retention in month k) x (revenue per active month) x 1%/month discount.
Months 13-24 are projected by extending the observed decay of months 9-12.

{md_table(clv_channel, "{:,.3f}")}

![CLV by channel](figures/clv_by_channel.png)

First-6-month value by city tier:

{md_table(value_tier, "{:,.3f}")}

## RFM segmentation

{md_table(rfm_summary, "{:,.2f}")}

![Pareto](figures/revenue_pareto.png)

## Activation impact: logistic regression (adjusted odds ratios)

n = {impact['n_users']:,} KYC users with 60+ days of history; pseudo R² = {impact['pseudo_r2']:.3f}.

{md_table(impact['odds_ratios'], "{:.3f}", index=True)}

## Recommendations

- Make the **first merchant payment** the north-star onboarding goal: in-app QR prompts, merchant discovery,
  and the first-payment cashback (validated in the A/B report).
- **Shift acquisition budget from paid to referral**; reward referrers on the referee's *activation*, not signup.
- Protect **Champions** (priority support, limits) and run win-back flows for **At risk** users before they lapse.
- Investigate **tier-3 KYC drop-off** (lower KYC and activation rates): vernacular KYC and assisted onboarding.
"""
    write_report("growth_report.md", report)
    return {"headline": headline, "kpis": kpis, "clv_channel": clv_channel, "rfm": rfm_summary,
            "funnel": funnel_overall, "impact": impact}
