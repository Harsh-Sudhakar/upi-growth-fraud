"""Interactive growth dashboard: ``make dashboard`` (streamlit run app/dashboard.py)."""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
import streamlit as st

from upi import config, db, growth

st.set_page_config(page_title="UPI Growth Analytics", layout="wide")


@st.cache_resource
def connection() -> sqlite3.Connection:
    con = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    if not db.marts_exist(con):
        db.build_marts(con)
    return con


if not config.DB_PATH.exists():
    st.error("No data yet. Run `make analytics` first.")
    st.stop()

con = connection()


@st.cache_data
def load(name: str, **params) -> pd.DataFrame:
    return db.query(con, name, **params)


st.title("UPI growth analytics")
st.caption("Simulated platform: 100K users, 18 months. Every number comes from the SQL marts in `data/simulated/upi.db`.")

kpis = load("01_monthly_kpis.sql")
last, prev = kpis.iloc[-1], kpis.iloc[-2]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Monthly active users", f"{last['mau']:,.0f}", f"{last['mau_growth_pct']:+.1f}% MoM")
c2.metric("Transactions (month)", f"{last['transactions']:,.0f}",
          f"{(last['transactions'] / prev['transactions'] - 1):+.1%} MoM")
c3.metric("Revenue (month)", f"Rs {last['revenue']:,.0f}", f"{(last['revenue'] / prev['revenue'] - 1):+.1%} MoM")
c4.metric("ARPU (month)", f"Rs {last['arpu']:.2f}")

tab_growth, tab_retention, tab_value, tab_ab, tab_fraud = st.tabs(
    ["Growth", "Retention", "Value & segments", "A/B tests", "Fraud model"])

with tab_growth:
    left, right = st.columns(2)
    left.subheader("New users per month")
    left.bar_chart(kpis.set_index("month")["new_users"])
    right.subheader("Monthly active users")
    right.line_chart(kpis.set_index("month")["mau"])
    st.subheader("Onboarding funnel")
    funnel, by_channel = growth.funnel(con)
    st.dataframe(funnel.style.format({"users": "{:,.0f}", "pct_of_signups": "{:.1%}", "step_conversion": "{:.1%}"}),
                 hide_index=True, use_container_width=True)
    st.dataframe(by_channel.set_index("channel").style.format("{:.1%}"), use_container_width=True)

with tab_retention:
    segment = st.selectbox("Split retention by", ["activated_d7", "channel", "city_tier", "device", "age_group"])
    curves = growth.retention_curves(con, segment)
    curves.columns = [str(c) for c in curves.columns]
    st.line_chart(curves)
    st.caption("Share of KYC users active k months after signup (censoring-aware).")
    st.subheader("Cohort retention triangle")
    matrix = growth.cohort_matrix(con)
    st.dataframe(matrix.style.format("{:.0%}", na_rep="").background_gradient(cmap="Blues", axis=None),
                 use_container_width=True)

with tab_value:
    seg = st.selectbox("Segment", ["channel", "city_tier", "device", "age_group"], key="value_seg")
    clv = growth.clv_by_segment(con, seg)
    left, right = st.columns(2)
    left.subheader("24-month CLV per KYC user")
    left.bar_chart(clv.set_index("segment")["clv_24m"])
    right.subheader("First-6-month value")
    right.dataframe(load("06_segment_value.sql", segment=seg), hide_index=True, use_container_width=True)
    st.subheader("RFM segments")
    um = pd.read_sql_query("SELECT * FROM user_metrics", con)
    _, rfm = growth.rfm_segments(um)
    st.dataframe(rfm.style.format({"share_of_users": "{:.1%}", "share_of_revenue": "{:.1%}",
                                   "avg_recency_days": "{:.0f}", "avg_txns": "{:.1f}", "avg_revenue": "{:.2f}"}),
                 hide_index=True, use_container_width=True)

with tab_ab:
    path = config.REPORTS_DIR / "metrics" / "abtest.json"
    if path.exists():
        ab = json.loads(path.read_text())
        st.dataframe(pd.DataFrame(ab["summary"])[["experiment", "n_control", "n_treatment", "srm_p",
                                                 "achieved_power", "decision"]],
                     hide_index=True, use_container_width=True)
        res = pd.DataFrame(ab["results"])
        st.dataframe(res[["experiment", "metric", "role", "control", "treatment", "diff", "ci_low", "ci_high",
                          "p_holm", "significant"]], hide_index=True, use_container_width=True)
        st.image(str(config.FIG_DIR / "ab_test_results.png"))
    else:
        st.info("Run `make abtest` to populate this tab.")

with tab_fraud:
    path = config.REPORTS_DIR / "metrics" / "fraud.json"
    if path.exists():
        fr = json.loads(path.read_text())
        t, b = fr["test"], fr["business"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PR-AUC (test)", f"{t['pr_auc']:.3f}")
        c2.metric("Recall", f"{t['recall']:.1%}")
        c3.metric("Precision", f"{t['precision']:.1%}")
        c4.metric("Fraud $ prevented", f"{b['share_of_fraud_amount_prevented']:.1%}")
        for fig in ("fraud_pr_curves", "fraud_threshold_cost", "fraud_shap_summary"):
            p = config.FIG_DIR / f"{fig}.png"
            if p.exists():
                st.image(str(p))
    else:
        st.info("Run `make data` then `make fraud` to populate this tab.")
