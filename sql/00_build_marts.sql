-- =====================================================================
-- Analytics layer: raw tables -> modelled tables every query builds on.
--   fct_transactions      successful transactions + tenure columns
--   fct_user_months       one row per user per active month (retention)
--   user_metrics          one row per user: lifecycle + value features
-- =====================================================================

DROP TABLE IF EXISTS fct_transactions;
CREATE TABLE fct_transactions AS
SELECT
    t.txn_id,
    t.user_id,
    t.txn_date,
    strftime('%Y-%m', t.txn_date)                               AS txn_month,
    t.category,
    t.amount,
    t.revenue,
    julianday(t.txn_date) - julianday(u.signup_date)            AS days_since_signup,
    (CAST(strftime('%Y', t.txn_date) AS INTEGER) - CAST(strftime('%Y', u.signup_date) AS INTEGER)) * 12
      + CAST(strftime('%m', t.txn_date) AS INTEGER) - CAST(strftime('%m', u.signup_date) AS INTEGER)
                                                                AS months_since_signup
FROM transactions t
JOIN users u USING (user_id)
WHERE t.status = 'success';

CREATE INDEX ix_fct_user ON fct_transactions(user_id);
CREATE INDEX ix_fct_month ON fct_transactions(txn_month);


DROP TABLE IF EXISTS fct_user_months;
CREATE TABLE fct_user_months AS
SELECT
    user_id,
    months_since_signup,
    COUNT(*)        AS txns,
    SUM(revenue)    AS revenue
FROM fct_transactions
GROUP BY user_id, months_since_signup;

CREATE UNIQUE INDEX ix_fum ON fct_user_months(user_id, months_since_signup);


DROP TABLE IF EXISTS user_metrics;
CREATE TABLE user_metrics AS
WITH bounds AS (
    SELECT MAX(txn_date) AS end_date FROM transactions
)
SELECT
    u.user_id,
    u.signup_date,
    strftime('%Y-%m', u.signup_date)                                    AS cohort_month,
    u.channel,
    u.city_tier,
    u.age_group,
    u.device,
    (u.kyc_completed_at IS NOT NULL)                                    AS kyc_completed,
    -- full calendar months observable after the signup month (right-censoring guard)
    (CAST(strftime('%Y', b.end_date) AS INTEGER) - CAST(strftime('%Y', u.signup_date) AS INTEGER)) * 12
      + CAST(strftime('%m', b.end_date) AS INTEGER) - CAST(strftime('%m', u.signup_date) AS INTEGER)
                                                                        AS months_observed,
    CAST(julianday(b.end_date) - julianday(u.signup_date) AS INTEGER)  AS days_observed,
    COUNT(t.txn_id)                                                     AS total_txns,
    COALESCE(ROUND(SUM(t.amount), 2), 0)                                AS total_gmv,
    COALESCE(ROUND(SUM(t.revenue), 2), 0)                               AS total_revenue,
    MIN(t.txn_date)                                                     AS first_txn_date,
    MAX(t.txn_date)                                                     AS last_txn_date,
    COUNT(DISTINCT t.txn_month)                                         AS active_months,
    COALESCE(MAX(t.category = 'merchant' AND t.days_since_signup < 7), 0) AS activated_d7,
    COALESCE(SUM(t.days_since_signup < 30), 0)                          AS txns_first_30d,
    COALESCE(MAX(t.days_since_signup BETWEEN 30 AND 59), 0)             AS active_d30_60,
    COALESCE(SUM(CASE WHEN t.months_since_signup <= 5 THEN t.revenue END), 0) AS revenue_first_6m
FROM users u
CROSS JOIN bounds b
LEFT JOIN fct_transactions t USING (user_id)
GROUP BY u.user_id;

CREATE UNIQUE INDEX ix_um_user ON user_metrics(user_id);
