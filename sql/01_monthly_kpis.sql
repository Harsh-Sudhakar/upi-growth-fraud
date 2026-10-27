-- Monthly platform KPIs: acquisition, MAU, volume, revenue and growth rates.
WITH signups AS (
    SELECT cohort_month AS month, COUNT(*) AS new_users
    FROM user_metrics
    GROUP BY 1
),
activity AS (
    SELECT
        txn_month               AS month,
        COUNT(DISTINCT user_id) AS mau,
        COUNT(*)                AS transactions,
        SUM(amount)             AS gmv,
        SUM(revenue)            AS revenue
    FROM fct_transactions
    GROUP BY 1
)
SELECT
    s.month,
    s.new_users,
    SUM(s.new_users) OVER (ORDER BY s.month)                         AS cumulative_users,
    a.mau,
    a.transactions,
    ROUND(a.gmv, 0)                                                  AS gmv,
    ROUND(a.revenue, 0)                                              AS revenue,
    ROUND(a.gmv / a.transactions, 1)                                 AS avg_txn_value,
    ROUND(1.0 * a.transactions / a.mau, 2)                           AS txns_per_mau,
    ROUND(a.revenue / a.mau, 2)                                      AS arpu,
    ROUND(100.0 * (a.mau - LAG(a.mau) OVER w) / LAG(a.mau) OVER w, 1) AS mau_growth_pct
FROM signups s
LEFT JOIN activity a USING (month)
WINDOW w AS (ORDER BY s.month)
ORDER BY s.month;
