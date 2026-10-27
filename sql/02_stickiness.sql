-- Engagement depth: average DAU, MAU and DAU/MAU stickiness per month.
WITH daily AS (
    SELECT txn_date, txn_month, COUNT(DISTINCT user_id) AS dau
    FROM fct_transactions
    GROUP BY 1, 2
),
monthly AS (
    SELECT txn_month, COUNT(DISTINCT user_id) AS mau
    FROM fct_transactions
    GROUP BY 1
)
SELECT
    d.txn_month                          AS month,
    ROUND(AVG(d.dau), 0)                 AS avg_dau,
    m.mau,
    ROUND(AVG(d.dau) / m.mau, 4)         AS stickiness
FROM daily d
JOIN monthly m USING (txn_month)
GROUP BY d.txn_month
ORDER BY d.txn_month;
