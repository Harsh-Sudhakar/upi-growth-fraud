-- Average revenue and transactions per *active* user-month, by segment (KYC users).
-- Combined with the retention curve this gives CLV = sum_k retention_k * ARPAU * discount_k.
SELECT
    um.{segment}                        AS segment,
    COUNT(*)                            AS active_user_months,
    ROUND(AVG(f.revenue), 4)            AS revenue_per_active_month,
    ROUND(AVG(f.txns), 3)               AS txns_per_active_month
FROM fct_user_months f
JOIN user_metrics um USING (user_id)
WHERE um.kyc_completed = 1
GROUP BY um.{segment}
ORDER BY um.{segment};
