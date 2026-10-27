-- Value by segment over each user's first 6 months (only users observed >= 6 months).
SELECT
    {segment}                                        AS segment,
    COUNT(*)                                         AS users,
    ROUND(AVG(kyc_completed), 4)                     AS kyc_rate,
    ROUND(AVG(activated_d7), 4)                      AS activation_rate,
    ROUND(AVG(active_d30_60), 4)                     AS d30_60_retention,
    ROUND(AVG(revenue_first_6m), 2)                  AS revenue_6m_per_user,
    ROUND(SUM(revenue_first_6m), 0)                  AS revenue_6m_total
FROM user_metrics
WHERE months_observed >= 5
GROUP BY {segment}
ORDER BY revenue_6m_per_user DESC;
