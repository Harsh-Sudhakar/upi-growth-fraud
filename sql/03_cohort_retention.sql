-- Monthly cohort retention triangle.
-- retention = users with >= 1 successful transaction in month k / users who signed up in the cohort.
WITH cohort_size AS (
    SELECT cohort_month, COUNT(*) AS cohort_size
    FROM user_metrics
    GROUP BY 1
)
SELECT
    um.cohort_month,
    fum.months_since_signup,
    COUNT(*)                                   AS active_users,
    cs.cohort_size,
    ROUND(1.0 * COUNT(*) / cs.cohort_size, 4)  AS retention
FROM fct_user_months fum
JOIN user_metrics um USING (user_id)
JOIN cohort_size cs USING (cohort_month)
GROUP BY um.cohort_month, fum.months_since_signup
ORDER BY um.cohort_month, fum.months_since_signup;
