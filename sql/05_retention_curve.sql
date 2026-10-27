-- Retention curve by segment: share of KYC-completed users active k months after signup.
-- A user only counts towards month k if they are old enough to have been observed then,
-- so recent cohorts don't drag the curve down (right-censoring).
-- {segment} is substituted from a whitelist in Python (upi.db.SEGMENTS).
WITH RECURSIVE k(m) AS (
    SELECT 0 UNION ALL SELECT m + 1 FROM k WHERE m < {max_month}
),
eligible AS (
    SELECT k.m AS months_since_signup, um.user_id, um.{segment} AS segment
    FROM user_metrics um
    JOIN k ON um.months_observed >= k.m
    WHERE um.kyc_completed = 1
)
SELECT
    e.segment,
    e.months_since_signup,
    COUNT(*)                                        AS eligible_users,
    COUNT(f.user_id)                                AS active_users,
    ROUND(1.0 * COUNT(f.user_id) / COUNT(*), 4)     AS retention
FROM eligible e
LEFT JOIN fct_user_months f
       ON f.user_id = e.user_id
      AND f.months_since_signup = e.months_since_signup
GROUP BY e.segment, e.months_since_signup
ORDER BY e.segment, e.months_since_signup;
