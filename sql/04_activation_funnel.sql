-- Onboarding funnel by acquisition channel. Steps are nested: each is a subset of the previous one.
-- Only users with >= 60 days of history, so every step is fully observable.
SELECT
    channel,
    COUNT(*)                                  AS signed_up,
    SUM(kyc_completed)                        AS kyc_completed,
    SUM(total_txns > 0)                       AS first_transaction,
    SUM(activated_d7)                         AS activated_d7,
    SUM(activated_d7 AND active_d30_60)       AS activated_retained_d30_60
FROM user_metrics
WHERE days_observed >= 60
GROUP BY channel
ORDER BY channel;
