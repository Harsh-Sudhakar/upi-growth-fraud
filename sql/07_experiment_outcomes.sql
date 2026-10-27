-- One row per experiment participant with every candidate outcome metric.
SELECT
    e.experiment_name,
    e.variant,
    e.user_id,
    um.channel,
    um.city_tier,
    um.device,
    um.days_observed,
    um.kyc_completed,
    um.activated_d7,
    um.txns_first_30d,
    um.active_d30_60
FROM experiments e
JOIN user_metrics um USING (user_id)
ORDER BY e.experiment_name, e.user_id;
