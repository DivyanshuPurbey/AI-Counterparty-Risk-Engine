-- =====================================================================
-- AI-Counterparty-Risk-Engine : analytical SQL (SQLite)
-- Each query is introduced by a "-- name: <query_name>" header and can be
-- run via  src.database.run_named_query(conn, "<query_name>", **params)
-- Amounts are EUR millions (suffix _mn). Parameters use :name placeholders.
-- Note: avoid colons inside comments/strings below - they confuse parsers.
-- =====================================================================

-- name: top_counterparties_by_ead
-- Largest counterparties by Exposure at Default. Parameter n = number of rows.
SELECT c.counterparty_id,
       c.counterparty_name,
       c.industry,
       c.country,
       c.rating,
       e.product_type,
       ROUND(e.current_exposure, 2) AS current_exposure_mn,
       ROUND(r.pfe, 2)              AS pfe_mn,
       ROUND(r.eligible_collateral, 2) AS eligible_collateral_mn,
       ROUND(r.ead, 2)              AS ead_mn,
       ROUND(100.0 * r.pd, 2)       AS pd_pct,
       ROUND(r.expected_loss, 3)    AS expected_loss_mn
FROM risk_metrics r
JOIN counterparties c ON c.counterparty_id = r.counterparty_id
JOIN exposures      e ON e.counterparty_id = r.counterparty_id
ORDER BY r.ead DESC
LIMIT :n;

-- name: top_industries_by_exposure
-- Exposure, expected loss and share of portfolio EAD by industry.
SELECT c.industry,
       COUNT(*)                                             AS n_counterparties,
       ROUND(SUM(e.current_exposure), 1)                    AS current_exposure_mn,
       ROUND(SUM(r.pfe), 1)                                 AS pfe_mn,
       ROUND(SUM(r.ead), 1)                                 AS ead_mn,
       ROUND(100.0 * SUM(r.ead) / SUM(SUM(r.ead)) OVER (), 2) AS ead_share_pct,
       ROUND(SUM(r.expected_loss), 2)                       AS expected_loss_mn,
       ROUND(SUM(r.pd * r.ead) / NULLIF(SUM(r.ead), 0) * 100.0, 2) AS ead_weighted_pd_pct
FROM risk_metrics r
JOIN counterparties c ON c.counterparty_id = r.counterparty_id
JOIN exposures      e ON e.counterparty_id = r.counterparty_id
GROUP BY c.industry
ORDER BY SUM(r.ead) DESC;

-- name: top_countries_by_exposure
-- Exposure, expected loss and share of portfolio EAD by country.
SELECT c.country,
       c.region,
       COUNT(*)                                             AS n_counterparties,
       ROUND(SUM(r.ead), 1)                                 AS ead_mn,
       ROUND(100.0 * SUM(r.ead) / SUM(SUM(r.ead)) OVER (), 2) AS ead_share_pct,
       ROUND(SUM(r.expected_loss), 2)                       AS expected_loss_mn
FROM risk_metrics r
JOIN counterparties c ON c.counterparty_id = r.counterparty_id
GROUP BY c.country, c.region
ORDER BY SUM(r.ead) DESC;

-- name: expected_loss_by_rating
-- Expected loss by rating grade, ordered from AAA to CCC.
SELECT c.rating,
       COUNT(*)                                             AS n_counterparties,
       ROUND(SUM(r.ead), 1)                                 AS ead_mn,
       ROUND(100.0 * SUM(r.ead) / SUM(SUM(r.ead)) OVER (), 2) AS ead_share_pct,
       ROUND(SUM(r.expected_loss), 2)                       AS expected_loss_mn,
       ROUND(100.0 * SUM(r.expected_loss) / SUM(SUM(r.expected_loss)) OVER (), 2) AS el_share_pct,
       ROUND(100.0 * SUM(r.pd * r.ead) / NULLIF(SUM(r.ead), 0), 3) AS ead_weighted_pd_pct,
       ROUND(1e4 * SUM(r.expected_loss) / NULLIF(SUM(r.ead), 0), 1) AS el_to_ead_bps
FROM risk_metrics r
JOIN counterparties c ON c.counterparty_id = r.counterparty_id
GROUP BY c.rating
ORDER BY CASE c.rating WHEN 'AAA' THEN 1 WHEN 'AA' THEN 2 WHEN 'A' THEN 3
                       WHEN 'BBB' THEN 4 WHEN 'BB' THEN 5 WHEN 'B' THEN 6
                       WHEN 'CCC' THEN 7 END;

-- name: expected_loss_by_product
-- Expected loss and PFE by product type.
SELECT e.product_type,
       COUNT(*)                          AS n_counterparties,
       ROUND(SUM(e.notional), 0)         AS notional_mn,
       ROUND(SUM(r.pfe), 1)              AS pfe_mn,
       ROUND(SUM(r.ead), 1)              AS ead_mn,
       ROUND(SUM(r.expected_loss), 2)    AS expected_loss_mn
FROM risk_metrics r
JOIN exposures e ON e.counterparty_id = r.counterparty_id
GROUP BY e.product_type
ORDER BY SUM(r.expected_loss) DESC;

-- name: high_risk_counterparties
-- Illustrative screen - PD at or above :pd_threshold (decimal, e.g. 0.10),
-- or rated B/CCC with EAD in the top decile of the portfolio.
WITH ranked AS (
    SELECT counterparty_id, NTILE(10) OVER (ORDER BY ead DESC) AS ead_decile
    FROM risk_metrics
)
SELECT v.counterparty_id,
       v.counterparty_name,
       v.industry,
       v.country,
       v.rating,
       ROUND(100.0 * v.pd, 2)    AS pd_pct,
       ROUND(v.ead, 2)           AS ead_mn,
       ROUND(v.expected_loss, 3) AS expected_loss_mn,
       ROUND(v.debt_to_ebitda, 2) AS debt_to_ebitda,
       ROUND(v.interest_coverage, 2) AS interest_coverage
FROM v_counterparty_risk v
JOIN ranked k ON k.counterparty_id = v.counterparty_id
WHERE v.pd >= :pd_threshold
   OR (v.rating IN ('B', 'CCC') AND k.ead_decile = 1)
ORDER BY v.expected_loss DESC
LIMIT :n;

-- name: high_exposure_low_rating
-- Counterparties rated BB or below whose EAD is in the top decile.
WITH ranked AS (
    SELECT counterparty_id, NTILE(10) OVER (ORDER BY ead DESC) AS ead_decile
    FROM risk_metrics
)
SELECT v.counterparty_id,
       v.counterparty_name,
       v.industry,
       v.country,
       v.rating,
       v.product_type,
       ROUND(v.ead, 2)           AS ead_mn,
       ROUND(100.0 * v.pd, 2)    AS pd_pct,
       ROUND(v.expected_loss, 3) AS expected_loss_mn
FROM v_counterparty_risk v
JOIN ranked k ON k.counterparty_id = v.counterparty_id
WHERE v.rating IN ('BB', 'B', 'CCC')
  AND k.ead_decile = 1
ORDER BY v.ead DESC
LIMIT :n;

-- name: top_unsecured_exposures
-- Largest unsecured current exposure (current exposure net of eligible collateral).
SELECT v.counterparty_id,
       v.counterparty_name,
       v.industry,
       v.rating,
       ROUND(v.current_exposure, 2)     AS current_exposure_mn,
       ROUND(v.eligible_collateral, 2)  AS eligible_collateral_mn,
       ROUND(v.unsecured_exposure, 2)   AS unsecured_exposure_mn,
       ROUND(v.collateral_coverage, 3)  AS collateral_coverage
FROM v_counterparty_risk v
ORDER BY v.unsecured_exposure DESC
LIMIT :n;

-- name: top_pfe_counterparties
-- Counterparties with the largest simplified PFE.
SELECT v.counterparty_id,
       v.counterparty_name,
       v.industry,
       v.rating,
       v.product_type,
       ROUND(v.notional, 1)        AS notional_mn,
       ROUND(v.maturity_years, 2)  AS maturity_years,
       ROUND(r.pfe, 2)             AS pfe_mn,
       ROUND(100.0 * r.pfe / v.notional, 2) AS pfe_pct_of_notional
FROM v_counterparty_risk v
JOIN risk_metrics r ON r.counterparty_id = v.counterparty_id
ORDER BY r.pfe DESC
LIMIT :n;

-- name: industry_concentration_hhi
-- Herfindahl-Hirschman Index of EAD across industries (0-10,000 scale).
-- Effective number = 1 / HHI (share form) = number of equally sized industries.
WITH by_industry AS (
    SELECT c.industry, SUM(r.ead) AS ead
    FROM risk_metrics r
    JOIN counterparties c ON c.counterparty_id = r.counterparty_id
    GROUP BY c.industry
), shares AS (
    SELECT industry, ead / SUM(ead) OVER () AS share FROM by_industry
)
SELECT COUNT(*)                                AS n_industries,
       ROUND(SUM(share * share) * 10000.0, 0)  AS hhi,
       ROUND(1.0 / SUM(share * share), 2)      AS effective_number,
       ROUND(100.0 * MAX(share), 2)            AS largest_industry_share_pct
FROM shares;

-- name: country_concentration_hhi
-- Herfindahl-Hirschman Index of EAD across countries (0-10,000 scale).
WITH by_country AS (
    SELECT c.country, SUM(r.ead) AS ead
    FROM risk_metrics r
    JOIN counterparties c ON c.counterparty_id = r.counterparty_id
    GROUP BY c.country
), shares AS (
    SELECT country, ead / SUM(ead) OVER () AS share FROM by_country
)
SELECT COUNT(*)                                AS n_countries,
       ROUND(SUM(share * share) * 10000.0, 0)  AS hhi,
       ROUND(1.0 / SUM(share * share), 2)      AS effective_number,
       ROUND(100.0 * MAX(share), 2)            AS largest_country_share_pct
FROM shares;

-- name: top_n_exposure_share
-- Share of portfolio EAD held by the 10 and 20 largest counterparties.
WITH ranked AS (
    SELECT ead, ROW_NUMBER() OVER (ORDER BY ead DESC) AS rn
    FROM risk_metrics
)
SELECT ROUND(100.0 * SUM(CASE WHEN rn <= 10 THEN ead ELSE 0 END) / SUM(ead), 2) AS top10_share_pct,
       ROUND(100.0 * SUM(CASE WHEN rn <= 20 THEN ead ELSE 0 END) / SUM(ead), 2) AS top20_share_pct,
       ROUND(SUM(ead), 1) AS total_ead_mn
FROM ranked;

-- name: stress_loss_by_industry
-- Base versus stressed expected loss by industry for one scenario (parameter scenario).
SELECT c.industry,
       s.scenario,
       ROUND(SUM(r.expected_loss), 2)                   AS base_el_mn,
       ROUND(SUM(s.stressed_expected_loss), 2)          AS stressed_el_mn,
       ROUND(SUM(s.incremental_expected_loss), 2)       AS incremental_el_mn,
       ROUND(100.0 * SUM(s.incremental_expected_loss)
             / NULLIF(SUM(r.expected_loss), 0), 1)      AS el_pct_increase,
       ROUND(SUM(s.incremental_ead), 1)                 AS incremental_ead_mn
FROM stress_results s
JOIN counterparties c ON c.counterparty_id = s.counterparty_id
JOIN risk_metrics   r ON r.counterparty_id = s.counterparty_id
WHERE s.scenario = :scenario
GROUP BY c.industry, s.scenario
ORDER BY SUM(s.incremental_expected_loss) DESC;

-- name: stress_summary_by_scenario
-- Portfolio base versus stressed EAD and expected loss for every scenario.
SELECT s.scenario,
       d.scenario_type,
       ROUND(SUM(r.ead), 1)                          AS base_ead_mn,
       ROUND(SUM(s.stressed_ead), 1)                 AS stressed_ead_mn,
       ROUND(SUM(r.expected_loss), 2)                AS base_el_mn,
       ROUND(SUM(s.stressed_expected_loss), 2)       AS stressed_el_mn,
       ROUND(SUM(s.incremental_expected_loss), 2)    AS incremental_el_mn,
       ROUND(100.0 * SUM(s.incremental_expected_loss)
             / NULLIF(SUM(r.expected_loss), 0), 1)   AS el_pct_increase
FROM stress_results s
JOIN risk_metrics         r ON r.counterparty_id = s.counterparty_id
JOIN scenario_definitions d ON d.scenario = s.scenario
GROUP BY s.scenario, d.scenario_type
ORDER BY d.scenario_type, SUM(s.incremental_expected_loss) DESC;

-- name: rating_downgrades
-- Counterparties downgraded versus the prior year, largest EAD first.
WITH notches AS (
    SELECT counterparty_id, counterparty_name, industry, rating, rating_prior_year,
           CASE rating WHEN 'AAA' THEN 0 WHEN 'AA' THEN 1 WHEN 'A' THEN 2 WHEN 'BBB' THEN 3
                       WHEN 'BB' THEN 4 WHEN 'B' THEN 5 WHEN 'CCC' THEN 6 END AS r_now,
           CASE rating_prior_year WHEN 'AAA' THEN 0 WHEN 'AA' THEN 1 WHEN 'A' THEN 2 WHEN 'BBB' THEN 3
                       WHEN 'BB' THEN 4 WHEN 'B' THEN 5 WHEN 'CCC' THEN 6 END AS r_prior
    FROM counterparties
)
SELECT n.counterparty_id,
       n.counterparty_name,
       n.industry,
       n.rating_prior_year,
       n.rating,
       n.r_now - n.r_prior          AS notches_downgraded,
       ROUND(r.ead, 2)              AS ead_mn,
       ROUND(100.0 * r.pd, 2)       AS pd_pct
FROM notches n
JOIN risk_metrics r ON r.counterparty_id = n.counterparty_id
WHERE n.r_now > n.r_prior
ORDER BY notches_downgraded DESC, r.ead DESC
LIMIT :n;

-- name: early_warning_screen
-- Illustrative screen - Debt/EBITDA above :max_leverage and interest coverage
-- below :min_coverage (defaults in the project are 6.0x and 2.0x).
SELECT v.counterparty_id,
       v.counterparty_name,
       v.industry,
       v.rating,
       ROUND(v.debt_to_ebitda, 2)     AS debt_to_ebitda,
       ROUND(v.interest_coverage, 2)  AS interest_coverage,
       ROUND(v.ead, 2)                AS ead_mn,
       ROUND(100.0 * v.pd, 2)         AS pd_pct
FROM v_counterparty_risk v
WHERE v.debt_to_ebitda > :max_leverage
  AND v.interest_coverage < :min_coverage
ORDER BY v.ead DESC
LIMIT :n;
