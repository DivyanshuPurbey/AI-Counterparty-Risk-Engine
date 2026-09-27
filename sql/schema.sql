-- =====================================================================
-- AI-Counterparty-Risk-Engine : SQLite schema
-- All data is SYNTHETIC. Monetary amounts are EUR millions.
-- Re-running this script rebuilds every table from scratch.
-- =====================================================================
PRAGMA foreign_keys = ON;

DROP VIEW  IF EXISTS v_counterparty_risk;
DROP TABLE IF EXISTS stress_results;
DROP TABLE IF EXISTS scenario_definitions;
DROP TABLE IF EXISTS risk_metrics;
DROP TABLE IF EXISTS exposures;
DROP TABLE IF EXISTS counterparties;

-- ---------------------------------------------------------------------
-- Reference data, financial statements and derived credit ratios
-- ---------------------------------------------------------------------
CREATE TABLE counterparties (
    counterparty_id                 TEXT PRIMARY KEY,
    counterparty_name               TEXT NOT NULL,
    industry                        TEXT NOT NULL,
    country                         TEXT NOT NULL,
    region                          TEXT NOT NULL,
    rating                          TEXT NOT NULL
        CHECK (rating IN ('AAA','AA','A','BBB','BB','B','CCC')),
    rating_prior_year               TEXT NOT NULL
        CHECK (rating_prior_year IN ('AAA','AA','A','BBB','BB','B','CCC')),
    annual_revenue                  REAL NOT NULL CHECK (annual_revenue >= 0),
    revenue_prior_year              REAL NOT NULL CHECK (revenue_prior_year >= 0),
    total_assets                    REAL NOT NULL CHECK (total_assets >= 0),
    total_debt                      REAL NOT NULL CHECK (total_debt >= 0),
    cash                            REAL NOT NULL CHECK (cash >= 0),
    current_assets                  REAL NOT NULL CHECK (current_assets >= 0),
    current_liabilities             REAL NOT NULL CHECK (current_liabilities >= 0),
    ebitda                          REAL NOT NULL,
    interest_expense                REAL NOT NULL CHECK (interest_expense >= 0),
    operating_cash_flow             REAL NOT NULL,
    operating_cash_flow_prior_year  REAL NOT NULL,
    debt_to_ebitda                  REAL,
    interest_coverage               REAL,
    current_ratio                   REAL,
    debt_to_assets                  REAL,
    cash_to_debt                    REAL,
    ocf_to_debt                     REAL,
    ebitda_margin                   REAL,
    revenue_growth                  REAL,
    ocf_growth                      REAL,
    credit_health_score             REAL,
    credit_score_band               TEXT,
    default_flag                    INTEGER NOT NULL CHECK (default_flag IN (0, 1))
);

-- ---------------------------------------------------------------------
-- Exposure inputs: one netting set / one product per counterparty
-- ---------------------------------------------------------------------
CREATE TABLE exposures (
    counterparty_id                 TEXT PRIMARY KEY
        REFERENCES counterparties (counterparty_id),
    product_type                    TEXT NOT NULL
        CHECK (product_type IN ('Interest Rate','FX','Equity','Commodity','Securities Financing')),
    notional                        REAL NOT NULL CHECK (notional >= 0),
    maturity_years                  REAL NOT NULL CHECK (maturity_years > 0),
    current_exposure                REAL NOT NULL CHECK (current_exposure >= 0),
    current_exposure_prior_quarter  REAL NOT NULL CHECK (current_exposure_prior_quarter >= 0),
    fx_exposure                     REAL NOT NULL CHECK (fx_exposure >= 0),
    interest_rate_exposure          REAL NOT NULL CHECK (interest_rate_exposure >= 0),
    collateral                      REAL NOT NULL CHECK (collateral >= 0),
    collateral_type                 TEXT NOT NULL
        CHECK (collateral_type IN ('Cash','Government Bonds','Corporate Bonds','Equities','None'))
);

-- ---------------------------------------------------------------------
-- Outputs of the deterministic risk engine (Python) - base case
-- ---------------------------------------------------------------------
CREATE TABLE risk_metrics (
    counterparty_id                 TEXT PRIMARY KEY
        REFERENCES counterparties (counterparty_id),
    pd                              REAL NOT NULL CHECK (pd BETWEEN 0 AND 1),
    lgd                             REAL NOT NULL CHECK (lgd BETWEEN 0 AND 1),
    addon_factor                    REAL NOT NULL,
    maturity_factor                 REAL NOT NULL,
    pfe                             REAL NOT NULL CHECK (pfe >= 0),
    collateral_haircut              REAL NOT NULL CHECK (collateral_haircut BETWEEN 0 AND 1),
    eligible_collateral             REAL NOT NULL CHECK (eligible_collateral >= 0),
    unsecured_exposure              REAL NOT NULL CHECK (unsecured_exposure >= 0),
    collateral_coverage             REAL,              -- NULL when current exposure = 0
    ead                             REAL NOT NULL CHECK (ead >= 0),
    expected_loss                   REAL NOT NULL CHECK (expected_loss >= 0)
);

-- ---------------------------------------------------------------------
-- Scenario parameters (macro scenarios and thematic scenarios)
-- ---------------------------------------------------------------------
CREATE TABLE scenario_definitions (
    scenario                        TEXT PRIMARY KEY,
    scenario_type                   TEXT NOT NULL CHECK (scenario_type IN ('MACRO','THEMATIC')),
    description                     TEXT,
    pd_mult                         REAL NOT NULL,
    lgd_mult                        REAL NOT NULL,
    exposure_mult                   REAL NOT NULL,
    fx_exposure_mult                REAL NOT NULL,
    ir_exposure_mult                REAL NOT NULL,
    collateral_haircut_add          REAL NOT NULL,
    overrides_json                  TEXT               -- industry/country/product overrides
);

-- ---------------------------------------------------------------------
-- Stressed metrics per counterparty per scenario
-- ---------------------------------------------------------------------
CREATE TABLE stress_results (
    counterparty_id                 TEXT NOT NULL REFERENCES counterparties (counterparty_id),
    scenario                        TEXT NOT NULL REFERENCES scenario_definitions (scenario),
    stressed_pd                     REAL NOT NULL,
    stressed_lgd                    REAL NOT NULL,
    stressed_current_exposure       REAL NOT NULL,
    stressed_pfe                    REAL NOT NULL,
    stressed_eligible_collateral    REAL NOT NULL,
    stressed_ead                    REAL NOT NULL,
    stressed_expected_loss          REAL NOT NULL,
    incremental_ead                 REAL NOT NULL,
    incremental_expected_loss       REAL NOT NULL,
    PRIMARY KEY (counterparty_id, scenario)
);

-- ---------------------------------------------------------------------
-- Indexes for the common access paths
-- ---------------------------------------------------------------------
CREATE INDEX idx_cpty_industry  ON counterparties (industry);
CREATE INDEX idx_cpty_country   ON counterparties (country);
CREATE INDEX idx_cpty_rating    ON counterparties (rating);
CREATE INDEX idx_exposure_prod  ON exposures (product_type);
CREATE INDEX idx_risk_ead       ON risk_metrics (ead DESC);
CREATE INDEX idx_stress_scen    ON stress_results (scenario);

-- ---------------------------------------------------------------------
-- Convenience view joining the three core tables
-- ---------------------------------------------------------------------
CREATE VIEW v_counterparty_risk AS
SELECT c.counterparty_id, c.counterparty_name, c.industry, c.country, c.region,
       c.rating, c.rating_prior_year, c.debt_to_ebitda, c.interest_coverage,
       c.credit_health_score, c.default_flag,
       e.product_type, e.notional, e.maturity_years, e.current_exposure,
       e.collateral, e.collateral_type,
       r.pd, r.lgd, r.pfe, r.eligible_collateral, r.unsecured_exposure,
       r.collateral_coverage, r.ead, r.expected_loss
FROM counterparties c
JOIN exposures    e ON e.counterparty_id = c.counterparty_id
JOIN risk_metrics r ON r.counterparty_id = c.counterparty_id;
