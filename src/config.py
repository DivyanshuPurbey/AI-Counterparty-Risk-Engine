"""Central configuration for the AI-Counterparty-Risk-Engine.

IMPORTANT
---------
Every numeric parameter in this module is ILLUSTRATIVE. Values were chosen to
produce plausible, internally consistent *synthetic* data and to keep the
educational risk methodology transparent. They are NOT calibrated to market or
default data, and they are NOT taken from any regulatory framework or from any
bank's internal methodology.

Monetary amounts are expressed in EUR millions throughout the project.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
SQL_DIR: Path = PROJECT_ROOT / "sql"

RAW_DATA_PATH: Path = DATA_RAW_DIR / "counterparties_raw.csv"
PROCESSED_DATA_PATH: Path = DATA_PROCESSED_DIR / "counterparty_risk_table.csv"
STRESS_RESULTS_PATH: Path = DATA_PROCESSED_DIR / "stress_results.csv"
DATABASE_PATH: Path = DATA_PROCESSED_DIR / "risk_engine.db"

SCHEMA_SQL_PATH: Path = SQL_DIR / "schema.sql"
QUERIES_SQL_PATH: Path = SQL_DIR / "queries.sql"

# --------------------------------------------------------------------------- #
# Reproducibility / size
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42
# 5,000 names -> ~120 synthetic defaults, enough for a (noisy but usable) PD model.
DEFAULT_N_COUNTERPARTIES: int = 5000
MIN_N_COUNTERPARTIES: int = 100  # guard rail for quick experiments only

# --------------------------------------------------------------------------- #
# Rating scale (best -> worst). Index 0 = AAA ... 6 = CCC.
# --------------------------------------------------------------------------- #
RATINGS: list[str] = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
RATING_INDEX: dict[str, int] = {r: i for i, r in enumerate(RATINGS)}

# Illustrative portfolio rating mix (investment-grade heavy, as in a
# derivatives-counterparty book).
RATING_MIX: dict[str, float] = {
    "AAA": 0.02,
    "AA": 0.06,
    "A": 0.20,
    "BBB": 0.32,
    "BB": 0.22,
    "B": 0.13,
    "CCC": 0.05,
}

# Illustrative one-year PD anchors and financial "typical" values per rating.
#   pd             : 1-year PD anchor (illustrative, not an agency default study)
#   lgd            : senior-unsecured-style LGD anchor (after netting; collateral
#                    is handled separately in EAD, so it is NOT double counted)
#   debt_ebitda    : typical Debt/EBITDA (x)
#   cost_of_debt   : typical interest rate on debt
#   debt_to_assets : typical Debt/Assets
#   cash_to_debt   : typical Cash/Debt
#   current_ratio  : typical current assets / current liabilities
#   csa_prob       : probability the counterparty is under a collateral (CSA)
#                    agreement. Lower ratings are more likely to be required to
#                    post collateral.
RATING_PARAMS: dict[str, dict[str, float]] = {
    "AAA": {"pd": 0.0003, "lgd": 0.35, "debt_ebitda": 0.8, "cost_of_debt": 0.025,
            "debt_to_assets": 0.12, "cash_to_debt": 0.60, "current_ratio": 1.90,
            "csa_prob": 0.35},
    "AA": {"pd": 0.0007, "lgd": 0.37, "debt_ebitda": 1.2, "cost_of_debt": 0.028,
           "debt_to_assets": 0.16, "cash_to_debt": 0.50, "current_ratio": 1.70,
           "csa_prob": 0.45},
    "A": {"pd": 0.0015, "lgd": 0.40, "debt_ebitda": 1.8, "cost_of_debt": 0.033,
          "debt_to_assets": 0.22, "cash_to_debt": 0.40, "current_ratio": 1.50,
          "csa_prob": 0.60},
    "BBB": {"pd": 0.0040, "lgd": 0.43, "debt_ebitda": 2.6, "cost_of_debt": 0.042,
            "debt_to_assets": 0.28, "cash_to_debt": 0.30, "current_ratio": 1.35,
            "csa_prob": 0.75},
    "BB": {"pd": 0.0150, "lgd": 0.47, "debt_ebitda": 3.8, "cost_of_debt": 0.058,
           "debt_to_assets": 0.35, "cash_to_debt": 0.22, "current_ratio": 1.20,
           "csa_prob": 0.85},
    "B": {"pd": 0.0600, "lgd": 0.53, "debt_ebitda": 5.5, "cost_of_debt": 0.078,
          "debt_to_assets": 0.45, "cash_to_debt": 0.15, "current_ratio": 1.05,
          "csa_prob": 0.90},
    "CCC": {"pd": 0.2200, "lgd": 0.60, "debt_ebitda": 8.0, "cost_of_debt": 0.110,
            "debt_to_assets": 0.58, "cash_to_debt": 0.08, "current_ratio": 0.90,
            "csa_prob": 0.90},
}

# --------------------------------------------------------------------------- #
# Industries
#   weight          : share of counterparties
#   ebitda_margin   : typical EBITDA margin
#   leverage_mult   : multiplier on the rating-typical Debt/EBITDA
#   size_log_adj    : additive adjustment to log(revenue) (bigger/smaller names)
#   pd_logit_shift  : additive shift to the PD logit (sector risk premium)
#   lgd_shift       : additive shift to LGD
# --------------------------------------------------------------------------- #
INDUSTRY_PARAMS: dict[str, dict[str, float]] = {
    "Financial Services": {"weight": 0.14, "ebitda_margin": 0.35, "leverage_mult": 1.30,
                           "size_log_adj": 0.30, "pd_logit_shift": 0.10, "lgd_shift": -0.03},
    "Energy": {"weight": 0.10, "ebitda_margin": 0.22, "leverage_mult": 1.10,
               "size_log_adj": 0.25, "pd_logit_shift": 0.15, "lgd_shift": 0.00},
    "Technology": {"weight": 0.11, "ebitda_margin": 0.24, "leverage_mult": 0.80,
                   "size_log_adj": 0.10, "pd_logit_shift": 0.05, "lgd_shift": 0.05},
    "Manufacturing": {"weight": 0.12, "ebitda_margin": 0.14, "leverage_mult": 1.00,
                      "size_log_adj": 0.00, "pd_logit_shift": 0.00, "lgd_shift": 0.00},
    "Healthcare": {"weight": 0.09, "ebitda_margin": 0.22, "leverage_mult": 0.90,
                   "size_log_adj": 0.05, "pd_logit_shift": -0.15, "lgd_shift": 0.00},
    "Telecommunications": {"weight": 0.06, "ebitda_margin": 0.32, "leverage_mult": 1.20,
                           "size_log_adj": 0.15, "pd_logit_shift": -0.05, "lgd_shift": -0.02},
    "Real Estate": {"weight": 0.10, "ebitda_margin": 0.55, "leverage_mult": 1.50,
                    "size_log_adj": -0.20, "pd_logit_shift": 0.20, "lgd_shift": -0.05},
    "Consumer Goods": {"weight": 0.10, "ebitda_margin": 0.15, "leverage_mult": 0.95,
                       "size_log_adj": 0.05, "pd_logit_shift": -0.05, "lgd_shift": 0.02},
    "Industrials": {"weight": 0.10, "ebitda_margin": 0.13, "leverage_mult": 1.00,
                    "size_log_adj": 0.00, "pd_logit_shift": 0.05, "lgd_shift": 0.00},
    "Transportation": {"weight": 0.08, "ebitda_margin": 0.16, "leverage_mult": 1.15,
                       "size_log_adj": -0.05, "pd_logit_shift": 0.10, "lgd_shift": 0.00},
}
INDUSTRIES: list[str] = list(INDUSTRY_PARAMS)

# --------------------------------------------------------------------------- #
# Countries / regions (EMEA)
# --------------------------------------------------------------------------- #
COUNTRY_PARAMS: dict[str, dict[str, object]] = {
    "UK": {"region": "UK & Ireland", "weight": 0.22},
    "Germany": {"region": "Western Europe", "weight": 0.18},
    "France": {"region": "Western Europe", "weight": 0.15},
    "Netherlands": {"region": "Western Europe", "weight": 0.08},
    "Spain": {"region": "Southern Europe", "weight": 0.08},
    "Italy": {"region": "Southern Europe", "weight": 0.09},
    "Ireland": {"region": "UK & Ireland", "weight": 0.05},
    "Switzerland": {"region": "Western Europe", "weight": 0.07},
    "UAE": {"region": "Middle East", "weight": 0.08},
}
COUNTRIES: list[str] = list(COUNTRY_PARAMS)

LEGAL_FORMS: dict[str, list[str]] = {
    "UK": ["PLC", "Ltd"],
    "Germany": ["AG", "GmbH"],
    "France": ["SA", "SAS"],
    "Netherlands": ["N.V.", "B.V."],
    "Spain": ["S.A."],
    "Italy": ["S.p.A."],
    "Ireland": ["PLC", "DAC"],
    "Switzerland": ["AG", "SA"],
    "UAE": ["PJSC", "LLC"],
}

# --------------------------------------------------------------------------- #
# Products
# --------------------------------------------------------------------------- #
PRODUCT_TYPES: list[str] = [
    "Interest Rate", "FX", "Equity", "Commodity", "Securities Financing",
]

# Probability of each product type by industry (rows are normalised in code).
INDUSTRY_PRODUCT_MIX: dict[str, list[float]] = {
    #                         IR    FX    EQ    COM   SFT
    "Financial Services": [0.35, 0.20, 0.15, 0.05, 0.25],
    "Energy": [0.15, 0.15, 0.05, 0.55, 0.10],
    "Technology": [0.25, 0.40, 0.25, 0.02, 0.08],
    "Manufacturing": [0.30, 0.40, 0.05, 0.20, 0.05],
    "Healthcare": [0.35, 0.35, 0.15, 0.02, 0.13],
    "Telecommunications": [0.55, 0.25, 0.05, 0.02, 0.13],
    "Real Estate": [0.65, 0.10, 0.10, 0.02, 0.13],
    "Consumer Goods": [0.25, 0.40, 0.08, 0.17, 0.10],
    "Industrials": [0.28, 0.30, 0.07, 0.25, 0.10],
    "Transportation": [0.25, 0.30, 0.03, 0.32, 0.10],
}

# DATA-GENERATION assumptions per product (used by the generator only).
#   ce_frac        : median current exposure as a fraction of notional
#   mat_median     : median maturity in years (log-normal)
#   mat_sigma      : log-normal sigma for maturity
#   mat_min/max    : clipping bounds for maturity (years)
#   notional_scale : relative notional size
#   lgd_shift      : additive LGD shift
PRODUCT_GENERATION_PARAMS: dict[str, dict[str, float]] = {
    "Interest Rate": {"ce_frac": 0.015, "mat_median": 5.0, "mat_sigma": 0.60,
                      "mat_min": 0.5, "mat_max": 30.0, "notional_scale": 1.5, "lgd_shift": 0.0},
    "FX": {"ce_frac": 0.020, "mat_median": 0.6, "mat_sigma": 0.70,
           "mat_min": 0.03, "mat_max": 5.0, "notional_scale": 1.0, "lgd_shift": 0.0},
    "Equity": {"ce_frac": 0.050, "mat_median": 1.5, "mat_sigma": 0.60,
               "mat_min": 0.10, "mat_max": 5.0, "notional_scale": 0.5, "lgd_shift": 0.02},
    "Commodity": {"ce_frac": 0.060, "mat_median": 1.0, "mat_sigma": 0.60,
                  "mat_min": 0.10, "mat_max": 5.0, "notional_scale": 0.4, "lgd_shift": 0.0},
    "Securities Financing": {"ce_frac": 0.015, "mat_median": 0.15, "mat_sigma": 0.80,
                             "mat_min": 0.02, "mat_max": 1.0, "notional_scale": 0.8,
                             "lgd_shift": -0.05},
}

# --------------------------------------------------------------------------- #
# EXPOSURE METHODOLOGY PARAMETERS (used by the risk engine)
#
# PFE = Notional x Add-on factor x Maturity factor
#
# Add-on factors are ILLUSTRATIVE and only loosely of the same order of
# magnitude as regulatory supervisory factors. THIS IS NOT SA-CCR.
# --------------------------------------------------------------------------- #
PRODUCT_ADDON_FACTORS: dict[str, float] = {
    "Interest Rate": 0.010,
    "FX": 0.040,
    "Equity": 0.120,
    "Commodity": 0.150,
    "Securities Financing": 0.030,
}

# Maturity factor = sqrt(clip(maturity, MIN, MAX)); square-root-of-time scaling
# of a diffusive risk factor. The floor (~10 business days) and cap (25y)
# prevent extreme values. Unlike SA-CCR, exposure keeps growing with maturity.
MIN_MATURITY_YEARS: float = 0.04
MAX_MATURITY_YEARS: float = 25.0

# Collateral: value is reduced by a haircut depending on collateral type.
COLLATERAL_TYPES: list[str] = ["Cash", "Government Bonds", "Corporate Bonds", "Equities"]
COLLATERAL_TYPE_MIX: list[float] = [0.50, 0.30, 0.12, 0.08]
COLLATERAL_HAIRCUTS: dict[str, float] = {
    "Cash": 0.00,
    "Government Bonds": 0.04,
    "Corporate Bonds": 0.10,
    "Equities": 0.20,
    "None": 0.00,
}

# --------------------------------------------------------------------------- #
# Credit health score specification (see preprocessing.py)
# metric -> (breakpoints, scores at breakpoints, weight, higher_is_better)
# Scores are linearly interpolated between breakpoints and clamped at the ends.
# Breakpoints are ILLUSTRATIVE rule-of-thumb bands, not calibrated to defaults.
# --------------------------------------------------------------------------- #
CREDIT_SCORE_SPEC: dict[str, dict[str, object]] = {
    "debt_to_ebitda": {"x": [0, 1, 2, 3, 4, 6, 8], "y": [100, 95, 80, 65, 50, 25, 0],
                       "weight": 0.25},
    "interest_coverage": {"x": [0, 1, 2, 3, 5, 8, 12], "y": [0, 10, 30, 50, 70, 90, 100],
                          "weight": 0.25},
    "cash_to_debt": {"x": [0, 0.10, 0.25, 0.50, 1.00], "y": [0, 20, 50, 80, 100],
                     "weight": 0.10},
    "current_ratio": {"x": [0.5, 0.8, 1.0, 1.25, 1.5, 2.0], "y": [0, 15, 40, 65, 85, 100],
                      "weight": 0.10},
    "debt_to_assets": {"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.7], "y": [100, 90, 75, 55, 35, 0],
                       "weight": 0.10},
    "ocf_to_debt": {"x": [0, 0.05, 0.10, 0.20, 0.40], "y": [0, 25, 50, 80, 100],
                    "weight": 0.10},
    "ebitda_margin": {"x": [0.0, 0.05, 0.10, 0.20, 0.30], "y": [0, 20, 45, 80, 100],
                      "weight": 0.10},
}
CREDIT_SCORE_BANDS: list[tuple[float, str]] = [
    (30.0, "Very Weak"), (50.0, "Weak"), (70.0, "Adequate"), (100.01, "Strong"),
]

# Caps used when a ratio's denominator is zero/negative.
LEVERAGE_CAP: float = 20.0
COVERAGE_CAP: float = 100.0
CASH_TO_DEBT_CAP: float = 10.0
