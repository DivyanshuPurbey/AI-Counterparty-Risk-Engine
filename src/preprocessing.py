"""Data validation and credit-analytics feature engineering.

Builds the financial-risk ratios listed in the project brief and a simple,
transparent *credit health score*.

Credit health score - methodology
---------------------------------
* Each ratio is mapped to a 0-100 sub-score using piecewise-linear bands
  (``config.CREDIT_SCORE_SPEC``). 100 = strongest, 0 = weakest.
* The final score is a fixed weighted average of the sub-scores
  (Debt/EBITDA 25%, Interest Coverage 25%, other five ratios 10% each).
* Because it is a plain weighted average with published bands, every score can
  be decomposed by the reader (``score_<metric>`` columns).

What it is NOT: it is not calibrated to defaults, is not a rating model, and
does not resemble any proprietary bank scorecard. Bands ignore sector
specifics (e.g. leverage norms for real estate or financials differ). In a
production setting scorecards are statistically calibrated, sector-specific,
and independently validated.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from src import config as cfg

logger = logging.getLogger(__name__)

REQUIRED_RAW_COLUMNS: list[str] = [
    "counterparty_id", "counterparty_name", "industry", "country", "region",
    "rating", "rating_prior_year", "annual_revenue", "revenue_prior_year",
    "total_assets", "total_debt", "cash", "current_assets", "current_liabilities",
    "ebitda", "interest_expense", "operating_cash_flow", "operating_cash_flow_prior_year",
    "product_type", "notional", "maturity_years", "current_exposure",
    "current_exposure_prior_quarter", "fx_exposure", "interest_rate_exposure",
    "collateral", "collateral_type", "pd", "lgd", "default_flag",
]

_NON_NEGATIVE = [
    "annual_revenue", "revenue_prior_year", "total_assets", "total_debt", "cash",
    "current_assets", "current_liabilities", "interest_expense", "notional",
    "maturity_years", "current_exposure", "current_exposure_prior_quarter",
    "fx_exposure", "interest_rate_exposure", "collateral",
]

RATIO_COLUMNS: list[str] = [
    "debt_to_ebitda", "interest_coverage", "current_ratio", "debt_to_assets",
    "cash_to_debt", "ocf_to_debt", "ebitda_margin",
]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate schema and basic integrity of the raw counterparty table.

    Raises:
        ValueError: on missing columns, duplicate IDs, unknown categories,
            negative amounts, PD/LGD outside [0, 1], or inconsistent exposure
            splits.
    """
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df["counterparty_id"].duplicated().any():
        raise ValueError("Duplicate counterparty_id values found")
    if df[REQUIRED_RAW_COLUMNS].isna().any().any():
        bad = df[REQUIRED_RAW_COLUMNS].columns[df[REQUIRED_RAW_COLUMNS].isna().any()].tolist()
        raise ValueError(f"Null values found in columns: {bad}")
    for col, allowed in (
        ("rating", cfg.RATINGS), ("rating_prior_year", cfg.RATINGS),
        ("product_type", cfg.PRODUCT_TYPES), ("collateral_type", list(cfg.COLLATERAL_HAIRCUTS)),
    ):
        unknown = set(df[col].unique()) - set(allowed)
        if unknown:
            raise ValueError(f"Unknown values in '{col}': {sorted(unknown)}")
    for col in _NON_NEGATIVE:
        if (df[col] < 0).any():
            raise ValueError(f"Negative values found in '{col}'")
    for col in ("pd", "lgd"):
        if ((df[col] < 0) | (df[col] > 1)).any():
            raise ValueError(f"'{col}' must lie in [0, 1]")
    if (df["fx_exposure"] + df["interest_rate_exposure"] > df["current_exposure"] + 1e-6).any():
        raise ValueError("fx_exposure + interest_rate_exposure exceeds current_exposure")


# --------------------------------------------------------------------------- #
# Ratios
# --------------------------------------------------------------------------- #
def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division returning NaN where the denominator is <= 0."""
    den = denominator.where(denominator > 0)
    return numerator / den


def compute_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add the seven credit ratios (returns a copy).

    Zero/negative denominators map to conservative caps from ``config``:
    EBITDA <= 0 -> leverage cap; zero interest -> coverage cap; zero debt ->
    cash/debt cap.
    """
    out = df.copy()
    out["debt_to_ebitda"] = safe_divide(out["total_debt"], out["ebitda"]).fillna(cfg.LEVERAGE_CAP).clip(upper=cfg.LEVERAGE_CAP)
    coverage = safe_divide(out["ebitda"], out["interest_expense"])
    out["interest_coverage"] = coverage.fillna(cfg.COVERAGE_CAP).clip(upper=cfg.COVERAGE_CAP)
    out["current_ratio"] = safe_divide(out["current_assets"], out["current_liabilities"])
    out["debt_to_assets"] = safe_divide(out["total_debt"], out["total_assets"])
    out["cash_to_debt"] = safe_divide(out["cash"], out["total_debt"]).fillna(cfg.CASH_TO_DEBT_CAP).clip(upper=cfg.CASH_TO_DEBT_CAP)
    out["ocf_to_debt"] = safe_divide(out["operating_cash_flow"], out["total_debt"]).fillna(0.0)
    out["ebitda_margin"] = safe_divide(out["ebitda"], out["annual_revenue"])
    return out


def compute_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add period-on-period change features used for early-warning analysis.

    * revenue_growth, ocf_growth  : (current - prior) / |prior|
    * exposure_growth             : (CE - prior CE) / max(prior CE, 1.0 EUR mn)
    * notches_downgraded          : positive = downgraded vs prior year
    """
    out = df.copy()
    out["revenue_growth"] = (out["annual_revenue"] - out["revenue_prior_year"]) / out["revenue_prior_year"].abs().clip(lower=1e-9)
    out["ocf_growth"] = (out["operating_cash_flow"] - out["operating_cash_flow_prior_year"]) / out["operating_cash_flow_prior_year"].abs().clip(lower=1e-9)
    out["exposure_growth"] = (out["current_exposure"] - out["current_exposure_prior_quarter"]) / out["current_exposure_prior_quarter"].clip(lower=1.0)
    out["rating_index"] = out["rating"].map(cfg.RATING_INDEX).astype(int)
    out["notches_downgraded"] = out["rating_index"] - out["rating_prior_year"].map(cfg.RATING_INDEX).astype(int)
    return out


# --------------------------------------------------------------------------- #
# Credit health score
# --------------------------------------------------------------------------- #
def compute_credit_health_score(
    df: pd.DataFrame, spec: dict[str, dict[str, object]] | None = None
) -> pd.DataFrame:
    """Add sub-scores, the weighted ``credit_health_score`` (0-100) and a band.

    Args:
        df: Frame that already contains the ratio columns.
        spec: Optional override of ``config.CREDIT_SCORE_SPEC``.
    """
    spec = spec or cfg.CREDIT_SCORE_SPEC
    weights = np.array([float(s["weight"]) for s in spec.values()])
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Score weights must sum to 1.0, got {weights.sum():.4f}")
    missing = [m for m in spec if m not in df.columns]
    if missing:
        raise ValueError(f"Ratio columns missing (run compute_financial_ratios first): {missing}")

    out = df.copy()
    total = np.zeros(len(out))
    for metric, s in spec.items():
        values = out[metric].fillna(out[metric].median()).to_numpy(dtype=float)
        sub = np.interp(values, np.asarray(s["x"], dtype=float), np.asarray(s["y"], dtype=float))
        out[f"score_{metric}"] = sub
        total += float(s["weight"]) * sub
    out["credit_health_score"] = total

    edges = [b[0] for b in cfg.CREDIT_SCORE_BANDS]
    labels = [b[1] for b in cfg.CREDIT_SCORE_BANDS]
    band_idx = np.searchsorted(np.asarray(edges), total, side="right")
    out["credit_score_band"] = np.asarray(labels, dtype=object)[np.clip(band_idx, 0, len(labels) - 1)]
    return out


def build_feature_table(df: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
    """Run validation, ratios, growth features and the credit health score."""
    if validate:
        validate_raw_data(df)
    out = compute_financial_ratios(df)
    out = compute_growth_features(out)
    out = compute_credit_health_score(out)
    logger.info("Built feature table: %d rows x %d columns", *out.shape)
    return out


def missing_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    """Return the names in ``required`` that are absent from ``df``."""
    return [c for c in required if c not in df.columns]
