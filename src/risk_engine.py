"""Deterministic counterparty exposure and expected-loss engine.

*** EDUCATIONAL METHODOLOGY - NOT A REGULATORY IMPLEMENTATION ***
This module is NOT SA-CCR, NOT IMM and NOT any bank's internal methodology.
It deliberately uses transparent, simplified formulas so each number can be
reproduced by hand.

Formulas (all amounts in EUR mn)
--------------------------------
Current Exposure (CE)
    ``max(MtM, 0)`` of one netting set per counterparty (supplied as input).

Maturity factor
    ``MF = sqrt(clip(maturity_years, MIN, MAX))`` - square-root-of-time
    scaling of a diffusive risk factor. Longer maturity -> larger PFE.

Simplified Potential Future Exposure
    ``PFE = Notional x AddOn(product) x MF``
    The add-on is a product-level percentage (``config.PRODUCT_ADDON_FACTORS``).
    It is a stand-in for a high-quantile increase in exposure; it is NOT
    calibrated to a specific confidence level or margin period of risk.

Eligible collateral
    ``Collateral x (1 - haircut(collateral_type))``

Exposure at Default
    ``EAD = max(CE + PFE - Eligible Collateral, 0)``

Expected Loss
    ``EL = PD x LGD x EAD``   (one-year horizon, PD and LGD are inputs)

What a production / regulatory implementation would additionally need
----------------------------------------------------------------------
Trade-level data and legally enforceable netting sets; SA-CCR replacement cost
and PFE add-on (asset-class hedging sets, supervisory delta/duration,
multiplier for over-collateralisation, alpha); or IMM with Monte Carlo
exposure simulation, margin-period-of-risk, effective EPE, stressed
calibration and backtesting; CVA and wrong-way-risk treatment; CSA terms
(thresholds, MTA, independent amounts, dispute/cure periods); and model
validation and governance. None of this is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Union

import numpy as np
import pandas as pd

from src import config as cfg

ArrayLike = Union[pd.Series, np.ndarray, float]

REQUIRED_ENGINE_COLUMNS: list[str] = [
    "counterparty_id", "industry", "country", "rating", "product_type", "notional",
    "maturity_years", "current_exposure", "collateral", "collateral_type", "pd", "lgd",
]


@dataclass(frozen=True)
class ExposureParams:
    """Configurable parameters of the exposure methodology."""

    addon_factors: Mapping[str, float] = field(default_factory=lambda: dict(cfg.PRODUCT_ADDON_FACTORS))
    collateral_haircuts: Mapping[str, float] = field(default_factory=lambda: dict(cfg.COLLATERAL_HAIRCUTS))
    min_maturity_years: float = cfg.MIN_MATURITY_YEARS
    max_maturity_years: float = cfg.MAX_MATURITY_YEARS

    def __post_init__(self) -> None:
        if self.min_maturity_years <= 0 or self.max_maturity_years < self.min_maturity_years:
            raise ValueError("Require 0 < min_maturity_years <= max_maturity_years")
        if any(v < 0 for v in self.addon_factors.values()):
            raise ValueError("Add-on factors must be non-negative")
        if any(not 0.0 <= v <= 1.0 for v in self.collateral_haircuts.values()):
            raise ValueError("Collateral haircuts must lie in [0, 1]")


# --------------------------------------------------------------------------- #
# Core formulas (pure functions - easy to unit test)
# --------------------------------------------------------------------------- #
def maturity_factor(
    maturity_years: ArrayLike,
    min_maturity: float = cfg.MIN_MATURITY_YEARS,
    max_maturity: float = cfg.MAX_MATURITY_YEARS,
) -> ArrayLike:
    """Square-root-of-time maturity factor with a floor and a cap on maturity."""
    clipped = np.clip(maturity_years, min_maturity, max_maturity)
    return np.sqrt(clipped)


def calculate_pfe(notional: ArrayLike, addon_factor: ArrayLike, mat_factor: ArrayLike) -> ArrayLike:
    """Simplified PFE = Notional x Add-on x Maturity factor (NOT SA-CCR)."""
    return notional * addon_factor * mat_factor


def calculate_eligible_collateral(collateral: ArrayLike, haircut: ArrayLike) -> ArrayLike:
    """Collateral value after haircut: ``collateral x (1 - haircut)``."""
    return collateral * (1.0 - haircut)


def calculate_ead(current_exposure: ArrayLike, pfe: ArrayLike, eligible_collateral: ArrayLike) -> ArrayLike:
    """EAD = max(CE + PFE - eligible collateral, 0)."""
    return np.maximum(current_exposure + pfe - eligible_collateral, 0.0)


def calculate_expected_loss(pd_: ArrayLike, lgd: ArrayLike, ead: ArrayLike) -> ArrayLike:
    """EL = PD x LGD x EAD (one-year horizon)."""
    return pd_ * lgd * ead


# --------------------------------------------------------------------------- #
# DataFrame-level engine
# --------------------------------------------------------------------------- #
def _check_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")


def _map_or_raise(series: pd.Series, mapping: Mapping[str, float], label: str) -> pd.Series:
    mapped = series.map(mapping)
    if mapped.isna().any():
        unknown = sorted(series[mapped.isna()].unique().tolist())
        raise ValueError(f"Unknown {label}: {unknown}. Known: {sorted(mapping)}")
    return mapped.astype(float)


def run_risk_engine(df: pd.DataFrame, params: ExposureParams | None = None) -> pd.DataFrame:
    """Compute PFE, eligible collateral, EAD and Expected Loss per counterparty.

    Args:
        df: Counterparty table with at least ``REQUIRED_ENGINE_COLUMNS``.
        params: Methodology parameters (defaults from ``config``).

    Returns:
        Copy of ``df`` with added columns: ``addon_factor``, ``maturity_factor``,
        ``pfe``, ``collateral_haircut``, ``eligible_collateral``,
        ``unsecured_exposure`` (CE net of eligible collateral, floored at 0),
        ``collateral_coverage`` (eligible collateral / CE, NaN if CE = 0),
        ``ead`` and ``expected_loss``.
    """
    _check_columns(df, REQUIRED_ENGINE_COLUMNS)
    params = params or ExposureParams()
    out = df.copy()

    out["addon_factor"] = _map_or_raise(out["product_type"], params.addon_factors, "product_type")
    out["maturity_factor"] = maturity_factor(
        out["maturity_years"], params.min_maturity_years, params.max_maturity_years
    )
    out["pfe"] = calculate_pfe(out["notional"], out["addon_factor"], out["maturity_factor"])

    out["collateral_haircut"] = _map_or_raise(out["collateral_type"], params.collateral_haircuts, "collateral_type")
    out["eligible_collateral"] = calculate_eligible_collateral(out["collateral"], out["collateral_haircut"])
    out["unsecured_exposure"] = np.maximum(out["current_exposure"] - out["eligible_collateral"], 0.0)
    out["collateral_coverage"] = (
        out["eligible_collateral"] / out["current_exposure"].where(out["current_exposure"] > 0)
    )

    out["ead"] = calculate_ead(out["current_exposure"], out["pfe"], out["eligible_collateral"])
    out["expected_loss"] = calculate_expected_loss(out["pd"], out["lgd"], out["ead"])
    return out


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate_risk(df: pd.DataFrame, by: str | list[str], sort_by: str = "expected_loss") -> pd.DataFrame:
    """Aggregate exposure and expected loss by one or more dimensions.

    ``ead_weighted_pd`` and ``ead_weighted_lgd`` are EAD-weighted averages.
    ``ead_share`` and ``el_share`` are shares of the portfolio total.
    When grouping by ``rating`` the result follows the rating scale (AAA..CCC).
    """
    _check_columns(df, ["ead", "expected_loss", "pfe", "eligible_collateral", "current_exposure", "notional"])
    work = df.assign(_pd_ead=df["pd"] * df["ead"], _lgd_ead=df["lgd"] * df["ead"])
    grouped = work.groupby(by, observed=True).agg(
        n_counterparties=("counterparty_id", "count"),
        notional=("notional", "sum"),
        current_exposure=("current_exposure", "sum"),
        pfe=("pfe", "sum"),
        eligible_collateral=("eligible_collateral", "sum"),
        ead=("ead", "sum"),
        expected_loss=("expected_loss", "sum"),
        _pd_ead=("_pd_ead", "sum"),
        _lgd_ead=("_lgd_ead", "sum"),
    )
    ead = grouped["ead"].where(grouped["ead"] > 0)
    grouped["ead_weighted_pd"] = grouped.pop("_pd_ead") / ead
    grouped["ead_weighted_lgd"] = grouped.pop("_lgd_ead") / ead
    total_ead, total_el = df["ead"].sum(), df["expected_loss"].sum()
    grouped["ead_share"] = grouped["ead"] / total_ead if total_ead > 0 else np.nan
    grouped["el_share"] = grouped["expected_loss"] / total_el if total_el > 0 else np.nan
    grouped["el_to_ead_bps"] = 1e4 * grouped["expected_loss"] / ead

    if by == "rating":
        grouped = grouped.reindex([r for r in cfg.RATINGS if r in grouped.index])
    else:
        grouped = grouped.sort_values(sort_by, ascending=False)
    return grouped.reset_index()


def expected_loss_by_dimension(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Expected loss at industry, country, rating and product level."""
    return {dim: aggregate_risk(df, dim) for dim in ("industry", "country", "rating", "product_type")}


def portfolio_summary(df: pd.DataFrame) -> dict[str, float]:
    """Headline portfolio metrics (EUR mn unless stated)."""
    _check_columns(df, ["ead", "expected_loss", "pfe", "current_exposure", "eligible_collateral"])
    total_ead = float(df["ead"].sum())
    return {
        "n_counterparties": int(len(df)),
        "total_notional": float(df["notional"].sum()),
        "total_current_exposure": float(df["current_exposure"].sum()),
        "total_pfe": float(df["pfe"].sum()),
        "total_eligible_collateral": float(df["eligible_collateral"].sum()),
        "total_ead": total_ead,
        "total_expected_loss": float(df["expected_loss"].sum()),
        "average_pd": float(df["pd"].mean()),
        "ead_weighted_pd": float((df["pd"] * df["ead"]).sum() / total_ead) if total_ead else float("nan"),
        "ead_weighted_lgd": float((df["lgd"] * df["ead"]).sum() / total_ead) if total_ead else float("nan"),
        "el_to_ead_bps": float(1e4 * df["expected_loss"].sum() / total_ead) if total_ead else float("nan"),
    }
