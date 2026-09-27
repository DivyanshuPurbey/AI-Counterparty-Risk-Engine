"""Scenario and thematic stress testing on top of the deterministic risk engine.

*** EDUCATIONAL, ILLUSTRATIVE SCENARIOS - NOT CCAR / EBA / ICAAP ***
The scenarios below are simple multiplicative shocks to risk drivers. They are
not derived from a macro-economic model, are not supervisory scenarios and are
not calibrated to historical stress episodes. In a production stress-testing
framework, scenario narratives are translated into risk-factor paths
(GDP, unemployment, rates, FX, spreads, commodity prices) which then drive
re-pricing of trades, re-simulation of exposure, rating migration and PD/LGD
models, with governance, challenge and reverse stress testing.

Mechanics (per counterparty)
----------------------------
* Stressed PD  = min(PD  x pd_mult  x industry_pd_mult x country_pd_mult, 1)
* Stressed LGD = min(LGD x lgd_mult x industry_lgd_mult, 1)
* General exposure multiplier
      g = exposure_mult x industry_exposure_mult x product_exposure_mult
* Stressed CE  = g x (other_CE + FX_CE x fx_exposure_mult + IR_CE x ir_exposure_mult)
  where other_CE = CE - FX_CE - IR_CE.
* Stressed PFE = PFE x g x (fx_exposure_mult for FX products,
                            ir_exposure_mult for Interest Rate products, else 1).
  Rationale: a larger market shock raises volatility, hence the add-on.
* Stressed eligible collateral = collateral x (1 - (haircut + extra haircut)).
  Collateral quantities are held constant (no additional margin calls), which
  is a conservative simplification.
* Stressed EAD = max(stressed CE + stressed PFE - stressed collateral, 0)
* Stressed EL  = stressed PD x stressed LGD x stressed EAD

The change in EL is decomposed sequentially into PD, LGD and EAD effects
(order: PD, then LGD, then EAD). The three effects sum exactly to the total
incremental EL; the split between them depends on the ordering.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, replace
from typing import Mapping

import numpy as np
import pandas as pd

from src import config as cfg
from src.risk_engine import calculate_ead, calculate_expected_loss

logger = logging.getLogger(__name__)

REQUIRED_STRESS_COLUMNS: list[str] = [
    "counterparty_id", "industry", "country", "rating", "product_type", "pd", "lgd",
    "current_exposure", "fx_exposure", "interest_rate_exposure", "collateral",
    "pfe", "collateral_haircut", "eligible_collateral", "ead", "expected_loss",
]

_MULTIPLIER_FIELDS = (
    "pd_mult", "lgd_mult", "exposure_mult", "fx_exposure_mult", "ir_exposure_mult",
)
_MAPPING_FIELDS = {
    "industry_pd_mult": cfg.INDUSTRIES,
    "industry_lgd_mult": cfg.INDUSTRIES,
    "industry_exposure_mult": cfg.INDUSTRIES,
    "country_pd_mult": cfg.COUNTRIES,
    "product_exposure_mult": cfg.PRODUCT_TYPES,
}


# --------------------------------------------------------------------------- #
# Scenario definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scenario:
    """A stress scenario expressed as multipliers on risk drivers.

    All parameters can be overridden dynamically with :meth:`with_overrides`.
    """

    name: str
    description: str = ""
    pd_mult: float = 1.0
    lgd_mult: float = 1.0
    exposure_mult: float = 1.0
    fx_exposure_mult: float = 1.0
    ir_exposure_mult: float = 1.0
    collateral_haircut_add: float = 0.0
    industry_pd_mult: Mapping[str, float] = field(default_factory=dict)
    industry_lgd_mult: Mapping[str, float] = field(default_factory=dict)
    industry_exposure_mult: Mapping[str, float] = field(default_factory=dict)
    country_pd_mult: Mapping[str, float] = field(default_factory=dict)
    product_exposure_mult: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in _MULTIPLIER_FIELDS:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative (got {getattr(self, name)})")
        if not 0.0 <= self.collateral_haircut_add <= 1.0:
            raise ValueError("collateral_haircut_add must lie in [0, 1]")
        for name, allowed in _MAPPING_FIELDS.items():
            mapping = getattr(self, name)
            unknown = set(mapping) - set(allowed)
            if unknown:
                raise ValueError(f"{name}: unknown keys {sorted(unknown)}")
            if any(v < 0 for v in mapping.values()):
                raise ValueError(f"{name}: multipliers must be non-negative")

    def with_overrides(self, **changes: object) -> "Scenario":
        """Return a copy with selected parameters changed."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, object]:
        """Plain-dict representation (used for the scenario_definitions table)."""
        return asdict(self)


# Macro scenarios - the multipliers follow the project brief; the extra
# collateral haircut is an ADDITION to the brief (set to 0.0 to switch off).
DEFAULT_SCENARIOS: dict[str, Scenario] = {
    "BASE": Scenario("BASE", "No shock; reproduces the base risk engine."),
    "RECESSION": Scenario(
        "RECESSION", "Moderate downturn: higher PD/LGD/exposure, mild collateral value decline.",
        pd_mult=1.5, lgd_mult=1.1, exposure_mult=1.10, collateral_haircut_add=0.05,
    ),
    "SEVERE_RECESSION": Scenario(
        "SEVERE_RECESSION", "Deep downturn: PD doubles, LGD +20%, exposure +25%, larger collateral haircuts.",
        pd_mult=2.0, lgd_mult=1.2, exposure_mult=1.25, collateral_haircut_add=0.10,
    ),
    "MARKET_SHOCK": Scenario(
        "MARKET_SHOCK", "Market dislocation: FX exposure +30%, rate exposure +20%, PD +30%, LGD +10%.",
        pd_mult=1.3, lgd_mult=1.1, fx_exposure_mult=1.3, ir_exposure_mult=1.2,
        collateral_haircut_add=0.05,
    ),
}

# Thematic scenarios (illustrative sector / risk-factor stories).
THEMES: dict[str, Scenario] = {
    "ENERGY_SHOCK": Scenario(
        "ENERGY_SHOCK", "Sharp commodity/energy price move hitting energy producers and energy-intensive sectors.",
        pd_mult=1.10, lgd_mult=1.05,
        industry_pd_mult={"Energy": 2.5, "Transportation": 1.6, "Industrials": 1.3,
                          "Manufacturing": 1.3, "Consumer Goods": 1.15},
        industry_lgd_mult={"Energy": 1.15},
        product_exposure_mult={"Commodity": 1.5},
    ),
    "FX_SHOCK": Scenario(
        "FX_SHOCK", "Large EMEA FX move: FX exposures rise; exporters/importers under pressure.",
        pd_mult=1.10, fx_exposure_mult=1.5,
        industry_pd_mult={"Manufacturing": 1.2, "Consumer Goods": 1.2, "Technology": 1.1},
    ),
    "INTEREST_RATE_SHOCK": Scenario(
        "INTEREST_RATE_SHOCK", "Sharp rate rise: IR exposures rise; leveraged, rate-sensitive sectors under pressure.",
        pd_mult=1.15, ir_exposure_mult=1.4,
        industry_pd_mult={"Real Estate": 1.8, "Financial Services": 1.25, "Telecommunications": 1.2},
    ),
    "REAL_ESTATE_DOWNTURN": Scenario(
        "REAL_ESTATE_DOWNTURN", "Property price fall: real-estate defaults, lower recoveries, spillover to financials.",
        pd_mult=1.05, collateral_haircut_add=0.10,
        industry_pd_mult={"Real Estate": 3.0, "Financial Services": 1.4, "Industrials": 1.2},
        industry_lgd_mult={"Real Estate": 1.3, "Financial Services": 1.1},
        industry_exposure_mult={"Real Estate": 1.1},
    ),
    "EMEA_RECESSION": Scenario(
        "EMEA_RECESSION", "Region-wide recession with heavier impact on southern Europe and the UK.",
        pd_mult=1.6, lgd_mult=1.1, exposure_mult=1.10, collateral_haircut_add=0.05,
        country_pd_mult={"Italy": 1.25, "Spain": 1.20, "UK": 1.10, "Germany": 1.10,
                         "France": 1.05, "Ireland": 1.05},
    ),
    "FINANCIAL_SECTOR_STRESS": Scenario(
        "FINANCIAL_SECTOR_STRESS", "Funding/liquidity stress in financial institutions; collateral values fall.",
        pd_mult=1.10, collateral_haircut_add=0.05,
        industry_pd_mult={"Financial Services": 3.0, "Real Estate": 1.5},
        industry_lgd_mult={"Financial Services": 1.15},
        industry_exposure_mult={"Financial Services": 1.25},
    ),
}

ALL_SCENARIOS: dict[str, Scenario] = {**DEFAULT_SCENARIOS, **THEMES}


def pd_shock_scenario(pct_increase: float, name: str | None = None) -> Scenario:
    """Uniform PD shock, e.g. ``pd_shock_scenario(50)`` for 'PD increases by 50%'."""
    if pct_increase <= -100:
        raise ValueError("pct_increase must be > -100")
    label = name or f"PD_{pct_increase:+g}PCT"
    return Scenario(label, f"All PDs multiplied by {1 + pct_increase / 100:.2f}", pd_mult=1 + pct_increase / 100)


def scenario_label(key: str) -> str:
    """Human-readable label, e.g. ``SEVERE_RECESSION`` -> ``Severe Recession``."""
    return key.replace("_", " ").title()


# --------------------------------------------------------------------------- #
# Applying a scenario
# --------------------------------------------------------------------------- #
def _lookup_mult(values: pd.Series, mapping: Mapping[str, float]) -> pd.Series:
    """Per-row multiplier from ``mapping`` (default 1.0)."""
    if not mapping:
        return pd.Series(1.0, index=values.index)
    return values.map(dict(mapping)).fillna(1.0).astype(float)


def apply_scenario(df: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    """Apply one scenario to the output of :func:`risk_engine.run_risk_engine`.

    Returns one row per counterparty with base and stressed metrics, the
    incremental EAD/EL, the % change in EL and the PD/LGD/EAD attribution.
    """
    missing = [c for c in REQUIRED_STRESS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Run risk_engine.run_risk_engine first; missing columns: {missing}")

    pd_mult = scenario.pd_mult * _lookup_mult(df["industry"], scenario.industry_pd_mult) \
        * _lookup_mult(df["country"], scenario.country_pd_mult)
    lgd_mult = scenario.lgd_mult * _lookup_mult(df["industry"], scenario.industry_lgd_mult)
    general = scenario.exposure_mult * _lookup_mult(df["industry"], scenario.industry_exposure_mult) \
        * _lookup_mult(df["product_type"], scenario.product_exposure_mult)

    stressed_pd = np.minimum(df["pd"] * pd_mult, 1.0)
    stressed_lgd = np.minimum(df["lgd"] * lgd_mult, 1.0)

    other_ce = np.maximum(df["current_exposure"] - df["fx_exposure"] - df["interest_rate_exposure"], 0.0)
    stressed_ce = general * (
        other_ce + df["fx_exposure"] * scenario.fx_exposure_mult
        + df["interest_rate_exposure"] * scenario.ir_exposure_mult
    )
    product_vol = df["product_type"].map(
        {"FX": scenario.fx_exposure_mult, "Interest Rate": scenario.ir_exposure_mult}
    ).fillna(1.0).astype(float)
    stressed_pfe = df["pfe"] * general * product_vol

    stressed_haircut = np.clip(df["collateral_haircut"] + scenario.collateral_haircut_add, 0.0, 1.0)
    stressed_collateral = df["collateral"] * (1.0 - stressed_haircut)

    stressed_ead = calculate_ead(stressed_ce, stressed_pfe, stressed_collateral)
    stressed_el = calculate_expected_loss(stressed_pd, stressed_lgd, stressed_ead)

    base_el = df["expected_loss"]
    incremental_el = stressed_el - base_el
    result = pd.DataFrame(
        {
            "counterparty_id": df["counterparty_id"],
            "counterparty_name": df["counterparty_name"] if "counterparty_name" in df else df["counterparty_id"],
            "industry": df["industry"],
            "country": df["country"],
            "rating": df["rating"],
            "product_type": df["product_type"],
            "scenario": scenario.name,
            "base_pd": df["pd"],
            "stressed_pd": stressed_pd,
            "base_lgd": df["lgd"],
            "stressed_lgd": stressed_lgd,
            "base_current_exposure": df["current_exposure"],
            "stressed_current_exposure": stressed_ce,
            "base_pfe": df["pfe"],
            "stressed_pfe": stressed_pfe,
            "stressed_eligible_collateral": stressed_collateral,
            "base_ead": df["ead"],
            "stressed_ead": stressed_ead,
            "incremental_ead": stressed_ead - df["ead"],
            "base_expected_loss": base_el,
            "stressed_expected_loss": stressed_el,
            "incremental_expected_loss": incremental_el,
            "expected_loss_pct_change": np.where(base_el > 0, incremental_el / base_el.where(base_el > 0), np.nan),
            # Sequential attribution (PD -> LGD -> EAD); sums to incremental_el.
            "el_effect_pd": (stressed_pd - df["pd"]) * df["lgd"] * df["ead"],
            "el_effect_lgd": stressed_pd * (stressed_lgd - df["lgd"]) * df["ead"],
            "el_effect_ead": stressed_pd * stressed_lgd * (stressed_ead - df["ead"]),
        },
        index=df.index,
    )
    return result


def summarise_scenario(stress: pd.DataFrame) -> dict[str, float]:
    """Portfolio-level summary of one scenario's per-counterparty results."""
    base_el = float(stress["base_expected_loss"].sum())
    stress_el = float(stress["stressed_expected_loss"].sum())
    base_ead = float(stress["base_ead"].sum())
    stress_ead = float(stress["stressed_ead"].sum())
    return {
        "scenario": str(stress["scenario"].iloc[0]),
        "base_ead": base_ead,
        "stressed_ead": stress_ead,
        "incremental_ead": stress_ead - base_ead,
        "ead_pct_change": (stress_ead / base_ead - 1.0) if base_ead > 0 else float("nan"),
        "base_expected_loss": base_el,
        "stressed_expected_loss": stress_el,
        "incremental_expected_loss": stress_el - base_el,
        "expected_loss_pct_change": (stress_el / base_el - 1.0) if base_el > 0 else float("nan"),
        "el_effect_pd": float(stress["el_effect_pd"].sum()),
        "el_effect_lgd": float(stress["el_effect_lgd"].sum()),
        "el_effect_ead": float(stress["el_effect_ead"].sum()),
        "ead_weighted_stressed_pd": float(
            (stress["stressed_pd"] * stress["stressed_ead"]).sum() / stress_ead
        ) if stress_ead > 0 else float("nan"),
    }


def run_scenarios(
    df: pd.DataFrame, scenarios: Mapping[str, Scenario] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run several scenarios.

    Args:
        df: Output of the risk engine.
        scenarios: Mapping of key -> Scenario (default: the four macro scenarios).

    Returns:
        ``(results, summary)``: per-counterparty results for every scenario
        (long format, keyed by ``scenario``) and a one-row-per-scenario summary.
    """
    scenarios = scenarios or DEFAULT_SCENARIOS
    frames, rows = [], []
    for key, scenario in scenarios.items():
        renamed = scenario if scenario.name == key else scenario.with_overrides(name=key)
        stress = apply_scenario(df, renamed)
        frames.append(stress)
        rows.append(summarise_scenario(stress))
        logger.info("Scenario %s: EL %+.1f%%", key, 100 * rows[-1]["expected_loss_pct_change"])
    return pd.concat(frames, ignore_index=True), pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Aggregation and thematic analysis
# --------------------------------------------------------------------------- #
def aggregate_stress(stress: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Aggregate one scenario's results by industry/country/rating/product.

    Sorted by incremental expected loss (largest first), except by rating,
    which follows the rating scale.
    """
    grouped = stress.groupby(by, observed=True).agg(
        n_counterparties=("counterparty_id", "count"),
        base_ead=("base_ead", "sum"),
        stressed_ead=("stressed_ead", "sum"),
        incremental_ead=("incremental_ead", "sum"),
        base_expected_loss=("base_expected_loss", "sum"),
        stressed_expected_loss=("stressed_expected_loss", "sum"),
        incremental_expected_loss=("incremental_expected_loss", "sum"),
    )
    base = grouped["base_expected_loss"].where(grouped["base_expected_loss"] > 0)
    grouped["expected_loss_pct_change"] = grouped["incremental_expected_loss"] / base
    total_incr = grouped["incremental_expected_loss"].sum()
    grouped["share_of_incremental_el"] = grouped["incremental_expected_loss"] / total_incr if total_incr else np.nan
    if by == "rating":
        grouped = grouped.reindex([r for r in cfg.RATINGS if r in grouped.index])
    else:
        grouped = grouped.sort_values("incremental_expected_loss", ascending=False)
    return grouped.reset_index()


def analyze_theme(df: pd.DataFrame, scenario: Scenario, top_n: int = 10) -> dict[str, object]:
    """Thematic vulnerability analysis for one scenario.

    Returns a dict with ``summary`` (portfolio totals), ``top_industries``,
    ``top_countries`` and ``top_counterparties`` (ranked by incremental EL),
    plus the full per-counterparty ``results``.
    """
    stress = apply_scenario(df, scenario)
    cols = ["counterparty_id", "counterparty_name", "industry", "country", "rating", "product_type",
            "base_ead", "stressed_ead", "incremental_ead", "base_expected_loss",
            "stressed_expected_loss", "incremental_expected_loss", "expected_loss_pct_change"]
    return {
        "scenario": scenario,
        "summary": summarise_scenario(stress),
        "top_industries": aggregate_stress(stress, "industry").head(top_n),
        "top_countries": aggregate_stress(stress, "country").head(top_n),
        "top_counterparties": stress.nlargest(top_n, "incremental_expected_loss")[cols].reset_index(drop=True),
        "results": stress,
    }


def industry_sensitivity_matrix(
    df: pd.DataFrame,
    scenarios: Mapping[str, Scenario] | None = None,
    metric: str = "expected_loss_pct_change",
) -> pd.DataFrame:
    """Industry x scenario matrix of a stress metric (default: % change in EL).

    Useful for questions like "which industries are most sensitive to an FX
    shock?". Rows are ordered by the maximum sensitivity across scenarios.
    """
    scenarios = scenarios or ALL_SCENARIOS
    columns = {}
    for key, scenario in scenarios.items():
        renamed = scenario if scenario.name == key else scenario.with_overrides(name=key)
        agg = aggregate_stress(apply_scenario(df, renamed), "industry").set_index("industry")
        columns[key] = agg[metric]
    matrix = pd.DataFrame(columns)
    return matrix.loc[matrix.max(axis=1).sort_values(ascending=False).index]
