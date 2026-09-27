"""Synthetic counterparty dataset generator.

Generates a fully SYNTHETIC EMEA corporate counterparty portfolio. No real
customer, bank or company data is used; all names are randomly assembled
fictional strings (any resemblance to a real entity is coincidental).

Generative story (every step is an assumption, documented in
``docs/synthetic_data_assumptions.md``)
------------------------------------------------------------------------
1. Industry, country and rating are drawn independently from the mixes in
   ``config.py``.
2. Financial statements are built from the rating: better ratings get lower
   leverage, cheaper debt, more cash and higher margins. Log-normal noise keeps
   names within a rating heterogeneous. Interest expense = Debt x cost of debt,
   so interest coverage is *mechanically* consistent with leverage.
3. The internal PD estimate (``pd``) starts from the rating anchor and is
   shifted (on the logit scale) by leverage, interest coverage, cash, liquidity,
   growth, recent rating migration and an industry effect.
4. The *realised* ``default_flag`` is a Bernoulli draw from that PD plus an
   UNOBSERVABLE idiosyncratic shock that no feature contains. This keeps any
   PD model honest: even a perfect model cannot reach unrealistic AUCs.
5. Exposure fields (product, notional, maturity, current exposure, collateral)
   are generated from product-specific assumptions. Current exposure is
   ``max(MtM, 0)`` of a single netting set per counterparty; ~25% of
   counterparties are out-of-the-money (zero current exposure).
6. Exposure is generated independently of PD (no wrong-way risk) - a
   documented limitation.

Monetary amounts are EUR millions.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr

from src import config as cfg

logger = logging.getLogger(__name__)

# Data-generating constants (documented in docs/synthetic_data_assumptions.md)
LEV_SIGMA = 0.35            # log-normal sigma of Debt/EBITDA around the rating norm
RATE_SIGMA = 0.15           # log-normal sigma of cost of debt
LATENT_SHOCK_SD = 1.00      # unobservable logit shock in realised defaults
P_DETERIORATING = 0.08      # share of names with a deteriorating trend
P_ZERO_EXPOSURE = 0.25      # share of names with zero current exposure (OTM)
NOTIONAL_SIGMA = 1.30       # heavy-ish tail in notional -> a few large names
PD_FLOOR, PD_CAP = 0.0002, 0.40

RAW_COLUMNS: list[str] = [
    "counterparty_id", "counterparty_name", "industry", "country", "region",
    "rating", "rating_prior_year",
    "annual_revenue", "revenue_prior_year", "total_assets", "total_debt", "cash",
    "current_assets", "current_liabilities", "ebitda", "interest_expense",
    "operating_cash_flow", "operating_cash_flow_prior_year",
    "product_type", "notional", "maturity_years", "current_exposure",
    "current_exposure_prior_quarter", "fx_exposure", "interest_rate_exposure",
    "collateral", "collateral_type",
    "pd", "lgd", "default_flag",
]

_NAME_PREFIXES = [
    "Alder", "Aurelia", "Baltic", "Belmont", "Calder", "Cardinal", "Cedar", "Corvus",
    "Crestline", "Delmar", "Eastgate", "Elmwood", "Falcon", "Fenwick", "Galena", "Garnet",
    "Halden", "Harrow", "Ironwood", "Jasper", "Kestrel", "Kingsford", "Larch", "Lyndon",
    "Marlow", "Meridian", "Northbridge", "Novara", "Oakhurst", "Orchard", "Pinnacle",
    "Quillon", "Ravenscroft", "Redwood", "Sable", "Silverline", "Stanmore", "Talbot",
    "Thornfield", "Umbra", "Valemont", "Vantor", "Westmere", "Wexford", "Yarrow", "Zephyr",
    "Ardent", "Bexley", "Caldwell", "Dunmore", "Everest", "Fairhaven", "Glenmoor",
    "Hartwell", "Islay", "Juniper", "Keswick", "Lockhart", "Montrose", "Norwood",
    "Overton", "Prestwick",
]

_INDUSTRY_NOUNS: dict[str, list[str]] = {
    "Financial Services": ["Capital", "Financial", "Holdings", "Asset Partners", "Trust",
                           "Securities", "Credit", "Insurance Group", "Investments", "Fund Services"],
    "Energy": ["Energy", "Petroleum", "Power", "Resources", "Oil & Gas",
               "Renewables", "Utilities", "Fuels", "Gas", "Offshore"],
    "Technology": ["Technologies", "Systems", "Software", "Digital", "Semiconductors",
                   "Cloud", "Data", "Networks", "Labs", "Robotics"],
    "Manufacturing": ["Manufacturing", "Industries", "Components", "Materials", "Steelworks",
                      "Machinery", "Plastics", "Chemicals", "Fabrication", "Engineering"],
    "Healthcare": ["Healthcare", "Pharma", "Biosciences", "Medical", "Therapeutics",
                   "Diagnostics", "Health Group", "Life Sciences", "Clinics", "MedTech"],
    "Telecommunications": ["Telecom", "Communications", "Mobile", "Broadband", "Wireless",
                           "Fibre", "Connect", "Satellite", "Media", "Signal"],
    "Real Estate": ["Properties", "Real Estate", "Estates", "Land", "Developments",
                    "Realty", "Housing", "Property Trust", "Logistics Parks", "Urban"],
    "Consumer Goods": ["Foods", "Brands", "Consumer", "Beverages", "Apparel",
                       "Retail", "Household", "Goods", "Cosmetics", "Home"],
    "Industrials": ["Industrial", "Engineering Group", "Aerospace", "Construction", "Equipment",
                    "Contractors", "Infrastructure", "Automation", "Marine", "Tools"],
    "Transportation": ["Logistics", "Shipping", "Airlines", "Freight", "Rail",
                       "Transport", "Ports", "Aviation", "Haulage", "Mobility"],
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _draw(rng: np.random.Generator, options: list[str], weights: list[float], size: int) -> np.ndarray:
    """Draw ``size`` categorical samples with (auto-normalised) weights."""
    p = np.asarray(weights, dtype=float)
    return rng.choice(np.asarray(options, dtype=object), size=size, p=p / p.sum())


def _lookup(values: np.ndarray, table: dict[str, dict], key: str) -> np.ndarray:
    """Map categorical ``values`` to a numeric parameter ``key`` in ``table``."""
    mapping = {k: float(v[key]) for k, v in table.items()}
    return pd.Series(values).map(mapping).to_numpy(dtype=float)


def _generate_names(
    rng: np.random.Generator, industries: np.ndarray, countries: np.ndarray
) -> list[str]:
    """Build unique fictional company names."""
    used: set[str] = set()
    names: list[str] = []
    for industry, country in zip(industries, countries):
        base = (
            f"{rng.choice(_NAME_PREFIXES)} {rng.choice(_INDUSTRY_NOUNS[industry])} "
            f"{rng.choice(cfg.LEGAL_FORMS[country])}"
        )
        name, counter = base, 1
        while name in used:  # extremely rare with the word lists above at n<=5,000
            counter += 1
            name = f"{base} ({counter})"
        used.add(name)
        names.append(name)
    return names


# --------------------------------------------------------------------------- #
# Main generator
# --------------------------------------------------------------------------- #
def generate_counterparties(
    n: int = cfg.DEFAULT_N_COUNTERPARTIES, seed: int = cfg.RANDOM_SEED
) -> pd.DataFrame:
    """Generate a synthetic counterparty portfolio.

    Args:
        n: Number of counterparties (>= ``config.MIN_N_COUNTERPARTIES``; the
            project brief asks for at least 2,000).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with the columns in :data:`RAW_COLUMNS`. Money in EUR mn.
    """
    if n < cfg.MIN_N_COUNTERPARTIES:
        raise ValueError(f"n must be >= {cfg.MIN_N_COUNTERPARTIES}, got {n}")
    rng = np.random.default_rng(seed)

    # ---- 1. categorical attributes -------------------------------------- #
    industries = _draw(rng, cfg.INDUSTRIES, [cfg.INDUSTRY_PARAMS[i]["weight"] for i in cfg.INDUSTRIES], n)
    countries = _draw(rng, cfg.COUNTRIES, [cfg.COUNTRY_PARAMS[c]["weight"] for c in cfg.COUNTRIES], n)
    ratings = _draw(rng, cfg.RATINGS, [cfg.RATING_MIX[r] for r in cfg.RATINGS], n)
    regions = pd.Series(countries).map({c: v["region"] for c, v in cfg.COUNTRY_PARAMS.items()}).to_numpy()
    idx = pd.Series(ratings).map(cfg.RATING_INDEX).to_numpy(dtype=int)

    r_pd = _lookup(ratings, cfg.RATING_PARAMS, "pd")
    r_lgd = _lookup(ratings, cfg.RATING_PARAMS, "lgd")
    r_lev = _lookup(ratings, cfg.RATING_PARAMS, "debt_ebitda")
    r_rate = _lookup(ratings, cfg.RATING_PARAMS, "cost_of_debt")
    r_dta = _lookup(ratings, cfg.RATING_PARAMS, "debt_to_assets")
    r_cash = _lookup(ratings, cfg.RATING_PARAMS, "cash_to_debt")
    r_cr = _lookup(ratings, cfg.RATING_PARAMS, "current_ratio")
    r_csa = _lookup(ratings, cfg.RATING_PARAMS, "csa_prob")
    i_margin = _lookup(industries, cfg.INDUSTRY_PARAMS, "ebitda_margin")
    i_lev = _lookup(industries, cfg.INDUSTRY_PARAMS, "leverage_mult")
    i_size = _lookup(industries, cfg.INDUSTRY_PARAMS, "size_log_adj")
    i_pd = _lookup(industries, cfg.INDUSTRY_PARAMS, "pd_logit_shift")
    i_lgd = _lookup(industries, cfg.INDUSTRY_PARAMS, "lgd_shift")

    # ---- 2. trend / migration ------------------------------------------- #
    deteriorating = rng.random(n) < P_DETERIORATING
    move = np.where(
        deteriorating,
        rng.choice([0, 1, 2], p=[0.40, 0.45, 0.15], size=n),      # notches downgraded
        rng.choice([-1, 0, 1], p=[0.05, 0.90, 0.05], size=n),     # -1 = upgraded
    )
    prior_idx = np.clip(idx - move, 0, len(cfg.RATINGS) - 1)
    rating_prior = np.asarray(cfg.RATINGS, dtype=object)[prior_idx]
    notch_downgrade = idx - prior_idx

    rev_growth = 0.035 - 0.012 * (idx - 3) + 0.09 * rng.standard_normal(n)
    rev_growth = rev_growth - np.where(deteriorating, rng.uniform(0.10, 0.30, n), 0.0)
    rev_growth = np.clip(rev_growth, -0.60, 0.60)
    ocf_growth = 1.2 * rev_growth + 0.10 * rng.standard_normal(n)
    ocf_growth = ocf_growth - np.where(deteriorating, rng.uniform(0.0, 0.15, n), 0.0)
    ocf_growth = np.clip(ocf_growth, -0.70, 0.90)

    # ---- 3. financial statements (EUR mn) ------------------------------- #
    log_rev = np.log(1200.0) + 0.20 * (3 - idx) + i_size + 0.90 * rng.standard_normal(n)
    revenue = np.clip(np.exp(log_rev), 50.0, 150_000.0)
    margin = np.clip(
        i_margin * np.exp(0.25 * rng.standard_normal(n)) * (1.0 - 0.03 * (idx - 3)), 0.02, 0.75
    )
    ebitda = revenue * margin

    e_lev = rng.standard_normal(n)
    lev_typ = r_lev * i_lev
    leverage = np.clip(lev_typ * np.exp(LEV_SIGMA * e_lev), 0.1, 20.0)
    debt = leverage * ebitda

    cost_of_debt = r_rate * np.exp(RATE_SIGMA * rng.standard_normal(n))
    interest = debt * cost_of_debt
    coverage = ebitda / interest  # mechanically consistent with leverage & rate
    coverage_typ = 1.0 / (lev_typ * r_rate)

    debt_to_assets = np.clip(r_dta * np.sqrt(i_lev) * np.exp(0.25 * rng.standard_normal(n)), 0.03, 0.85)
    assets = debt / debt_to_assets

    e_cash = rng.standard_normal(n)
    cash = np.minimum(debt * r_cash * np.exp(0.40 * e_cash), 0.40 * assets)

    e_cr = rng.standard_normal(n)
    current_liabilities = revenue * 0.18 * np.exp(0.30 * rng.standard_normal(n))
    current_assets = current_liabilities * r_cr * np.exp(0.20 * e_cr)
    current_assets = np.minimum(np.maximum(current_assets, 1.05 * cash), 0.90 * assets)

    conversion = np.clip(0.75 - 0.03 * (idx - 3) + 0.12 * rng.standard_normal(n), 0.20, 1.05)
    ocf = ebitda * conversion - 0.30 * interest

    revenue_prior = revenue / (1.0 + rev_growth)
    ocf_prior = ocf / (1.0 + ocf_growth)

    # ---- 4. PD (internal estimate) and realised default ------------------ #
    lev_dev = np.log(leverage / lev_typ) / LEV_SIGMA
    cov_dev = np.log(coverage / coverage_typ) / np.sqrt(LEV_SIGMA**2 + RATE_SIGMA**2)
    logit_adj = (
        0.30 * lev_dev            # higher leverage -> higher PD
        - 0.20 * cov_dev          # stronger coverage -> lower PD
        - 0.10 * e_cash           # more cash -> lower PD
        - 0.08 * e_cr             # better liquidity -> lower PD
        - 1.20 * rev_growth       # shrinking revenue -> higher PD
        - 0.60 * ocf_growth       # falling cash flow -> higher PD
        + 0.25 * notch_downgrade  # rating momentum
        + i_pd                    # industry effect
    )
    pd_est = np.clip(expit(logit(r_pd) + logit_adj), PD_FLOOR, PD_CAP)

    latent = LATENT_SHOCK_SD * rng.standard_normal(n) - 0.5 * LATENT_SHOCK_SD**2  # mean-preserving
    true_pd = expit(logit(pd_est) + latent)
    default_flag = (rng.random(n) < true_pd).astype(int)

    # ---- 5. LGD ---------------------------------------------------------- #
    # ---- 6. exposures ---------------------------------------------------- #
    product = np.empty(n, dtype=object)
    for industry in cfg.INDUSTRIES:
        mask = industries == industry
        if mask.any():
            product[mask] = _draw(rng, cfg.PRODUCT_TYPES, cfg.INDUSTRY_PRODUCT_MIX[industry], int(mask.sum()))

    p_lgd_shift = _lookup(product, cfg.PRODUCT_GENERATION_PARAMS, "lgd_shift")
    lgd = np.clip(r_lgd + i_lgd + p_lgd_shift + 0.05 * rng.standard_normal(n), 0.10, 0.90)

    gp = cfg.PRODUCT_GENERATION_PARAMS
    p_scale = _lookup(product, gp, "notional_scale")
    notional = np.clip(
        60.0 * p_scale * np.exp(0.35 * (np.log(revenue) - np.log(1200.0)) + NOTIONAL_SIGMA * rng.standard_normal(n)),
        1.0, 10_000.0,
    )
    maturity = np.exp(
        np.log(_lookup(product, gp, "mat_median")) + _lookup(product, gp, "mat_sigma") * rng.standard_normal(n)
    )
    maturity = np.clip(maturity, _lookup(product, gp, "mat_min"), _lookup(product, gp, "mat_max"))

    # Current exposure = max(MtM, 0). Longer maturity -> larger MtM dispersion.
    ce_frac = _lookup(product, gp, "ce_frac") * maturity**0.35 * np.exp(0.80 * rng.standard_normal(n))
    in_the_money = rng.random(n) > P_ZERO_EXPOSURE
    current_exposure = np.where(in_the_money, notional * ce_frac, 0.0)

    # Prior-quarter exposure: small drift, deteriorating names grow faster.
    drift = rng.normal(0.03, 0.12, n) + np.where(deteriorating, 0.12, 0.0)
    prior_from_ce = current_exposure / np.exp(drift)
    prior_when_zero = np.where(rng.random(n) < 0.25, notional * ce_frac * 0.5, 0.0)
    ce_prior = np.where(current_exposure > 0, prior_from_ce, prior_when_zero)

    # Risk-factor split of current exposure (fx + ir <= current exposure).
    primary = rng.uniform(0.80, 1.00, n)
    secondary = rng.uniform(0.0, 1.0, n) * (1.0 - primary)
    other_fx, other_ir = rng.uniform(0.0, 0.25, n), rng.uniform(0.0, 0.25, n)
    fx_share = np.select([product == "FX", product == "Interest Rate"], [primary, secondary], other_fx)
    ir_share = np.select([product == "FX", product == "Interest Rate"], [secondary, primary], other_ir)
    fx_exposure = current_exposure * fx_share
    ir_exposure = current_exposure * ir_share

    # Collateral: posted against current exposure under a CSA; coverage ~ 0-125%.
    has_csa = rng.random(n) < r_csa
    coverage_ratio = 1.25 * rng.beta(4.0, 2.5, n)
    collateral = np.where(has_csa, current_exposure * coverage_ratio, 0.0)
    coll_type = _draw(rng, cfg.COLLATERAL_TYPES, cfg.COLLATERAL_TYPE_MIX, n)
    coll_type = np.where(collateral > 0, coll_type, "None")

    # ---- assemble --------------------------------------------------------- #
    df = pd.DataFrame(
        {
            "counterparty_id": [f"CP{i:05d}" for i in range(1, n + 1)],
            "counterparty_name": _generate_names(rng, industries, countries),
            "industry": industries,
            "country": countries,
            "region": regions,
            "rating": ratings,
            "rating_prior_year": rating_prior,
            "annual_revenue": revenue,
            "revenue_prior_year": revenue_prior,
            "total_assets": assets,
            "total_debt": debt,
            "cash": cash,
            "current_assets": current_assets,
            "current_liabilities": current_liabilities,
            "ebitda": ebitda,
            "interest_expense": interest,
            "operating_cash_flow": ocf,
            "operating_cash_flow_prior_year": ocf_prior,
            "product_type": product,
            "notional": notional,
            "maturity_years": maturity,
            "current_exposure": current_exposure,
            "current_exposure_prior_quarter": ce_prior,
            "fx_exposure": fx_exposure,
            "interest_rate_exposure": ir_exposure,
            "collateral": collateral,
            "collateral_type": coll_type,
            "pd": pd_est,
            "lgd": lgd,
            "default_flag": default_flag,
        }
    )[RAW_COLUMNS]

    money_cols = [
        "annual_revenue", "revenue_prior_year", "total_assets", "total_debt", "cash",
        "current_assets", "current_liabilities", "ebitda", "interest_expense",
        "operating_cash_flow", "operating_cash_flow_prior_year", "notional",
        "current_exposure", "current_exposure_prior_quarter", "fx_exposure",
        "interest_rate_exposure", "collateral",
    ]
    df[money_cols] = df[money_cols].round(4)
    df["maturity_years"] = df["maturity_years"].round(3)
    df["pd"] = df["pd"].round(6)
    df["lgd"] = df["lgd"].round(4)
    logger.info("Generated %d counterparties (seed=%d)", n, seed)
    return df


# --------------------------------------------------------------------------- #
# Diagnostics and I/O
# --------------------------------------------------------------------------- #
def relationship_diagnostics(df: pd.DataFrame) -> dict[str, object]:
    """Check that the synthetic data has the intended economic relationships.

    Returns a dict with a by-rating table and Spearman rank correlations. Used
    by tests and by ``notebooks/01_data_generation.ipynb``.
    """
    work = df.assign(
        debt_to_ebitda=df["total_debt"] / df["ebitda"],
        interest_coverage=df["ebitda"] / df["interest_expense"],
    )
    by_rating = (
        work.groupby("rating", observed=True)
        .agg(n=("counterparty_id", "count"), mean_pd=("pd", "mean"), mean_lgd=("lgd", "mean"),
             default_rate=("default_flag", "mean"), median_debt_to_ebitda=("debt_to_ebitda", "median"),
             median_interest_coverage=("interest_coverage", "median"))
        .reindex(cfg.RATINGS)
    )
    rho_lev = spearmanr(work["debt_to_ebitda"], work["pd"])[0]
    rho_cov = spearmanr(work["interest_coverage"], work["pd"])[0]
    return {
        "by_rating": by_rating,
        "spearman_leverage_vs_pd": float(rho_lev),
        "spearman_coverage_vs_pd": float(rho_cov),
        "overall_mean_pd": float(work["pd"].mean()),
        "overall_default_rate": float(work["default_flag"].mean()),
        "n_defaults": int(work["default_flag"].sum()),
        "share_zero_current_exposure": float((work["current_exposure"] == 0).mean()),
        "share_collateralised": float((work["collateral"] > 0).mean()),
    }


def save_dataset(df: pd.DataFrame, path: Path = cfg.RAW_DATA_PATH) -> Path:
    """Write the dataset to CSV, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows to %s", len(df), path)
    return path


def load_dataset(path: Path = cfg.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset from CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python run_pipeline.py` or "
                                "`python -m src.data_generator` first.")
    return pd.read_csv(path)


def main() -> None:
    """CLI entry point: ``python -m src.data_generator --n 5000 --seed 42``."""
    parser = argparse.ArgumentParser(description="Generate the synthetic counterparty dataset.")
    parser.add_argument("--n", type=int, default=cfg.DEFAULT_N_COUNTERPARTIES)
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    parser.add_argument("--out", type=Path, default=cfg.RAW_DATA_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = generate_counterparties(args.n, args.seed)
    save_dataset(df, args.out)
    diag = relationship_diagnostics(df)
    print(diag["by_rating"].round(4).to_string())
    print(f"Mean PD {diag['overall_mean_pd']:.4f} | default rate {diag['overall_default_rate']:.4f} "
          f"| defaults {diag['n_defaults']}")


if __name__ == "__main__":
    main()
