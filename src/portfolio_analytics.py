"""Credit concentration analytics and an Early Warning System (EWS).

*** EDUCATIONAL, ILLUSTRATIVE THRESHOLDS ***
Every threshold in ``EWS_TRIGGERS`` below is a commonly-used, textbook-style
rule of thumb (e.g. Debt/EBITDA > 6x, Interest Coverage < 2x). None of them
is calibrated to this bank's, or any bank's, actual default experience, and
none of them reproduces a specific rating agency or regulatory framework.
In a production setting these thresholds would be calibrated, back-tested and
governed (model risk sign-off, periodic recalibration, override process).

This module intentionally does NOT call any LLM: everything here is plain
pandas/numpy arithmetic so results are deterministic and auditable. The AI
Risk Analyst (``ai_risk_analyst.py``) explains these outputs in natural
language, but never recomputes them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk_engine import aggregate_risk

# --------------------------------------------------------------------------- #
# Concentration analytics (PART 5)
# --------------------------------------------------------------------------- #


def top_n_by(df: pd.DataFrame, metric: str, n: int = 10) -> pd.DataFrame:
    """Top-``n`` counterparties by a given metric (e.g. 'ead', 'expected_loss')."""
    cols = [
        "counterparty_id", "counterparty_name", "industry", "country", "rating",
        "product_type", metric,
    ]
    cols = [c for c in cols if c in df.columns]
    return df.sort_values(metric, ascending=False).head(n)[cols].reset_index(drop=True)


def _herfindahl(shares: pd.Series) -> float:
    """Herfindahl-Hirschman Index (HHI) on a set of exposure shares (0-1).

    HHI = sum(share_i^2). Ranges from ~1/N (fully diversified) to 1 (single
    name). This is a standard concentration statistic, used here in its
    simplest form (no regulatory granularity adjustment).
    """
    s = shares.dropna()
    return float((s ** 2).sum()) if len(s) else float("nan")


def concentration_summary(df: pd.DataFrame) -> dict[str, object]:
    """Headline concentration measures: top-10 exposure share + HHI by dimension."""
    total_ead = float(df["ead"].sum())
    top10_ead_share = (
        df.sort_values("ead", ascending=False).head(10)["ead"].sum() / total_ead
        if total_ead > 0 else float("nan")
    )
    total_el = float(df["expected_loss"].sum())
    top10_el_share = (
        df.sort_values("expected_loss", ascending=False).head(10)["expected_loss"].sum() / total_el
        if total_el > 0 else float("nan")
    )

    out: dict[str, object] = {
        "top10_ead_share": top10_ead_share,
        "top10_el_share": top10_el_share,
    }
    for dim in ("industry", "country", "rating"):
        agg = aggregate_risk(df, dim)
        out[f"{dim}_hhi_ead"] = _herfindahl(agg["ead_share"])
        out[f"{dim}_top1_ead_share"] = float(agg["ead_share"].max()) if len(agg) else float("nan")
    return out


def concentration_tables(df: pd.DataFrame, n: int = 10) -> dict[str, pd.DataFrame]:
    """All concentration tables needed by the dashboard, in one call."""
    return {
        "top_counterparties_ead": top_n_by(df, "ead", n),
        "top_counterparties_el": top_n_by(df, "expected_loss", n),
        "by_industry": aggregate_risk(df, "industry"),
        "by_country": aggregate_risk(df, "country"),
        "by_rating": aggregate_risk(df, "rating"),
        "by_product": aggregate_risk(df, "product_type"),
    }


# --------------------------------------------------------------------------- #
# Risk flags used by concentration screens (PART 5)
# --------------------------------------------------------------------------- #


def flag_high_risk_counterparties(
    df: pd.DataFrame,
    high_ead_pctile: float = 0.90,
    high_pd: float = 0.10,
    low_rating: tuple[str, ...] = ("B", "CCC"),
    high_pfe_pctile: float = 0.90,
    unsecured_pctile: float = 0.90,
) -> pd.DataFrame:
    """Add boolean screen flags (not the full EWS traffic-light, see below).

    Flags added:
        flag_high_ead_low_rating : EAD in top decile AND rating in ``low_rating``
        flag_high_ead_high_pd    : EAD in top decile AND PD > ``high_pd``
        flag_high_unsecured      : unsecured_exposure in top decile
        flag_high_pfe            : PFE in top decile
        flag_deteriorating       : rating downgraded vs prior year (notches_downgraded > 0)
    """
    out = df.copy()
    ead_cut = out["ead"].quantile(high_ead_pctile)
    pfe_cut = out["pfe"].quantile(high_pfe_pctile)
    unsec_cut = out["unsecured_exposure"].quantile(unsecured_pctile)

    out["flag_high_ead_low_rating"] = (out["ead"] >= ead_cut) & (out["rating"].isin(low_rating))
    out["flag_high_ead_high_pd"] = (out["ead"] >= ead_cut) & (out["pd"] >= high_pd)
    out["flag_high_unsecured"] = out["unsecured_exposure"] >= unsec_cut
    out["flag_high_pfe"] = out["pfe"] >= pfe_cut
    if "notches_downgraded" in out.columns:
        out["flag_deteriorating"] = out["notches_downgraded"] > 0
    else:
        out["flag_deteriorating"] = False
    return out


# --------------------------------------------------------------------------- #
# Early Warning System (PART 8)
# --------------------------------------------------------------------------- #

# Each trigger: (column, comparison, threshold, rationale)
# comparison is one of "gt", "lt", "ge", "le", "isin"
EWS_TRIGGERS: list[dict[str, object]] = [
    {
        "key": "high_leverage", "column": "debt_to_ebitda", "op": "gt", "threshold": 6.0,
        "rationale": "Debt/EBITDA > 6x is a widely-used leveraged-credit rule of thumb "
                     "(illustrative; not calibrated to this portfolio's defaults).",
    },
    {
        "key": "weak_coverage", "column": "interest_coverage", "op": "lt", "threshold": 2.0,
        "rationale": "Interest coverage < 2x means EBITDA barely covers interest expense.",
    },
    {
        "key": "high_pd", "column": "pd", "op": "gt", "threshold": 0.10,
        "rationale": "1-year PD > 10% is a speculative-grade-and-worse illustrative anchor.",
    },
    {
        "key": "weak_rating", "column": "rating", "op": "isin", "threshold": ("B", "CCC"),
        "rationale": "Rating at or below B is treated as sub-investment-grade-weak here.",
    },
    {
        "key": "exposure_growth", "column": "exposure_growth", "op": "gt", "threshold": 0.20,
        "rationale": "Current-exposure growth > 20% quarter-on-quarter flags rapidly building exposure.",
    },
    {
        "key": "low_collateral_coverage", "column": "collateral_coverage", "op": "lt", "threshold": 0.30,
        "rationale": "Eligible collateral covering < 30% of current exposure leaves most of it unsecured.",
    },
    {
        "key": "revenue_decline", "column": "revenue_growth", "op": "lt", "threshold": -0.10,
        "rationale": "Revenue decline > 10% year-on-year signals a deteriorating top line.",
    },
    {
        "key": "ocf_decline", "column": "ocf_growth", "op": "lt", "threshold": -0.15,
        "rationale": "Operating cash flow decline > 15% signals weakening cash generation.",
    },
]


def _evaluate_trigger(df: pd.DataFrame, trigger: dict[str, object]) -> pd.Series:
    col, op, thr = trigger["column"], trigger["op"], trigger["threshold"]
    s = df[col]
    if op == "gt":
        hit = s > thr
    elif op == "ge":
        hit = s >= thr
    elif op == "lt":
        hit = s < thr
    elif op == "le":
        hit = s <= thr
    elif op == "isin":
        hit = s.isin(thr)
    else:
        raise ValueError(f"Unknown operator: {op}")
    return hit.fillna(False)


def apply_early_warning_system(
    df: pd.DataFrame, triggers: list[dict[str, object]] | None = None
) -> pd.DataFrame:
    """Evaluate every trigger and assign a GREEN / AMBER / RED early-warning status.

    Banding (illustrative, documented rather than "black box"):
        0 triggers            -> GREEN
        1-2 triggers          -> AMBER
        3+ triggers           -> RED

    Adds one boolean column per trigger (``ews_<key>``), ``ews_triggers_hit``
    (count) and ``ews_status``.
    """
    triggers = triggers or EWS_TRIGGERS
    out = df.copy()
    hit_cols = []
    for trig in triggers:
        col_name = f"ews_{trig['key']}"
        out[col_name] = _evaluate_trigger(out, trig)
        hit_cols.append(col_name)

    out["ews_triggers_hit"] = out[hit_cols].sum(axis=1)
    out["ews_status"] = np.select(
        [out["ews_triggers_hit"] >= 3, out["ews_triggers_hit"] >= 1],
        ["RED", "AMBER"],
        default="GREEN",
    )
    return out


def ews_summary(df_with_ews: pd.DataFrame) -> pd.DataFrame:
    """Counts, EAD and Expected Loss by EWS status (GREEN/AMBER/RED)."""
    if "ews_status" not in df_with_ews.columns:
        raise ValueError("Run apply_early_warning_system() first.")
    return aggregate_risk(df_with_ews, "ews_status")


def trigger_frequency(df_with_ews: pd.DataFrame, triggers: list[dict[str, object]] | None = None) -> pd.DataFrame:
    """How many counterparties tripped each individual EWS trigger."""
    triggers = triggers or EWS_TRIGGERS
    rows = []
    for trig in triggers:
        col = f"ews_{trig['key']}"
        if col not in df_with_ews.columns:
            continue
        rows.append({
            "trigger": trig["key"],
            "n_counterparties": int(df_with_ews[col].sum()),
            "pct_of_portfolio": float(df_with_ews[col].mean()),
            "rationale": trig["rationale"],
        })
    return pd.DataFrame(rows).sort_values("n_counterparties", ascending=False).reset_index(drop=True)
