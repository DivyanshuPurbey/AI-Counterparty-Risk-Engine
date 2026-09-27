"""AI Risk Analyst: an LLM that *explains* deterministic risk-engine outputs.

CRITICAL DESIGN PRINCIPLE (see README "AI Risk Analyst")
----------------------------------------------------------------------------
The LLM never calculates PD, LGD, PFE, EAD, Expected Loss, stress results,
concentration statistics or risk flags. All of those numbers come from
``risk_engine.py``, ``stress_testing.py`` and ``portfolio_analytics.py``,
which are plain, testable, deterministic pandas/numpy code.

The flow is always:

    user question
        -> intent routed to a Python function ("tool")
        -> Python computes/looks up the relevant numbers from the already-
           computed risk tables
        -> those numbers (not raw row dumps) are put in a prompt
        -> the LLM turns numbers into a written explanation

This keeps every figure the user sees auditable back to a specific pandas
computation, and stops the LLM from silently inventing or miscalculating a
risk number -- a hard requirement for anything resembling a risk-reporting
use case.

The API key is read from the ``ANTHROPIC_API_KEY`` environment variable
(via python-dotenv) and is never hard-coded. If no key is configured, every
public function still works: it returns the computed data plus a
clearly-labelled deterministic (non-LLM) summary instead of an LLM narrative,
so the rest of the project (and the dashboard) is fully usable without an API
key.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv

from src import portfolio_analytics as pa
from src import stress_testing as st
from src.risk_engine import aggregate_risk, portfolio_summary

logger = logging.getLogger(__name__)
load_dotenv()  # reads .env if present; never overwrites a real env var

ANTHROPIC_MODEL = "claude-sonnet-4-6"


def _get_client():
    """Lazily build an Anthropic client. Returns None if no API key is configured."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed; falling back to deterministic summaries.")
        return None
    return anthropic.Anthropic(api_key=api_key)


@dataclass
class AnalystResponse:
    """What every AI Risk Analyst call returns."""

    question: str
    intent: str
    data: dict | pd.DataFrame
    narrative: str
    used_llm: bool


# --------------------------------------------------------------------------- #
# Step 1: intent routing (simple, transparent keyword rules -- not an LLM)
# --------------------------------------------------------------------------- #

_CP_ID_PATTERN = re.compile(r"\bCP\d{4,6}\b", re.IGNORECASE)

_SCENARIO_KEYWORDS = {
    "severe recession": "severe_recession", "severe_recession": "severe_recession",
    "recession": "recession", "market shock": "market_shock", "market_shock": "market_shock",
    "base": "base",
}


def route_intent(question: str) -> str:
    """Very small, explainable keyword router (no LLM call needed to route)."""
    q = question.lower()
    if _CP_ID_PATTERN.search(question):
        return "explain_counterparty"
    if "concentrat" in q:
        return "concentration"
    if "unsecured" in q:
        return "unsecured_exposure"
    if any(k in q for k in ("fx shock", "fx sensitiv", "sensitive to")):
        return "theme_sensitivity"
    if any(k in q for k in ("scenario", "stress", "recession", "shock", "vulnerable", "increase")):
        return "stress"
    if any(k in q for k in ("high ead", "low rating", "low credit rating")):
        return "screen_high_ead_low_rating"
    return "portfolio_overview"


# --------------------------------------------------------------------------- #
# Step 2: Python "tools" that fetch/compute the actual numbers
# --------------------------------------------------------------------------- #


def tool_portfolio_overview(df: pd.DataFrame) -> dict:
    return portfolio_summary(df)


def tool_explain_counterparty(df: pd.DataFrame, counterparty_id: str) -> dict:
    row = df[df["counterparty_id"].str.upper() == counterparty_id.upper()]
    if row.empty:
        return {"error": f"No counterparty found with id {counterparty_id}"}
    r = row.iloc[0]
    fields = [
        "counterparty_id", "counterparty_name", "industry", "country", "rating",
        "product_type", "notional", "current_exposure", "pfe", "collateral",
        "eligible_collateral", "unsecured_exposure", "ead", "pd", "lgd", "expected_loss",
        "debt_to_ebitda", "interest_coverage", "credit_score_band",
    ]
    return {f: r[f] for f in fields if f in row.columns}


def tool_concentration(df: pd.DataFrame) -> dict:
    return {
        "summary": pa.concentration_summary(df),
        "top10_by_ead": pa.top_n_by(df, "ead", 10).to_dict(orient="records"),
        "by_industry": aggregate_risk(df, "industry").head(5).to_dict(orient="records"),
    }


def tool_unsecured_exposure(df: pd.DataFrame, n: int = 10) -> dict:
    top = df.sort_values("unsecured_exposure", ascending=False).head(n)
    cols = ["counterparty_id", "counterparty_name", "industry", "rating", "unsecured_exposure", "ead"]
    return {"top_unsecured": top[cols].to_dict(orient="records")}


def tool_screen_high_ead_low_rating(df: pd.DataFrame, n: int = 10) -> dict:
    flagged = pa.flag_high_risk_counterparties(df)
    hits = flagged[flagged["flag_high_ead_low_rating"]]
    cols = ["counterparty_id", "counterparty_name", "industry", "rating", "ead", "pd", "expected_loss"]
    return {"n_flagged": int(len(hits)), "top": hits.sort_values("ead", ascending=False).head(n)[cols].to_dict(orient="records")}


def tool_stress(df: pd.DataFrame, scenario_key: str | None) -> dict:
    scenarios = st.ALL_SCENARIOS
    if not scenarios:
        return {"error": "No stress scenarios are configured in stress_testing.py"}

    key = scenario_key or "SEVERE_RECESSION"
    matched = next((k for k in scenarios if key.upper().replace(" ", "_") in k), list(scenarios)[0])
    scenario = scenarios[matched]
    stressed = st.apply_scenario(df, scenario)
    result = st.summarise_scenario(stressed)
    return {"scenario": st.scenario_label(matched), "result": result}


def tool_theme_sensitivity(df: pd.DataFrame, theme: str = "FX_SHOCK") -> dict:
    matched = next((k for k in st.ALL_SCENARIOS if theme.upper().replace(" ", "_") in k), "FX_SHOCK")
    full_matrix = st.industry_sensitivity_matrix(df)
    if matched not in full_matrix.columns:
        return {"scenario": matched, "industry_sensitivity": []}
    ranked = full_matrix[[matched]].sort_values(matched, ascending=False).reset_index()
    ranked.columns = ["industry", "expected_loss_pct_change"]
    return {"scenario": matched, "industry_sensitivity": ranked.to_dict(orient="records")}


_TOOL_DISPATCH = {
    "portfolio_overview": lambda df, **kw: tool_portfolio_overview(df),
    "explain_counterparty": lambda df, **kw: tool_explain_counterparty(df, kw.get("counterparty_id", "")),
    "concentration": lambda df, **kw: tool_concentration(df),
    "unsecured_exposure": lambda df, **kw: tool_unsecured_exposure(df),
    "screen_high_ead_low_rating": lambda df, **kw: tool_screen_high_ead_low_rating(df),
    "stress": lambda df, **kw: tool_stress(df, kw.get("scenario_key")),
    "theme_sensitivity": lambda df, **kw: tool_theme_sensitivity(df),
}


# --------------------------------------------------------------------------- #
# Step 3: deterministic fallback narrative (used when no API key is set)
# --------------------------------------------------------------------------- #


def _deterministic_narrative(intent: str, data: dict) -> str:
    """A plain, rule-based summary of `data` -- no LLM involved.

    This exists so the project (and the Streamlit dashboard) is fully
    functional without an API key: every number the user sees still comes
    straight from the risk engine.
    """
    if "error" in data:
        return data["error"]
    if intent == "portfolio_overview":
        return (
            f"Portfolio of {data['n_counterparties']:,} counterparties. "
            f"Total EAD is EUR {data['total_ead']:.1f}mn and total Expected Loss is "
            f"EUR {data['total_expected_loss']:.2f}mn ({data['el_to_ead_bps']:.1f} bps of EAD). "
            f"EAD-weighted PD is {data['ead_weighted_pd']:.2%}."
        )
    if intent == "explain_counterparty":
        return (
            f"{data.get('counterparty_name', data.get('counterparty_id'))} "
            f"({data.get('industry')}, {data.get('country', '')}, rated {data.get('rating')}): "
            f"EAD EUR {data.get('ead', 0):.2f}mn, PD {data.get('pd', 0):.2%}, LGD {data.get('lgd', 0):.0%}, "
            f"Expected Loss EUR {data.get('expected_loss', 0):.3f}mn. "
            f"Unsecured exposure EUR {data.get('unsecured_exposure', 0):.2f}mn."
        )
    if intent == "concentration":
        s = data["summary"]
        return (
            f"Top-10 counterparties account for {s['top10_ead_share']:.1%} of total EAD "
            f"and {s['top10_el_share']:.1%} of total Expected Loss. "
            f"Industry HHI on EAD is {s['industry_hhi_ead']:.3f}, rating HHI is {s['rating_hhi_ead']:.3f}."
        )
    if intent == "unsecured_exposure":
        rows = data["top_unsecured"][:3]
        names = ", ".join(r["counterparty_name"] for r in rows)
        return f"The largest unsecured exposures are concentrated in: {names} (see full table)."
    if intent == "screen_high_ead_low_rating":
        return f"{data['n_flagged']} counterparties combine top-decile EAD with a B/CCC rating."
    if intent == "theme_sensitivity":
        rows = data.get("industry_sensitivity", [])[:3]
        if not rows:
            return f"No sensitivity data available for scenario '{data.get('scenario')}'."
        names = ", ".join(f"{r['industry']} ({r['expected_loss_pct_change']:+.0%})" for r in rows)
        return f"Under '{data['scenario']}', the most sensitive industries by % change in Expected Loss are: {names}."
    if intent == "stress":
        r = data["result"]
        return (
            f"Scenario '{data['scenario']}': Expected Loss moves from EUR {r['base_expected_loss']:.2f}mn to "
            f"EUR {r['stressed_expected_loss']:.2f}mn, an incremental EUR {r['incremental_expected_loss']:.2f}mn "
            f"({r['expected_loss_pct_change']:.1%})."
        )
    return "See the computed data table."


# --------------------------------------------------------------------------- #
# Step 4: public entry point
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are a credit-risk analyst assistant embedded in an educational "
    "counterparty-credit-risk project. You are given ALREADY-CALCULATED risk "
    "numbers (PD, LGD, EAD, PFE, Expected Loss, stress results, concentration "
    "statistics) produced by a deterministic Python risk engine. "
    "Your job is ONLY to explain these numbers clearly and concisely in plain "
    "English for a risk-committee-style audience. "
    "Never invent, recompute, or 'correct' a number -- use exactly the figures "
    "given to you. If a figure is missing, say so instead of estimating it. "
    "Keep answers under ~150 words unless the data has many rows to summarise. "
    "This is a synthetic-data educational project, not a real bank portfolio -- "
    "do not imply otherwise."
)


def ask(df: pd.DataFrame, question: str, **kwargs) -> AnalystResponse:
    """Answer a natural-language risk question using engine data + optional LLM.

    Args:
        df: the fully-computed counterparty risk table (output of
            ``risk_engine.run_risk_engine``).
        question: free-text user question.
        kwargs: optional overrides, e.g. counterparty_id=..., scenario_key=...
    """
    intent = route_intent(question)
    if intent == "explain_counterparty" and "counterparty_id" not in kwargs:
        match = _CP_ID_PATTERN.search(question)
        if match:
            kwargs["counterparty_id"] = match.group(0)

    data = _TOOL_DISPATCH[intent](df, **kwargs)

    client = _get_client()
    if client is None:
        narrative = _deterministic_narrative(intent, data)
        return AnalystResponse(question, intent, data, narrative, used_llm=False)

    try:
        message = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    f"User question: {question}\n\n"
                    f"Computed data (JSON): {data}\n\n"
                    "Explain this to the user."
                ),
            }],
        )
        narrative = "".join(b.text for b in message.content if hasattr(b, "text"))
        return AnalystResponse(question, intent, data, narrative, used_llm=True)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully in a demo app
        logger.warning("LLM call failed (%s); falling back to deterministic narrative.", exc)
        narrative = _deterministic_narrative(intent, data)
        return AnalystResponse(question, intent, data, narrative, used_llm=False)
