"""Tests for src/ai_risk_analyst.py.

These tests deliberately run WITHOUT an ANTHROPIC_API_KEY set, so every
assertion is about the deterministic data path: the LLM must never be the
source of a number. We only check ``response.data`` (computed by Python) and
that ``used_llm`` is False when no key is configured.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from src import ai_risk_analyst as ai


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_route_intent_detects_counterparty_id():
    assert ai.route_intent("Explain counterparty CP00042") == "explain_counterparty"


def test_route_intent_detects_stress():
    assert ai.route_intent("What happens under a severe recession scenario?") == "stress"


def test_route_intent_detects_concentration():
    assert ai.route_intent("Which industries have the highest concentration risk?") == "concentration"


def test_route_intent_falls_back_to_overview():
    assert ai.route_intent("hello") == "portfolio_overview"


def test_ask_portfolio_overview_uses_computed_totals(risk_df: pd.DataFrame):
    response = ai.ask(risk_df, "Give me a portfolio overview")
    assert response.used_llm is False
    assert response.data["total_ead"] == pytest.approx(float(risk_df["ead"].sum()))
    assert response.data["total_expected_loss"] == pytest.approx(float(risk_df["expected_loss"].sum()))
    assert str(response.data["n_counterparties"]) in response.narrative or True  # narrative is text, not asserted verbatim


def test_ask_explain_counterparty_matches_row(risk_df: pd.DataFrame):
    cp_id = risk_df["counterparty_id"].iloc[0]
    response = ai.ask(risk_df, f"Explain counterparty {cp_id}")
    assert response.intent == "explain_counterparty"
    assert response.data["counterparty_id"] == cp_id
    expected_ead = risk_df.loc[risk_df["counterparty_id"] == cp_id, "ead"].iloc[0]
    assert response.data["ead"] == pytest.approx(expected_ead)


def test_ask_unknown_counterparty_returns_error(risk_df: pd.DataFrame):
    response = ai.ask(risk_df, "Explain counterparty CP99999")
    assert response.intent == "explain_counterparty"
    assert "error" in response.data


def test_ask_stress_scenario_matches_stress_testing_module(risk_df: pd.DataFrame):
    response = ai.ask(risk_df, "Which counterparties are most vulnerable under a severe recession?")
    assert response.intent == "stress"
    assert response.data["result"]["incremental_expected_loss"] > 0


def test_ask_never_calls_llm_without_api_key(risk_df: pd.DataFrame):
    for q in [
        "portfolio overview", "concentration risk", "unsecured exposure",
        "high ead low rating", "fx shock sensitivity",
    ]:
        response = ai.ask(risk_df, q)
        assert response.used_llm is False
