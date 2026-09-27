"""Tests for src/portfolio_analytics.py — concentration analytics and EWS."""

from __future__ import annotations

import pandas as pd

from src.portfolio_analytics import (
    apply_early_warning_system,
    concentration_summary,
    concentration_tables,
    ews_summary,
    flag_high_risk_counterparties,
    top_n_by,
    trigger_frequency,
)


def test_top_n_by_returns_sorted_subset(risk_df: pd.DataFrame):
    top5 = top_n_by(risk_df, "ead", n=5)
    assert len(top5) == 5
    assert list(top5["ead"]) == sorted(top5["ead"], reverse=True)
    assert top5["ead"].iloc[0] == risk_df["ead"].max()


def test_concentration_summary_shares_are_bounded(risk_df: pd.DataFrame):
    summary = concentration_summary(risk_df)
    assert 0.0 <= summary["top10_ead_share"] <= 1.0
    assert 0.0 <= summary["top10_el_share"] <= 1.0
    # HHI on shares that sum to 1 across N buckets is in [1/N, 1]
    assert 0.0 < summary["industry_hhi_ead"] <= 1.0
    assert 0.0 < summary["rating_hhi_ead"] <= 1.0


def test_top10_share_at_least_as_large_as_average_share(risk_df: pd.DataFrame):
    """The 10 largest names must hold >= what 10 average-sized names would."""
    summary = concentration_summary(risk_df)
    average_10_share = 10 / len(risk_df)
    assert summary["top10_ead_share"] >= average_10_share


def test_concentration_tables_returns_expected_keys(risk_df: pd.DataFrame):
    tables = concentration_tables(risk_df, n=5)
    assert set(tables) == {
        "top_counterparties_ead", "top_counterparties_el",
        "by_industry", "by_country", "by_rating", "by_product",
    }
    assert len(tables["top_counterparties_ead"]) == 5


def test_flag_high_risk_counterparties_adds_boolean_columns(risk_df: pd.DataFrame):
    flagged = flag_high_risk_counterparties(risk_df)
    for col in (
        "flag_high_ead_low_rating", "flag_high_ead_high_pd",
        "flag_high_unsecured", "flag_high_pfe", "flag_deteriorating",
    ):
        assert col in flagged.columns
        assert flagged[col].dtype == bool

    # By construction, ~10% of names should sit in the top EAD decile.
    ead_cut = risk_df["ead"].quantile(0.90)
    n_top_decile = int((risk_df["ead"] >= ead_cut).sum())
    assert 0 < n_top_decile < len(risk_df)


def test_early_warning_system_status_matches_trigger_count(risk_df: pd.DataFrame):
    ews_df = apply_early_warning_system(risk_df)
    assert set(ews_df["ews_status"].unique()) <= {"GREEN", "AMBER", "RED"}

    red = ews_df[ews_df["ews_status"] == "RED"]
    amber = ews_df[ews_df["ews_status"] == "AMBER"]
    green = ews_df[ews_df["ews_status"] == "GREEN"]
    assert (red["ews_triggers_hit"] >= 3).all()
    assert ((amber["ews_triggers_hit"] >= 1) & (amber["ews_triggers_hit"] <= 2)).all()
    assert (green["ews_triggers_hit"] == 0).all()


def test_red_status_carries_higher_average_pd_than_green(risk_df: pd.DataFrame):
    """Sanity check: RED-flagged names should look riskier than GREEN on average."""
    ews_df = apply_early_warning_system(risk_df)
    red_pd = ews_df.loc[ews_df["ews_status"] == "RED", "pd"].mean()
    green_pd = ews_df.loc[ews_df["ews_status"] == "GREEN", "pd"].mean()
    assert red_pd > green_pd


def test_ews_summary_and_trigger_frequency_run(risk_df: pd.DataFrame):
    ews_df = apply_early_warning_system(risk_df)
    summary = ews_summary(ews_df)
    assert set(summary["ews_status"]) <= {"GREEN", "AMBER", "RED"}
    assert summary["ead"].sum() > 0

    freq = trigger_frequency(ews_df)
    assert set(freq.columns) == {"trigger", "n_counterparties", "pct_of_portfolio", "rationale"}
    assert (freq["pct_of_portfolio"] >= 0).all() and (freq["pct_of_portfolio"] <= 1).all()
