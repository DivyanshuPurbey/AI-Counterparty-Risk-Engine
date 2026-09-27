"""Unit tests for scenario, stress and thematic calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk_engine import run_risk_engine
from src.stress_testing import (
    ALL_SCENARIOS,
    DEFAULT_SCENARIOS,
    THEMES,
    Scenario,
    aggregate_stress,
    analyze_theme,
    apply_scenario,
    industry_sensitivity_matrix,
    pd_shock_scenario,
    run_scenarios,
    scenario_label,
    summarise_scenario,
)


# ------------------------------ scenario objects --------------------------- #
def test_default_scenarios_follow_the_brief():
    s = DEFAULT_SCENARIOS
    assert list(s) == ["BASE", "RECESSION", "SEVERE_RECESSION", "MARKET_SHOCK"]
    assert (s["BASE"].pd_mult, s["BASE"].lgd_mult, s["BASE"].exposure_mult) == (1.0, 1.0, 1.0)
    assert (s["RECESSION"].pd_mult, s["RECESSION"].lgd_mult, s["RECESSION"].exposure_mult) == (1.5, 1.1, 1.10)
    assert (s["SEVERE_RECESSION"].pd_mult, s["SEVERE_RECESSION"].lgd_mult,
            s["SEVERE_RECESSION"].exposure_mult) == (2.0, 1.2, 1.25)
    m = s["MARKET_SHOCK"]
    assert (m.pd_mult, m.lgd_mult, m.fx_exposure_mult, m.ir_exposure_mult) == (1.3, 1.1, 1.3, 1.2)


def test_scenario_parameters_can_be_changed_dynamically():
    harsher = DEFAULT_SCENARIOS["RECESSION"].with_overrides(pd_mult=2.5)
    assert harsher.pd_mult == 2.5
    assert DEFAULT_SCENARIOS["RECESSION"].pd_mult == 1.5  # original untouched (frozen)


def test_invalid_scenarios_are_rejected():
    with pytest.raises(ValueError):
        Scenario("bad", pd_mult=-1.0)
    with pytest.raises(ValueError):
        Scenario("bad", collateral_haircut_add=1.5)
    with pytest.raises(ValueError, match="unknown keys"):
        Scenario("bad", industry_pd_mult={"Space Mining": 2.0})


def test_pd_shock_helper_and_labels():
    assert pd_shock_scenario(50).pd_mult == pytest.approx(1.5)
    assert pd_shock_scenario(-20).pd_mult == pytest.approx(0.8)
    assert scenario_label("SEVERE_RECESSION") == "Severe Recession"
    with pytest.raises(ValueError):
        pd_shock_scenario(-100)


# ---------------------------- hand-calculated cases ------------------------- #
def test_base_scenario_reproduces_base_engine(tiny_df):
    risk = run_risk_engine(tiny_df)
    stress = apply_scenario(risk, DEFAULT_SCENARIOS["BASE"])
    np.testing.assert_allclose(stress["stressed_ead"], risk["ead"])
    np.testing.assert_allclose(stress["stressed_expected_loss"], risk["expected_loss"])
    np.testing.assert_allclose(stress["incremental_expected_loss"], 0.0, atol=1e-12)


def test_recession_hand_calculation_cp1(tiny_df):
    risk = run_risk_engine(tiny_df)
    row = apply_scenario(risk, DEFAULT_SCENARIOS["RECESSION"]).set_index("counterparty_id").loc["CP1"]
    # PD 0.02x1.5 ; LGD 0.45x1.1 ; CE 10x1.10 ; PFE 2.0x1.10 ; haircut 10%+5% on 6.0 collateral
    assert row["stressed_pd"] == pytest.approx(0.03)
    assert row["stressed_lgd"] == pytest.approx(0.495)
    assert row["stressed_current_exposure"] == pytest.approx(11.0)
    assert row["stressed_pfe"] == pytest.approx(2.2)
    assert row["stressed_eligible_collateral"] == pytest.approx(5.1)
    assert row["stressed_ead"] == pytest.approx(8.1)
    assert row["stressed_expected_loss"] == pytest.approx(0.03 * 0.495 * 8.1)
    assert row["incremental_expected_loss"] == pytest.approx(0.03 * 0.495 * 8.1 - 0.02 * 0.45 * 6.6)


def test_market_shock_scales_fx_and_rate_exposure_separately(tiny_df):
    risk = run_risk_engine(tiny_df)
    res = apply_scenario(risk, DEFAULT_SCENARIOS["MARKET_SHOCK"]).set_index("counterparty_id")
    # CP2 is an FX trade: CE 4 x1.3 = 5.2 ; PFE 2.0 x1.3 = 2.6 ; no collateral
    assert res.loc["CP2", "stressed_current_exposure"] == pytest.approx(5.2)
    assert res.loc["CP2", "stressed_pfe"] == pytest.approx(2.6)
    assert res.loc["CP2", "stressed_ead"] == pytest.approx(7.8)
    assert res.loc["CP2", "stressed_expected_loss"] == pytest.approx(0.065 * 0.55 * 7.8)
    # CP1 is an IR trade: CE = 2 (other) + 8 x1.2 = 11.6 ; PFE 2.0 x1.2 = 2.4
    assert res.loc["CP1", "stressed_current_exposure"] == pytest.approx(11.6)
    assert res.loc["CP1", "stressed_pfe"] == pytest.approx(2.4)
    assert res.loc["CP1", "stressed_ead"] == pytest.approx(11.6 + 2.4 - 5.1)


def test_stressed_pd_and_lgd_are_capped_at_one(tiny_df):
    risk = run_risk_engine(tiny_df)
    res = apply_scenario(risk, Scenario("EXTREME", pd_mult=4.0, lgd_mult=3.0))
    assert (res["stressed_pd"] <= 1.0).all() and (res["stressed_lgd"] <= 1.0).all()
    assert res.set_index("counterparty_id").loc["CP3", "stressed_pd"] == pytest.approx(1.0)


def test_industry_specific_multiplier_only_hits_that_industry(tiny_df):
    risk = run_risk_engine(tiny_df)
    res = apply_scenario(risk, Scenario("RE_ONLY", industry_pd_mult={"Real Estate": 2.0})).set_index("counterparty_id")
    assert res.loc["CP3", "stressed_pd"] == pytest.approx(0.60)
    assert res.loc["CP1", "stressed_pd"] == pytest.approx(0.02)
    assert res.loc["CP2", "stressed_pd"] == pytest.approx(0.05)


def test_el_attribution_sums_to_incremental_el(tiny_df, risk_df):
    for frame in (run_risk_engine(tiny_df), risk_df):
        for scenario in ALL_SCENARIOS.values():
            res = apply_scenario(frame, scenario)
            attribution = res["el_effect_pd"] + res["el_effect_lgd"] + res["el_effect_ead"]
            np.testing.assert_allclose(attribution, res["incremental_expected_loss"], atol=1e-9)


def test_apply_scenario_requires_risk_engine_output(tiny_df):
    with pytest.raises(ValueError, match="risk_engine"):
        apply_scenario(tiny_df, DEFAULT_SCENARIOS["RECESSION"])


# ------------------------------ portfolio level ----------------------------- #
def test_scenario_severity_ordering(risk_df):
    _, summary = run_scenarios(risk_df)
    el = summary.set_index("scenario")["stressed_expected_loss"]
    assert el["BASE"] < el["RECESSION"] < el["SEVERE_RECESSION"]
    assert el["BASE"] < el["MARKET_SHOCK"]


def test_summary_columns_are_consistent(risk_df):
    results, summary = run_scenarios(risk_df)
    for _, row in summary.iterrows():
        assert row["incremental_expected_loss"] == pytest.approx(row["stressed_expected_loss"] - row["base_expected_loss"])
        assert row["expected_loss_pct_change"] == pytest.approx(row["stressed_expected_loss"] / row["base_expected_loss"] - 1.0)
        assert row["incremental_ead"] == pytest.approx(row["stressed_ead"] - row["base_ead"])
        assert row["el_effect_pd"] + row["el_effect_lgd"] + row["el_effect_ead"] == pytest.approx(row["incremental_expected_loss"])
    assert len(results) == len(risk_df) * len(DEFAULT_SCENARIOS)
    base = summary.set_index("scenario").loc["BASE"]
    assert base["incremental_expected_loss"] == pytest.approx(0.0, abs=1e-9)


def test_run_scenarios_accepts_custom_scenario_dict(risk_df):
    _, summary = run_scenarios(risk_df, {"PD_PLUS_50": pd_shock_scenario(50, "PD_PLUS_50")})
    row = summary.iloc[0]
    # PD-only shock: LGD/EAD effects are zero and EL rises by exactly +50% unless PDs cap at 1
    assert row["el_effect_lgd"] == pytest.approx(0.0, abs=1e-9)
    assert row["el_effect_ead"] == pytest.approx(0.0, abs=1e-9)
    assert row["expected_loss_pct_change"] == pytest.approx(0.5, rel=1e-6)


def test_aggregate_stress_reconciles(risk_df):
    stress = apply_scenario(risk_df, DEFAULT_SCENARIOS["SEVERE_RECESSION"])
    for dim in ("industry", "country", "rating", "product_type"):
        agg = aggregate_stress(stress, dim)
        assert agg["incremental_expected_loss"].sum() == pytest.approx(stress["incremental_expected_loss"].sum()), dim
        assert agg["stressed_ead"].sum() == pytest.approx(stress["stressed_ead"].sum()), dim
        assert agg["share_of_incremental_el"].sum() == pytest.approx(1.0), dim


def test_summarise_scenario_returns_scenario_name(risk_df):
    stress = apply_scenario(risk_df, DEFAULT_SCENARIOS["RECESSION"])
    assert summarise_scenario(stress)["scenario"] == "RECESSION"


# --------------------------------- themes ----------------------------------- #
def test_theme_names_cover_the_brief():
    assert set(THEMES) == {
        "ENERGY_SHOCK", "FX_SHOCK", "INTEREST_RATE_SHOCK",
        "REAL_ESTATE_DOWNTURN", "EMEA_RECESSION", "FINANCIAL_SECTOR_STRESS",
    }


def test_theme_hits_intended_industry_hardest(risk_df):
    cases = {
        "REAL_ESTATE_DOWNTURN": "Real Estate",
        "ENERGY_SHOCK": "Energy",
        "FINANCIAL_SECTOR_STRESS": "Financial Services",
    }
    for key, industry in cases.items():
        top = analyze_theme(risk_df, THEMES[key])["top_industries"]
        pct = top.set_index("industry")["expected_loss_pct_change"]
        assert pct.idxmax() == industry, key


def test_fx_shock_increases_fx_heavy_exposure_most(risk_df):
    result = analyze_theme(risk_df, THEMES["FX_SHOCK"])
    by_product = aggregate_stress(result["results"], "product_type").set_index("product_type")
    assert by_product["stressed_ead"].div(by_product["base_ead"]).idxmax() == "FX"


def test_analyze_theme_structure(risk_df):
    out = analyze_theme(risk_df, THEMES["EMEA_RECESSION"], top_n=5)
    assert set(out) >= {"summary", "top_industries", "top_countries", "top_counterparties", "results"}
    assert len(out["top_counterparties"]) == 5
    assert out["top_counterparties"]["incremental_expected_loss"].is_monotonic_decreasing
    assert out["summary"]["incremental_expected_loss"] > 0


def test_industry_sensitivity_matrix_shape(risk_df):
    matrix = industry_sensitivity_matrix(risk_df, {k: ALL_SCENARIOS[k] for k in ("FX_SHOCK", "ENERGY_SHOCK")})
    assert list(matrix.columns) == ["FX_SHOCK", "ENERGY_SHOCK"]
    assert len(matrix) == 10
    assert isinstance(matrix, pd.DataFrame)
