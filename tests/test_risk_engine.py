"""Unit tests for PFE, EAD, Expected Loss and aggregation logic."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.risk_engine import (
    ExposureParams,
    aggregate_risk,
    calculate_ead,
    calculate_eligible_collateral,
    calculate_expected_loss,
    calculate_pfe,
    expected_loss_by_dimension,
    maturity_factor,
    portfolio_summary,
    run_risk_engine,
)


# ----------------------------- maturity factor ----------------------------- #
def test_maturity_factor_is_square_root_of_time():
    assert maturity_factor(1.0) == pytest.approx(1.0)
    assert maturity_factor(4.0) == pytest.approx(2.0)
    assert maturity_factor(9.0) == pytest.approx(3.0)


def test_maturity_factor_floor_and_cap():
    assert maturity_factor(0.001) == pytest.approx(math.sqrt(cfg.MIN_MATURITY_YEARS))
    assert maturity_factor(100.0) == pytest.approx(math.sqrt(cfg.MAX_MATURITY_YEARS))


def test_maturity_factor_strictly_increasing_inside_bounds():
    grid = np.linspace(0.05, 24.0, 60)
    assert np.all(np.diff(maturity_factor(grid)) > 0)


# ---------------------------------- PFE ------------------------------------ #
def test_pfe_formula_hand_calculation():
    # 100 notional x 1% add-on x sqrt(4y) = 2.0
    assert calculate_pfe(100.0, 0.01, maturity_factor(4.0)) == pytest.approx(2.0)


def test_pfe_scales_linearly_with_notional_and_addon():
    base = calculate_pfe(100.0, 0.02, 1.5)
    assert calculate_pfe(200.0, 0.02, 1.5) == pytest.approx(2 * base)
    assert calculate_pfe(100.0, 0.04, 1.5) == pytest.approx(2 * base)


def test_longer_maturity_gives_higher_pfe_for_same_trade():
    short = calculate_pfe(100.0, 0.04, maturity_factor(0.5))
    long = calculate_pfe(100.0, 0.04, maturity_factor(5.0))
    assert long > short


def test_pfe_is_product_dependent(tiny_df):
    out = run_risk_engine(tiny_df)
    assert out.loc[0, "addon_factor"] == pytest.approx(cfg.PRODUCT_ADDON_FACTORS["Interest Rate"])
    assert out.loc[1, "addon_factor"] == pytest.approx(cfg.PRODUCT_ADDON_FACTORS["FX"])
    assert out.loc[2, "addon_factor"] == pytest.approx(cfg.PRODUCT_ADDON_FACTORS["Equity"])


# ---------------------------------- EAD ------------------------------------ #
def test_eligible_collateral_applies_haircut():
    assert calculate_eligible_collateral(100.0, 0.10) == pytest.approx(90.0)
    assert calculate_eligible_collateral(100.0, 0.0) == pytest.approx(100.0)


def test_ead_formula_hand_calculation():
    assert calculate_ead(10.0, 5.0, 3.0) == pytest.approx(12.0)


def test_ead_floored_at_zero_when_over_collateralised():
    assert calculate_ead(1.0, 1.0, 10.0) == pytest.approx(0.0)


def test_ead_vectorised_over_series():
    ead = calculate_ead(pd.Series([10.0, 1.0]), pd.Series([5.0, 1.0]), pd.Series([3.0, 10.0]))
    np.testing.assert_allclose(ead.to_numpy(), [12.0, 0.0])


def test_more_collateral_never_increases_ead():
    collateral = np.linspace(0, 30, 31)
    ead = calculate_ead(10.0, 5.0, collateral)
    assert np.all(np.diff(ead) <= 1e-12)


# ------------------------------ expected loss ------------------------------ #
def test_expected_loss_formula_hand_calculation():
    assert calculate_expected_loss(0.02, 0.45, 1000.0) == pytest.approx(9.0)


def test_expected_loss_zero_when_any_driver_is_zero():
    assert calculate_expected_loss(0.0, 0.45, 100.0) == 0.0
    assert calculate_expected_loss(0.02, 0.0, 100.0) == 0.0
    assert calculate_expected_loss(0.02, 0.45, 0.0) == 0.0


# ------------------------------ full engine -------------------------------- #
def test_run_risk_engine_matches_hand_calculations(tiny_df):
    out = run_risk_engine(tiny_df).set_index("counterparty_id")

    # CP1: PFE = 100*0.01*sqrt(4)=2.0 ; eligible = 6*0.90=5.4 ; EAD = 10+2-5.4
    assert out.loc["CP1", "pfe"] == pytest.approx(2.0)
    assert out.loc["CP1", "eligible_collateral"] == pytest.approx(5.4)
    assert out.loc["CP1", "ead"] == pytest.approx(6.6)
    assert out.loc["CP1", "expected_loss"] == pytest.approx(0.02 * 0.45 * 6.6)
    assert out.loc["CP1", "unsecured_exposure"] == pytest.approx(4.6)
    assert out.loc["CP1", "collateral_coverage"] == pytest.approx(0.54)

    # CP2: PFE = 50*0.04*1 = 2.0 ; no collateral ; EAD = 4+2
    assert out.loc["CP2", "ead"] == pytest.approx(6.0)
    assert out.loc["CP2", "expected_loss"] == pytest.approx(0.05 * 0.50 * 6.0)

    # CP3: PFE = 20*0.12*sqrt(2.25)=3.6 ; cash 3.0 ; EAD = 0+3.6-3.0
    assert out.loc["CP3", "pfe"] == pytest.approx(3.6)
    assert out.loc["CP3", "ead"] == pytest.approx(0.6)
    assert math.isnan(out.loc["CP3", "collateral_coverage"])  # undefined when CE = 0


def test_run_risk_engine_does_not_mutate_input(tiny_df):
    before = tiny_df.copy()
    run_risk_engine(tiny_df)
    pd.testing.assert_frame_equal(tiny_df, before)


def test_higher_collateral_reduces_ead_in_engine(tiny_df):
    base = run_risk_engine(tiny_df)
    more = tiny_df.assign(collateral=tiny_df["collateral"] + 2.0, collateral_type="Cash")
    assert (run_risk_engine(more)["ead"] <= base["ead"] + 1e-12).all()
    assert run_risk_engine(more).loc[0, "ead"] < base.loc[0, "ead"]


def test_higher_haircut_reduces_eligible_collateral(tiny_df):
    low = run_risk_engine(tiny_df, ExposureParams(collateral_haircuts={**cfg.COLLATERAL_HAIRCUTS, "Corporate Bonds": 0.05}))
    high = run_risk_engine(tiny_df, ExposureParams(collateral_haircuts={**cfg.COLLATERAL_HAIRCUTS, "Corporate Bonds": 0.30}))
    assert high.loc[0, "eligible_collateral"] < low.loc[0, "eligible_collateral"]
    assert high.loc[0, "ead"] > low.loc[0, "ead"]


def test_custom_addon_factors_change_pfe(tiny_df):
    params = ExposureParams(addon_factors={**cfg.PRODUCT_ADDON_FACTORS, "FX": 0.08})
    assert run_risk_engine(tiny_df, params).loc[1, "pfe"] == pytest.approx(4.0)


def test_unknown_product_type_raises(tiny_df):
    bad = tiny_df.assign(product_type=["Interest Rate", "Weather Derivative", "FX"])
    with pytest.raises(ValueError, match="Unknown product_type"):
        run_risk_engine(bad)


def test_unknown_collateral_type_raises(tiny_df):
    bad = tiny_df.assign(collateral_type=["Cash", "Gold", "None"])
    with pytest.raises(ValueError, match="Unknown collateral_type"):
        run_risk_engine(bad)


def test_missing_columns_raise(tiny_df):
    with pytest.raises(ValueError, match="missing required columns"):
        run_risk_engine(tiny_df.drop(columns=["pd"]))


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        ExposureParams(min_maturity_years=0.0)
    with pytest.raises(ValueError):
        ExposureParams(collateral_haircuts={"Cash": 1.5})
    with pytest.raises(ValueError):
        ExposureParams(addon_factors={"FX": -0.01})


# ---------------------- properties on the synthetic book ------------------- #
def test_engine_outputs_are_internally_consistent(risk_df):
    assert (risk_df["pfe"] >= 0).all()
    assert (risk_df["ead"] >= 0).all()
    assert (risk_df["expected_loss"] >= 0).all()
    assert (risk_df["expected_loss"] <= risk_df["ead"] + 1e-12).all()  # PD x LGD <= 1
    np.testing.assert_allclose(
        risk_df["ead"], np.maximum(risk_df["current_exposure"] + risk_df["pfe"] - risk_df["eligible_collateral"], 0.0)
    )
    np.testing.assert_allclose(risk_df["expected_loss"], risk_df["pd"] * risk_df["lgd"] * risk_df["ead"])


def test_collateralised_names_have_lower_ead_ratio(risk_df):
    gross = risk_df["current_exposure"] + risk_df["pfe"]
    ratio = risk_df["ead"] / gross
    assert ratio[risk_df["collateral"] > 0].mean() < ratio[risk_df["collateral"] == 0].mean()


def test_pfe_ratio_rises_with_maturity_within_each_product(risk_df):
    for product, grp in risk_df.groupby("product_type"):
        long = grp["maturity_years"] > grp["maturity_years"].median()
        ratio = grp["pfe"] / grp["notional"]
        assert ratio[long].mean() > ratio[~long].mean(), product


# ------------------------------- aggregation -------------------------------- #
def test_aggregations_reconcile_to_portfolio_totals(risk_df):
    for dim, table in expected_loss_by_dimension(risk_df).items():
        assert table["expected_loss"].sum() == pytest.approx(risk_df["expected_loss"].sum()), dim
        assert table["ead"].sum() == pytest.approx(risk_df["ead"].sum()), dim
        assert table["ead_share"].sum() == pytest.approx(1.0), dim
        assert table["n_counterparties"].sum() == len(risk_df), dim


def test_rating_aggregation_follows_rating_scale(risk_df):
    assert aggregate_risk(risk_df, "rating")["rating"].tolist() == cfg.RATINGS


def test_expected_loss_rate_rises_with_worse_rating(risk_df):
    bps = aggregate_risk(risk_df, "rating")["el_to_ead_bps"].to_numpy()
    assert np.all(np.diff(bps) > 0)


def test_portfolio_summary_matches_column_sums(risk_df):
    summary = portfolio_summary(risk_df)
    assert summary["total_ead"] == pytest.approx(risk_df["ead"].sum())
    assert summary["total_expected_loss"] == pytest.approx(risk_df["expected_loss"].sum())
    assert summary["n_counterparties"] == len(risk_df)
    assert summary["average_pd"] == pytest.approx(risk_df["pd"].mean())
