"""Shared fixtures for the test-suite."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_generator import generate_counterparties
from src.preprocessing import build_feature_table
from src.risk_engine import run_risk_engine


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """A 3,000-name synthetic portfolio (fixed seed)."""
    return generate_counterparties(3000, seed=123)


@pytest.fixture(scope="session")
def features_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return build_feature_table(raw_df)


@pytest.fixture(scope="session")
def risk_df(features_df: pd.DataFrame) -> pd.DataFrame:
    return run_risk_engine(features_df)


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """Three hand-built counterparties whose metrics are checked by hand.

    CP1: IR swap, BBB energy, part-collateralised with corporate bonds (10% haircut)
    CP2: FX forward, BB technology, uncollateralised
    CP3: Equity trade, CCC real estate, out-of-the-money but holds cash collateral
    """
    return pd.DataFrame(
        {
            "counterparty_id": ["CP1", "CP2", "CP3"],
            "counterparty_name": ["Alpha Energy PLC", "Beta Tech AG", "Gamma Estates S.p.A."],
            "industry": ["Energy", "Technology", "Real Estate"],
            "country": ["UK", "Germany", "Italy"],
            "rating": ["BBB", "BB", "CCC"],
            "product_type": ["Interest Rate", "FX", "Equity"],
            "notional": [100.0, 50.0, 20.0],
            "maturity_years": [4.0, 1.0, 2.25],
            "current_exposure": [10.0, 4.0, 0.0],
            "fx_exposure": [0.0, 4.0, 0.0],
            "interest_rate_exposure": [8.0, 0.0, 0.0],
            "collateral": [6.0, 0.0, 3.0],
            "collateral_type": ["Corporate Bonds", "None", "Cash"],
            "pd": [0.02, 0.05, 0.30],
            "lgd": [0.45, 0.50, 0.60],
        }
    )
