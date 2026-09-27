"""Tests for src/pd_model.py — PD machine-learning model training and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pd_model import (
    ALL_FEATURES,
    TARGET,
    best_model,
    calibration_table,
    confusion_matrices,
    metrics_table,
    prepare_xy,
    train_all_models,
)


@pytest.fixture(scope="module")
def model_results(risk_df: pd.DataFrame):
    return train_all_models(risk_df, test_size=0.3, random_state=7)


def test_prepare_xy_drops_missing_and_selects_features(risk_df: pd.DataFrame):
    X, y = prepare_xy(risk_df)
    assert list(X.columns) == ALL_FEATURES
    assert y.name == TARGET
    assert X.isna().sum().sum() == 0
    assert set(y.unique()) <= {0, 1}


def test_train_all_models_returns_multiple_fitted_models(model_results):
    assert len(model_results) >= 2
    for result in model_results.values():
        assert 0.0 <= result.metrics["roc_auc"] <= 1.0
        assert 0.0 <= result.metrics["precision"] <= 1.0
        assert 0.0 <= result.metrics["recall"] <= 1.0


def test_models_beat_random_guessing(model_results):
    """With a synthetic default_flag derived from PD, models should clearly beat AUC=0.5."""
    for name, result in model_results.items():
        assert result.metrics["roc_auc"] > 0.6, f"{name} ROC-AUC too low: {result.metrics['roc_auc']}"


def test_metrics_table_sorted_by_roc_auc_desc(model_results):
    table = metrics_table(model_results)
    assert list(table["roc_auc"]) == sorted(table["roc_auc"], reverse=True)


def test_best_model_has_highest_auc(model_results):
    best = best_model(model_results)
    table = metrics_table(model_results)
    assert best.metrics["roc_auc"] == table["roc_auc"].max()


def test_confusion_matrices_shapes(model_results):
    matrices = confusion_matrices(model_results)
    for name, cm in matrices.items():
        assert cm.shape == (2, 2)
        # rows/cols sum to number of test-set observations
        n_test = len(next(iter(model_results.values())).y_test)
        assert cm.sum() == n_test


def test_calibration_table_has_expected_columns(model_results):
    best = best_model(model_results)
    table = calibration_table(best, n_bins=5)
    assert set(table.columns) == {"mean_predicted_pd", "observed_default_rate"}
    assert (table["mean_predicted_pd"] >= 0).all()
