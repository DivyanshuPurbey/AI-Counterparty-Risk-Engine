"""Probability-of-Default (PD) machine-learning model.

*** EDUCATIONAL MODEL - NOT A REGULATORY (IRB/CECL) PD MODEL ***
This trains a classifier on the project's synthetic ``default_flag`` using a
handful of standard credit-analytics features. It is meant to demonstrate an
end-to-end modelling workflow (train/test split, multiple algorithms,
standard classification metrics, calibration) — not to reproduce a bank's
internal-ratings-based (IRB) PD estimation, which would require far longer
default histories, through-the-cycle/point-in-time calibration, low-default-
portfolio techniques, and independent model validation.

The synthetic ``default_flag`` was generated from ``pd`` (see
``data_generator.py``), so a well-specified model recovers the generating
signal reasonably well. This is expected and is called out explicitly in the
README as a limitation: performance here is a demonstration of workflow, not
evidence of real-world predictive power.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:  # pragma: no cover - environment without lightgbm
    _HAS_LGBM = False
    logger.warning("lightgbm not installed; falling back to GradientBoostingClassifier.")
    from sklearn.ensemble import GradientBoostingClassifier

NUMERIC_FEATURES: list[str] = [
    "debt_to_ebitda", "interest_coverage", "current_ratio", "cash_to_debt",
    "annual_revenue", "ebitda_margin", "current_exposure", "collateral",
]
CATEGORICAL_FEATURES: list[str] = ["rating", "industry"]
ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET: str = "default_flag"

RANDOM_STATE: int = 42


@dataclass
class ModelResult:
    """Container for a fitted model plus its evaluation on the held-out test set."""

    name: str
    pipeline: Pipeline
    metrics: dict[str, float]
    y_test: np.ndarray
    y_proba: np.ndarray
    fpr: np.ndarray = field(default_factory=lambda: np.array([]))
    tpr: np.ndarray = field(default_factory=lambda: np.array([]))


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select model features and target, dropping rows with missing values."""
    work = df[ALL_FEATURES + [TARGET]].dropna()
    return work[ALL_FEATURES], work[TARGET].astype(int)


def train_test_split_stratified(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.25, random_state: int = RANDOM_STATE
):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "default_rate_test": float(np.mean(y_true)),
    }


def _fit_one(
    name: str, estimator, X_train, X_test, y_train, y_test, threshold: float = 0.5
) -> ModelResult:
    pipe = Pipeline([("prep", _build_preprocessor()), ("clf", estimator)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = _evaluate(y_test.to_numpy(), pred, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    logger.info("%s: ROC-AUC=%.3f  F1=%.3f", name, metrics["roc_auc"], metrics["f1"])
    return ModelResult(name=name, pipeline=pipe, metrics=metrics, y_test=y_test.to_numpy(),
                        y_proba=proba, fpr=fpr, tpr=tpr)


def train_all_models(
    df: pd.DataFrame, test_size: float = 0.25, random_state: int = RANDOM_STATE
) -> dict[str, ModelResult]:
    """Train Logistic Regression, Random Forest and LightGBM (or GB fallback).

    Class imbalance (defaults are rare) is handled with ``class_weight="balanced"``
    for logistic regression / random forest, and ``scale_pos_weight`` for LightGBM,
    rather than oversampling, to keep the workflow simple and interpretable.
    """
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size, random_state)

    results: dict[str, ModelResult] = {}

    results["logistic_regression"] = _fit_one(
        "Logistic Regression",
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        X_train, X_test, y_train, y_test,
    )

    results["random_forest"] = _fit_one(
        "Random Forest",
        RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        ),
        X_train, X_test, y_train, y_test,
    )

    if _HAS_LGBM:
        pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        results["lightgbm"] = _fit_one(
            "LightGBM",
            LGBMClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                scale_pos_weight=pos_weight, random_state=random_state, verbosity=-1,
            ),
            X_train, X_test, y_train, y_test,
        )
    else:
        results["gradient_boosting"] = _fit_one(
            "Gradient Boosting",
            GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=random_state),
            X_train, X_test, y_train, y_test,
        )

    return results


def metrics_table(results: dict[str, ModelResult]) -> pd.DataFrame:
    rows = [{"model": r.name, **r.metrics} for r in results.values()]
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def confusion_matrices(results: dict[str, ModelResult]) -> dict[str, np.ndarray]:
    return {
        key: confusion_matrix(r.y_test, (r.y_proba >= 0.5).astype(int))
        for key, r in results.items()
    }


def calibration_table(result: ModelResult, n_bins: int = 10) -> pd.DataFrame:
    """Observed vs predicted default rate by predicted-probability bin (reliability table)."""
    frac_pos, mean_pred = calibration_curve(result.y_test, result.y_proba, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted_pd": mean_pred, "observed_default_rate": frac_pos})


def best_model(results: dict[str, ModelResult]) -> ModelResult:
    """Select the model with the highest ROC-AUC on the held-out test set."""
    return max(results.values(), key=lambda r: r.metrics["roc_auc"])
