"""Model validation & explainability helpers for the PD model (PART 10).

*** EDUCATIONAL - NOT A REGULATORY MODEL-VALIDATION FRAMEWORK ***
This module demonstrates the *kinds* of checks an independent model-validation
function would run (discrimination, calibration, feature importance, stability)
using standard open-source tooling (scikit-learn, SHAP). It is not a
replacement for a bank's Model Risk Management (MRM) policy, which would also
cover: independent replication, conceptual soundness review, sensitivity/
benchmarking analysis, outcomes analysis against realised defaults over
multiple years, ongoing monitoring with defined escalation, and governance
sign-off (e.g. SR 11-7 / EBA GL 2017/16-style processes).

Limitations of the analysis in this project (documented explicitly, see
README "Limitations"):
  * Single synthetic snapshot, no true out-of-time validation window.
  * default_flag was generated from the same `pd` anchors used as a model
    feature route (via rating/financial ratios), so discrimination here is
    expected to look better than a real-world PD model's.
  * No independent replication or challenger-model benchmarking.
  * Population Stability Index below compares two *random* portfolio splits
    as a stand-in for "development vs current" populations, since only one
    time period of data exists.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

try:
    import shap
    _HAS_SHAP = True
except ImportError:  # pragma: no cover
    _HAS_SHAP = False
    logger.warning("shap not installed; SHAP-based explanations will be skipped.")


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #


def get_feature_names(pipeline) -> list[str]:
    """Recover post-encoding feature names from a fitted sklearn Pipeline."""
    prep = pipeline.named_steps["prep"]
    return list(prep.get_feature_names_out())


def tree_feature_importance(pipeline) -> pd.DataFrame:
    """Feature importance for a tree-based estimator (Random Forest / LightGBM / GB).

    Returns an empty frame if the fitted estimator has no ``feature_importances_``
    (e.g. Logistic Regression -- use ``logistic_coefficients`` instead).
    """
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    names = get_feature_names(pipeline)
    imp = clf.feature_importances_
    out = pd.DataFrame({"feature": names, "importance": imp})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def logistic_coefficients(pipeline) -> pd.DataFrame:
    """Standardised coefficients for a fitted Logistic Regression pipeline."""
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return pd.DataFrame(columns=["feature", "coefficient"])
    names = get_feature_names(pipeline)
    out = pd.DataFrame({"feature": names, "coefficient": clf.coef_[0]})
    out["abs_coefficient"] = out["coefficient"].abs()
    return out.sort_values("abs_coefficient", ascending=False).drop(columns="abs_coefficient").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #


def compute_shap_values(pipeline, X: pd.DataFrame, max_rows: int = 500):
    """Compute SHAP values for a fitted pipeline's tree-based classifier.

    Uses ``shap.TreeExplainer`` for tree models (fast, exact). For non-tree
    models (e.g. Logistic Regression) returns ``None`` -- use
    ``logistic_coefficients`` for that model's explainability instead.
    Sub-samples to ``max_rows`` for speed on larger portfolios.
    """
    if not _HAS_SHAP:
        return None
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return None  # not a tree model

    X_sample = X.sample(min(max_rows, len(X)), random_state=42)
    X_transformed = pipeline.named_steps["prep"].transform(X_sample)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):  # binary classifier returns [class0, class1]
        shap_values = shap_values[1]
    names = get_feature_names(pipeline)
    return {
        "shap_values": shap_values,
        "X_transformed": pd.DataFrame(X_transformed, columns=names),
        "feature_names": names,
    }


def shap_summary_table(shap_result: dict | None) -> pd.DataFrame:
    """Mean absolute SHAP value per feature -- a global importance ranking."""
    if shap_result is None:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    mean_abs = np.abs(shap_result["shap_values"]).mean(axis=0)
    out = pd.DataFrame({"feature": shap_result["feature_names"], "mean_abs_shap": mean_abs})
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def explain_single_counterparty(shap_result: dict | None, row_index: int) -> pd.DataFrame:
    """SHAP contribution breakdown for one row of the SHAP sample (local explanation)."""
    if shap_result is None:
        return pd.DataFrame(columns=["feature", "shap_value", "feature_value"])
    values = shap_result["shap_values"][row_index]
    feat_vals = shap_result["X_transformed"].iloc[row_index]
    out = pd.DataFrame({
        "feature": shap_result["feature_names"],
        "shap_value": values,
        "feature_value": feat_vals.values,
    })
    out["abs_shap"] = out["shap_value"].abs()
    return out.sort_values("abs_shap", ascending=False).drop(columns="abs_shap").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stability: Population Stability Index (PSI) and prediction-stability check
# --------------------------------------------------------------------------- #


def population_stability_index(
    expected: pd.Series, actual: pd.Series, n_bins: int = 10
) -> tuple[float, pd.DataFrame]:
    """PSI between two distributions of a score/probability (e.g. predicted PD).

    PSI = sum[(actual_pct - expected_pct) * ln(actual_pct / expected_pct)]
    Rule-of-thumb interpretation (illustrative, industry-common):
        PSI < 0.10           : no significant shift
        0.10 <= PSI < 0.25    : moderate shift, investigate
        PSI >= 0.25           : significant shift, model may need review
    """
    edges = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0, pd.DataFrame()

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)
    exp_pct = np.where(exp_counts == 0, 1e-4, exp_counts / exp_counts.sum())
    act_pct = np.where(act_counts == 0, 1e-4, act_counts / act_counts.sum())

    psi_per_bin = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    table = pd.DataFrame({
        "bin_upper_edge": edges[1:],
        "expected_pct": exp_pct,
        "actual_pct": act_pct,
        "psi_contribution": psi_per_bin,
    })
    return float(psi_per_bin.sum()), table


def train_test_prediction_psi(result, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    """PSI between two random halves of the test-set predicted probabilities.

    Stand-in for "development sample vs current portfolio" drift monitoring,
    since this project has only one time period of data (see module docstring).
    """
    proba = pd.Series(result.y_proba)
    half = len(proba) // 2
    return population_stability_index(proba.iloc[:half], proba.iloc[half:], n_bins=n_bins)


def validation_notes() -> dict[str, list[str]]:
    """Static text used by the dashboard / README to describe what is (and is not) covered."""
    return {
        "covered_in_this_project": [
            "Train/test split with stratification on the rare default class",
            "Discrimination: ROC-AUC, average precision",
            "Threshold-based metrics: precision, recall, F1, confusion matrix",
            "Calibration: reliability table (predicted vs observed default rate by bin)",
            "Global feature importance (tree impurity-based and/or SHAP mean |value|)",
            "Local explanations for individual counterparties (SHAP)",
            "A simple Population Stability Index (PSI) demonstration",
        ],
        "required_for_a_production_pd_model_but_not_done_here": [
            "Multi-year, through-the-cycle or point-in-time default history",
            "Out-of-time and out-of-sample (different vintage) validation",
            "Independent model validation function, replication and challenger models",
            "Formal PSI/CSI monitoring on a live scoring population over time",
            "Low-default-portfolio calibration techniques where defaults are rare",
            "Rating philosophy documentation (PIT vs TTC), master scale mapping",
            "Governance: model risk sign-off, periodic recalibration, use-test evidence",
        ],
    }
