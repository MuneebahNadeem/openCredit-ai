"""
Explainability — SHAP-based feature importance for the risk pipeline.

Uses SHAP's KernelExplainer to approximate per-feature contributions to the
trustworthiness and business potential scores produced by the risk engine.
Because the risk engine is a deterministic function (no trained model needed),
KernelExplainer can wrap it directly and produce valid Shapley values.

Key types
---------
FeatureImportance
    One feature's name, its SHAP value (signed contribution to the score),
    and its actual value in this investigation.

ExplainabilityReport
    All feature importances for one assessment (trust or potential), plus
    the baseline score (expected value) and the final predicted score.

Usage::

    from ml.explainability import explain_assessment

    report = explain_assessment(investigation_result)
    for fi in report.trustworthiness[:5]:          # top-5 trust drivers
        print(fi.feature_name, fi.shap_value)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List

import numpy as np
import shap

from agent.schemas.result import InvestigationResult
from ml.feature_extractor import extract_features
from ml.risk_engine import _score_business_potential, _score_trustworthiness
from ml.credibility_scorer import score_credibility
from ml.sentiment import score_evidence_texts


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class FeatureImportance:
    """SHAP contribution of a single feature to one assessment score."""

    feature_name: str
    shap_value: float       # signed contribution; positive = raises score
    feature_value: float    # actual value of this feature in the investigation

    def __repr__(self) -> str:
        sign = "+" if self.shap_value >= 0 else ""
        return (
            f"FeatureImportance({self.feature_name!r}, "
            f"shap={sign}{self.shap_value:.4f}, "
            f"value={self.feature_value:.4f})"
        )


@dataclass
class ExplainabilityReport:
    """
    SHAP explainability report for one InvestigationResult.

    Attributes
    ----------
    trustworthiness:
        Feature importances for the trustworthiness score, sorted by
        absolute SHAP value descending (biggest driver first).
    business_potential:
        Feature importances for the business potential score, same order.
    trust_baseline:
        Expected (mean) trustworthiness score over the background dataset.
    potential_baseline:
        Expected (mean) business potential score over the background dataset.
    trust_predicted:
        Actual trustworthiness score for this investigation.
    potential_predicted:
        Actual business potential score for this investigation.
    """

    trustworthiness: List[FeatureImportance]
    business_potential: List[FeatureImportance]
    trust_baseline: float
    potential_baseline: float
    trust_predicted: float
    potential_predicted: float

    def top_trust_drivers(self, n: int = 5) -> List[FeatureImportance]:
        """Return the top-n features driving the trustworthiness score."""
        return self.trustworthiness[:n]

    def top_potential_drivers(self, n: int = 5) -> List[FeatureImportance]:
        """Return the top-n features driving the business potential score."""
        return self.business_potential[:n]


# ── Background dataset ───────────────────────────────────────────────────────

# A small synthetic background used by KernelExplainer as the reference
# distribution.  Each row is a plausible feature vector (all 44 features).
# These represent a "neutral" investigation with mid-range values.
_BACKGROUND_SIZE = 20


def _make_background(feature_names: List[str]) -> np.ndarray:
    """
    Build a small background dataset for SHAP KernelExplainer.

    Each row is a synthetic neutral investigation.  The values are chosen
    to be mid-range and realistic so the baseline score represents an
    "average" investigation rather than an empty or perfect one.
    """
    rng = np.random.default_rng(seed=42)
    n_features = len(feature_names)
    rows = []

    for _ in range(_BACKGROUND_SIZE):
        row = np.zeros(n_features)
        for i, name in enumerate(feature_names):
            if name.endswith("_ratio"):
                row[i] = rng.uniform(0.3, 0.7)
            elif name.endswith("_count") or name.endswith("_total"):
                row[i] = rng.uniform(2.0, 8.0)
            elif name.startswith("status_"):
                # one-hot: randomly pick one status flag as 1
                row[i] = 0.0
            elif name.startswith("confidence_"):
                row[i] = rng.uniform(0.4, 0.8)
            elif name.startswith("features_cat_"):
                row[i] = rng.uniform(0.0, 2.0)
            elif name in ("searches_performed", "sources_examined",
                          "unique_sources_count"):
                row[i] = rng.uniform(3.0, 10.0)
            else:
                row[i] = rng.uniform(0.2, 0.8)
        # Ensure status_complete = 1 for exactly one flag
        for status in ("status_complete", "status_limit_reached",
                       "status_partial", "status_failed"):
            if status in feature_names:
                row[feature_names.index(status)] = 0.0
        if "status_complete" in feature_names:
            row[feature_names.index("status_complete")] = 1.0

        rows.append(row)

    return np.array(rows, dtype=np.float64)


# ── Prediction functions wrapped for SHAP ────────────────────────────────────

def _make_trust_predictor(feature_names, sentiment, credibility):
    """Return a function (np.ndarray → np.ndarray) for trustworthiness."""

    def _predict(X: np.ndarray) -> np.ndarray:
        results = []
        for row in X:
            feat = dict(zip(feature_names, row.tolist()))
            score = _score_trustworthiness(feat, sentiment, credibility)
            results.append(score)
        return np.array(results)

    return _predict


def _make_potential_predictor(feature_names, sentiment, credibility):
    """Return a function (np.ndarray → np.ndarray) for business potential."""

    def _predict(X: np.ndarray) -> np.ndarray:
        results = []
        for row in X:
            feat = dict(zip(feature_names, row.tolist()))
            score = _score_business_potential(feat, sentiment, credibility)
            results.append(score)
        return np.array(results)

    return _predict


# ── Core explainer ────────────────────────────────────────────────────────────

def explain_assessment(
    result: InvestigationResult,
    n_samples: int = 100,
) -> ExplainabilityReport:
    """
    Produce SHAP feature importances for a given InvestigationResult.

    Parameters
    ----------
    result:
        A populated InvestigationResult (evidence, features, signals present).
    n_samples:
        Number of samples KernelExplainer uses when approximating Shapley
        values.  Higher = more accurate but slower.  100 is a good default
        for this pipeline size.

    Returns
    -------
    ExplainabilityReport
        Two sorted lists of FeatureImportance (trust + potential) plus
        baseline and predicted scores.
    """
    # ── Extract features for this result ─────────────────────────────────
    feat_dict = extract_features(result)
    feature_names = list(feat_dict.keys())
    feature_values = np.array(list(feat_dict.values()), dtype=np.float64)

    # ── Run sentiment + credibility (shared by both scorers) ─────────────
    sentiment = score_evidence_texts(result.evidence)
    credibility = score_credibility(result)

    # ── Build background dataset ──────────────────────────────────────────
    background = _make_background(feature_names)

    # ── Trustworthiness SHAP ──────────────────────────────────────────────
    trust_fn = _make_trust_predictor(feature_names, sentiment, credibility)
    trust_predicted = float(trust_fn(feature_values.reshape(1, -1))[0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trust_explainer = shap.KernelExplainer(trust_fn, background)
        trust_shap = trust_explainer.shap_values(
            feature_values, nsamples=n_samples, silent=True
        )

    trust_baseline = float(trust_explainer.expected_value)
    trust_importances = _build_importances(
        feature_names, feature_values, trust_shap
    )

    # ── Business potential SHAP ───────────────────────────────────────────
    potential_fn = _make_potential_predictor(feature_names, sentiment, credibility)
    potential_predicted = float(potential_fn(feature_values.reshape(1, -1))[0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        potential_explainer = shap.KernelExplainer(potential_fn, background)
        potential_shap = potential_explainer.shap_values(
            feature_values, nsamples=n_samples, silent=True
        )

    potential_baseline = float(potential_explainer.expected_value)
    potential_importances = _build_importances(
        feature_names, feature_values, potential_shap
    )

    return ExplainabilityReport(
        trustworthiness=trust_importances,
        business_potential=potential_importances,
        trust_baseline=trust_baseline,
        potential_baseline=potential_baseline,
        trust_predicted=trust_predicted,
        potential_predicted=potential_predicted,
    )


def _build_importances(
    feature_names: List[str],
    feature_values: np.ndarray,
    shap_values: np.ndarray,
) -> List[FeatureImportance]:
    """Convert raw SHAP output into a sorted list of FeatureImportance."""
    importances = [
        FeatureImportance(
            feature_name=name,
            shap_value=float(sv),
            feature_value=float(fv),
        )
        for name, sv, fv in zip(feature_names, shap_values, feature_values)
    ]
    # Sort by absolute contribution descending — biggest driver first.
    importances.sort(key=lambda fi: abs(fi.shap_value), reverse=True)
    return importances


# ── Convenience: summary string ───────────────────────────────────────────────

def format_report(report: ExplainabilityReport, top_n: int = 5) -> str:
    """
    Return a human-readable summary of the top drivers in each assessment.

    Useful for logging or displaying in the backend.
    """
    lines = [
        f"Trustworthiness: predicted={report.trust_predicted:.4f}  "
        f"baseline={report.trust_baseline:.4f}",
        "  Top drivers:",
    ]
    for fi in report.top_trust_drivers(top_n):
        sign = "+" if fi.shap_value >= 0 else ""
        lines.append(
            f"    {fi.feature_name:<40} {sign}{fi.shap_value:.4f}"
        )

    lines += [
        "",
        f"Business Potential: predicted={report.potential_predicted:.4f}  "
        f"baseline={report.potential_baseline:.4f}",
        "  Top drivers:",
    ]
    for fi in report.top_potential_drivers(top_n):
        sign = "+" if fi.shap_value >= 0 else ""
        lines.append(
            f"    {fi.feature_name:<40} {sign}{fi.shap_value:.4f}"
        )

    return "\n".join(lines)
