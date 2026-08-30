"""
Tests for ml/explainability.py — SHAP-based feature importance.

Run with:  python -m pytest tests/ml/test_explainability.py -v

These tests use a small n_samples value (10) to keep execution fast while
still exercising the full SHAP pipeline.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from agent.schemas.input import BusinessInput
from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability
from agent.schemas.feature import DiscoveredFeature, FeatureCategory
from agent.schemas.result import (
    InvestigationResult,
    InvestigationStatus,
    Signal,
)
from ml.explainability import (
    ExplainabilityReport,
    FeatureImportance,
    explain_assessment,
    format_report,
    _make_background,
)
from ml.feature_extractor import extract_features


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_business() -> BusinessInput:
    return BusinessInput(name="Acme Store")


def make_evidence(
    field="revenue",
    value="50000",
    etype=EvidenceType.OBSERVED,
    reliability=SourceReliability.HIGH,
    confidence=0.85,
    source_name="official_site",
    raw_snippet="Annual revenue is approximately $50,000",
) -> EvidenceItem:
    return EvidenceItem(
        field_name=field,
        value=value,
        evidence_type=etype,
        source_name=source_name,
        source_reliability=reliability,
        confidence=confidence,
        raw_snippet=raw_snippet,
    )


def make_feature(
    name="instagram_followers",
    category=FeatureCategory.AUDIENCE,
    value="5000",
) -> DiscoveredFeature:
    return DiscoveredFeature(
        name=name,
        category=category,
        value=value,
        reason="Social media presence indicator",
        searched=True,
    )


def make_rich_result() -> InvestigationResult:
    """A well-populated InvestigationResult with evidence and features."""
    evidence = [
        make_evidence("revenue", "50000", EvidenceType.OBSERVED,
                      SourceReliability.HIGH, 0.9, "gov_registry",
                      "Revenue is strong"),
        make_evidence("reviews", "4.5 stars", EvidenceType.CORROBORATED,
                      SourceReliability.MEDIUM, 0.8, "review_platform",
                      "excellent product quality"),
        make_evidence("social_followers", "12000", EvidenceType.OBSERVED,
                      SourceReliability.MEDIUM, 0.75, "instagram",
                      "good engagement"),
        make_evidence("complaints", "none found", EvidenceType.OBSERVED,
                      SourceReliability.HIGH, 0.85, "consumer_board",
                      "no complaints found"),
        make_evidence("website_age", "5 years", EvidenceType.OBSERVED,
                      SourceReliability.HIGH, 0.9, "whois",
                      "established website"),
    ]
    features = [
        make_feature("instagram_followers", FeatureCategory.AUDIENCE, "12000"),
        make_feature("customer_reviews", FeatureCategory.REPUTATION, "4.5"),
        make_feature("monthly_revenue", FeatureCategory.DEMAND, "50000"),
    ]
    positive_signals = [
        Signal(label="Active social presence", detail="12k Instagram followers",
               evidence_refs=["social_followers"]),
        Signal(label="Strong reviews", detail="4.5-star average",
               evidence_refs=["reviews"]),
    ]
    return InvestigationResult(
        business_input=make_business(),
        status=InvestigationStatus.COMPLETE,
        searches_performed=8,
        sources_examined=5,
        evidence=evidence,
        features=features,
        positive_signals=positive_signals,
        risk_signals=[],
        missing_information=[],
        sources=["https://example.com"],
        justification="",
    )


def make_empty_result() -> InvestigationResult:
    """An InvestigationResult with no evidence."""
    return InvestigationResult(
        business_input=make_business(),
        status=InvestigationStatus.FAILED,
        searches_performed=0,
        sources_examined=0,
        evidence=[],
        features=[],
        positive_signals=[],
        risk_signals=[],
        missing_information=[],
        sources=[],
        justification="",
    )


# ── FeatureImportance tests ───────────────────────────────────────────────────

class TestFeatureImportance:
    def test_fields_stored_correctly(self):
        fi = FeatureImportance(
            feature_name="evidence_count_total",
            shap_value=0.05,
            feature_value=10.0,
        )
        assert fi.feature_name == "evidence_count_total"
        assert fi.shap_value == pytest.approx(0.05)
        assert fi.feature_value == pytest.approx(10.0)

    def test_repr_contains_name(self):
        fi = FeatureImportance("confidence_mean", 0.03, 0.75)
        assert "confidence_mean" in repr(fi)

    def test_repr_positive_shap_has_plus(self):
        fi = FeatureImportance("x", 0.02, 1.0)
        assert "+" in repr(fi)

    def test_repr_negative_shap_has_minus(self):
        fi = FeatureImportance("x", -0.02, 1.0)
        assert "-" in repr(fi)


# ── Background dataset tests ──────────────────────────────────────────────────

class TestMakeBackground:
    def test_shape(self):
        feat_dict = extract_features(make_empty_result())
        names = list(feat_dict.keys())
        bg = _make_background(names)
        assert bg.shape == (20, len(names))

    def test_values_are_finite(self):
        import numpy as np
        feat_dict = extract_features(make_empty_result())
        names = list(feat_dict.keys())
        bg = _make_background(names)
        assert np.all(np.isfinite(bg))

    def test_ratio_features_in_range(self):
        feat_dict = extract_features(make_empty_result())
        names = list(feat_dict.keys())
        bg = _make_background(names)
        ratio_indices = [i for i, n in enumerate(names) if n.endswith("_ratio")]
        for idx in ratio_indices:
            col = bg[:, idx]
            assert col.min() >= 0.0
            assert col.max() <= 1.0


# ── ExplainabilityReport structure tests ─────────────────────────────────────

class TestExplainabilityReportStructure:
    @pytest.fixture(scope="class")
    def report(self):
        return explain_assessment(make_rich_result(), n_samples=10)

    def test_returns_report_type(self, report):
        assert isinstance(report, ExplainabilityReport)

    def test_trustworthiness_is_list(self, report):
        assert isinstance(report.trustworthiness, list)

    def test_business_potential_is_list(self, report):
        assert isinstance(report.business_potential, list)

    def test_trustworthiness_contains_feature_importances(self, report):
        assert all(isinstance(fi, FeatureImportance) for fi in report.trustworthiness)

    def test_business_potential_contains_feature_importances(self, report):
        assert all(isinstance(fi, FeatureImportance) for fi in report.business_potential)

    def test_trustworthiness_length_equals_feature_count(self, report):
        feat_count = len(extract_features(make_rich_result()))
        assert len(report.trustworthiness) == feat_count

    def test_business_potential_length_equals_feature_count(self, report):
        feat_count = len(extract_features(make_rich_result()))
        assert len(report.business_potential) == feat_count

    def test_trust_predicted_is_float(self, report):
        assert isinstance(report.trust_predicted, float)

    def test_potential_predicted_is_float(self, report):
        assert isinstance(report.potential_predicted, float)

    def test_trust_baseline_is_float(self, report):
        assert isinstance(report.trust_baseline, float)

    def test_potential_baseline_is_float(self, report):
        assert isinstance(report.potential_baseline, float)


# ── Score bounds ──────────────────────────────────────────────────────────────

class TestScoreBounds:
    @pytest.fixture(scope="class")
    def report(self):
        return explain_assessment(make_rich_result(), n_samples=10)

    def test_trust_predicted_in_range(self, report):
        assert 0.0 <= report.trust_predicted <= 1.0

    def test_potential_predicted_in_range(self, report):
        assert 0.0 <= report.potential_predicted <= 1.0

    def test_trust_baseline_in_range(self, report):
        assert 0.0 <= report.trust_baseline <= 1.0

    def test_potential_baseline_in_range(self, report):
        assert 0.0 <= report.potential_baseline <= 1.0


# ── Sort order ────────────────────────────────────────────────────────────────

class TestSortOrder:
    @pytest.fixture(scope="class")
    def report(self):
        return explain_assessment(make_rich_result(), n_samples=10)

    def test_trustworthiness_sorted_by_abs_shap_descending(self, report):
        values = [abs(fi.shap_value) for fi in report.trustworthiness]
        assert values == sorted(values, reverse=True)

    def test_business_potential_sorted_by_abs_shap_descending(self, report):
        values = [abs(fi.shap_value) for fi in report.business_potential]
        assert values == sorted(values, reverse=True)


# ── Top drivers helpers ───────────────────────────────────────────────────────

class TestTopDrivers:
    @pytest.fixture(scope="class")
    def report(self):
        return explain_assessment(make_rich_result(), n_samples=10)

    def test_top_trust_drivers_returns_n_items(self, report):
        assert len(report.top_trust_drivers(3)) == 3

    def test_top_potential_drivers_returns_n_items(self, report):
        assert len(report.top_potential_drivers(3)) == 3

    def test_top_trust_drivers_default_5(self, report):
        assert len(report.top_trust_drivers()) == 5

    def test_top_potential_drivers_default_5(self, report):
        assert len(report.top_potential_drivers()) == 5

    def test_top_drivers_are_first_in_full_list(self, report):
        top3 = report.top_trust_drivers(3)
        assert top3 == report.trustworthiness[:3]


# ── Feature names match extractor ─────────────────────────────────────────────

class TestFeatureNames:
    @pytest.fixture(scope="class")
    def report(self):
        return explain_assessment(make_rich_result(), n_samples=10)

    def test_trust_feature_names_match_extractor(self, report):
        expected_names = set(extract_features(make_rich_result()).keys())
        actual_names = {fi.feature_name for fi in report.trustworthiness}
        assert actual_names == expected_names

    def test_potential_feature_names_match_extractor(self, report):
        expected_names = set(extract_features(make_rich_result()).keys())
        actual_names = {fi.feature_name for fi in report.business_potential}
        assert actual_names == expected_names

    def test_feature_values_match_extractor(self, report):
        feat_dict = extract_features(make_rich_result())
        for fi in report.trustworthiness:
            assert fi.feature_value == pytest.approx(feat_dict[fi.feature_name])


# ── Empty evidence edge case ──────────────────────────────────────────────────

class TestEmptyEvidence:
    @pytest.fixture(scope="class")
    def report(self):
        return explain_assessment(make_empty_result(), n_samples=10)

    def test_returns_report_for_empty_result(self, report):
        assert isinstance(report, ExplainabilityReport)

    def test_trust_predicted_zero_for_empty(self, report):
        # With no evidence the trust score should be very low (near 0)
        assert report.trust_predicted <= 0.2

    def test_potential_predicted_zero_for_empty(self, report):
        assert report.potential_predicted <= 0.2

    def test_importances_list_not_empty(self, report):
        assert len(report.trustworthiness) > 0
        assert len(report.business_potential) > 0


# ── format_report ─────────────────────────────────────────────────────────────

class TestFormatReport:
    @pytest.fixture(scope="class")
    def report(self):
        return explain_assessment(make_rich_result(), n_samples=10)

    def test_returns_string(self, report):
        assert isinstance(format_report(report), str)

    def test_contains_trustworthiness_label(self, report):
        assert "Trustworthiness" in format_report(report)

    def test_contains_business_potential_label(self, report):
        assert "Business Potential" in format_report(report)

    def test_contains_predicted_score(self, report):
        text = format_report(report)
        assert "predicted=" in text

    def test_contains_baseline_score(self, report):
        text = format_report(report)
        assert "baseline=" in text

    def test_top_n_respected(self, report):
        text = format_report(report, top_n=3)
        # Count lines containing shap values (lines with feature names)
        driver_lines = [l for l in text.splitlines()
                        if l.strip() and "=" not in l and "driver" not in l.lower()
                        and l.startswith("    ")]
        assert len(driver_lines) == 6  # 3 trust + 3 potential
