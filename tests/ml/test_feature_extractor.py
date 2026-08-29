"""
Tests for ml/feature_extractor.py — InvestigationResult → numeric features.

Run with:  python -m pytest tests/ml/test_feature_extractor.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math
import pytest

from agent.schemas.input import BusinessInput
from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability
from agent.schemas.feature import DiscoveredFeature, FeatureCategory
from agent.schemas.result import (
    InvestigationResult,
    InvestigationStatus,
    AssessmentScore,
    AssessmentLevel,
    Signal,
)
from ml.feature_extractor import (
    extract_features,
    extract_evidence_counts,
    extract_confidence_stats,
    extract_signal_ratios,
    extract_source_reliability,
    extract_feature_categories,
    extract_investigation_meta,
)


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def make_business() -> BusinessInput:
    return BusinessInput(name="Test Business", location="Lagos")


def make_evidence(
    field="field_a",
    value="100",
    etype=EvidenceType.OBSERVED,
    reliability=SourceReliability.MEDIUM,
    confidence=0.8,
) -> EvidenceItem:
    return EvidenceItem(
        field_name=field,
        value=value,
        evidence_type=etype,
        source_name="TestSource",
        source_reliability=reliability,
        confidence=confidence,
    )


def make_feature(
    name="feature_a",
    category=FeatureCategory.AUDIENCE,
    value="25000",
    reason="Relevant signal.",
    confidence=0.85,
    searched=True,
) -> DiscoveredFeature:
    return DiscoveredFeature(
        name=name,
        category=category,
        value=value,
        reason=reason,
        confidence=confidence,
        searched=searched,
    )


def make_signal(label="Test Signal") -> Signal:
    return Signal(
        label=label,
        detail="One-sentence explanation.",
        evidence_refs=["field_a"],
    )


def make_result(**kwargs) -> InvestigationResult:
    """Build an InvestigationResult with sensible defaults; override any field."""
    defaults = dict(
        business_input=make_business(),
        status=InvestigationStatus.COMPLETE,
        searches_performed=5,
        sources_examined=4,
        evidence=[
            make_evidence("ev1", "10", EvidenceType.OBSERVED, SourceReliability.HIGH, 0.9),
            make_evidence("ev2", "20", EvidenceType.CORROBORATED, SourceReliability.MEDIUM, 0.8),
            make_evidence("ev3", "30", EvidenceType.INFERENCE, SourceReliability.LOW, 0.5),
            make_evidence("ev4", "40", EvidenceType.UNKNOWN, SourceReliability.UNKNOWN, 0.3),
        ],
        features=[
            make_feature("feat_aud", FeatureCategory.AUDIENCE, "500"),
            make_feature("feat_rep", FeatureCategory.REPUTATION, "4.5"),
            make_feature("feat_risk", FeatureCategory.RISK, None, searched=True),
        ],
        trustworthiness=AssessmentScore(
            level=AssessmentLevel.MODERATE, score=0.6, evidence_count=3,
        ),
        business_potential=AssessmentScore(
            level=AssessmentLevel.HIGH, score=0.8, evidence_count=4,
        ),
        positive_signals=[make_signal("Positive A"), make_signal("Positive B")],
        risk_signals=[make_signal("Risk A")],
        missing_information=["Tax records", "Registration number"],
        sources=["https://example.com", "https://reviews.com"],
        justification="Test justification.",
    )
    defaults.update(kwargs)
    return InvestigationResult(**defaults)


# ── extract_evidence_counts ───────────────────────────────────────────────────

class TestExtractEvidenceCounts:

    def test_total_count(self):
        evidence = [make_evidence(), make_evidence(), make_evidence()]
        f = extract_evidence_counts(evidence)
        assert f["evidence_count_total"] == 3.0

    def test_reliable_count(self):
        # Only HIGH/MEDIUM + OBSERVED/CORROBORATED + confidence >= 0.7 pass is_reliable().
        reliable = make_evidence(
            field="r", etype=EvidenceType.OBSERVED,
            reliability=SourceReliability.HIGH, confidence=0.9,
        )
        unreliable = make_evidence(
            field="u", etype=EvidenceType.INFERENCE,
            reliability=SourceReliability.LOW, confidence=0.4,
        )
        f = extract_evidence_counts([reliable, unreliable])
        assert f["evidence_count_reliable"] == 1.0

    def test_reliable_ratio(self):
        reliable = make_evidence(
            field="r", etype=EvidenceType.CORROBORATED,
            reliability=SourceReliability.MEDIUM, confidence=0.8,
        )
        unreliable = make_evidence(
            field="u", etype=EvidenceType.UNKNOWN,
            reliability=SourceReliability.UNKNOWN, confidence=0.2,
        )
        f = extract_evidence_counts([reliable, unreliable])
        assert f["evidence_reliable_ratio"] == 0.5

    def test_type_breakdown(self):
        evidence = [
            make_evidence("a", etype=EvidenceType.OBSERVED),
            make_evidence("b", etype=EvidenceType.OBSERVED),
            make_evidence("c", etype=EvidenceType.CORROBORATED),
            make_evidence("d", etype=EvidenceType.INFERENCE),
            make_evidence("e", etype=EvidenceType.UNKNOWN),
        ]
        f = extract_evidence_counts(evidence)
        assert f["evidence_observed_count"] == 2.0
        assert f["evidence_corroborated_count"] == 1.0
        assert f["evidence_inference_count"] == 1.0
        assert f["evidence_unknown_count"] == 1.0

    def test_empty_evidence(self):
        f = extract_evidence_counts([])
        assert f["evidence_count_total"] == 0.0
        assert f["evidence_reliable_ratio"] == 0.0
        assert f["evidence_observed_count"] == 0.0


# ── extract_confidence_stats ──────────────────────────────────────────────────

class TestExtractConfidenceStats:

    def test_mean(self):
        evidence = [
            make_evidence(confidence=0.6),
            make_evidence(confidence=0.8),
            make_evidence(confidence=1.0),
        ]
        f = extract_confidence_stats(evidence)
        assert f["confidence_mean"] == pytest.approx(0.8)

    def test_min_max(self):
        evidence = [
            make_evidence(confidence=0.3),
            make_evidence(confidence=0.9),
        ]
        f = extract_confidence_stats(evidence)
        assert f["confidence_min"] == pytest.approx(0.3)
        assert f["confidence_max"] == pytest.approx(0.9)

    def test_std_single_item(self):
        evidence = [make_evidence(confidence=0.7)]
        f = extract_confidence_stats(evidence)
        assert f["confidence_mean"] == pytest.approx(0.7)
        assert f["confidence_std"] == pytest.approx(0.0)

    def test_std_multiple_items(self):
        evidence = [
            make_evidence(confidence=0.4),
            make_evidence(confidence=0.6),
            make_evidence(confidence=0.8),
        ]
        f = extract_confidence_stats(evidence)
        # Mean = 0.6, variance = ((0.04 + 0.0 + 0.04) / 3) ≈ 0.02667
        expected_std = math.sqrt(0.08 / 3)
        assert f["confidence_std"] == pytest.approx(expected_std, abs=1e-6)

    def test_empty_returns_zeros(self):
        f = extract_confidence_stats([])
        assert f["confidence_mean"] == 0.0
        assert f["confidence_min"] == 0.0
        assert f["confidence_max"] == 0.0
        assert f["confidence_std"] == 0.0


# ── extract_signal_ratios ─────────────────────────────────────────────────────

class TestExtractSignalRatios:

    def test_counts(self):
        r = make_result(
            positive_signals=[make_signal("P1"), make_signal("P2"), make_signal("P3")],
            risk_signals=[make_signal("R1")],
        )
        f = extract_signal_ratios(r)
        assert f["positive_signal_count"] == 3.0
        assert f["risk_signal_count"] == 1.0
        assert f["signal_count_total"] == 4.0

    def test_ratios(self):
        r = make_result(
            positive_signals=[make_signal("P1"), make_signal("P2")],
            risk_signals=[make_signal("R1"), make_signal("R2")],
        )
        f = extract_signal_ratios(r)
        assert f["positive_signal_ratio"] == pytest.approx(0.5)
        assert f["risk_signal_ratio"] == pytest.approx(0.5)

    def test_no_signals(self):
        r = make_result(positive_signals=[], risk_signals=[])
        f = extract_signal_ratios(r)
        assert f["positive_signal_ratio"] == 0.0
        assert f["risk_signal_ratio"] == 0.0
        assert f["signal_count_total"] == 0.0

    def test_missing_information_count(self):
        r = make_result(missing_information=["A", "B", "C"])
        f = extract_signal_ratios(r)
        assert f["missing_information_count"] == 3.0


# ── extract_source_reliability ────────────────────────────────────────────────

class TestExtractSourceReliability:

    def test_distribution(self):
        evidence = [
            make_evidence("a", reliability=SourceReliability.HIGH),
            make_evidence("b", reliability=SourceReliability.HIGH),
            make_evidence("c", reliability=SourceReliability.MEDIUM),
            make_evidence("d", reliability=SourceReliability.LOW),
        ]
        f = extract_source_reliability(evidence)
        assert f["source_reliability_high_ratio"] == pytest.approx(0.5)
        assert f["source_reliability_medium_ratio"] == pytest.approx(0.25)
        assert f["source_reliability_low_ratio"] == pytest.approx(0.25)
        assert f["source_reliability_unknown_ratio"] == pytest.approx(0.0)

    def test_all_unknown(self):
        evidence = [
            make_evidence(reliability=SourceReliability.UNKNOWN),
            make_evidence(reliability=SourceReliability.UNKNOWN),
        ]
        f = extract_source_reliability(evidence)
        assert f["source_reliability_unknown_ratio"] == 1.0
        assert f["source_reliability_high_ratio"] == 0.0

    def test_empty_evidence(self):
        f = extract_source_reliability([])
        assert f["source_reliability_high_ratio"] == 0.0
        assert f["source_reliability_medium_ratio"] == 0.0
        assert f["source_reliability_low_ratio"] == 0.0
        assert f["source_reliability_unknown_ratio"] == 0.0

    def test_ratios_sum_to_one(self):
        evidence = [
            make_evidence("a", reliability=SourceReliability.HIGH),
            make_evidence("b", reliability=SourceReliability.MEDIUM),
            make_evidence("c", reliability=SourceReliability.LOW),
            make_evidence("d", reliability=SourceReliability.UNKNOWN),
        ]
        f = extract_source_reliability(evidence)
        total = (
            f["source_reliability_high_ratio"]
            + f["source_reliability_medium_ratio"]
            + f["source_reliability_low_ratio"]
            + f["source_reliability_unknown_ratio"]
        )
        assert total == pytest.approx(1.0)


# ── extract_feature_categories ────────────────────────────────────────────────

class TestExtractFeatureCategories:

    def test_total_and_found(self):
        features = [
            make_feature("f1", FeatureCategory.AUDIENCE, value="100"),
            make_feature("f2", FeatureCategory.REPUTATION, value="4.5"),
            make_feature("f3", FeatureCategory.RISK, value=None),
        ]
        f = extract_feature_categories(features)
        assert f["features_total"] == 3.0
        assert f["features_found"] == 2.0

    def test_found_ratio(self):
        features = [
            make_feature("f1", value="yes"),
            make_feature("f2", value=None),
        ]
        f = extract_feature_categories(features)
        assert f["feature_found_ratio"] == pytest.approx(0.5)

    def test_searched_count_and_ratio(self):
        features = [
            make_feature("f1", searched=True),
            make_feature("f2", searched=True),
            make_feature("f3", searched=False),
        ]
        f = extract_feature_categories(features)
        assert f["features_searched"] == 2.0
        assert f["feature_searched_ratio"] == pytest.approx(2 / 3)

    def test_category_counts(self):
        features = [
            make_feature("a1", FeatureCategory.AUDIENCE),
            make_feature("a2", FeatureCategory.AUDIENCE),
            make_feature("r1", FeatureCategory.REPUTATION),
            make_feature("rsk", FeatureCategory.RISK),
        ]
        f = extract_feature_categories(features)
        assert f["features_cat_audience"] == 2.0
        assert f["features_cat_reputation"] == 1.0
        assert f["features_cat_risk"] == 1.0
        # Categories not present should still be zero.
        assert f["features_cat_identity"] == 0.0
        assert f["features_cat_demand"] == 0.0

    def test_all_categories_have_keys(self):
        """Every FeatureCategory member must appear in the output, even if zero."""
        f = extract_feature_categories([])
        for cat in FeatureCategory:
            assert f"features_cat_{cat.value}" in f

    def test_empty_features(self):
        f = extract_feature_categories([])
        assert f["features_total"] == 0.0
        assert f["features_found"] == 0.0
        assert f["feature_found_ratio"] == 0.0
        assert f["features_searched"] == 0.0
        assert f["feature_searched_ratio"] == 0.0


# ── extract_investigation_meta ────────────────────────────────────────────────

class TestExtractInvestigationMeta:

    def test_effort_indicators(self):
        r = make_result(searches_performed=8, sources_examined=6)
        f = extract_investigation_meta(r)
        assert f["searches_performed"] == 8.0
        assert f["sources_examined"] == 6.0

    def test_unique_sources(self):
        r = make_result(sources=["https://a.com", "https://b.com", "https://c.com"])
        f = extract_investigation_meta(r)
        assert f["unique_sources_count"] == 3.0

    def test_status_complete(self):
        r = make_result(status=InvestigationStatus.COMPLETE)
        f = extract_investigation_meta(r)
        assert f["status_complete"] == 1.0
        assert f["status_limit_reached"] == 0.0
        assert f["status_partial"] == 0.0
        assert f["status_failed"] == 0.0

    def test_status_failed(self):
        r = make_result(status=InvestigationStatus.FAILED)
        f = extract_investigation_meta(r)
        assert f["status_complete"] == 0.0
        assert f["status_failed"] == 1.0

    def test_status_limit_reached(self):
        r = make_result(status=InvestigationStatus.LIMIT_REACHED)
        f = extract_investigation_meta(r)
        assert f["status_limit_reached"] == 1.0
        assert f["status_complete"] == 0.0

    def test_status_partial(self):
        r = make_result(status=InvestigationStatus.PARTIAL)
        f = extract_investigation_meta(r)
        assert f["status_partial"] == 1.0
        assert f["status_complete"] == 0.0

    def test_exactly_one_status_flag(self):
        for status in InvestigationStatus:
            r = make_result(status=status)
            f = extract_investigation_meta(r)
            flags = [
                f["status_complete"],
                f["status_limit_reached"],
                f["status_partial"],
                f["status_failed"],
            ]
            assert sum(flags) == 1.0, f"Expected exactly one flag for {status}"


# ── extract_features (combined) ──────────────────────────────────────────────

class TestExtractFeatures:

    def test_returns_flat_dict(self):
        f = extract_features(make_result())
        assert isinstance(f, dict)
        for key, val in f.items():
            assert isinstance(key, str)
            assert isinstance(val, float)

    def test_contains_all_subextractors(self):
        result = make_result()
        combined = extract_features(result)

        evidence_keys = extract_evidence_counts(result.evidence).keys()
        confidence_keys = extract_confidence_stats(result.evidence).keys()
        signal_keys = extract_signal_ratios(result).keys()
        reliability_keys = extract_source_reliability(result.evidence).keys()
        category_keys = extract_feature_categories(result.features).keys()
        meta_keys = extract_investigation_meta(result).keys()

        for k in evidence_keys:
            assert k in combined, f"Missing evidence key: {k}"
        for k in confidence_keys:
            assert k in combined, f"Missing confidence key: {k}"
        for k in signal_keys:
            assert k in combined, f"Missing signal key: {k}"
        for k in reliability_keys:
            assert k in combined, f"Missing reliability key: {k}"
        for k in category_keys:
            assert k in combined, f"Missing category key: {k}"
        for k in meta_keys:
            assert k in combined, f"Missing meta key: {k}"

    def test_values_match_subextractors(self):
        result = make_result()
        combined = extract_features(result)
        sub = extract_evidence_counts(result.evidence)
        for k, v in sub.items():
            assert combined[k] == v

    def test_empty_result(self):
        """An InvestigationResult with no evidence/features should still produce a full dict."""
        r = InvestigationResult(
            business_input=make_business(),
            status=InvestigationStatus.FAILED,
        )
        f = extract_features(r)
        assert isinstance(f, dict)
        assert f["evidence_count_total"] == 0.0
        assert f["confidence_mean"] == 0.0
        assert f["features_total"] == 0.0
        assert f["status_failed"] == 1.0

    def test_feature_count_is_stable(self):
        """Regression guard: the total number of extracted features should not change silently."""
        f = extract_features(make_result())
        # 7 evidence + 4 confidence + 6 signal + 4 reliability + 16 category + 7 meta = 44
        assert len(f) == 44

    def test_dominant_evidence_type(self):
        """When all evidence is corroborated, observed/corrob ratio should reflect that."""
        evidence = [
            make_evidence("a", etype=EvidenceType.CORROBORATED,
                          reliability=SourceReliability.HIGH, confidence=0.95),
            make_evidence("b", etype=EvidenceType.CORROBORATED,
                          reliability=SourceReliability.HIGH, confidence=0.90),
        ]
        r = make_result(evidence=evidence)
        f = extract_features(r)
        assert f["evidence_corroborated_count"] == 2.0
        assert f["evidence_observed_count"] == 0.0
        assert f["evidence_reliable_ratio"] == 1.0

    def test_high_risk_scenario(self):
        """A business with more risk signals than positive should reflect in ratios."""
        r = make_result(
            positive_signals=[make_signal("P1")],
            risk_signals=[make_signal("R1"), make_signal("R2"), make_signal("R3")],
        )
        f = extract_features(r)
        assert f["risk_signal_ratio"] > f["positive_signal_ratio"]
        assert f["risk_signal_count"] == 3.0
        assert f["positive_signal_count"] == 1.0
