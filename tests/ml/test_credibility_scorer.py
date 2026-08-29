"""
Tests for ml/credibility_scorer.py — evidence credibility scoring.

Run with:  python -m pytest tests/ml/test_credibility_scorer.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math
import pytest

from agent.schemas.input import BusinessInput
from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability
from agent.schemas.result import InvestigationResult, InvestigationStatus
from ml.credibility_scorer import (
    CredibilityScore,
    score_credibility,
    _score_source_reliability,
    _score_evidence_quality,
    _score_confidence,
    _score_reliable_ratio,
    _score_source_diversity,
    _score_corroboration,
    _score_evidence_depth,
    _classify_level,
)


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def make_business() -> BusinessInput:
    return BusinessInput(name="Test Business")


def make_evidence(
    field="field_a",
    value="100",
    etype=EvidenceType.OBSERVED,
    reliability=SourceReliability.MEDIUM,
    confidence=0.8,
    source_name="SourceA",
) -> EvidenceItem:
    return EvidenceItem(
        field_name=field,
        value=value,
        evidence_type=etype,
        source_name=source_name,
        source_reliability=reliability,
        confidence=confidence,
    )


def make_result(evidence=None, **kwargs) -> InvestigationResult:
    defaults = dict(
        business_input=make_business(),
        status=InvestigationStatus.COMPLETE,
        searches_performed=5,
        sources_examined=4,
        evidence=evidence or [],
    )
    defaults.update(kwargs)
    return InvestigationResult(**defaults)


# ── CredibilityScore dataclass ────────────────────────────────────────────────

class TestCredibilityScore:

    def test_creation(self):
        s = CredibilityScore(
            source_reliability_score=0.8,
            evidence_quality_score=0.7,
            confidence_score=0.9,
            reliable_ratio=0.6,
            source_diversity_score=0.5,
            corroboration_score=0.4,
            evidence_depth_score=0.7,
            overall_score=0.65,
            level="moderate",
        )
        assert s.overall_score == 0.65
        assert s.level == "moderate"

    def test_frozen(self):
        s = CredibilityScore(
            source_reliability_score=0.5, evidence_quality_score=0.5,
            confidence_score=0.5, reliable_ratio=0.5,
            source_diversity_score=0.5, corroboration_score=0.5,
            evidence_depth_score=0.5, overall_score=0.5, level="moderate",
        )
        with pytest.raises(AttributeError):
            s.overall_score = 1.0

    def test_clamping(self):
        s = CredibilityScore(
            source_reliability_score=1.5, evidence_quality_score=-0.1,
            confidence_score=0.5, reliable_ratio=0.5,
            source_diversity_score=0.5, corroboration_score=0.5,
            evidence_depth_score=0.5, overall_score=2.0, level="high",
        )
        assert s.source_reliability_score == 1.0
        assert s.evidence_quality_score == 0.0
        assert s.overall_score == 1.0


# ── _score_source_reliability ─────────────────────────────────────────────────

class TestScoreSourceReliability:

    def test_all_high(self):
        evidence = [
            make_evidence(reliability=SourceReliability.HIGH),
            make_evidence(reliability=SourceReliability.HIGH),
        ]
        assert _score_source_reliability(evidence) == pytest.approx(1.0)

    def test_all_low(self):
        evidence = [make_evidence(reliability=SourceReliability.LOW)]
        assert _score_source_reliability(evidence) == pytest.approx(0.33)

    def test_mixed(self):
        evidence = [
            make_evidence("a", reliability=SourceReliability.HIGH),
            make_evidence("b", reliability=SourceReliability.LOW),
        ]
        # (1.0 + 0.33) / 2 = 0.665
        assert _score_source_reliability(evidence) == pytest.approx(0.665)

    def test_unknown(self):
        evidence = [make_evidence(reliability=SourceReliability.UNKNOWN)]
        assert _score_source_reliability(evidence) == pytest.approx(0.0)

    def test_empty(self):
        assert _score_source_reliability([]) == 0.0


# ── _score_evidence_quality ───────────────────────────────────────────────────

class TestScoreEvidenceQuality:

    def test_all_corroborated(self):
        evidence = [make_evidence(etype=EvidenceType.CORROBORATED)]
        assert _score_evidence_quality(evidence) == pytest.approx(1.0)

    def test_all_observed(self):
        evidence = [make_evidence(etype=EvidenceType.OBSERVED)]
        assert _score_evidence_quality(evidence) == pytest.approx(0.75)

    def test_all_inference(self):
        evidence = [make_evidence(etype=EvidenceType.INFERENCE)]
        assert _score_evidence_quality(evidence) == pytest.approx(0.25)

    def test_mixed(self):
        evidence = [
            make_evidence("a", etype=EvidenceType.CORROBORATED),
            make_evidence("b", etype=EvidenceType.INFERENCE),
        ]
        # (1.0 + 0.25) / 2 = 0.625
        assert _score_evidence_quality(evidence) == pytest.approx(0.625)

    def test_empty(self):
        assert _score_evidence_quality([]) == 0.0


# ── _score_confidence ────────────────────────────────────────────────────────

class TestScoreConfidence:

    def test_high_confidence(self):
        evidence = [make_evidence(confidence=0.95)]
        assert _score_confidence(evidence) == pytest.approx(0.95)

    def test_average(self):
        evidence = [
            make_evidence(confidence=0.6),
            make_evidence(confidence=0.8),
        ]
        assert _score_confidence(evidence) == pytest.approx(0.7)

    def test_empty(self):
        assert _score_confidence([]) == 0.0


# ── _score_reliable_ratio ────────────────────────────────────────────────────

class TestScoreReliableRatio:

    def test_all_reliable(self):
        evidence = [
            make_evidence(etype=EvidenceType.OBSERVED,
                          reliability=SourceReliability.HIGH, confidence=0.9),
            make_evidence(etype=EvidenceType.CORROBORATED,
                          reliability=SourceReliability.MEDIUM, confidence=0.8),
        ]
        assert _score_reliable_ratio(evidence) == pytest.approx(1.0)

    def test_none_reliable(self):
        evidence = [
            make_evidence(etype=EvidenceType.INFERENCE,
                          reliability=SourceReliability.LOW, confidence=0.4),
        ]
        assert _score_reliable_ratio(evidence) == pytest.approx(0.0)

    def test_half_reliable(self):
        reliable = make_evidence("r", etype=EvidenceType.OBSERVED,
                                 reliability=SourceReliability.HIGH, confidence=0.9)
        unreliable = make_evidence("u", etype=EvidenceType.INFERENCE,
                                   reliability=SourceReliability.LOW, confidence=0.3)
        assert _score_reliable_ratio([reliable, unreliable]) == pytest.approx(0.5)

    def test_empty(self):
        assert _score_reliable_ratio([]) == 0.0


# ── _score_source_diversity ──────────────────────────────────────────────────

class TestScoreSourceDiversity:

    def test_all_different_sources(self):
        evidence = [
            make_evidence("a", source_name="Google"),
            make_evidence("b", source_name="Instagram"),
            make_evidence("c", source_name="Yelp"),
        ]
        assert _score_source_diversity(evidence) == pytest.approx(1.0)

    def test_single_source(self):
        evidence = [
            make_evidence("a", source_name="Google"),
            make_evidence("b", source_name="Google"),
            make_evidence("c", source_name="Google"),
        ]
        # 1 unique / 3 total
        assert _score_source_diversity(evidence) == pytest.approx(1 / 3)

    def test_no_source_names(self):
        evidence = [make_evidence(source_name=None)]
        assert _score_source_diversity(evidence) == 0.0

    def test_empty(self):
        assert _score_source_diversity([]) == 0.0


# ── _score_corroboration ─────────────────────────────────────────────────────

class TestScoreCorroboration:

    def test_fully_corroborated(self):
        """Same field_name from 2+ different sources."""
        evidence = [
            make_evidence("followers", source_name="Instagram"),
            make_evidence("followers", source_name="SocialBlade"),
        ]
        assert _score_corroboration(evidence) == pytest.approx(1.0)

    def test_no_corroboration(self):
        """Each field_name from a single source."""
        evidence = [
            make_evidence("followers", source_name="Instagram"),
            make_evidence("reviews", source_name="Yelp"),
        ]
        assert _score_corroboration(evidence) == pytest.approx(0.0)

    def test_partial_corroboration(self):
        evidence = [
            make_evidence("followers", source_name="Instagram"),
            make_evidence("followers", source_name="SocialBlade"),
            make_evidence("reviews", source_name="Yelp"),
        ]
        # "followers" corroborated (2 sources), "reviews" not → 1/2 = 0.5
        assert _score_corroboration(evidence) == pytest.approx(0.5)

    def test_same_source_same_field_not_corroborated(self):
        evidence = [
            make_evidence("followers", source_name="Instagram"),
            make_evidence("followers", source_name="Instagram"),
        ]
        assert _score_corroboration(evidence) == pytest.approx(0.0)

    def test_empty(self):
        assert _score_corroboration([]) == 0.0


# ── _score_evidence_depth ────────────────────────────────────────────────────

class TestScoreEvidenceDepth:

    def test_empty(self):
        assert _score_evidence_depth([]) == 0.0

    def test_single_item(self):
        # sqrt(1/10) ≈ 0.316
        evidence = [make_evidence()]
        assert _score_evidence_depth(evidence) == pytest.approx(math.sqrt(0.1))

    def test_at_threshold(self):
        evidence = [make_evidence(field=f"f{i}") for i in range(10)]
        assert _score_evidence_depth(evidence) == pytest.approx(1.0)

    def test_above_threshold(self):
        evidence = [make_evidence(field=f"f{i}") for i in range(20)]
        assert _score_evidence_depth(evidence) == pytest.approx(1.0)

    def test_diminishing_returns(self):
        few = [make_evidence(field=f"f{i}") for i in range(3)]
        more = [make_evidence(field=f"f{i}") for i in range(8)]
        # Going from 3 to 8 items gives a larger gain than going from 8 to 13.
        score_few = _score_evidence_depth(few)
        score_more = _score_evidence_depth(more)
        assert score_more > score_few


# ── _classify_level ──────────────────────────────────────────────────────────

class TestClassifyLevel:

    def test_high(self):
        assert _classify_level(0.80, 5) == "high"

    def test_high_at_threshold(self):
        assert _classify_level(0.70, 5) == "high"

    def test_moderate(self):
        assert _classify_level(0.55, 5) == "moderate"

    def test_moderate_at_threshold(self):
        assert _classify_level(0.45, 5) == "moderate"

    def test_low(self):
        assert _classify_level(0.20, 5) == "low"

    def test_insufficient_evidence(self):
        assert _classify_level(0.0, 0) == "insufficient_evidence"
        assert _classify_level(0.99, 0) == "insufficient_evidence"


# ── score_credibility (integration) ──────────────────────────────────────────

class TestScoreCredibility:

    def test_empty_result(self):
        r = make_result(evidence=[])
        s = score_credibility(r)
        assert s.level == "insufficient_evidence"
        assert s.overall_score == 0.0

    def test_high_credibility_scenario(self):
        evidence = [
            make_evidence("f1", etype=EvidenceType.CORROBORATED,
                          reliability=SourceReliability.HIGH,
                          confidence=0.95, source_name="GovRegistry"),
            make_evidence("f1", etype=EvidenceType.CORROBORATED,
                          reliability=SourceReliability.HIGH,
                          confidence=0.90, source_name="NewsOutlet"),
            make_evidence("f2", etype=EvidenceType.OBSERVED,
                          reliability=SourceReliability.HIGH,
                          confidence=0.85, source_name="OfficialSite"),
            make_evidence("f3", etype=EvidenceType.OBSERVED,
                          reliability=SourceReliability.MEDIUM,
                          confidence=0.80, source_name="ReviewPlatform"),
            make_evidence("f4", etype=EvidenceType.OBSERVED,
                          reliability=SourceReliability.MEDIUM,
                          confidence=0.75, source_name="Marketplace"),
        ]
        r = make_result(evidence=evidence)
        s = score_credibility(r)
        assert s.level == "high"
        assert s.overall_score >= 0.70
        assert s.source_reliability_score > 0.7
        assert s.evidence_quality_score > 0.7
        assert s.reliable_ratio == 1.0

    def test_low_credibility_scenario(self):
        evidence = [
            make_evidence("f1", etype=EvidenceType.INFERENCE,
                          reliability=SourceReliability.LOW,
                          confidence=0.3, source_name="ForumPost"),
            make_evidence("f2", etype=EvidenceType.UNKNOWN,
                          reliability=SourceReliability.UNKNOWN,
                          confidence=0.2, source_name="ForumPost"),
        ]
        r = make_result(evidence=evidence)
        s = score_credibility(r)
        assert s.level == "low"
        assert s.overall_score < 0.45

    def test_moderate_credibility(self):
        evidence = [
            make_evidence("f1", etype=EvidenceType.OBSERVED,
                          reliability=SourceReliability.MEDIUM,
                          confidence=0.7, source_name="ReviewSite"),
            make_evidence("f2", etype=EvidenceType.OBSERVED,
                          reliability=SourceReliability.MEDIUM,
                          confidence=0.75, source_name="SocialMedia"),
        ]
        r = make_result(evidence=evidence)
        s = score_credibility(r)
        assert s.level in ("moderate", "high", "low")
        assert 0.0 <= s.overall_score <= 1.0

    def test_all_subscores_populated(self):
        evidence = [
            make_evidence("f1", etype=EvidenceType.OBSERVED,
                          reliability=SourceReliability.HIGH,
                          confidence=0.8, source_name="SourceA"),
        ]
        r = make_result(evidence=evidence)
        s = score_credibility(r)
        assert s.source_reliability_score > 0.0
        assert s.evidence_quality_score > 0.0
        assert s.confidence_score > 0.0
        assert s.reliable_ratio > 0.0
        assert s.source_diversity_score > 0.0
        assert s.evidence_depth_score > 0.0
        assert s.overall_score > 0.0

    def test_overall_bounded(self):
        """Overall score should always be in [0, 1] regardless of input."""
        for count in [1, 5, 15, 50]:
            evidence = [
                make_evidence(
                    field=f"f{i}",
                    etype=EvidenceType.OBSERVED,
                    reliability=SourceReliability.HIGH,
                    confidence=1.0,
                    source_name=f"Source{i}",
                )
                for i in range(count)
            ]
            r = make_result(evidence=evidence)
            s = score_credibility(r)
            assert 0.0 <= s.overall_score <= 1.0

    def test_corroboration_boosts_score(self):
        """Corroborated evidence from multiple sources should score higher."""
        no_corrob = [
            make_evidence("f1", source_name="Google"),
            make_evidence("f2", source_name="Yelp"),
        ]
        with_corrob = [
            make_evidence("f1", source_name="Google"),
            make_evidence("f1", source_name="Yelp"),
        ]
        r1 = make_result(evidence=no_corrob)
        r2 = make_result(evidence=with_corrob)
        s1 = score_credibility(r1)
        s2 = score_credibility(r2)
        assert s2.corroboration_score > s1.corroboration_score
