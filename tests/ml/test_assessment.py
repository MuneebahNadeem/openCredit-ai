"""
Tests for ml/assessment.py — final assessment wrapper.

Run with:  python -m pytest tests/ml/test_assessment.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from agent.schemas.input import BusinessInput
from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability
from agent.schemas.feature import DiscoveredFeature, FeatureCategory
from agent.schemas.result import (
    AssessmentLevel,
    AssessmentScore,
    InvestigationResult,
    InvestigationStatus,
    Signal,
)
from ml.assessment import (
    generate_assessment,
    generate_justification,
    generate_recommendation,
    _RECOMMEND_APPROVE,
    _RECOMMEND_APPROVE_CONDITIONAL,
    _RECOMMEND_DECLINE,
    _RECOMMEND_REVIEW,
    _RECOMMEND_INSUFFICIENT,
)
from ml.risk_engine import RiskAssessment


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
    raw_snippet=None,
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
    name="feat_a",
    category=FeatureCategory.AUDIENCE,
    value="500",
    reason="Relevant signal.",
    confidence=0.85,
    searched=True,
) -> DiscoveredFeature:
    return DiscoveredFeature(
        name=name, category=category, value=value,
        reason=reason, confidence=confidence, searched=searched,
    )


def make_signal(label="Signal A") -> Signal:
    return Signal(label=label, detail="Explanation.", evidence_refs=["field_a"])


def make_assessment_score(
    level=AssessmentLevel.MODERATE,
    score=0.55,
    evidence_count=3,
    explanation="Test explanation.",
) -> AssessmentScore:
    return AssessmentScore(
        level=level, score=score,
        evidence_count=evidence_count, explanation=explanation,
    )


def make_risk_assessment(
    trust_level=AssessmentLevel.MODERATE,
    trust_score=0.55,
    potential_level=AssessmentLevel.MODERATE,
    potential_score=0.55,
) -> RiskAssessment:
    """Build a minimal RiskAssessment for recommendation tests."""
    from ml.credibility_scorer import CredibilityScore
    from ml.sentiment import SentimentScore

    trust = make_assessment_score(trust_level, trust_score)
    potential = make_assessment_score(potential_level, potential_score)

    credibility = CredibilityScore(
        source_reliability_score=0.5, evidence_quality_score=0.5,
        confidence_score=0.5, reliable_ratio=0.5,
        source_diversity_score=0.5, corroboration_score=0.5,
        evidence_depth_score=0.5, overall_score=0.5, level="moderate",
    )
    sentiment = SentimentScore(
        positive=0.5, negative=0.5, neutral=0.0,
        compound=0.0, label="neutral",
    )

    return RiskAssessment(
        trustworthiness=trust,
        business_potential=potential,
        credibility=credibility,
        sentiment=sentiment,
        features={},
    )


def make_result(**kwargs) -> InvestigationResult:
    defaults = dict(
        business_input=make_business(),
        status=InvestigationStatus.COMPLETE,
        searches_performed=5,
        sources_examined=4,
        evidence=[
            make_evidence("ev1", "excellent quality",
                          EvidenceType.CORROBORATED, SourceReliability.HIGH,
                          0.9, "GovRegistry",
                          "The business provides excellent quality products."),
            make_evidence("ev2", "25000",
                          EvidenceType.OBSERVED, SourceReliability.MEDIUM,
                          0.8, "Instagram",
                          "Business has 25000 followers on Instagram."),
            make_evidence("ev3", "4.5",
                          EvidenceType.OBSERVED, SourceReliability.MEDIUM,
                          0.75, "ReviewSite",
                          "Average rating is 4.5 stars."),
        ],
        features=[
            make_feature("audience", FeatureCategory.AUDIENCE, "25000"),
            make_feature("engagement", FeatureCategory.ENGAGEMENT, "high"),
            make_feature("reputation", FeatureCategory.REPUTATION, "4.5"),
        ],
        positive_signals=[
            make_signal("Active social presence"),
            make_signal("Good reviews"),
        ],
        risk_signals=[],
        missing_information=[],
        sources=["https://gov.registry.com", "https://instagram.com/biz"],
        justification="",
    )
    defaults.update(kwargs)
    return InvestigationResult(**defaults)


# ── generate_justification ───────────────────────────────────────────────────

class TestGenerateJustification:

    def test_high_trust_high_potential(self):
        r = make_result()
        risk = make_risk_assessment(
            trust_level=AssessmentLevel.HIGH, trust_score=0.85,
            potential_level=AssessmentLevel.HIGH, potential_score=0.80,
        )
        j = generate_justification(r, risk)
        assert "high" in j.lower()
        assert "strong" in j.lower() or "high" in j.lower()
        assert "3 evidence" in j

    def test_moderate_trust_moderate_potential(self):
        r = make_result()
        risk = make_risk_assessment(
            trust_level=AssessmentLevel.MODERATE, trust_score=0.55,
            potential_level=AssessmentLevel.MODERATE, potential_score=0.50,
        )
        j = generate_justification(r, risk)
        assert "moderate" in j.lower()

    def test_low_trust(self):
        r = make_result()
        risk = make_risk_assessment(
            trust_level=AssessmentLevel.LOW, trust_score=0.25,
            potential_level=AssessmentLevel.MODERATE, potential_score=0.50,
        )
        j = generate_justification(r, risk)
        assert "low" in j.lower()

    def test_insufficient_evidence(self):
        r = make_result(evidence=[], features=[])
        risk = make_risk_assessment(
            trust_level=AssessmentLevel.INSUFFICIENT_EVIDENCE, trust_score=None,
            potential_level=AssessmentLevel.INSUFFICIENT_EVIDENCE, potential_score=None,
        )
        j = generate_justification(r, risk)
        assert "insufficient" in j.lower()

    def test_ends_with_period(self):
        r = make_result()
        risk = make_risk_assessment()
        j = generate_justification(r, risk)
        assert j.endswith(".")

    def test_includes_evidence_count(self):
        r = make_result()
        risk = make_risk_assessment()
        j = generate_justification(r, risk)
        assert "3 evidence items" in j

    def test_includes_reliable_count(self):
        r = make_result()
        risk = make_risk_assessment()
        j = generate_justification(r, risk)
        # At least one of the 3 evidence items should be reliable
        assert "reliable" in j.lower()


# ── generate_recommendation ──────────────────────────────────────────────────

class TestGenerateRecommendation:

    def test_both_high(self):
        risk = make_risk_assessment(
            trust_level=AssessmentLevel.HIGH,
            potential_level=AssessmentLevel.HIGH,
        )
        assert generate_recommendation(risk) == _RECOMMEND_APPROVE

    def test_both_moderate(self):
        risk = make_risk_assessment(
            trust_level=AssessmentLevel.MODERATE,
            potential_level=AssessmentLevel.MODERATE,
        )
        assert generate_recommendation(risk) == _RECOMMEND_APPROVE_CONDITIONAL

    def test_high_trust_moderate_potential(self):
        risk = make_risk_assessment(
            trust_level=AssessmentLevel.HIGH,
            potential_level=AssessmentLevel.MODERATE,
        )
        assert generate_recommendation(risk) == _RECOMMEND_APPROVE_CONDITIONAL

    def test_moderate_trust_high_potential(self):
        risk = make_risk_assessment(
            trust_level=AssessmentLevel.MODERATE,
            potential_level=AssessmentLevel.HIGH,
        )
        assert generate_recommendation(risk) == _RECOMMEND_APPROVE_CONDITIONAL

    def test_low_trust(self):
        risk = make_risk_assessment(
            trust_level=AssessmentLevel.LOW,
            potential_level=AssessmentLevel.HIGH,
        )
        assert generate_recommendation(risk) == _RECOMMEND_DECLINE

    def test_low_trust_low_potential(self):
        risk = make_risk_assessment(
            trust_level=AssessmentLevel.LOW,
            potential_level=AssessmentLevel.LOW,
        )
        # Low trust → decline, regardless of potential
        assert generate_recommendation(risk) == _RECOMMEND_DECLINE

    def test_moderate_trust_low_potential(self):
        risk = make_risk_assessment(
            trust_level=AssessmentLevel.MODERATE,
            potential_level=AssessmentLevel.LOW,
        )
        assert generate_recommendation(risk) == _RECOMMEND_REVIEW

    def test_high_trust_low_potential(self):
        risk = make_risk_assessment(
            trust_level=AssessmentLevel.HIGH,
            potential_level=AssessmentLevel.LOW,
        )
        assert generate_recommendation(risk) == _RECOMMEND_REVIEW

    def test_insufficient_trust(self):
        risk = make_risk_assessment(
            trust_level=AssessmentLevel.INSUFFICIENT_EVIDENCE,
            potential_level=AssessmentLevel.HIGH,
        )
        assert generate_recommendation(risk) == _RECOMMEND_INSUFFICIENT

    def test_insufficient_potential(self):
        risk = make_risk_assessment(
            trust_level=AssessmentLevel.HIGH,
            potential_level=AssessmentLevel.INSUFFICIENT_EVIDENCE,
        )
        assert generate_recommendation(risk) == _RECOMMEND_INSUFFICIENT

    def test_both_insufficient(self):
        risk = make_risk_assessment(
            trust_level=AssessmentLevel.INSUFFICIENT_EVIDENCE,
            potential_level=AssessmentLevel.INSUFFICIENT_EVIDENCE,
        )
        assert generate_recommendation(risk) == _RECOMMEND_INSUFFICIENT


# ── generate_assessment (integration) ────────────────────────────────────────

class TestGenerateAssessment:

    def test_returns_investigation_result(self):
        r = make_result()
        enriched = generate_assessment(r)
        assert isinstance(enriched, InvestigationResult)

    def test_trustworthiness_populated(self):
        r = make_result()
        enriched = generate_assessment(r)
        assert enriched.trustworthiness.score is not None
        assert enriched.trustworthiness.level != AssessmentLevel.INSUFFICIENT_EVIDENCE

    def test_business_potential_populated(self):
        r = make_result()
        enriched = generate_assessment(r)
        assert enriched.business_potential.score is not None
        assert enriched.business_potential.level != AssessmentLevel.INSUFFICIENT_EVIDENCE

    def test_justification_generated(self):
        r = make_result()
        enriched = generate_assessment(r)
        assert len(enriched.justification) > 0
        assert enriched.justification.endswith(".")

    def test_preserves_original_evidence(self):
        r = make_result()
        enriched = generate_assessment(r)
        assert len(enriched.evidence) == len(r.evidence)
        for orig, copy in zip(r.evidence, enriched.evidence):
            assert orig.field_name == copy.field_name

    def test_preserves_original_features(self):
        r = make_result()
        enriched = generate_assessment(r)
        assert len(enriched.features) == len(r.features)

    def test_preserves_original_signals(self):
        r = make_result()
        enriched = generate_assessment(r)
        assert len(enriched.positive_signals) == len(r.positive_signals)
        assert len(enriched.risk_signals) == len(r.risk_signals)

    def test_preserves_business_input(self):
        r = make_result()
        enriched = generate_assessment(r)
        assert enriched.business_input.name == r.business_input.name

    def test_preserves_status(self):
        r = make_result()
        enriched = generate_assessment(r)
        assert enriched.status == r.status

    def test_preserves_sources(self):
        r = make_result()
        enriched = generate_assessment(r)
        assert enriched.sources == r.sources

    def test_no_evidence_result(self):
        r = make_result(
            evidence=[], features=[],
            positive_signals=[], risk_signals=[],
        )
        enriched = generate_assessment(r)
        assert enriched.trustworthiness.level == AssessmentLevel.INSUFFICIENT_EVIDENCE
        assert enriched.business_potential.level == AssessmentLevel.INSUFFICIENT_EVIDENCE
        assert enriched.trustworthiness.score is None
        assert "insufficient" in enriched.justification.lower()

    def test_scores_are_bounded(self):
        r = make_result()
        enriched = generate_assessment(r)
        assert 0.0 <= enriched.trustworthiness.score <= 1.0
        assert 0.0 <= enriched.business_potential.score <= 1.0

    def test_level_matches_score(self):
        r = make_result()
        enriched = generate_assessment(r)
        for assessment in [enriched.trustworthiness, enriched.business_potential]:
            if assessment.score >= 0.70:
                assert assessment.level == AssessmentLevel.HIGH
            elif assessment.score >= 0.45:
                assert assessment.level == AssessmentLevel.MODERATE
            else:
                assert assessment.level == AssessmentLevel.LOW

    def test_valid_pydantic_model(self):
        """The enriched result should serialise to JSON without errors."""
        r = make_result()
        enriched = generate_assessment(r)
        json_data = enriched.model_dump_json()
        assert len(json_data) > 0

    def test_risky_business(self):
        """A business with risk signals should get lower trust."""
        evidence = [
            make_evidence("ev1", "complaints",
                          EvidenceType.INFERENCE, SourceReliability.LOW,
                          0.3, "Forum"),
            make_evidence("ev2", "no website",
                          EvidenceType.OBSERVED, SourceReliability.UNKNOWN,
                          0.4, "Search"),
        ]
        r = make_result(
            evidence=evidence,
            features=[make_feature("risk", FeatureCategory.RISK, "high")],
            positive_signals=[],
            risk_signals=[
                make_signal("Multiple complaints"),
                make_signal("No online presence"),
                make_signal("Suspicious activity"),
            ],
        )
        enriched = generate_assessment(r)
        assert enriched.trustworthiness.score < 0.5

    def test_summary_works(self):
        """The enriched result's summary() should work."""
        r = make_result()
        enriched = generate_assessment(r)
        s = enriched.summary()
        assert "Test Business" in s
        assert "trust=" in s
        assert "potential=" in s
