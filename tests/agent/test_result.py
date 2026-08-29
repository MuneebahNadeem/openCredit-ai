"""
Tests for agent/schemas/result.py — InvestigationResult model.

Run with:  python -m pytest tests/agent/test_result.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from pydantic import ValidationError
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


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def make_business() -> BusinessInput:
    return BusinessInput(name="Sunshine Bakery", location="Lagos, Nigeria")


def make_evidence(field="instagram_followers", value="25000",
                  etype=EvidenceType.OBSERVED,
                  reliability=SourceReliability.MEDIUM,
                  confidence=0.9) -> EvidenceItem:
    return EvidenceItem(
        field_name=field,
        value=value,
        evidence_type=etype,
        source_name="Instagram",
        source_reliability=reliability,
        confidence=confidence,
    )


def make_feature() -> DiscoveredFeature:
    return DiscoveredFeature(
        name="instagram_follower_count",
        category=FeatureCategory.AUDIENCE,
        value="25000",
        unit="followers",
        reason="Social reach may signal customer demand.",
        evidence=[make_evidence()],
        confidence=0.85,
        searched=True,
    )


def make_signal(label="Active Instagram") -> Signal:
    return Signal(
        label=label,
        detail="Business has 25,000 followers with recent posts.",
        evidence_refs=["instagram_followers"],
    )


def make_result(**kwargs) -> InvestigationResult:
    defaults = dict(
        business_input=make_business(),
        status=InvestigationStatus.COMPLETE,
        searches_performed=5,
        sources_examined=4,
        evidence=[make_evidence()],
        features=[make_feature()],
        trustworthiness=AssessmentScore(
            level=AssessmentLevel.MODERATE,
            score=0.6,
            evidence_count=3,
            explanation="Consistent identity across two sources.",
        ),
        business_potential=AssessmentScore(
            level=AssessmentLevel.HIGH,
            score=0.8,
            evidence_count=4,
            explanation="Strong social audience and positive reviews.",
        ),
        positive_signals=[make_signal("Active Instagram")],
        risk_signals=[make_signal("No website found")],
        missing_information=["Company registration number"],
        sources=["https://instagram.com/sunshinebakery"],
        justification=(
            "The business shows a consistent public identity and positive customer presence. "
            "Financial information could not be independently verified."
        ),
    )
    defaults.update(kwargs)
    return InvestigationResult(**defaults)


# ── Valid construction ────────────────────────────────────────────────────────

def test_minimal_result():
    r = InvestigationResult(
        business_input=make_business(),
        status=InvestigationStatus.COMPLETE,
    )
    assert r.status == InvestigationStatus.COMPLETE
    assert r.evidence == []
    assert r.features == []
    assert r.trustworthiness.level == AssessmentLevel.INSUFFICIENT_EVIDENCE
    assert r.business_potential.level == AssessmentLevel.INSUFFICIENT_EVIDENCE
    assert r.has_sufficient_evidence is False


def test_full_result_construction():
    r = make_result()
    assert r.business_input.name == "Sunshine Bakery"
    assert r.searches_performed == 5
    assert len(r.evidence) == 1
    assert len(r.features) == 1
    assert r.trustworthiness.level == AssessmentLevel.MODERATE
    assert r.business_potential.score == 0.8


def test_investigated_at_is_set_automatically():
    r = make_result()
    assert r.investigated_at is not None


# ── Assessment scores ─────────────────────────────────────────────────────────

def test_assessment_score_valid():
    s = AssessmentScore(level=AssessmentLevel.HIGH, score=0.9, evidence_count=5)
    assert s.score == 0.9


def test_assessment_score_above_1_raises():
    with pytest.raises(ValidationError):
        AssessmentScore(level=AssessmentLevel.HIGH, score=1.1)


def test_assessment_score_below_0_raises():
    with pytest.raises(ValidationError):
        AssessmentScore(level=AssessmentLevel.LOW, score=-0.1)


def test_assessment_score_none_allowed():
    s = AssessmentScore(level=AssessmentLevel.INSUFFICIENT_EVIDENCE, score=None)
    assert s.score is None


# ── Signal validation ─────────────────────────────────────────────────────────

def test_signal_valid():
    s = make_signal()
    assert s.label == "Active Instagram"


def test_signal_blank_label_raises():
    with pytest.raises(ValidationError):
        Signal(label="   ", detail="some detail")


def test_signal_blank_detail_raises():
    with pytest.raises(ValidationError):
        Signal(label="some label", detail="   ")


# ── has_sufficient_evidence ───────────────────────────────────────────────────

def test_has_sufficient_evidence_true_when_trust_scored():
    r = make_result()
    assert r.has_sufficient_evidence is True


def test_has_sufficient_evidence_false_when_both_insufficient():
    r = InvestigationResult(
        business_input=make_business(),
        status=InvestigationStatus.COMPLETE,
    )
    assert r.has_sufficient_evidence is False


# ── Computed helpers ──────────────────────────────────────────────────────────

def test_evidence_count_total():
    r = make_result()
    assert r.evidence_count_total == 1


def test_reliable_evidence_filters_correctly():
    reliable = make_evidence(confidence=0.9, reliability=SourceReliability.MEDIUM)
    unreliable = make_evidence(field="low_conf", value="x", confidence=0.3)
    r = make_result(evidence=[reliable, unreliable])
    assert len(r.reliable_evidence()) == 1


def test_get_features_by_category():
    r = make_result()
    audience = r.get_features_by_category("audience")
    assert len(audience) == 1
    assert audience[0].name == "instagram_follower_count"


def test_get_features_by_category_no_match():
    r = make_result()
    assert r.get_features_by_category("risk") == []


# ── summary() ────────────────────────────────────────────────────────────────

def test_summary_contains_key_fields():
    r = make_result()
    s = r.summary()
    assert "Sunshine Bakery" in s
    assert "moderate" in s
    assert "high" in s
    assert "complete" in s


# ── Status variants ───────────────────────────────────────────────────────────

def test_limit_reached_status():
    r = make_result(status=InvestigationStatus.LIMIT_REACHED)
    assert r.status == InvestigationStatus.LIMIT_REACHED


def test_failed_status_with_no_evidence():
    r = InvestigationResult(
        business_input=make_business(),
        status=InvestigationStatus.FAILED,
    )
    assert r.has_sufficient_evidence is False
    assert r.evidence == []
