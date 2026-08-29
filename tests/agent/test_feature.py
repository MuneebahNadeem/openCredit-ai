"""
Tests for agent/schemas/feature.py — DiscoveredFeature model.

Run with:  python -m pytest tests/agent/test_feature.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from pydantic import ValidationError
from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability
from agent.schemas.feature import DiscoveredFeature, FeatureCategory


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_evidence_item(evidence_type=EvidenceType.OBSERVED) -> EvidenceItem:
    return EvidenceItem(
        field_name="instagram_followers",
        value="25000",
        evidence_type=evidence_type,
        source_name="Instagram",
        source_reliability=SourceReliability.MEDIUM,
        confidence=0.9,
    )


def make_feature(**kwargs) -> DiscoveredFeature:
    defaults = dict(
        name="instagram_follower_count",
        category=FeatureCategory.AUDIENCE,
        value="25000",
        unit="followers",
        reason="Large social following may indicate public reach.",
        confidence=0.85,
        searched=True,
    )
    defaults.update(kwargs)
    return DiscoveredFeature(**defaults)


# ── Valid construction ────────────────────────────────────────────────────────

def test_minimal_feature():
    f = DiscoveredFeature(
        name="has_website",
        reason="A website is the most basic public business presence.",
    )
    assert f.name == "has_website"
    assert f.value is None
    assert f.confidence == 0.0
    assert f.category == FeatureCategory.UNKNOWN
    assert f.searched is False


def test_full_feature_with_evidence():
    e = make_evidence_item()
    f = make_feature(evidence=[e])
    assert f.name == "instagram_follower_count"
    assert f.is_found() is True
    assert len(f.evidence) == 1


def test_feature_not_found_when_value_is_none():
    f = make_feature(value=None)
    assert f.is_found() is False


def test_feature_category_defaults_to_unknown():
    f = DiscoveredFeature(name="mystery_signal", reason="Testing defaults.")
    assert f.category == FeatureCategory.UNKNOWN


# ── Validation errors ─────────────────────────────────────────────────────────

def test_blank_name_raises():
    with pytest.raises(ValidationError):
        DiscoveredFeature(name="   ", reason="some reason")


def test_blank_reason_raises():
    with pytest.raises(ValidationError):
        DiscoveredFeature(name="some_feature", reason="   ")


def test_missing_reason_raises():
    with pytest.raises(ValidationError):
        DiscoveredFeature(name="some_feature")


def test_confidence_above_1_raises():
    with pytest.raises(ValidationError):
        make_feature(confidence=1.5)


def test_confidence_below_0_raises():
    with pytest.raises(ValidationError):
        make_feature(confidence=-0.1)


# ── evidence_type() helper ────────────────────────────────────────────────────

def test_evidence_type_no_evidence_returns_unknown():
    f = make_feature(evidence=[])
    assert f.evidence_type() == EvidenceType.UNKNOWN


def test_evidence_type_observed():
    f = make_feature(evidence=[make_evidence_item(EvidenceType.OBSERVED)])
    assert f.evidence_type() == EvidenceType.OBSERVED


def test_evidence_type_corroborated_wins_over_observed():
    f = make_feature(evidence=[
        make_evidence_item(EvidenceType.OBSERVED),
        make_evidence_item(EvidenceType.CORROBORATED),
    ])
    assert f.evidence_type() == EvidenceType.CORROBORATED


def test_evidence_type_inference_when_only_inference():
    f = make_feature(evidence=[make_evidence_item(EvidenceType.INFERENCE)])
    assert f.evidence_type() == EvidenceType.INFERENCE


# ── summary() helper ──────────────────────────────────────────────────────────

def test_summary_found_feature():
    f = make_feature()
    s = f.summary()
    assert "instagram_follower_count" in s
    assert "25000" in s
    assert "audience" in s


def test_summary_not_found_feature():
    f = make_feature(value=None)
    s = f.summary()
    assert "not found" in s
