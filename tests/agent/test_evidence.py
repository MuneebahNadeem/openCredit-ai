"""
Tests for agent/schemas/evidence.py — EvidenceItem model.

Run with:  python -m pytest tests/agent/test_evidence.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from pydantic import ValidationError
from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_evidence(**kwargs) -> EvidenceItem:
    defaults = dict(
        field_name="instagram_followers",
        value="25000",
        evidence_type=EvidenceType.OBSERVED,
        source_name="Instagram",
        source_url="https://instagram.com/testbiz",
        source_reliability=SourceReliability.MEDIUM,
        confidence=0.9,
    )
    defaults.update(kwargs)
    return EvidenceItem(**defaults)


# ── Valid construction ────────────────────────────────────────────────────────

def test_minimal_evidence():
    e = EvidenceItem(
        field_name="website_present",
        value="true",
        evidence_type=EvidenceType.OBSERVED,
    )
    assert e.field_name == "website_present"
    assert e.evidence_type == EvidenceType.OBSERVED
    assert e.confidence == 1.0
    assert e.source_reliability == SourceReliability.UNKNOWN


def test_full_evidence():
    e = make_evidence()
    assert e.field_name == "instagram_followers"
    assert e.value == "25000"
    assert e.confidence == 0.9


def test_inference_type_allowed():
    e = make_evidence(evidence_type=EvidenceType.INFERENCE, confidence=0.5)
    assert e.evidence_type == EvidenceType.INFERENCE


def test_unknown_type_allowed():
    e = make_evidence(evidence_type=EvidenceType.UNKNOWN, confidence=0.0)
    assert e.evidence_type == EvidenceType.UNKNOWN


def test_corroborated_type_allowed():
    e = make_evidence(evidence_type=EvidenceType.CORROBORATED, confidence=1.0)
    assert e.evidence_type == EvidenceType.CORROBORATED


# ── Validation errors ─────────────────────────────────────────────────────────

def test_blank_field_name_raises():
    with pytest.raises(ValidationError):
        make_evidence(field_name="   ")


def test_blank_value_raises():
    with pytest.raises(ValidationError):
        make_evidence(value="   ")


def test_confidence_above_1_raises():
    with pytest.raises(ValidationError):
        make_evidence(confidence=1.1)


def test_confidence_below_0_raises():
    with pytest.raises(ValidationError):
        make_evidence(confidence=-0.1)


def test_invalid_source_url_raises():
    with pytest.raises(ValidationError):
        make_evidence(source_url="not-a-url")


# ── is_reliable() helper ──────────────────────────────────────────────────────

def test_is_reliable_high_confidence_medium_source():
    e = make_evidence(confidence=0.9, source_reliability=SourceReliability.MEDIUM,
                      evidence_type=EvidenceType.OBSERVED)
    assert e.is_reliable() is True


def test_not_reliable_low_confidence():
    e = make_evidence(confidence=0.5)
    assert e.is_reliable() is False


def test_not_reliable_low_source():
    e = make_evidence(confidence=0.9, source_reliability=SourceReliability.LOW)
    assert e.is_reliable() is False


def test_not_reliable_inference():
    e = make_evidence(confidence=0.9, source_reliability=SourceReliability.HIGH,
                      evidence_type=EvidenceType.INFERENCE)
    assert e.is_reliable() is False


# ── summary() helper ──────────────────────────────────────────────────────────

def test_summary_contains_field_name_and_value():
    e = make_evidence()
    s = e.summary()
    assert "instagram_followers" in s
    assert "25000" in s
    assert "observed" in s
    assert "Instagram" in s
