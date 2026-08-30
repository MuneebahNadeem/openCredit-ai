"""
Tests for agent/state.py — InvestigationState behaviour.

Run with:  python -m pytest tests/agent/test_state.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from agent.config import InvestigationConfig
from agent.state import InvestigationState
from agent.schemas.input import BusinessInput
from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability
from agent.schemas.feature import DiscoveredFeature, FeatureCategory
from agent.schemas.result import InvestigationStatus, Signal


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_config(**kwargs) -> InvestigationConfig:
    defaults = dict(max_searches=5, max_sources=5, max_iterations=10, max_evidence_items=10)
    defaults.update(kwargs)
    return InvestigationConfig(**defaults)


def make_state(**kwargs) -> InvestigationState:
    return InvestigationState(config=make_config(**kwargs))


def make_evidence(field="ev", reliability=SourceReliability.HIGH, confidence=0.9) -> EvidenceItem:
    return EvidenceItem(
        field_name=field,
        value="test",
        evidence_type=EvidenceType.OBSERVED,
        source_name="Source",
        source_reliability=reliability,
        confidence=confidence,
    )


def make_feature(name="feat") -> DiscoveredFeature:
    return DiscoveredFeature(
        name=name,
        category=FeatureCategory.AUDIENCE,
        value="100",
        reason="Test",
        confidence=0.8,
    )


def make_business() -> BusinessInput:
    return BusinessInput(name="Test Co", location="Karachi")


# ── Search recording ──────────────────────────────────────────────────────────

class TestRecordSearch:

    def test_new_query_returns_true(self):
        s = make_state()
        assert s.record_search("test query") is True

    def test_duplicate_query_returns_false(self):
        s = make_state()
        s.record_search("test query")
        assert s.record_search("test query") is False

    def test_case_insensitive_dedup(self):
        s = make_state()
        s.record_search("Test Query")
        assert s.record_search("test query") is False

    def test_whitespace_normalised(self):
        s = make_state()
        s.record_search("  hello  ")
        assert s.record_search("hello") is False

    def test_counter_increments_on_new(self):
        s = make_state()
        s.record_search("q1")
        s.record_search("q2")
        assert s.searches_performed == 2

    def test_counter_does_not_increment_on_duplicate(self):
        s = make_state()
        s.record_search("q1")
        s.record_search("q1")
        assert s.searches_performed == 1

    def test_has_searched_true_after_record(self):
        s = make_state()
        s.record_search("karachi fashion")
        assert s.has_searched("Karachi Fashion") is True

    def test_has_searched_false_before(self):
        s = make_state()
        assert s.has_searched("karachi fashion") is False


# ── Source recording ──────────────────────────────────────────────────────────

class TestAddSource:

    def test_new_url_returns_true(self):
        s = make_state()
        assert s.add_source("https://example.com") is True

    def test_duplicate_url_returns_false(self):
        s = make_state()
        s.add_source("https://example.com")
        assert s.add_source("https://example.com") is False

    def test_trailing_slash_normalised(self):
        s = make_state()
        s.add_source("https://example.com/")
        assert s.add_source("https://example.com") is False

    def test_has_visited_true_after_add(self):
        s = make_state()
        s.add_source("https://x.com")
        assert s.has_visited("https://x.com") is True

    def test_has_visited_false_before(self):
        s = make_state()
        assert s.has_visited("https://x.com") is False


# ── Evidence accumulation ─────────────────────────────────────────────────────

class TestEvidenceAccumulation:

    def test_add_evidence_increases_count(self):
        s = make_state()
        s.add_evidence(make_evidence())
        assert s.evidence_count == 1

    def test_add_evidence_batch(self):
        s = make_state()
        s.add_evidence_batch([make_evidence("a"), make_evidence("b"), make_evidence("c")])
        assert s.evidence_count == 3

    def test_reliable_evidence_count_counts_reliable(self):
        s = make_state()
        s.add_evidence(make_evidence("r", SourceReliability.HIGH, 0.9))
        s.add_evidence(make_evidence("u", SourceReliability.LOW, 0.2))
        # reliable: HIGH + OBSERVED + confidence>=0.7
        assert s.reliable_evidence_count == 1

    def test_reliable_evidence_count_zero_initially(self):
        s = make_state()
        assert s.reliable_evidence_count == 0


# ── Feature accumulation ──────────────────────────────────────────────────────

class TestFeatureAccumulation:

    def test_add_feature_appends(self):
        s = make_state()
        s.add_feature(make_feature("f1"))
        assert len(s.features) == 1

    def test_add_feature_replaces_by_name(self):
        s = make_state()
        s.add_feature(make_feature("f1"))
        updated = DiscoveredFeature(
            name="f1", category=FeatureCategory.REPUTATION,
            value="new_value", reason="Updated", confidence=0.9,
        )
        s.add_feature(updated)
        assert len(s.features) == 1
        assert s.get_feature("f1").value == "new_value"

    def test_get_feature_returns_none_if_missing(self):
        s = make_state()
        assert s.get_feature("nonexistent") is None

    def test_get_feature_finds_by_name(self):
        s = make_state()
        s.add_feature(make_feature("target"))
        assert s.get_feature("target") is not None


# ── Signal and missing info ───────────────────────────────────────────────────

class TestSignals:

    def test_add_positive_signal(self):
        s = make_state()
        s.add_positive_signal(Signal(label="Good", detail="Good signal", evidence_refs=[]))
        assert len(s.positive_signals) == 1

    def test_add_risk_signal(self):
        s = make_state()
        s.add_risk_signal(Signal(label="Bad", detail="Risk signal", evidence_refs=[]))
        assert len(s.risk_signals) == 1

    def test_add_missing_info_dedup(self):
        s = make_state()
        s.add_missing_info("SECP registration")
        s.add_missing_info("SECP registration")
        assert s.missing_information.count("SECP registration") == 1

    def test_add_missing_info_multiple_distinct(self):
        s = make_state()
        s.add_missing_info("SECP registration")
        s.add_missing_info("Tax ID")
        assert len(s.missing_information) == 2


# ── Stop conditions ───────────────────────────────────────────────────────────

class TestStopConditions:

    def test_should_stop_false_initially(self):
        s = make_state()
        assert s.should_stop() is False

    def test_stops_at_max_searches(self):
        s = make_state(max_searches=2)
        s.record_search("q1")
        s.record_search("q2")
        assert s.should_stop() is True
        assert s.stop_reason == "max_searches_reached"

    def test_stops_at_max_sources(self):
        s = make_state(max_sources=2)
        s.add_source("https://a.com")
        s.add_source("https://b.com")
        assert s.should_stop() is True
        assert s.stop_reason == "max_sources_reached"

    def test_stops_at_max_iterations(self):
        s = make_state(max_iterations=3)
        s.iteration = 3
        assert s.should_stop() is True
        assert s.stop_reason == "max_iterations_reached"

    def test_stops_at_max_evidence(self):
        s = make_state(max_evidence_items=2)
        s.add_evidence(make_evidence("e1"))
        s.add_evidence(make_evidence("e2"))
        assert s.should_stop() is True
        assert s.stop_reason == "max_evidence_reached"

    def test_explicit_stop(self):
        s = make_state()
        s.stop("agent_decided")
        assert s.should_stop() is True
        assert s.stop_reason == "agent_decided"

    def test_stop_reason_sticks_once_set(self):
        s = make_state(max_searches=1)
        s.record_search("q1")
        s.should_stop()
        assert s.stop_reason == "max_searches_reached"
        # subsequent call should still return True
        assert s.should_stop() is True


# ── Sufficient evidence ───────────────────────────────────────────────────────

class TestSufficientEvidence:

    def test_false_when_empty(self):
        s = make_state(min_evidence_to_conclude=3)
        assert s.has_sufficient_evidence() is False

    def test_false_when_unreliable_only(self):
        s = make_state(min_evidence_to_conclude=1)
        s.add_evidence(make_evidence("u", SourceReliability.LOW, 0.2))
        assert s.has_sufficient_evidence() is False

    def test_true_when_reliable_meets_threshold(self):
        s = make_state(min_evidence_to_conclude=2)
        s.add_evidence(make_evidence("r1", SourceReliability.HIGH, 0.9))
        s.add_evidence(make_evidence("r2", SourceReliability.MEDIUM, 0.8))
        assert s.has_sufficient_evidence() is True


# ── Status derivation ────────────────────────────────────────────────────────

class TestDeriveStatus:

    def test_no_stop_reason_is_complete(self):
        s = make_state()
        assert s._derive_status() == InvestigationStatus.COMPLETE

    def test_agent_decided_is_complete(self):
        s = make_state()
        s.stop("agent_decided")
        assert s._derive_status() == InvestigationStatus.COMPLETE

    def test_max_searches_is_limit_reached(self):
        s = make_state()
        s.stop("max_searches_reached")
        assert s._derive_status() == InvestigationStatus.LIMIT_REACHED

    def test_max_sources_is_limit_reached(self):
        s = make_state()
        s.stop("max_sources_reached")
        assert s._derive_status() == InvestigationStatus.LIMIT_REACHED

    def test_partial_failure_is_partial(self):
        s = make_state()
        s.stop("partial_failure")
        assert s._derive_status() == InvestigationStatus.PARTIAL

    def test_failed_is_failed(self):
        s = make_state()
        s.stop("failed")
        assert s._derive_status() == InvestigationStatus.FAILED


# ── build_result ─────────────────────────────────────────────────────────────

class TestBuildResult:

    def test_returns_investigation_result(self):
        from agent.schemas.result import InvestigationResult
        s = make_state()
        result = s.build_result(make_business())
        assert isinstance(result, InvestigationResult)

    def test_evidence_copied_to_result(self):
        s = make_state()
        s.add_evidence(make_evidence("field_x"))
        result = s.build_result(make_business())
        assert any(e.field_name == "field_x" for e in result.evidence)

    def test_searches_performed_in_result(self):
        s = make_state()
        s.record_search("query1")
        result = s.build_result(make_business())
        assert result.searches_performed == 1

    def test_sources_examined_matches_sources_read(self):
        s = make_state()
        s.add_source("https://a.com")
        s.add_source("https://b.com")
        result = s.build_result(make_business())
        assert result.sources_examined == 2

    def test_status_reflects_stop_reason(self):
        s = make_state()
        s.stop("max_searches_reached")
        result = s.build_result(make_business())
        assert result.status == InvestigationStatus.LIMIT_REACHED


# ── summary() ────────────────────────────────────────────────────────────────

class TestSummary:

    def test_summary_is_string(self):
        s = make_state()
        assert isinstance(s.summary(), str)

    def test_summary_contains_iteration(self):
        s = make_state()
        s.iteration = 4
        assert "4" in s.summary()
