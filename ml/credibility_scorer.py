"""
Credibility Scorer — evaluates how credible an investigation's evidence is.

Takes an ``InvestigationResult`` and produces a structured credibility score
that captures the overall trustworthiness of the collected evidence.  This
is **not** a judgment of the business itself — it measures the *quality
and reliability of the data* the agent gathered.

A high credibility score means:
  - Evidence comes from reliable sources
  - Facts are observed or corroborated (not just inferred)
  - Confidence levels are high
  - Multiple independent sources support the findings

Usage::

    from ml.credibility_scorer import score_credibility, CredibilityScore

    result: CredibilityScore = score_credibility(investigation_result)
    print(result.overall_score)  # 0.72
    print(result.level)          # "high"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability
from agent.schemas.result import InvestigationResult


# ── Credibility score ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CredibilityScore:
    """Structured credibility assessment over an investigation's evidence."""

    source_reliability_score: float   # 0.0 – 1.0  how reliable the sources are
    evidence_quality_score: float     # 0.0 – 1.0  observed/corroborated vs inference
    confidence_score: float           # 0.0 – 1.0  average agent confidence
    reliable_ratio: float             # 0.0 – 1.0  proportion passing is_reliable()
    source_diversity_score: float      # 0.0 – 1.0  distinct source coverage
    corroboration_score: float        # 0.0 – 1.0  cross-source confirmation
    evidence_depth_score: float       # 0.0 – 1.0  is there enough evidence
    overall_score: float              # 0.0 – 1.0  weighted combination
    level: str                        # "high" | "moderate" | "low" | "insufficient_evidence"

    def __post_init__(self) -> None:
        for field in (
            "source_reliability_score", "evidence_quality_score",
            "confidence_score", "reliable_ratio", "source_diversity_score",
            "corroboration_score", "evidence_depth_score", "overall_score",
        ):
            val = getattr(self, field)
            object.__setattr__(self, field, max(0.0, min(1.0, val)))


# ── Weight map for source reliability ────────────────────────────────────────

_RELIABILITY_WEIGHTS = {
    SourceReliability.HIGH: 1.0,
    SourceReliability.MEDIUM: 0.66,
    SourceReliability.LOW: 0.33,
    SourceReliability.UNKNOWN: 0.0,
}

# ── Weight map for evidence type ─────────────────────────────────────────────

_EVIDENCE_TYPE_WEIGHTS = {
    EvidenceType.CORROBORATED: 1.0,
    EvidenceType.OBSERVED: 0.75,
    EvidenceType.INFERENCE: 0.25,
    EvidenceType.UNKNOWN: 0.0,
}

# ── Evidence depth threshold ─────────────────────────────────────────────────

# Number of evidence items considered "sufficient depth".
_MIN_EVIDENCE_DEPTH = 10

# ── Overall score weights ────────────────────────────────────────────────────

_WEIGHTS = {
    "source_reliability": 0.15,
    "evidence_quality": 0.20,
    "confidence": 0.15,
    "reliable_ratio": 0.20,
    "source_diversity": 0.05,
    "corroboration": 0.10,
    "evidence_depth": 0.15,
}

# ── Level thresholds ─────────────────────────────────────────────────────────

_HIGH_THRESHOLD = 0.70
_MODERATE_THRESHOLD = 0.45


# ── Sub-scorers ──────────────────────────────────────────────────────────────

def _score_source_reliability(evidence: List[EvidenceItem]) -> float:
    """Average source reliability across all evidence items."""
    if not evidence:
        return 0.0
    total = sum(_RELIABILITY_WEIGHTS[e.source_reliability] for e in evidence)
    return total / len(evidence)


def _score_evidence_quality(evidence: List[EvidenceItem]) -> float:
    """Average evidence-type quality (corroborated > observed > inference)."""
    if not evidence:
        return 0.0
    total = sum(_EVIDENCE_TYPE_WEIGHTS[e.evidence_type] for e in evidence)
    return total / len(evidence)


def _score_confidence(evidence: List[EvidenceItem]) -> float:
    """Average confidence across all evidence items."""
    if not evidence:
        return 0.0
    return sum(e.confidence for e in evidence) / len(evidence)


def _score_reliable_ratio(evidence: List[EvidenceItem]) -> float:
    """Proportion of evidence items that pass is_reliable()."""
    if not evidence:
        return 0.0
    reliable = sum(1 for e in evidence if e.is_reliable())
    return reliable / len(evidence)


def _score_source_diversity(evidence: List[EvidenceItem]) -> float:
    """
    How many distinct named sources back the evidence.

    Uses source_name; items with no source_name are ignored.
    Score is unique_sources / total_sources, capped at 1.0.
    A single source backing everything → low diversity.
    """
    names = [e.source_name for e in evidence if e.source_name]
    if not names:
        return 0.0
    unique = len(set(names))
    total = len(names)
    return unique / total


def _score_corroboration(evidence: List[EvidenceItem]) -> float:
    """
    How well facts are cross-confirmed across sources.

    A field_name appearing in evidence from multiple distinct sources
    indicates corroboration.  Score is the proportion of field_names
    that appear in 2+ different sources.
    """
    if not evidence:
        return 0.0

    # Map each field_name to the set of source_names that mention it.
    field_sources: dict[str, set[str]] = {}
    for e in evidence:
        src = e.source_name or "__no_source__"
        field_sources.setdefault(e.field_name, set()).add(src)

    if not field_sources:
        return 0.0

    corroborated = sum(
        1 for sources in field_sources.values() if len(sources) >= 2
    )
    return corroborated / len(field_sources)


def _score_evidence_depth(evidence: List[EvidenceItem]) -> float:
    """
    Whether enough evidence has been collected.

    Uses a square-root curve: a few items ramp up quickly, then
    diminishing returns as we approach the depth threshold.
    """
    count = len(evidence)
    if count == 0:
        return 0.0
    # sqrt curve normalised so that _MIN_EVIDENCE_DEPTH → 1.0
    return min(1.0, math.sqrt(count / _MIN_EVIDENCE_DEPTH))


def _classify_level(overall: float, evidence_count: int) -> str:
    """Map an overall credibility score to a human-readable level."""
    if evidence_count == 0:
        return "insufficient_evidence"
    if overall >= _HIGH_THRESHOLD:
        return "high"
    if overall >= _MODERATE_THRESHOLD:
        return "moderate"
    return "low"


# ── Main entry point ─────────────────────────────────────────────────────────

def score_credibility(result: InvestigationResult) -> CredibilityScore:
    """
    Score the credibility of an investigation's evidence.

    Evaluates source reliability, evidence quality, confidence levels,
    reliable-evidence ratio, source diversity, corroboration, and evidence
    depth.  Returns a weighted overall score with a human-readable level.
    """
    evidence = result.evidence

    src_rel = _score_source_reliability(evidence)
    ev_qual = _score_evidence_quality(evidence)
    conf = _score_confidence(evidence)
    rel_ratio = _score_reliable_ratio(evidence)
    src_div = _score_source_diversity(evidence)
    corrob = _score_corroboration(evidence)
    depth = _score_evidence_depth(evidence)

    overall = (
        _WEIGHTS["source_reliability"] * src_rel
        + _WEIGHTS["evidence_quality"] * ev_qual
        + _WEIGHTS["confidence"] * conf
        + _WEIGHTS["reliable_ratio"] * rel_ratio
        + _WEIGHTS["source_diversity"] * src_div
        + _WEIGHTS["corroboration"] * corrob
        + _WEIGHTS["evidence_depth"] * depth
    )

    level = _classify_level(overall, len(evidence))

    return CredibilityScore(
        source_reliability_score=src_rel,
        evidence_quality_score=ev_qual,
        confidence_score=conf,
        reliable_ratio=rel_ratio,
        source_diversity_score=src_div,
        corroboration_score=corrob,
        evidence_depth_score=depth,
        overall_score=overall,
        level=level,
    )
