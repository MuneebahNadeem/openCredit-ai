"""
Feature Extractor — converts InvestigationResult into a flat numeric feature dict.

This is the bridge between Person 1's structured agent output and the ML
pipeline.  Every function returns a plain ``dict[str, float]`` so the result
can be fed directly into scikit-learn, XGBoost, or any other tabular model.

Usage::

    from ml.feature_extractor import extract_features

    features: dict[str, float] = extract_features(investigation_result)
"""

from __future__ import annotations

import math
from typing import Dict, List

from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability
from agent.schemas.feature import DiscoveredFeature, FeatureCategory
from agent.schemas.result import InvestigationResult, InvestigationStatus

# Type alias used throughout this module.
FeatureDict = Dict[str, float]


# ── Evidence counts ───────────────────────────────────────────────────────────

def extract_evidence_counts(evidence: List[EvidenceItem]) -> FeatureDict:
    """
    Count-based features derived from the raw evidence list.

    Returns total / reliable counts, the reliable ratio, and a breakdown
    by EvidenceType (observed, corroborated, inference, unknown).
    """
    total = len(evidence)
    reliable = sum(1 for e in evidence if e.is_reliable())

    type_counts = {t: 0 for t in EvidenceType}
    for e in evidence:
        type_counts[e.evidence_type] += 1

    return {
        "evidence_count_total": float(total),
        "evidence_count_reliable": float(reliable),
        "evidence_reliable_ratio": reliable / total if total > 0 else 0.0,
        "evidence_observed_count": float(type_counts[EvidenceType.OBSERVED]),
        "evidence_corroborated_count": float(type_counts[EvidenceType.CORROBORATED]),
        "evidence_inference_count": float(type_counts[EvidenceType.INFERENCE]),
        "evidence_unknown_count": float(type_counts[EvidenceType.UNKNOWN]),
    }


# ── Confidence statistics ─────────────────────────────────────────────────────

def extract_confidence_stats(evidence: List[EvidenceItem]) -> FeatureDict:
    """
    Descriptive statistics over per-evidence confidence scores.

    Returns mean, min, max, and population standard deviation.
    All values are 0.0 when no evidence items are present.
    """
    if not evidence:
        return {
            "confidence_mean": 0.0,
            "confidence_min": 0.0,
            "confidence_max": 0.0,
            "confidence_std": 0.0,
        }

    scores = [e.confidence for e in evidence]
    n = len(scores)
    mean = sum(scores) / n
    min_val = min(scores)
    max_val = max(scores)
    variance = sum((s - mean) ** 2 for s in scores) / n
    std = math.sqrt(variance)

    return {
        "confidence_mean": mean,
        "confidence_min": min_val,
        "confidence_max": max_val,
        "confidence_std": std,
    }


# ── Signal ratios ─────────────────────────────────────────────────────────────

def extract_signal_ratios(result: InvestigationResult) -> FeatureDict:
    """
    Ratio features derived from positive / risk signals and missing info.

    A high positive-to-total ratio is generally favourable; a high
    risk-to-total ratio is generally unfavourable.  Missing information
    count is included as an absolute feature (more gaps → less certainty).
    """
    pos = len(result.positive_signals)
    risk = len(result.risk_signals)
    total = pos + risk

    return {
        "positive_signal_count": float(pos),
        "risk_signal_count": float(risk),
        "signal_count_total": float(total),
        "positive_signal_ratio": pos / total if total > 0 else 0.0,
        "risk_signal_ratio": risk / total if total > 0 else 0.0,
        "missing_information_count": float(len(result.missing_information)),
    }


# ── Source reliability distribution ───────────────────────────────────────────

def extract_source_reliability(evidence: List[EvidenceItem]) -> FeatureDict:
    """
    Distribution of evidence across SourceReliability levels.

    Each value is the *ratio* of evidence items at that reliability level
    relative to the total, giving a normalised profile that is comparable
    across investigations of different sizes.
    """
    total = len(evidence)

    if total == 0:
        return {
            "source_reliability_high_ratio": 0.0,
            "source_reliability_medium_ratio": 0.0,
            "source_reliability_low_ratio": 0.0,
            "source_reliability_unknown_ratio": 0.0,
        }

    counts = {r: 0 for r in SourceReliability}
    for e in evidence:
        counts[e.source_reliability] += 1

    return {
        "source_reliability_high_ratio": counts[SourceReliability.HIGH] / total,
        "source_reliability_medium_ratio": counts[SourceReliability.MEDIUM] / total,
        "source_reliability_low_ratio": counts[SourceReliability.LOW] / total,
        "source_reliability_unknown_ratio": counts[SourceReliability.UNKNOWN] / total,
    }


# ── Feature category breakdown ───────────────────────────────────────────────

def extract_feature_categories(features: List[DiscoveredFeature]) -> FeatureDict:
    """
    Category-level features derived from the discovered feature list.

    Captures how many features were identified, how many were actually
    found (vs. searched-but-not-found), and the count per category.
    This gives the model a view of *breadth* (many categories covered)
    vs. *depth* (many features within a category).
    """
    total = len(features)
    found = sum(1 for f in features if f.is_found())
    searched = sum(1 for f in features if f.searched)

    # Count per category — every enum member gets a key, even if zero.
    cat_counts: FeatureDict = {}
    for cat in FeatureCategory:
        key = f"features_cat_{cat.value}"
        cat_counts[key] = 0.0

    for f in features:
        key = f"features_cat_{f.category.value}"
        cat_counts[key] += 1.0

    return {
        "features_total": float(total),
        "features_found": float(found),
        "feature_found_ratio": found / total if total > 0 else 0.0,
        "features_searched": float(searched),
        "feature_searched_ratio": searched / total if total > 0 else 0.0,
        **cat_counts,
    }


# ── Investigation metadata ───────────────────────────────────────────────────

def extract_investigation_meta(result: InvestigationResult) -> FeatureDict:
    """
    Lightweight metadata about the investigation run itself.

    Includes effort indicators (searches, sources) and one-hot flags for
    the InvestigationStatus so the model can learn that partial or failed
    investigations may carry less reliable evidence.
    """
    return {
        "searches_performed": float(result.searches_performed),
        "sources_examined": float(result.sources_examined),
        "unique_sources_count": float(len(result.sources)),
        "status_complete": 1.0 if result.status == InvestigationStatus.COMPLETE else 0.0,
        "status_limit_reached": 1.0 if result.status == InvestigationStatus.LIMIT_REACHED else 0.0,
        "status_partial": 1.0 if result.status == InvestigationStatus.PARTIAL else 0.0,
        "status_failed": 1.0 if result.status == InvestigationStatus.FAILED else 0.0,
    }


# ── Combined extraction ──────────────────────────────────────────────────────

def extract_features(result: InvestigationResult) -> FeatureDict:
    """
    Extract **all** numeric features from an InvestigationResult.

    This is the main entry point.  It delegates to the sub-extractors and
    merges their outputs into a single flat ``dict[str, float]`` suitable
    for any tabular ML model.

    Example::

        features = extract_features(result)
        # features["evidence_count_total"] == 12.0
        # features["confidence_mean"]      == 0.81
        # ...
    """
    features: FeatureDict = {}
    features.update(extract_evidence_counts(result.evidence))
    features.update(extract_confidence_stats(result.evidence))
    features.update(extract_signal_ratios(result))
    features.update(extract_source_reliability(result.evidence))
    features.update(extract_feature_categories(result.features))
    features.update(extract_investigation_meta(result))
    return features
