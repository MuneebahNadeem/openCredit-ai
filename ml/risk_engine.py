"""
Risk Engine — central scoring module that combines all ML components.

Takes an ``InvestigationResult`` from Person 1 and produces the two main
assessments (trustworthiness and business potential) by orchestrating:

1. **Feature extraction** — numeric evidence features (60 features)
2. **Sentiment analysis** — text-based evidence sentiment
3. **Credibility scoring** — evidence reliability assessment
4. **Trained models** — saved classifiers blended with the rule scores

The output is a ``RiskAssessment`` containing two ``AssessmentScore``
objects that are directly compatible with the ``InvestigationResult``
schema, plus the intermediate scores for transparency.

Usage::

    from ml.risk_engine import assess_risk, RiskAssessment

    assessment = assess_risk(investigation_result)
    print(assessment.trustworthiness.level)     # "moderate"
    print(assessment.business_potential.score)  # 0.72
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from agent.schemas.result import (
    AssessmentLevel,
    AssessmentScore,
    InvestigationResult,
)
from ml.credibility_scorer import CredibilityScore, score_credibility
from ml.feature_extractor import FeatureDict, extract_features
from ml.model_predictor import ModelPrediction, ModelPredictor, get_predictor
from ml.sentiment import SentimentScore, score_evidence_texts


# ── Risk assessment output ───────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskAssessment:
    """Complete risk assessment combining all ML module outputs."""

    trustworthiness: AssessmentScore
    business_potential: AssessmentScore
    credibility: CredibilityScore
    sentiment: SentimentScore
    features: FeatureDict
    # The trained-model prediction that fed the hybrid blend (None when
    # the models were unavailable or there was no evidence to score).
    model_prediction: Optional[ModelPrediction] = None


# ── Hybrid blend ─────────────────────────────────────────────────────────────

# Weight of the trained model in the final score; the rule engine keeps
# the remainder.  0.5 = equal hybrid.  The models are synthetic-trained,
# so the rule engine remains the evidence-grounded safety net.
_MODEL_WEIGHT = 0.5


def _blend_scores(
    rule_score: float,
    model_score: Optional[float],
    model_weight: float,
) -> float:
    """
    Blend a rule-engine score with a trained-model score.

    Falls back to the pure rule score when no model prediction is
    available — the rule engine is the safety net.
    """
    if model_score is None:
        return rule_score
    blended = model_weight * model_score + (1.0 - model_weight) * rule_score
    return max(0.0, min(1.0, blended))


# ── Score-to-level mapping ───────────────────────────────────────────────────

_HIGH_THRESHOLD = 0.70
_MODERATE_THRESHOLD = 0.45


def _score_to_level(score: float) -> AssessmentLevel:
    """Map a 0.0–1.0 numeric score to an AssessmentLevel."""
    if score >= _HIGH_THRESHOLD:
        return AssessmentLevel.HIGH
    if score >= _MODERATE_THRESHOLD:
        return AssessmentLevel.MODERATE
    return AssessmentLevel.LOW


# ── Explanation generation ───────────────────────────────────────────────────

def _explain_trustworthiness(
    score: float, credibility: CredibilityScore, features: FeatureDict,
    model_used: bool = False,
) -> str:
    """Generate a plain-English explanation for the trustworthiness assessment."""
    parts: List[str] = []

    if credibility.overall_score >= 0.70:
        parts.append("Evidence comes from highly credible sources")
    elif credibility.overall_score >= 0.45:
        parts.append("Evidence credibility is moderate")
    else:
        parts.append("Evidence credibility is low")

    reliable_pct = features.get("evidence_reliable_ratio", 0.0)
    if reliable_pct >= 0.75:
        parts.append("most evidence is reliable")
    elif reliable_pct >= 0.40:
        parts.append("some evidence is reliable")
    else:
        parts.append("limited reliable evidence found")

    risk_ratio = features.get("risk_signal_ratio", 0.0)
    if risk_ratio >= 0.50:
        parts.append("significant risk signals detected")
    elif risk_ratio > 0.0:
        parts.append("some risk signals present")

    if model_used:
        parts.append("score blends evidence-based rules with a trained ML model")

    return ". ".join(parts) + "."


def _explain_business_potential(
    score: float, sentiment: SentimentScore, features: FeatureDict,
    model_used: bool = False,
) -> str:
    """Generate a plain-English explanation for the business potential assessment."""
    parts: List[str] = []

    pos_ratio = features.get("positive_signal_ratio", 0.0)
    if pos_ratio >= 0.60:
        parts.append("Strong positive signals about market presence")
    elif pos_ratio >= 0.30:
        parts.append("Some positive market signals found")
    else:
        parts.append("Limited positive market signals")

    if sentiment.compound > 0.25:
        parts.append("public sentiment is positive")
    elif sentiment.compound < -0.25:
        parts.append("public sentiment is negative")
    else:
        parts.append("public sentiment is mixed or neutral")

    audience = features.get("features_cat_audience", 0.0)
    engagement = features.get("features_cat_engagement", 0.0)
    demand = features.get("features_cat_demand", 0.0)
    growth = features.get("features_cat_growth", 0.0)
    market = features.get("features_cat_market_presence", 0.0)
    potential_signals = audience + engagement + demand + growth + market
    if potential_signals >= 3.0:
        parts.append("multiple business potential indicators identified")
    elif potential_signals >= 1.0:
        parts.append("some business potential indicators found")

    if model_used:
        parts.append("score blends evidence-based rules with a trained ML model")

    return ". ".join(parts) + "."


# ── Trustworthiness scoring ──────────────────────────────────────────────────

# Weight map for trustworthiness score.
_TRUST_WEIGHTS = {
    "credibility": 0.35,
    "sentiment": 0.15,
    "positive_signals": 0.15,
    "risk_signals": 0.15,
    "reliable_evidence": 0.10,
    "source_quality": 0.10,
}


def _score_trustworthiness(
    features: FeatureDict,
    sentiment: SentimentScore,
    credibility: CredibilityScore,
) -> float:
    """
    Compute a trustworthiness score (0.0–1.0).

    Factors:
    - Evidence credibility (35%)
    - Sentiment (15%) — compound normalised from [-1, +1] to [0, 1]
    - Positive signal ratio (15%)
    - Risk signal ratio inverted (15%) — fewer risks = higher score
    - Reliable evidence ratio (10%)
    - Source quality — HIGH reliability ratio (10%)
    """
    # Normalise sentiment compound from [-1, +1] → [0, 1]
    sentiment_norm = max(0.0, sentiment.compound)

    risk_ratio = features.get("risk_signal_ratio", 0.0)

    score = (
        _TRUST_WEIGHTS["credibility"] * credibility.overall_score
        + _TRUST_WEIGHTS["sentiment"] * sentiment_norm
        + _TRUST_WEIGHTS["positive_signals"] * features.get("positive_signal_ratio", 0.0)
        + _TRUST_WEIGHTS["risk_signals"] * (1.0 - risk_ratio)
        + _TRUST_WEIGHTS["reliable_evidence"] * features.get("evidence_reliable_ratio", 0.0)
        + _TRUST_WEIGHTS["source_quality"] * features.get("source_reliability_high_ratio", 0.0)
    )

    return max(0.0, min(1.0, score))


# ── Business potential scoring ───────────────────────────────────────────────

_POTENTIAL_WEIGHTS = {
    "positive_signals": 0.25,
    "sentiment": 0.20,
    "credibility": 0.15,
    "reliable_evidence": 0.15,
    "business_features": 0.25,
}

def _score_business_potential(
    features: FeatureDict,
    sentiment: SentimentScore,
    credibility: CredibilityScore,
) -> float:
    """
    Compute a business potential score (0.0–1.0).

    Factors:
    - Positive signal ratio (25%)
    - Positive sentiment (20%)
    - Evidence credibility (15%)
    - Reliable evidence ratio (15%)
    - Business potential features (25%)
    """
    sentiment_norm = max(0.0, sentiment.compound)

    total_features = features.get("features_total", 0.0)

    if total_features > 0:
        bp_count = (
            features.get("features_cat_audience", 0.0)
            + features.get("features_cat_engagement", 0.0)
            + features.get("features_cat_demand", 0.0)
            + features.get("features_cat_growth", 0.0)
            + features.get("features_cat_market_presence", 0.0)
        )
        bp_ratio = min(1.0, bp_count / total_features)
    else:
        bp_ratio = 0.0

    score = (
        _POTENTIAL_WEIGHTS["positive_signals"]
        * features.get("positive_signal_ratio", 0.0)
        + _POTENTIAL_WEIGHTS["sentiment"] * sentiment_norm
        + _POTENTIAL_WEIGHTS["credibility"] * credibility.overall_score
        + _POTENTIAL_WEIGHTS["reliable_evidence"]
        * features.get("evidence_reliable_ratio", 0.0)
        + _POTENTIAL_WEIGHTS["business_features"] * bp_ratio
    )

    return max(0.0, min(1.0, score))

# ── Main entry point ─────────────────────────────────────────────────────────

def assess_risk(
    result: InvestigationResult,
    predictor: Optional[ModelPredictor] = None,
    model_weight: float = _MODEL_WEIGHT,
) -> RiskAssessment:
    """
    Produce a complete risk assessment for an investigation.

    Orchestrates feature extraction, sentiment analysis, and credibility
    scoring, then combines them into trustworthiness and business potential
    assessments.  Returns a ``RiskAssessment`` whose ``trustworthiness``
    and ``business_potential`` fields are ``AssessmentScore`` objects
    directly compatible with ``InvestigationResult``.

    **Hybrid scoring:** each final score blends the rule-engine score with
    the trained-model prediction (``model_weight`` = model share, default
    0.5).  When the saved models are unavailable — or ``predictor`` returns
    an unavailable prediction — the pure rule score is used, so the engine
    degrades gracefully and never fails an investigation.

    Parameters
    ----------
    result:
        The investigation output from Person 1's agent.
    predictor:
        The model predictor to use.  ``None`` (default) uses the
        process-wide singleton, which loads the saved artifacts from
        ``data/models/`` on first use.  Inject a stub in tests.
    model_weight:
        Weight of the model score in the blend (0.0 = pure rules,
        1.0 = pure model).  Must be within [0.0, 1.0].

    When no evidence is present, both assessments return
    ``INSUFFICIENT_EVIDENCE`` with no numeric score.
    """
    if not 0.0 <= model_weight <= 1.0:
        raise ValueError(
            f"model_weight must be within [0.0, 1.0], got {model_weight}"
        )

    # ── Run sub-modules ──────────────────────────────────────────────────
    features = extract_features(result)
    sentiment = score_evidence_texts(result.evidence)
    credibility = score_credibility(result)

    evidence_count = len(result.reliable_evidence())

    # ── No evidence → insufficient ───────────────────────────────────────
    if not result.evidence:
        insufficient = AssessmentScore(
            level=AssessmentLevel.INSUFFICIENT_EVIDENCE,
            score=None,
            evidence_count=0,
            explanation="No evidence was collected during the investigation.",
        )
        return RiskAssessment(
            trustworthiness=insufficient,
            business_potential=insufficient,
            credibility=credibility,
            sentiment=sentiment,
            features=features,
        )

    # ── Trained-model prediction (hybrid blend) ──────────────────────────
    if predictor is None:
        predictor = get_predictor()
    # Defensive: a broken predictor must never fail an investigation.
    try:
        prediction = predictor.predict(features)
    except Exception:
        prediction = ModelPrediction(
            trust_score=None, potential_score=None,
            trust_model=None, potential_model=None,
            available=False, reason="predictor raised during prediction",
        )
    model_used = prediction.available

    # ── Compute scores ───────────────────────────────────────────────────
    trust_rule = _score_trustworthiness(features, sentiment, credibility)
    potential_rule = _score_business_potential(features, sentiment, credibility)

    trust_raw = _blend_scores(trust_rule, prediction.trust_score, model_weight)
    potential_raw = _blend_scores(
        potential_rule, prediction.potential_score, model_weight,
    )

    trust_level = _score_to_level(trust_raw)
    potential_level = _score_to_level(potential_raw)

    trust_explanation = _explain_trustworthiness(
        trust_raw, credibility, features, model_used=model_used,
    )
    potential_explanation = _explain_business_potential(
        potential_raw, sentiment, features, model_used=model_used,
    )

    trustworthiness = AssessmentScore(
        level=trust_level,
        score=round(trust_raw, 4),
        evidence_count=evidence_count,
        explanation=trust_explanation,
    )

    business_potential = AssessmentScore(
        level=potential_level,
        score=round(potential_raw, 4),
        evidence_count=evidence_count,
        explanation=potential_explanation,
    )

    return RiskAssessment(
        trustworthiness=trustworthiness,
        business_potential=business_potential,
        credibility=credibility,
        sentiment=sentiment,
        features=features,
        model_prediction=prediction,
    )
