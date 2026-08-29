"""
Assessment — final wrapper that produces a complete InvestigationResult.

Takes an ``InvestigationResult`` from Person 1 (agent) that already contains
evidence, features, signals, and metadata, then runs all ML modules through
the risk engine to populate the two main assessments (trustworthiness and
business potential).  Also generates a justification and an overall
recommendation.

The output is a fully populated ``InvestigationResult`` ready for Person 3's
backend to consume — no further ML processing needed.

Usage::

    from ml.assessment import generate_assessment

    result: InvestigationResult = agent.invest(business_input)
    enriched: InvestigationResult = generate_assessment(result)
    print(enriched.trustworthiness.level)   # "moderate"
    print(enriched.justification)           # "Evidence credibility is moderate..."
"""

from __future__ import annotations

from typing import List

from agent.schemas.result import (
    AssessmentLevel,
    AssessmentScore,
    InvestigationResult,
)
from ml.credibility_scorer import CredibilityScore
from ml.risk_engine import RiskAssessment, assess_risk
from ml.sentiment import SentimentScore


# ── Recommendation categories ────────────────────────────────────────────────

_RECOMMEND_APPROVE = "approve"
_RECOMMEND_APPROVE_CONDITIONAL = "approve_with_conditions"
_RECOMMEND_DECLINE = "decline"
_RECOMMEND_REVIEW = "further_review"
_RECOMMEND_INSUFFICIENT = "insufficient_data"


# ── Justification generation ─────────────────────────────────────────────────

def generate_justification(
    result: InvestigationResult,
    assessment: RiskAssessment,
) -> str:
    """
    Generate an evidence-based justification (~2 lines).

    Combines the trustworthiness and business potential explanations
    with credibility and sentiment context.  Never speculative — only
    reflects what the evidence supports.
    """
    trust = assessment.trustworthiness
    potential = assessment.business_potential

    parts: List[str] = []

    # Trustworthiness summary
    if trust.level == AssessmentLevel.INSUFFICIENT_EVIDENCE:
        parts.append("Insufficient evidence to assess trustworthiness")
    elif trust.level == AssessmentLevel.HIGH:
        parts.append(
            f"Trustworthiness is high (score {trust.score})"
        )
    elif trust.level == AssessmentLevel.MODERATE:
        parts.append(
            f"Trustworthiness is moderate (score {trust.score})"
        )
    else:
        parts.append(
            f"Trustworthiness is low (score {trust.score})"
        )

    # Business potential summary
    if potential.level == AssessmentLevel.INSUFFICIENT_EVIDENCE:
        parts.append("insufficient evidence for business potential")
    elif potential.level == AssessmentLevel.HIGH:
        parts.append(
            f"business potential is strong (score {potential.score})"
        )
    elif potential.level == AssessmentLevel.MODERATE:
        parts.append(
            f"business potential is moderate (score {potential.score})"
        )
    else:
        parts.append(
            f"business potential is low (score {potential.score})"
        )

    # Evidence summary
    ev_count = len(result.evidence)
    reliable_count = len(result.reliable_evidence())
    parts.append(
        f"based on {ev_count} evidence items ({reliable_count} reliable)"
    )

    return ". ".join(parts) + "."


# ── Recommendation logic ─────────────────────────────────────────────────────

def generate_recommendation(assessment: RiskAssessment) -> str:
    """
    Produce an overall recommendation based on the two assessments.

    Returns one of:
    - ``"approve"`` — high trust AND high potential
    - ``"approve_with_conditions"`` — at least one is moderate, none low
    - ``"decline"`` — trustworthiness is low
    - ``"further_review"`` — potential is low but trust is adequate
    - ``"insufficient_data"`` — either assessment has insufficient evidence
    """
    trust = assessment.trustworthiness
    potential = assessment.business_potential

    # Insufficient evidence on either axis
    if (
        trust.level == AssessmentLevel.INSUFFICIENT_EVIDENCE
        or potential.level == AssessmentLevel.INSUFFICIENT_EVIDENCE
    ):
        return _RECOMMEND_INSUFFICIENT

    # Low trustworthiness → decline regardless of potential
    if trust.level == AssessmentLevel.LOW:
        return _RECOMMEND_DECLINE

    # Low potential but adequate trust → needs further review
    if potential.level == AssessmentLevel.LOW:
        return _RECOMMEND_REVIEW

    # Both high → approve
    if (
        trust.level == AssessmentLevel.HIGH
        and potential.level == AssessmentLevel.HIGH
    ):
        return _RECOMMEND_APPROVE

    # At least one moderate → conditional approval
    return _RECOMMEND_APPROVE_CONDITIONAL


# ── Main entry point ─────────────────────────────────────────────────────────

def generate_assessment(result: InvestigationResult) -> InvestigationResult:
    """
    Produce a fully assessed InvestigationResult.

    Takes a result from Person 1 (with evidence, features, and signals
    already populated), runs all ML modules through the risk engine,
    generates a justification and recommendation, and returns a new
    ``InvestigationResult`` with the assessments, justification, and
    all original data preserved.

    This is the single entry point Person 3's backend should call.
    """
    # ── Run the risk engine ──────────────────────────────────────────────
    risk = assess_risk(result)

    # ── Generate justification and recommendation ────────────────────────
    justification = generate_justification(result, risk)
    recommendation = generate_recommendation(risk)

    # ── Build enriched result ────────────────────────────────────────────
    # Copy all original fields and overlay the ML-generated assessments.
    enriched = result.model_copy(
        update={
            "trustworthiness": risk.trustworthiness,
            "business_potential": risk.business_potential,
            "justification": justification,
        }
    )

    return enriched
