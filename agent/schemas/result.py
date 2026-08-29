"""
InvestigationResult — the complete structured output of the investigation agent.

This is the contract between Person 1 (agent) and Person 2 (ML) / Person 3 (backend).
Nothing downstream should need to re-run web searches or re-parse raw pages.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from agent.schemas.evidence import EvidenceItem
from agent.schemas.feature import DiscoveredFeature
from agent.schemas.input import BusinessInput


# ── Assessment level enums ─────────────────────────────────────────────────────

class AssessmentLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class InvestigationStatus(str, Enum):
    COMPLETE = "complete"               # Agent finished normally
    LIMIT_REACHED = "limit_reached"     # Hit max search / iteration limit
    PARTIAL = "partial"                 # Some sources failed but others succeeded
    FAILED = "failed"                   # Agent could not complete investigation


# ── Sub-models ─────────────────────────────────────────────────────────────────

class Signal(BaseModel):
    """A single positive or risk signal surfaced by the agent."""
    label: str          # Short label, e.g. "Active Instagram presence"
    detail: str         # One-sentence explanation of why this matters
    # The evidence item(s) that support this signal
    evidence_refs: List[str] = []   # field_names of backing EvidenceItems

    @field_validator("label", "detail")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Signal label and detail must not be blank.")
        return v.strip()


class AssessmentScore(BaseModel):
    """
    One of the two main assessments: trustworthiness OR business potential.
    Kept deliberately separate — a business can score high on one and low on the other.
    """
    level: AssessmentLevel
    # 0.0–1.0 numeric score; None when evidence is insufficient to score
    score: Optional[float] = None
    # How many reliable evidence items back this assessment
    evidence_count: int = 0
    # Plain-English explanation of how we reached this level
    explanation: str = ""

    @field_validator("score")
    @classmethod
    def score_in_range(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("score must be between 0.0 and 1.0.")
        return v


# ── Main result ────────────────────────────────────────────────────────────────

class InvestigationResult(BaseModel):
    # ── Input echo ────────────────────────────────────────────────────────
    # Store what we were asked to investigate so the result is self-contained.
    business_input: BusinessInput

    # ── Investigation metadata ────────────────────────────────────────────
    status: InvestigationStatus
    searches_performed: int = 0
    sources_examined: int = 0
    investigated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ── Raw evidence and features collected ───────────────────────────────
    evidence: List[EvidenceItem] = []
    features: List[DiscoveredFeature] = []

    # ── The two independent assessments ──────────────────────────────────
    trustworthiness: AssessmentScore = Field(
        default_factory=lambda: AssessmentScore(
            level=AssessmentLevel.INSUFFICIENT_EVIDENCE
        )
    )
    business_potential: AssessmentScore = Field(
        default_factory=lambda: AssessmentScore(
            level=AssessmentLevel.INSUFFICIENT_EVIDENCE
        )
    )

    # ── Signals ───────────────────────────────────────────────────────────
    positive_signals: List[Signal] = []
    risk_signals: List[Signal] = []

    # ── Gaps ─────────────────────────────────────────────────────────────
    # Information the agent looked for but could not find.
    missing_information: List[str] = []

    # All unique URLs the agent read during investigation.
    sources: List[str] = []

    # ── Final justification ───────────────────────────────────────────────
    # Approximately 2 lines. Must be evidence-based, never speculative.
    justification: str = ""

    # ── Computed helpers ──────────────────────────────────────────────────

    @property
    def evidence_count_total(self) -> int:
        return len(self.evidence)

    @property
    def has_sufficient_evidence(self) -> bool:
        """True if at least one assessment has a real score."""
        return (
            self.trustworthiness.level != AssessmentLevel.INSUFFICIENT_EVIDENCE
            or self.business_potential.level != AssessmentLevel.INSUFFICIENT_EVIDENCE
        )

    def get_features_by_category(self, category_value: str) -> List[DiscoveredFeature]:
        """Return all features belonging to a given FeatureCategory value string."""
        return [f for f in self.features if f.category.value == category_value]

    def reliable_evidence(self) -> List[EvidenceItem]:
        """Return only evidence items that pass the is_reliable() threshold."""
        return [e for e in self.evidence if e.is_reliable()]

    def summary(self) -> str:
        trust = self.trustworthiness.level.value
        potential = self.business_potential.level.value
        return (
            f"{self.business_input.name} | "
            f"trust={trust} | potential={potential} | "
            f"evidence={len(self.evidence)} | status={self.status.value}"
        )
