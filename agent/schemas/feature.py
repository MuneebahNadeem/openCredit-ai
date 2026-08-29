"""
DiscoveredFeature — a business signal the agent identified during investigation.

Some features are known up front (e.g. website_present).
Others are discovered dynamically based on what the agent finds.
Every feature must carry the evidence that supports it — no unsupported claims.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, field_validator

from agent.schemas.evidence import EvidenceItem, EvidenceType


class FeatureCategory(str, Enum):
    # Signals relevant to trustworthiness
    IDENTITY = "identity"           # Name/registration consistency across sources
    REPUTATION = "reputation"       # Reviews, complaints, public sentiment
    TRANSPARENCY = "transparency"   # Contact info, about page, public records
    HISTORY = "history"             # Age, longevity, track record

    # Signals relevant to business/investment potential
    AUDIENCE = "audience"           # Followers, reach, traffic
    ENGAGEMENT = "engagement"       # Activity, interactions, recency
    DEMAND = "demand"               # Product reviews, marketplace activity
    GROWTH = "growth"               # Trend signals, expansion indicators
    MARKET_PRESENCE = "market_presence"  # Where the business appears

    # Cross-cutting
    RISK = "risk"                   # Complaints, inconsistencies, red flags
    UNKNOWN = "unknown"             # Agent could not classify


class DiscoveredFeature(BaseModel):
    # ── Identity ──────────────────────────────────────────────────────────
    name: str           # e.g. "instagram_follower_count", "has_google_listing"
    category: FeatureCategory = FeatureCategory.UNKNOWN

    # ── Value ─────────────────────────────────────────────────────────────
    # None means the agent looked but could not find this feature.
    value: Optional[str] = None
    unit: Optional[str] = None      # e.g. "followers", "stars", "years"

    # ── Why this feature matters ──────────────────────────────────────────
    # The agent must explain why it decided this signal is relevant.
    reason: str

    # ── Evidence backing this feature ─────────────────────────────────────
    evidence: List[EvidenceItem] = []

    # ── How certain is the agent about this feature ───────────────────────
    # Derived from the supporting evidence; set explicitly by the agent.
    confidence: float = 0.0

    # Whether the agent actively searched for this and found nothing,
    # vs. simply never looked.
    searched: bool = False

    # ── Validators ────────────────────────────────────────────────────────

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Feature name must not be blank.")
        return v.strip()

    @field_validator("reason")
    @classmethod
    def reason_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("A reason must be provided for every discovered feature.")
        return v.strip()

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0.")
        return v

    # ── Helpers ───────────────────────────────────────────────────────────

    def is_found(self) -> bool:
        """True if the agent actually found a value for this feature."""
        return self.value is not None

    def evidence_type(self) -> EvidenceType:
        """The strongest evidence type across all supporting evidence items."""
        if not self.evidence:
            return EvidenceType.UNKNOWN
        priority = [
            EvidenceType.CORROBORATED,
            EvidenceType.OBSERVED,
            EvidenceType.INFERENCE,
            EvidenceType.UNKNOWN,
        ]
        types_found = {e.evidence_type for e in self.evidence}
        for t in priority:
            if t in types_found:
                return t
        return EvidenceType.UNKNOWN

    def summary(self) -> str:
        val_str = f"{self.value} {self.unit or ''}".strip() if self.value else "not found"
        return (
            f"[{self.category.value}] {self.name}: {val_str} "
            f"(confidence={self.confidence:.2f}, evidence_items={len(self.evidence)})"
        )
