"""
EvidenceItem — one piece of information the agent collected during investigation.

The evidence_type field enforces the project rule that we must never
silently promote an inference into a verified fact.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, HttpUrl, field_validator


class EvidenceType(str, Enum):
    # Directly observed from a single source
    OBSERVED = "observed"
    # Same fact confirmed by two or more independent sources
    CORROBORATED = "corroborated"
    # A conclusion drawn from observed facts — not directly stated anywhere
    INFERENCE = "inference"
    # Could not find sufficient information
    UNKNOWN = "unknown"


class SourceReliability(str, Enum):
    HIGH = "high"        # Official / first-party (the business's own site, govt registry)
    MEDIUM = "medium"    # Independent third-party (news, review platforms)
    LOW = "low"          # User-generated / unverified (forums, social comments)
    UNKNOWN = "unknown"


class EvidenceItem(BaseModel):
    # ── What was found ────────────────────────────────────────────────────
    field_name: str          # e.g. "instagram_followers", "business_age"
    value: str               # Always stored as string; ML layer converts types
    unit: Optional[str] = None   # e.g. "followers", "years", "USD"

    # ── How certain are we ────────────────────────────────────────────────
    evidence_type: EvidenceType
    # 0.0 – 1.0: how confident the agent is in this specific item
    confidence: float = 1.0

    # ── Where it came from ────────────────────────────────────────────────
    source_url: Optional[HttpUrl] = None
    source_name: Optional[str] = None       # e.g. "Instagram", "Google Maps"
    source_reliability: SourceReliability = SourceReliability.UNKNOWN
    raw_snippet: Optional[str] = None       # The exact text or data extracted

    # ── When it was collected ─────────────────────────────────────────────
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Validators ────────────────────────────────────────────────────────

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0.")
        return v

    @field_validator("field_name")
    @classmethod
    def field_name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field_name must not be blank.")
        return v.strip()

    @field_validator("value")
    @classmethod
    def value_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("value must not be blank.")
        return v

    # ── Helpers ───────────────────────────────────────────────────────────

    def is_reliable(self) -> bool:
        """True if both confidence and source reliability are high enough to trust."""
        return (
            self.confidence >= 0.7
            and self.source_reliability in (SourceReliability.HIGH, SourceReliability.MEDIUM)
            and self.evidence_type in (EvidenceType.OBSERVED, EvidenceType.CORROBORATED)
        )

    def summary(self) -> str:
        unit_str = f" {self.unit}" if self.unit else ""
        src = f" (source: {self.source_name})" if self.source_name else ""
        return f"{self.field_name}: {self.value}{unit_str} [{self.evidence_type.value}]{src}"
