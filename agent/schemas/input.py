"""
BusinessInput — the structured input the agent receives from the user.

All fields except `name` are optional because real-world submissions
vary widely; the agent must work with whatever the user can provide.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, HttpUrl, field_validator, model_validator


class BusinessInput(BaseModel):
    # ── Required ──────────────────────────────────────────────────────────
    name: str

    # ── Identity / location ───────────────────────────────────────────────
    location: Optional[str] = None
    category: Optional[str] = None

    # ── Online presence ───────────────────────────────────────────────────
    website: Optional[HttpUrl] = None
    social_links: List[HttpUrl] = []
    marketplace_links: List[HttpUrl] = []

    # ── Narrative / user-supplied knowledge ───────────────────────────────
    description: Optional[str] = None

    # Free-form text the user can include: revenue figures, years trading,
    # number of employees, anything they choose to share.
    additional_info: Optional[str] = None

    # ── Validators ────────────────────────────────────────────────────────

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Business name must not be blank.")
        return v.strip()

    @field_validator("location", "category", "description", "additional_info", mode="before")
    @classmethod
    def strip_optional_strings(cls, v):
        if isinstance(v, str):
            stripped = v.strip()
            return stripped if stripped else None
        return v

    @model_validator(mode="after")
    def at_least_name_present(self) -> "BusinessInput":
        # Guard: name is the single required anchor for any investigation.
        if not self.name:
            raise ValueError("A business name is required to begin investigation.")
        return self

    # ── Helpers ───────────────────────────────────────────────────────────

    def has_online_presence(self) -> bool:
        """True if the user supplied at least one online pointer."""
        return bool(self.website or self.social_links or self.marketplace_links)

    def summary(self) -> str:
        """One-line human-readable summary for logging/prompts."""
        parts = [self.name]
        if self.location:
            parts.append(self.location)
        if self.category:
            parts.append(f"({self.category})")
        return " | ".join(parts)
