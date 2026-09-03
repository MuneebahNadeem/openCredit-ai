"""API request/response models and friendly validation.

The create-request model deliberately mirrors Person 1's ``BusinessInput``
shape (field names identical) but validates with plain strings and
human-readable messages, so the browser shows "Please enter a valid website
URL" instead of a raw Pydantic ``HttpUrl`` error. Conversion to the real
``BusinessInput`` happens once, at the service boundary.
"""

from __future__ import annotations

from typing import List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, field_validator


def _clean_url(value: str, field_label: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"Please enter a valid {field_label} URL.")
    if not value.startswith(("http://", "https://")):
        value = "https://" + value
    parsed = urlparse(value)
    host = parsed.hostname or ""
    if " " in value or "." not in host:
        raise ValueError(f"Please enter a valid {field_label} URL.")
    return value


class InvestigationCreateRequest(BaseModel):
    name: str
    location: Optional[str] = None
    category: Optional[str] = None
    website: Optional[str] = None
    social_links: List[str] = []
    marketplace_links: List[str] = []
    description: Optional[str] = None
    additional_info: Optional[str] = None
    trustworthiness_override: bool = False

    @field_validator("name")
    @classmethod
    def name_required(cls, v):
        if not v or not v.strip():
            raise ValueError("Business name is required to begin an investigation.")
        return v.strip()

    @field_validator("location", "category", "description", "additional_info")
    @classmethod
    def strip_optionals(cls, v):
        return v.strip() if isinstance(v, str) and v.strip() else None

    @field_validator("website")
    @classmethod
    def valid_website(cls, v):
        return _clean_url(v, "website") if v else v

    @field_validator("social_links")
    @classmethod
    def valid_social_links(cls, v):
        return [_clean_url(link, "social media") for link in v]

    @field_validator("marketplace_links")
    @classmethod
    def valid_marketplace_links(cls, v):
        return [_clean_url(link, "marketplace") for link in v]

    def to_business_input(self):
        from agent.schemas.input import BusinessInput

        return BusinessInput(
            name=self.name,
            location=self.location,
            category=self.category,
            website=self.website,
            social_links=self.social_links,
            marketplace_links=self.marketplace_links,
            description=self.description,
            additional_info=self.additional_info,
        )


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_required(cls, v):
        if not v or not v.strip():
            raise ValueError("Please type a question about this report.")
        return v.strip()


class SaveRequest(BaseModel):
    saved: bool


class TrustworthinessOverrideRequest(BaseModel):
    enabled: bool
