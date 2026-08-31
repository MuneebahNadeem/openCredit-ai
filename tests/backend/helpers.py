"""Shared factories for backend tests.

The fake agent produces a realistic ``InvestigationResult``; the ML adapter
used in tests is the REAL one (it is deterministic and makes no network
calls), so aggregation coverage is genuine.
"""

from __future__ import annotations

import time

from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability
from agent.schemas.input import BusinessInput
from agent.schemas.result import (
    InvestigationResult,
    InvestigationStatus,
    Signal,
)

TERMINAL = {"completed", "partial", "failed"}


def make_agent_result(
    status: InvestigationStatus = InvestigationStatus.COMPLETE,
) -> InvestigationResult:
    """A realistic mid-quality investigation result."""
    business = BusinessInput(
        name="Karachi Threads",
        location="Karachi",
        category="Clothing",
        additional_info="Home-based boutique, 10 years in business, "
        "monthly revenue Rs 250k, 4 tailors",
    )
    evidence = [
        EvidenceItem(
            field_name="instagram_followers",
            value="12500",
            unit="followers",
            evidence_type=EvidenceType.OBSERVED,
            confidence=0.9,
            source_url="https://instagram.com/karachithreads",
            source_name="Instagram",
            source_reliability=SourceReliability.MEDIUM,
            raw_snippet="12.5k followers, recent posts taking orders",
        ),
        EvidenceItem(
            field_name="reviews_rating",
            value="4.6",
            unit="stars",
            evidence_type=EvidenceType.OBSERVED,
            confidence=0.85,
            source_url="https://google.com/maps",
            source_name="Google Maps",
            source_reliability=SourceReliability.HIGH,
            raw_snippet="Rated 4.6 stars from 212 reviews, great stitching quality",
        ),
        EvidenceItem(
            field_name="units_sold",
            value="340",
            unit="items",
            evidence_type=EvidenceType.OBSERVED,
            confidence=0.8,
            source_url="https://daraz.pk/shop/karachi-threads",
            source_name="Daraz",
            source_reliability=SourceReliability.MEDIUM,
            raw_snippet="340 items sold on this listing",
        ),
        EvidenceItem(
            field_name="self_reported_revenue",
            value="250000",
            unit="PKR",
            evidence_type=EvidenceType.INFERENCE,
            confidence=0.5,
            source_url=None,
            source_name="User reported",
            source_reliability=SourceReliability.LOW,
            raw_snippet="monthly revenue Rs 250k",
        ),
    ]
    return InvestigationResult(
        business_input=business,
        status=status,
        searches_performed=5,
        sources_examined=3,
        evidence=evidence,
        positive_signals=[
            Signal(
                label="Active Instagram presence",
                detail="12.5k followers with recent order-taking posts.",
                evidence_refs=["instagram_followers"],
            ),
            Signal(
                label="Strong customer reviews",
                detail="4.6 stars across 212 Google reviews.",
                evidence_refs=["reviews_rating"],
            ),
        ],
        risk_signals=[
            Signal(
                label="Revenue is self-reported only",
                detail="No independent source confirms the stated revenue.",
                evidence_refs=["self_reported_revenue"],
            ),
        ],
        missing_information=["No official website found", "No business registration found"],
        sources=[
            "https://instagram.com/karachithreads",
            "https://google.com/maps",
            "https://daraz.pk/shop/karachi-threads",
            "https://example.com/empty-page",
        ],
    )


def wait_terminal(service, investigation_id: str, timeout: float = 5.0) -> dict:
    """Poll the record until it reaches a terminal status."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = service.get(investigation_id)
        if record is not None and record["status"] in TERMINAL:
            return record
        time.sleep(0.02)
    raise AssertionError(
        f"Investigation {investigation_id} did not reach a terminal status in time"
    )
