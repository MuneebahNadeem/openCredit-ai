"""
Product Analyzer — extracts product and demand signals from marketplace pages.

Analyses Daraz, OLX, and other marketplace listings to extract product
activity, sold counts, listing age, and demand indicators.

Usage::

    from agent.tools.product_analyzer import ProductAnalyzer

    analyzer = ProductAnalyzer()
    items = analyzer.analyze(page_text, url="https://daraz.pk/...", source_name="Daraz")
"""

from __future__ import annotations

import re
from typing import List

from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability


# ── Patterns ───────────────────────────────────────────────────────────────────

# Sold count: "1,234 sold", "500+ sold"
_SOLD_RE = re.compile(r"([\d,]+\+?)\s*sold", re.IGNORECASE)

# Product listing count: "234 products", "45 listings", "12 items"
_LISTING_COUNT_RE = re.compile(
    r"([\d,]+)\s*(products?|listings?|items?|results?)",
    re.IGNORECASE,
)

# Price presence: signals active commerce
_PRICE_RE = re.compile(r"(Rs\.?|PKR|₨)\s*[\d,]+", re.IGNORECASE)

# Listing age / freshness
_FRESHNESS_RE = re.compile(
    r"(today|yesterday|\d+\s*(hours?|mins?|days?|weeks?)\s*ago|just\s+now|new\s+arrival)",
    re.IGNORECASE,
)

# Marketplace platform reliability
_PLATFORM_RELIABILITY = {
    "daraz": SourceReliability.MEDIUM,
    "olx": SourceReliability.LOW,
    "amazon": SourceReliability.MEDIUM,
    "alibaba": SourceReliability.MEDIUM,
    "shopify": SourceReliability.MEDIUM,
    "default": SourceReliability.LOW,
}


def _reliability_for(source_name: str) -> SourceReliability:
    for key, rel in _PLATFORM_RELIABILITY.items():
        if key in source_name.lower():
            return rel
    return _PLATFORM_RELIABILITY["default"]


def _parse_count(raw: str) -> int:
    try:
        return int(raw.replace(",", "").replace("+", "").strip())
    except ValueError:
        return 0


class ProductAnalyzer:
    """
    Extracts product and demand evidence from marketplace page text.
    """

    def analyze(
        self,
        text: str,
        url: str = "",
        source_name: str = "marketplace",
    ) -> List[EvidenceItem]:
        """
        Extract product/demand signals from page text.

        Returns a list of EvidenceItems.
        """
        if not text:
            return []

        reliability = _reliability_for(source_name)
        items: List[EvidenceItem] = []

        # ── Units sold ────────────────────────────────────────────────────
        sold_match = _SOLD_RE.search(text)
        if sold_match:
            count = _parse_count(sold_match.group(1))
            if count > 0:
                items.append(EvidenceItem(
                    field_name="units_sold",
                    value=str(count),
                    unit="units",
                    evidence_type=EvidenceType.OBSERVED,
                    source_url=url or None,
                    source_name=source_name,
                    source_reliability=reliability,
                    confidence=0.80,
                    raw_snippet=sold_match.group(0).strip(),
                ))

        # ── Number of active listings ──────────────────────────────────────
        listing_match = _LISTING_COUNT_RE.search(text)
        if listing_match:
            count = _parse_count(listing_match.group(1))
            if count > 0:
                items.append(EvidenceItem(
                    field_name="active_listing_count",
                    value=str(count),
                    unit="listings",
                    evidence_type=EvidenceType.OBSERVED,
                    source_url=url or None,
                    source_name=source_name,
                    source_reliability=reliability,
                    confidence=0.75,
                    raw_snippet=listing_match.group(0).strip(),
                ))

        # ── Price activity (indicates active commerce) ─────────────────────
        price_count = len(_PRICE_RE.findall(text))
        if price_count > 0:
            items.append(EvidenceItem(
                field_name="marketplace_price_activity",
                value=str(price_count),
                unit="price_listings",
                evidence_type=EvidenceType.OBSERVED,
                source_url=url or None,
                source_name=source_name,
                source_reliability=reliability,
                confidence=0.65,
                raw_snippet=f"{price_count} price listings found",
            ))

        # ── Listing freshness ─────────────────────────────────────────────
        freshness_match = _FRESHNESS_RE.search(text)
        if freshness_match:
            items.append(EvidenceItem(
                field_name="listing_freshness",
                value=freshness_match.group(0).strip().lower(),
                evidence_type=EvidenceType.OBSERVED,
                source_url=url or None,
                source_name=source_name,
                source_reliability=reliability,
                confidence=0.70,
                raw_snippet=freshness_match.group(0).strip(),
            ))

        return items
