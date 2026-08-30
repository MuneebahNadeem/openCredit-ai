"""
Review Analyzer — extracts rating and sentiment evidence from review pages.

Handles Google Maps, Daraz, OLX, Facebook reviews, and generic review
blocks.  Extracts overall star ratings, review counts, and representative
review snippets for sentiment analysis.

Usage::

    from agent.tools.review_analyzer import ReviewAnalyzer

    analyzer = ReviewAnalyzer()
    items = analyzer.analyze(page_text, url="https://daraz.pk/...", source_name="Daraz")
    for item in items:
        print(item.field_name, item.value)
"""

from __future__ import annotations

import re
from typing import List, Optional

from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability


# ── Rating patterns ────────────────────────────────────────────────────────────

# Matches: "4.5 out of 5", "4.5/5", "4.5 stars", "Rated 4.5"
_RATING_RE = re.compile(
    r"(?:rated?|rating|score|stars?)[\s:]*([0-9]\.[0-9]|[0-9])\s*"
    r"(?:/\s*5|out\s+of\s*5|stars?)?",
    re.IGNORECASE,
)

# Matches: "1,234 reviews", "250 ratings", "47 customer reviews"
_REVIEW_COUNT_RE = re.compile(
    r"([\d,]+)\s*(customer\s+)?(reviews?|ratings?|feedbacks?)",
    re.IGNORECASE,
)

# Complaint / negative review indicators
_COMPLAINT_RE = re.compile(
    r"(complaint[s]?|reported|fraud|scam|fake|dispute[s]?|refund\s+issue)",
    re.IGNORECASE,
)

# Review snippets — sentences containing strong sentiment words
_SNIPPET_SENTENCE_RE = re.compile(
    r"[^.!?]*(?:excellent|outstanding|great|good|poor|terrible|scam|fraud"
    r"|recommend|satisfied|disappointed|horrible|amazing|awful)[^.!?]*[.!?]",
    re.IGNORECASE,
)


def _clean_count(raw: str) -> int:
    """Parse '1,234' → 1234."""
    try:
        return int(raw.replace(",", "").strip())
    except ValueError:
        return 0


# ── Source reliability by platform ────────────────────────────────────────────

_PLATFORM_RELIABILITY = {
    "google": SourceReliability.HIGH,
    "daraz": SourceReliability.MEDIUM,
    "amazon": SourceReliability.MEDIUM,
    "facebook": SourceReliability.MEDIUM,
    "olx": SourceReliability.LOW,
    "trustpilot": SourceReliability.MEDIUM,
    "yelp": SourceReliability.MEDIUM,
    "default": SourceReliability.MEDIUM,
}


def _reliability_for(source_name: str) -> SourceReliability:
    for key, rel in _PLATFORM_RELIABILITY.items():
        if key in source_name.lower():
            return rel
    return _PLATFORM_RELIABILITY["default"]


# ── Analyzer class ────────────────────────────────────────────────────────────

class ReviewAnalyzer:
    """
    Extracts star ratings, review counts, and complaint signals from page text.
    """

    def analyze(
        self,
        text: str,
        url: str = "",
        source_name: str = "reviews",
    ) -> List[EvidenceItem]:
        """
        Extract review evidence from the given page text.

        Returns a list of EvidenceItems (may be empty if nothing found).
        """
        if not text:
            return []

        reliability = _reliability_for(source_name)
        items: List[EvidenceItem] = []

        # ── Star rating ───────────────────────────────────────────────────
        rating_match = _RATING_RE.search(text)
        if rating_match:
            try:
                rating = float(rating_match.group(1))
                if 0.0 <= rating <= 5.0:
                    items.append(EvidenceItem(
                        field_name="star_rating",
                        value=str(rating),
                        unit="stars_out_of_5",
                        evidence_type=EvidenceType.OBSERVED,
                        source_url=url or None,
                        source_name=source_name,
                        source_reliability=reliability,
                        confidence=0.85,
                        raw_snippet=rating_match.group(0).strip(),
                    ))
            except ValueError:
                pass

        # ── Review count ──────────────────────────────────────────────────
        count_match = _REVIEW_COUNT_RE.search(text)
        if count_match:
            count = _clean_count(count_match.group(1))
            if count > 0:
                items.append(EvidenceItem(
                    field_name="review_count",
                    value=str(count),
                    unit="reviews",
                    evidence_type=EvidenceType.OBSERVED,
                    source_url=url or None,
                    source_name=source_name,
                    source_reliability=reliability,
                    confidence=0.80,
                    raw_snippet=count_match.group(0).strip(),
                ))

        # ── Representative review snippets (for sentiment) ────────────────
        snippets = _SNIPPET_SENTENCE_RE.findall(text)
        if snippets:
            combined = " | ".join(s.strip() for s in snippets[:3])
            items.append(EvidenceItem(
                field_name="review_snippets",
                value=combined[:300],
                evidence_type=EvidenceType.OBSERVED,
                source_url=url or None,
                source_name=source_name,
                source_reliability=reliability,
                confidence=0.70,
                raw_snippet=combined[:300],
            ))

        # ── Complaint signals ─────────────────────────────────────────────
        complaint_matches = _COMPLAINT_RE.findall(text)
        if complaint_matches:
            unique = list(set(m.lower() for m in complaint_matches))[:3]
            items.append(EvidenceItem(
                field_name="complaint_signals",
                value=", ".join(unique),
                evidence_type=EvidenceType.OBSERVED,
                source_url=url or None,
                source_name=source_name,
                source_reliability=reliability,
                confidence=0.65,
                raw_snippet=" | ".join(unique),
            ))

        return items
