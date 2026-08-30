"""
Social Analyzer — extracts social media signals from page content.

Analyses text from social media pages (Instagram, Facebook, Twitter/X,
LinkedIn, TikTok, WhatsApp Business) to extract structured signals:
follower counts, post frequency, engagement indicators, and order activity.

Handles the informal Pakistani market specifically — recognises phrases
like "taking orders", "DM for rates", "limited slots" that indicate
active demand even without formal metrics.

Usage::

    from agent.tools.social_analyzer import SocialAnalyzer

    analyzer = SocialAnalyzer()
    signals = analyzer.analyze(page_content, platform="instagram")
    for s in signals:
        print(s.field_name, s.value)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability


# ── Platform reliability mapping ──────────────────────────────────────────────

_PLATFORM_RELIABILITY: dict[str, SourceReliability] = {
    "instagram": SourceReliability.MEDIUM,
    "facebook": SourceReliability.MEDIUM,
    "twitter": SourceReliability.MEDIUM,
    "x": SourceReliability.MEDIUM,
    "linkedin": SourceReliability.MEDIUM,
    "tiktok": SourceReliability.LOW,
    "whatsapp": SourceReliability.LOW,
    "youtube": SourceReliability.MEDIUM,
    "unknown": SourceReliability.UNKNOWN,
}

# ── Informal demand phrases (Pakistani market) ────────────────────────────────

_DEMAND_PHRASES = [
    r"taking orders?",
    r"order(s)? open",
    r"dm (for |to )?(order|rates?|price|booking)",
    r"inbox (for |to )?(order|rates?|price)",
    r"whatsapp (for |to )?(order|rates?|price)",
    r"limited slots?",
    r"slots? available",
    r"booking(s)? open",
    r"pre[\s-]?order",
    r"custom (order|stitching|work)",
    r"orders? (being )?accept(ed|ing)",
    r"daily orders?",
    r"monthly orders?",
    r"\d+\s*orders?\s*(per|a|/)?\s*(day|week|month)",
    r"sold out",
    r"fully booked",
]

_DEMAND_RE = re.compile(
    "|".join(_DEMAND_PHRASES), re.IGNORECASE
)

# ── Follower/like count patterns ──────────────────────────────────────────────

_FOLLOWER_RE = re.compile(
    r"([\d,\.]+[kKmM]?)\s*(followers?|following|likes?|subscribers?)",
    re.IGNORECASE,
)

_NUMBER_NORMALISE = re.compile(r"[,\s]")


def _parse_count(raw: str) -> Optional[int]:
    """Parse '12.5K', '1,200', '2M' etc into an integer."""
    raw = raw.strip()
    multiplier = 1
    if raw.lower().endswith("k"):
        multiplier = 1_000
        raw = raw[:-1]
    elif raw.lower().endswith("m"):
        multiplier = 1_000_000
        raw = raw[:-1]
    raw = _NUMBER_NORMALISE.sub("", raw)
    try:
        return int(float(raw) * multiplier)
    except ValueError:
        return None


# ── Engagement patterns ───────────────────────────────────────────────────────

_ENGAGEMENT_RE = re.compile(
    r"([\d,\.]+[kKmM]?)\s*(likes?|comments?|shares?|views?|reacts?)",
    re.IGNORECASE,
)

_ACTIVE_RE = re.compile(
    r"(posted?|updated?|active|last\s+post|hours? ago|days? ago|mins? ago)",
    re.IGNORECASE,
)


# ── Analyzer class ────────────────────────────────────────────────────────────

class SocialAnalyzer:
    """
    Extracts structured evidence from social media page text.

    Designed for informal Pakistani business social profiles — recognises
    demand phrases common in WhatsApp/Instagram commerce that formal
    analytics tools miss.
    """

    def analyze(
        self,
        text: str,
        url: str = "",
        platform: str = "unknown",
    ) -> List[EvidenceItem]:
        """
        Analyse social media page text and return extracted EvidenceItems.

        Parameters
        ----------
        text:
            Plain text of the social media page (from WebpageExtractor).
        url:
            Source URL for traceability.
        platform:
            Platform name: instagram, facebook, whatsapp, etc.

        Returns
        -------
        List[EvidenceItem] — may be empty if no signals found.
        """
        if not text:
            return []

        reliability = _PLATFORM_RELIABILITY.get(platform.lower(), SourceReliability.UNKNOWN)
        items: List[EvidenceItem] = []

        # ── Follower / subscriber count ───────────────────────────────────
        for m in _FOLLOWER_RE.finditer(text):
            count = _parse_count(m.group(1))
            metric = m.group(2).lower().rstrip("s")  # normalise plural
            if count is not None:
                field = f"{platform}_{metric}_count"
                items.append(EvidenceItem(
                    field_name=field,
                    value=str(count),
                    unit=metric + "s",
                    evidence_type=EvidenceType.OBSERVED,
                    source_url=url or None,
                    source_name=platform.capitalize(),
                    source_reliability=reliability,
                    confidence=0.75,
                    raw_snippet=m.group(0),
                ))

        # ── Engagement signals ────────────────────────────────────────────
        engagement_found = []
        for m in _ENGAGEMENT_RE.finditer(text):
            count = _parse_count(m.group(1))
            metric = m.group(2).lower().rstrip("s")
            if count is not None:
                engagement_found.append(f"{count} {metric}s")

        if engagement_found:
            items.append(EvidenceItem(
                field_name=f"{platform}_engagement",
                value=", ".join(engagement_found[:5]),
                evidence_type=EvidenceType.OBSERVED,
                source_url=url or None,
                source_name=platform.capitalize(),
                source_reliability=reliability,
                confidence=0.70,
                raw_snippet=text[:200],
            ))

        # ── Active posting indicator ──────────────────────────────────────
        if _ACTIVE_RE.search(text):
            items.append(EvidenceItem(
                field_name=f"{platform}_recently_active",
                value="yes",
                evidence_type=EvidenceType.OBSERVED,
                source_url=url or None,
                source_name=platform.capitalize(),
                source_reliability=reliability,
                confidence=0.65,
                raw_snippet="Recent activity detected",
            ))

        # ── Informal demand signals (key for Pakistani micro-businesses) ──
        demand_matches = _DEMAND_RE.findall(text)
        if demand_matches:
            snippets = list(set(m for m in demand_matches if m))[:3]
            items.append(EvidenceItem(
                field_name="informal_demand_signal",
                value="active_ordering",
                evidence_type=EvidenceType.OBSERVED,
                source_url=url or None,
                source_name=platform.capitalize(),
                source_reliability=SourceReliability.LOW,
                confidence=0.60,
                raw_snippet=" | ".join(str(s) for s in snippets),
            ))

        return items

    @staticmethod
    def detect_platform(url: str) -> str:
        """Guess the social platform from a URL."""
        url_lower = url.lower()
        for platform in ("instagram", "facebook", "twitter", "linkedin",
                         "tiktok", "youtube", "whatsapp"):
            if platform in url_lower:
                return platform
        return "unknown"
