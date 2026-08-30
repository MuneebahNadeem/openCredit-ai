"""
User Input Parser — converts self-reported business information into low-confidence evidence.

Processes the ``additional_info`` and ``description`` fields from ``BusinessInput``
and extracts structured signals: revenue estimates, monthly order volumes,
years in business, staff count, product types, and sales channels.

All evidence produced here has:
- ``evidence_type = INFERENCE`` (self-reported, not independently verified)
- ``source_reliability = LOW``   (comes from the applicant themselves)
- ``confidence ≤ 0.50``          (needs independent verification)

This is important for informal Pakistani micro-businesses (e.g. home-based
tailors, WhatsApp sellers) who have no formal online presence but provide
meaningful information verbally or in a form.

Usage::

    from agent.tools.user_input_parser import UserInputParser
    from agent.schemas.input import BusinessInput

    parser = UserInputParser()
    evidence = parser.parse(business_input)
    for item in evidence:
        print(item.field_name, item.value, item.confidence)
"""

from __future__ import annotations

import re
from typing import List, Optional

from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability
from agent.schemas.input import BusinessInput


# ── Extraction patterns ────────────────────────────────────────────────────────

# Monthly revenue / sales: "50,000 per month", "Rs 30k monthly", "PKR 100,000/month"
_REVENUE_RE = re.compile(
    r"(?:rs\.?|pkr|rupees?|₨)?\s*([\d,]+[kKmM]?)"
    r"\s*(?:rupees?|rs\.?|pkr)?\s*"
    r"(?:per\s+month|monthly|/\s*month|a\s+month|pm\b)",
    re.IGNORECASE,
)

# Monthly orders/sales: "40 orders a month", "50 suits per month", "30 pieces monthly"
_ORDERS_RE = re.compile(
    r"([\d]+)\s*(?:to\s*[\d]+)?\s*"
    r"(orders?|suits?|pieces?|dresses?|shirts?|items?|units?|products?|clothes?)"
    r"\s*(?:per\s+month|monthly|/\s*month|a\s+month)",
    re.IGNORECASE,
)

# Years in business: "5 years", "since 2018", "established 2015"
_YEARS_RE = re.compile(
    r"(?:for\s+|last\s+|past\s+)?([\d]+)\s+years?\s*(?:in\s+business|old|experience|trading)?|"
    r"(?:since|established|founded|started|running since)\s+(\d{4})",
    re.IGNORECASE,
)

# Employee / staff count: "5 employees", "team of 3", "2 workers"
_STAFF_RE = re.compile(
    r"([\d]+)\s*(?:to\s*[\d]+)?\s*"
    r"(employees?|workers?|staff|helpers?|people|tailors?|stitchers?)",
    re.IGNORECASE,
)

# Sales channels mentioned
_CHANNEL_RE = re.compile(
    r"\b(whatsapp|facebook|instagram|daraz|olx|in[\s-]?store|shop|boutique"
    r"|home[\s-]?based|online|walk[\s-]?in|door[\s-]?to[\s-]?door"
    r"|word[\s-]?of[\s-]?mouth|referral|exhibition|mela)\b",
    re.IGNORECASE,
)

# Product types
_PRODUCT_RE = re.compile(
    r"\b(lawn|khaddar|silk|chiffon|cotton|embroidered?|stitching|tailoring"
    r"|ready[\s-]?made|pret|bridal|formal|casual|traditional|kurta|shalwar"
    r"|salwar|dupatta|suit|saree|abaya|hijab|kids?|children|men|women|ladies)\b",
    re.IGNORECASE,
)


def _parse_number(raw: str) -> Optional[int]:
    """Parse '30k', '1,500', '2M' → integer."""
    raw = raw.strip()
    multiplier = 1
    if raw.lower().endswith("k"):
        multiplier = 1_000
        raw = raw[:-1]
    elif raw.lower().endswith("m"):
        multiplier = 1_000_000
        raw = raw[:-1]
    try:
        return int(float(raw.replace(",", "")) * multiplier)
    except ValueError:
        return None


def _current_year() -> int:
    from datetime import datetime
    return datetime.now().year


class UserInputParser:
    """
    Extracts structured low-confidence evidence from user-supplied free text.

    All produced EvidenceItems are clearly marked as self-reported
    (INFERENCE type, LOW reliability) so the scoring layer weights them
    appropriately.
    """

    def parse(self, business_input: BusinessInput) -> List[EvidenceItem]:
        """
        Parse ``additional_info`` and ``description`` from BusinessInput.

        Returns a list of EvidenceItems — may be empty if nothing is found.
        """
        texts = []
        if business_input.additional_info:
            texts.append(business_input.additional_info)
        if business_input.description:
            texts.append(business_input.description)

        if not texts:
            return []

        combined = " ".join(texts)
        items: List[EvidenceItem] = []
        source_name = "self_reported"

        def _make(field: str, value: str, snippet: str, confidence: float = 0.40) -> EvidenceItem:
            return EvidenceItem(
                field_name=field,
                value=value,
                evidence_type=EvidenceType.INFERENCE,
                source_name=source_name,
                source_reliability=SourceReliability.LOW,
                confidence=confidence,
                raw_snippet=snippet[:200],
            )

        # ── Monthly revenue ───────────────────────────────────────────────
        rev_match = _REVENUE_RE.search(combined)
        if rev_match:
            amount = _parse_number(rev_match.group(1))
            if amount:
                items.append(_make(
                    "self_reported_monthly_revenue",
                    str(amount),
                    rev_match.group(0),
                    confidence=0.40,
                ))

        # ── Monthly orders ────────────────────────────────────────────────
        order_match = _ORDERS_RE.search(combined)
        if order_match:
            items.append(_make(
                "self_reported_monthly_orders",
                order_match.group(1),
                order_match.group(0),
                confidence=0.45,
            ))

        # ── Years in business ─────────────────────────────────────────────
        years_match = _YEARS_RE.search(combined)
        if years_match:
            if years_match.group(1):  # "5 years"
                years = years_match.group(1)
            else:  # "since 2018"
                year = int(years_match.group(2))
                years = str(_current_year() - year)
            items.append(_make(
                "self_reported_years_in_business",
                years,
                years_match.group(0),
                confidence=0.45,
            ))

        # ── Staff / employees ─────────────────────────────────────────────
        staff_match = _STAFF_RE.search(combined)
        if staff_match:
            items.append(_make(
                "self_reported_staff_count",
                staff_match.group(1),
                staff_match.group(0),
                confidence=0.40,
            ))

        # ── Sales channels ────────────────────────────────────────────────
        channels = list(set(
            m.lower() for m in _CHANNEL_RE.findall(combined)
        ))
        if channels:
            items.append(_make(
                "self_reported_sales_channels",
                ", ".join(sorted(channels)[:5]),
                "Sales channels mentioned: " + ", ".join(channels[:5]),
                confidence=0.50,
            ))

        # ── Product types ─────────────────────────────────────────────────
        products = list(set(
            m.lower() for m in _PRODUCT_RE.findall(combined)
        ))
        if products:
            items.append(_make(
                "self_reported_product_types",
                ", ".join(sorted(products)[:8]),
                "Products mentioned: " + ", ".join(products[:8]),
                confidence=0.50,
            ))

        return items
