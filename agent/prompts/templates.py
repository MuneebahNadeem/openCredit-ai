"""
Prompt templates for the investigation agent.

All prompts are plain strings with {placeholder} fields.
Call the module-level functions to get filled prompts ready to send to the LLM.

Design rules:
- Every prompt ends with a clear instruction on what to return.
- Prompts never ask the LLM to speculate — only to work with given evidence.
- Output format is always structured (JSON or numbered list) so it can be parsed.
"""

from __future__ import annotations

from typing import List, Optional


# ── Investigation planning prompt ─────────────────────────────────────────────

_INVESTIGATION_PLAN_TEMPLATE = """\
You are an AI business investigation agent for OpenCredit AI.
Your job is to assess the trustworthiness and business potential of a Pakistani business.

BUSINESS DETAILS:
Name: {business_name}
Location: {location}
Category: {category}
Website: {website}
Social links: {social_links}
Marketplace links: {marketplace_links}
Description: {description}
Additional info: {additional_info}

ALREADY SEARCHED:
{already_searched}

EVIDENCE COLLECTED SO FAR:
{evidence_summary}

Based on the above, decide the next 3 most valuable search queries to run.
Focus on: business registration, reviews, social media presence, marketplace listings, complaints.
For Pakistani businesses also consider: Daraz, OLX, Facebook, Instagram, WhatsApp Business.

Return ONLY a JSON array of query strings. Example:
["Karachi Textile Hub SECP registration", "Karachi Textile Hub Daraz reviews", "Karachi Textile Hub complaints Pakistan"]
"""


def investigation_plan_prompt(
    business_name: str,
    location: Optional[str],
    category: Optional[str],
    website: Optional[str],
    social_links: List[str],
    marketplace_links: List[str],
    description: Optional[str],
    additional_info: Optional[str],
    already_searched: List[str],
    evidence_summary: str,
) -> str:
    """Return a filled investigation planning prompt."""
    return _INVESTIGATION_PLAN_TEMPLATE.format(
        business_name=business_name,
        location=location or "Not provided",
        category=category or "Not provided",
        website=website or "Not provided",
        social_links=", ".join(social_links) if social_links else "None",
        marketplace_links=", ".join(marketplace_links) if marketplace_links else "None",
        description=description or "Not provided",
        additional_info=additional_info or "Not provided",
        already_searched="\n".join(f"- {q}" for q in already_searched) if already_searched else "None yet",
        evidence_summary=evidence_summary or "No evidence collected yet.",
    )


# ── Evidence extraction prompt ────────────────────────────────────────────────

_EXTRACTION_TEMPLATE = """\
You are extracting business evidence from a web page.

BUSINESS BEING INVESTIGATED: {business_name}
SOURCE URL: {url}
SOURCE TYPE: {source_type}

PAGE CONTENT (truncated):
{page_text}

Extract all factual information relevant to assessing this business.
For each fact found, return a JSON object with:
- "field_name": snake_case identifier (e.g. "instagram_followers", "star_rating", "registration_number")
- "value": the extracted value as a string
- "unit": optional unit (e.g. "followers", "stars", "years") or null
- "evidence_type": one of "observed", "corroborated", "inference"
- "confidence": 0.0-1.0 how certain you are this fact is accurate
- "raw_snippet": the exact text you extracted this from (max 100 chars)

Return ONLY a JSON array. If nothing relevant is found, return [].
Do NOT invent information. Only extract what is explicitly stated in the page content.
"""


def extraction_prompt(
    business_name: str,
    url: str,
    source_type: str,
    page_text: str,
    max_chars: int = 3000,
) -> str:
    """Return a filled evidence extraction prompt."""
    return _EXTRACTION_TEMPLATE.format(
        business_name=business_name,
        url=url,
        source_type=source_type,
        page_text=page_text[:max_chars],
    )


# ── Feature discovery prompt ──────────────────────────────────────────────────

_FEATURE_DISCOVERY_TEMPLATE = """\
You are identifying business signals from collected evidence.

BUSINESS: {business_name} ({location})
CATEGORY: {category}

COLLECTED EVIDENCE:
{evidence_list}

Based on this evidence, identify the most important business signals (features).
For each signal, return a JSON object with:
- "name": snake_case feature name (e.g. "has_social_presence", "monthly_order_volume")
- "category": one of "identity", "reputation", "transparency", "history", "audience", "engagement", "demand", "growth", "market_presence", "risk", "unknown"
- "value": the signal value as a string, or null if not found
- "reason": ONE sentence explaining why this signal matters for this business
- "confidence": 0.0-1.0
- "searched": true (agent looked for this), false (not looked for)

Return ONLY a JSON array of feature objects. Maximum 10 features.
Focus on signals most relevant to trustworthiness and business potential.
"""


def feature_discovery_prompt(
    business_name: str,
    location: Optional[str],
    category: Optional[str],
    evidence_list: str,
) -> str:
    """Return a filled feature discovery prompt."""
    return _FEATURE_DISCOVERY_TEMPLATE.format(
        business_name=business_name,
        location=location or "Pakistan",
        category=category or "Unknown",
        evidence_list=evidence_list or "No evidence collected yet.",
    )


# ── Assessment prompt ─────────────────────────────────────────────────────────

_ASSESSMENT_TEMPLATE = """\
You are producing a final assessment summary for a business investigation.

BUSINESS: {business_name}
TRUSTWORTHINESS SCORE: {trust_score} ({trust_level})
BUSINESS POTENTIAL SCORE: {potential_score} ({potential_level})

POSITIVE SIGNALS:
{positive_signals}

RISK SIGNALS:
{risk_signals}

MISSING INFORMATION:
{missing_info}

EVIDENCE COUNT: {evidence_count} items ({reliable_count} reliable)

Write a 2-sentence justification for this assessment.
Requirements:
- Sentence 1: Summarise trustworthiness based on evidence found.
- Sentence 2: Summarise business potential based on signals found.
- Never speculate. Only state what the evidence supports.
- Keep each sentence under 25 words.

Return ONLY the 2-sentence justification as plain text. No bullet points, no headers.
"""


def assessment_prompt(
    business_name: str,
    trust_score: Optional[float],
    trust_level: str,
    potential_score: Optional[float],
    potential_level: str,
    positive_signals: List[str],
    risk_signals: List[str],
    missing_info: List[str],
    evidence_count: int,
    reliable_count: int,
) -> str:
    """Return a filled assessment justification prompt."""
    return _ASSESSMENT_TEMPLATE.format(
        business_name=business_name,
        trust_score=f"{trust_score:.2f}" if trust_score is not None else "N/A",
        trust_level=trust_level,
        potential_score=f"{potential_score:.2f}" if potential_score is not None else "N/A",
        potential_level=potential_level,
        positive_signals="\n".join(f"- {s}" for s in positive_signals) if positive_signals else "None",
        risk_signals="\n".join(f"- {s}" for s in risk_signals) if risk_signals else "None",
        missing_info="\n".join(f"- {m}" for m in missing_info) if missing_info else "None",
        evidence_count=evidence_count,
        reliable_count=reliable_count,
    )
