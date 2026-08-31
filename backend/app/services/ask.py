"""Ask OpenCredit — question answering over a completed report.

Answers strictly from the investigation result so the endpoint can never
invent facts the agent did not find. Requires an LLM key; the route returns
503 with a clear message when it is not configured.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Optional

from backend.app.config import get_settings

logger = logging.getLogger("opencredit.ask")

_SYSTEM_PROMPT = (
    "You are OpenCredit's report assistant. You answer questions about a "
    "single business investigation report. Rules:\n"
    "1. Use ONLY the report data provided. Never invent facts, scores, or "
    "sources.\n"
    "2. If the report does not contain the answer, say so plainly.\n"
    "3. Keep answers to 2-4 sentences, plain English, no markdown headings.\n"
    "4. This is decision support, not a lending decision — never advise "
    "approve or decline; describe what the evidence shows."
)


def _report_context(record: dict) -> str:
    result = record.get("result") or {}
    business = record.get("business") or {}
    keep = {
        "business": business,
        "trustworthiness": result.get("trustworthiness"),
        "business_potential": result.get("business_potential"),
        "recommendation": result.get("recommendation"),
        "justification": result.get("justification"),
        "positive_signals": result.get("positive_signals"),
        "risk_signals": result.get("risk_signals"),
        "missing_information": result.get("missing_information"),
        "features": [
            {"name": f.get("name"), "value": f.get("value"), "reason": f.get("reason")}
            for f in (result.get("features") or [])[:40]
        ],
        "evidence": [
            {
                "field_name": e.get("field_name"),
                "value": e.get("value"),
                "evidence_type": e.get("evidence_type"),
                "source_url": e.get("source_url"),
                "confidence": e.get("confidence"),
            }
            for e in (result.get("evidence") or [])[:60]
        ],
    }
    return json.dumps(keep, default=str)


def ask(record: dict, question: str) -> str:
    """Answer ``question`` using only the completed report in ``record``."""
    settings = get_settings()
    prompt = (
        f"{_SYSTEM_PROMPT}\n\nREPORT DATA (JSON):\n{_report_context(record)}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )
    return _call_llm(prompt, settings.ask_model, settings.ask_temperature)


# Injectable so tests avoid real API calls.
_call_fn: Optional[Callable[[str, str, float], str]] = None


def set_call_fn(fn: Optional[Callable[[str, str, float], str]]) -> None:
    global _call_fn
    _call_fn = fn


def _call_llm(prompt: str, model: str, temperature: float) -> str:
    if _call_fn is not None:
        return _call_fn(prompt, model, temperature)

    import openai  # type: ignore

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=400,
    )
    return response.choices[0].message.content or ""
