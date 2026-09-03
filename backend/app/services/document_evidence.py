"""Convert extracted document text into EvidenceItem objects.

Document evidence is merged with the agent's web evidence in the orchestrator,
so the ML layer can score the combined picture without knowing where each item
came from.
"""

from __future__ import annotations

import re
from typing import List, Optional

from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability

_MAX_VALUE = 2_000
_MAX_SNIPPET = 500


def _field_name(filename: str) -> str:
    """Derive a compact, valid field_name from the uploaded filename."""
    base = filename.split("/")[-1].split("\\")[-1]
    stem = base.rsplit(".", 1)[0] or "document"
    safe = re.sub(r"[^\w]+", "_", stem).strip("_") or "document"
    return f"document_{safe}"[:64]


def evidence_from_document(
    text: str, filename: str, content_type: Optional[str] = None
) -> List[EvidenceItem]:
    """Build one or more EvidenceItems from extracted document text."""
    text = (text or "").strip()
    if not text:
        return []

    snippet = text[:_MAX_SNIPPET]
    value = text[:_MAX_VALUE]
    if len(text) > _MAX_VALUE:
        value = value.rstrip() + "…"

    return [
        EvidenceItem(
            field_name=_field_name(filename),
            value=value,
            unit=None,
            evidence_type=EvidenceType.OBSERVED,
            confidence=0.8,
            source_url=None,
            source_name=f"uploaded_document:{filename}",
            source_reliability=SourceReliability.MEDIUM,
            raw_snippet=snippet,
        )
    ]
