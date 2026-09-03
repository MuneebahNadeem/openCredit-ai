"""Investigation orchestration — the backend service layer.

Coordinates the end-to-end flow:

    BusinessInput
        → Person 1 agent (investigation, evidence)
        → Person 2 ML (assessments, justification)
        → aggregated API record

Jobs run in a thread pool because Person 1's agent is synchronous and
network-bound. Status transitions are the real orchestration phases —
no fabricated progress is reported.
"""

from __future__ import annotations

import logging
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlparse

from fastapi import UploadFile

from agent.schemas.input import BusinessInput
from agent.schemas.result import InvestigationResult, InvestigationStatus

from backend.app.config import get_settings
from backend.app.services.adapters.agent_adapter import AgentAdapter
from backend.app.services.adapters.ml_adapter import MLAdapter
from backend.app.services.document_evidence import evidence_from_document
from backend.app.services.documents import DocumentStorage
from backend.app.services.storage import InvestigationStorage

logger = logging.getLogger("opencredit.investigations")

TERMINAL_STATUSES = {"completed", "partial", "failed"}

_PHASE_LABELS = {
    "queued": "Queued",
    "investigating": "Investigating public sources",
    "analyzing": "Analyzing evidence and risk",
    "completed": "Investigation complete",
    "partial": "Completed with limited evidence",
    "failed": "Investigation failed",
}

_AGENT_ERROR_MESSAGE = (
    "Investigation temporarily unavailable. "
    "The investigation service could not complete this request. Please try again."
)
_ML_ERROR_MESSAGE = (
    "Investigation completed, but the risk assessment could not be completed."
)

# Display-level source grouping (mirrors the agent's URL classification
# for presentation only — no investigation logic lives here).
_SOURCE_TYPE_KEYWORDS = [
    ("social", ("instagram.", "facebook.", "twitter.", "x.com", "tiktok.",
                "youtube.", "linkedin.", "pinterest.")),
    ("marketplace", ("daraz", "olx", "amazon", "shopify", "alibaba", "etsy.")),
    ("review", ("trustpilot", "yelp", "review", "rating", "google.com/maps")),
    ("government", (".gov", ".gov.pk", "secp", "fbr.gov")),
]


def _source_type(url: str) -> str:
    if url.startswith("uploaded_document:"):
        return "document"
    parsed = urlparse(url)
    target = f"{(parsed.hostname or '').lower()}{parsed.path.lower()}"
    for label, keywords in _SOURCE_TYPE_KEYWORDS:
        if any(k in target for k in keywords):
            return label
    return "web"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class InvestigationService:
    def __init__(
        self,
        agent_adapter: Optional[AgentAdapter] = None,
        ml_adapter: Optional[MLAdapter] = None,
        storage: Optional[InvestigationStorage] = None,
    ) -> None:
        settings = get_settings()
        self._agent = agent_adapter or AgentAdapter()
        self._ml = ml_adapter or MLAdapter()
        self._storage = storage or InvestigationStorage(settings.investigations_dir)
        self._documents = DocumentStorage(settings.documents_dir)
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_investigations,
            thread_name_prefix="investigation",
        )

    # ── Public API ────────────────────────────────────────────────────────

    def create(
        self,
        business_input: BusinessInput,
        documents: Optional[List[UploadFile]] = None,
    ) -> dict:
        record = {
            "id": f"inv_{secrets.token_hex(5)}",
            "created_at": _now(),
            "started_at": None,
            "completed_at": None,
            "status": "queued",
            "phase_label": _PHASE_LABELS["queued"],
            "error": None,
            "saved": False,
            "business": business_input.model_dump(mode="json"),
            "documents": [],
            "result": None,
        }
        self._storage.save(record)

        if documents:
            record["documents"] = [
                self._documents.save(record["id"], upload)
                for upload in documents
            ]
            self._storage.save(record)

        # Snapshot before the worker can mutate the same dict object.
        snapshot = dict(record)
        self._executor.submit(self._execute, record["id"])
        return snapshot

    def get(self, investigation_id: str) -> Optional[dict]:
        return self._storage.get(investigation_id)

    def list_summaries(self) -> list[dict]:
        return [self._summary(r) for r in self._storage.list_all()]

    def set_saved(self, investigation_id: str, saved: bool) -> Optional[dict]:
        record = self._storage.get(investigation_id)
        if record is None:
            return None
        record["saved"] = bool(saved)
        self._storage.save(record)
        return record

    # ── Job execution ─────────────────────────────────────────────────────

    def _execute(self, investigation_id: str) -> None:
        record = self._storage.get(investigation_id)
        if record is None:
            return
        business_input = BusinessInput.model_validate(record["business"])

        # Phase 1 — Person 1: investigation
        record["status"] = "investigating"
        record["phase_label"] = _PHASE_LABELS["investigating"]
        record["started_at"] = _now()
        self._storage.save(record)

        try:
            result = self._agent.run(business_input)
        except Exception:
            logger.exception("Agent failed for %s", investigation_id)
            self._fail(record, _AGENT_ERROR_MESSAGE)
            return

        # Merge any uploaded-document evidence before the ML assessment.
        self._add_document_evidence(record, result)

        # Phase 2 — Person 2: ML / risk assessment
        record["status"] = "analyzing"
        record["phase_label"] = _PHASE_LABELS["analyzing"]
        self._storage.save(record)

        try:
            enriched, context = self._ml.run(result)
        except Exception:
            logger.exception("ML assessment failed for %s", investigation_id)
            self._fail(record, _ML_ERROR_MESSAGE)
            return

        record["status"] = self._final_status(enriched.status)
        record["phase_label"] = _PHASE_LABELS[record["status"]]
        record["completed_at"] = _now()
        record["result"] = self._aggregate(enriched, context)
        self._storage.save(record)

    def _fail(self, record: dict, message: str) -> None:
        record["status"] = "failed"
        record["phase_label"] = _PHASE_LABELS["failed"]
        record["error"] = message
        record["completed_at"] = _now()
        self._storage.save(record)

    @staticmethod
    def _final_status(agent_status: InvestigationStatus) -> str:
        if agent_status == InvestigationStatus.FAILED:
            return "failed"
        if agent_status == InvestigationStatus.PARTIAL:
            return "partial"
        return "completed"  # complete or limit_reached are both finished runs

    def _add_document_evidence(
        self, record: dict, result: InvestigationResult
    ) -> None:
        """Extract text from uploaded documents and append EvidenceItems."""
        docs = record.get("documents") or []
        if not docs:
            return

        changed = False
        for doc in docs:
            if doc.get("extracted"):
                text = doc.get("extracted_text") or ""
            else:
                text = self._documents.extract(doc)
                changed = True

            if text:
                items = evidence_from_document(
                    text, doc["filename"], doc.get("content_type")
                )
                result.evidence.extend(items)

        if changed:
            self._storage.save(record)

    # ── Result aggregation ────────────────────────────────────────────────

    def _aggregate(self, enriched, context: dict) -> dict:
        result = enriched.model_dump(mode="json")
        result.pop("business_input", None)  # already stored at record level

        result["agent_status"] = enriched.status.value
        result["recommendation"] = context.get("recommendation")
        result["analysis_context"] = {
            "credibility": context.get("credibility"),
            "sentiment": context.get("sentiment"),
        }
        result["sources_detail"] = self._build_sources_detail(enriched)
        return result

    @staticmethod
    def _source_key(ev) -> str:
        if ev.source_url:
            return str(ev.source_url)
        if ev.source_name and ev.source_name.startswith("uploaded_document:"):
            return ev.source_name
        return "unknown"

    @staticmethod
    def _source_display_name(url: str, fallback_name: str | None) -> str:
        if url.startswith("uploaded_document:"):
            return url.split(":", 1)[1]
        if fallback_name:
            return fallback_name
        return urlparse(url).hostname or url

    @staticmethod
    def _build_sources_detail(enriched) -> list[dict]:
        """Group evidence by source for the report's Sources section."""
        groups: dict[str, dict] = {}
        for ev in enriched.evidence:
            url = InvestigationService._source_key(ev)
            group = groups.setdefault(
                url,
                {
                    "url": url,
                    "name": InvestigationService._source_display_name(
                        url, ev.source_name
                    ),
                    "reliability": ev.source_reliability.value,
                    "evidence_count": 0,
                    "evidence_fields": [],
                },
            )
            group["evidence_count"] += 1
            if ev.field_name not in group["evidence_fields"]:
                group["evidence_fields"].append(ev.field_name)

        # Include sources the agent visited but extracted nothing from.
        for url in enriched.sources:
            if url not in groups:
                groups[url] = {
                    "url": url,
                    "name": urlparse(url).hostname or url,
                    "reliability": "unknown",
                    "evidence_count": 0,
                    "evidence_fields": [],
                }

        detail = list(groups.values())
        for item in detail:
            item["type"] = _source_type(item["url"])
        detail.sort(key=lambda s: s["evidence_count"], reverse=True)
        return detail

    # ── Serialization helpers ─────────────────────────────────────────────

    @staticmethod
    def _summary(record: dict) -> dict:
        result = record.get("result") or {}
        return {
            "id": record["id"],
            "created_at": record["created_at"],
            "completed_at": record["completed_at"],
            "status": record["status"],
            "phase_label": record["phase_label"],
            "error": record["error"],
            "saved": record["saved"],
            "business": record["business"],
            "trustworthiness": result.get("trustworthiness"),
            "business_potential": result.get("business_potential"),
            "justification": result.get("justification"),
        }


_service: Optional[InvestigationService] = None
_service_lock = threading.Lock()


def get_service() -> InvestigationService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = InvestigationService()
    return _service


def set_service(service: InvestigationService) -> None:
    """Replace the singleton (used by tests to inject mock adapters)."""
    global _service
    with _service_lock:
        _service = service
