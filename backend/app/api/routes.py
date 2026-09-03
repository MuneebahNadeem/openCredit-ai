"""HTTP API routes — the only surface Person 3's backend exposes.

Endpoints:
    POST   /api/investigations                start an investigation
    GET    /api/investigations                list summaries
    GET    /api/investigations/{id}           full record
    GET    /api/investigations/{id}/status    poll phase
    GET    /api/investigations/{id}/result    aggregated report
    POST   /api/investigations/{id}/save      save / unsave
    POST   /api/investigations/{id}/ask       ask about a completed report
    GET    /api/health                        service status

Statuses flow queued → investigating → analyzing → completed | partial | failed.
No endpoint fabricates progress: the phase comes from the orchestrator.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from backend.app.schemas import (
    AskRequest,
    InvestigationCreateRequest,
    SaveRequest,
)
from backend.app.services.adapters.agent_adapter import llm_configured
from backend.app.services.ask import ask
from backend.app.services.investigations import (
    TERMINAL_STATUSES,
    get_service,
)

logger = logging.getLogger("opencredit.api")

router = APIRouter(prefix="/api")


@router.post("/investigations", status_code=202)
def create_investigation(
    payload: str = Form(...),
    documents: list[UploadFile] = File(default=[]),
):
    try:
        request = InvestigationCreateRequest.model_validate_json(payload)
    except ValidationError as exc:
        messages = []
        for err in exc.errors():
            field = ".".join(str(loc) for loc in err.get("loc", []) if loc != "body")
            message = err.get("msg", "").removeprefix("Value error, ")
            if field:
                messages.append(f"{field}: {message}" if message else field)
            elif message:
                messages.append(message)
        raise HTTPException(
            status_code=422,
            detail=" ".join(messages) or "Invalid request.",
        )

    record = get_service().create(request.to_business_input(), documents=documents)
    return _public_record(record)


@router.get("/investigations")
def list_investigations():
    return get_service().list_summaries()


@router.get("/investigations/{investigation_id}")
def get_investigation(investigation_id: str):
    record = _require(investigation_id)
    return _public_record(record)


@router.get("/investigations/{investigation_id}/status")
def get_status(investigation_id: str):
    record = _require(investigation_id)
    return {
        "id": record["id"],
        "status": record["status"],
        "phase_label": record["phase_label"],
        "error": record["error"],
    }


@router.get("/investigations/{investigation_id}/result")
def get_result(investigation_id: str):
    record = _require(investigation_id)
    if record["status"] not in TERMINAL_STATUSES:
        raise HTTPException(
            status_code=409,
            detail="Investigation is still running. Poll the status endpoint.",
        )
    if record["status"] == "failed" or record["result"] is None:
        raise HTTPException(
            status_code=422,
            detail=record.get("error")
            or "No result is available for this investigation.",
        )
    return record["result"]


@router.post("/investigations/{investigation_id}/save")
def set_saved(investigation_id: str, payload: SaveRequest):
    record = _require(investigation_id)
    updated = get_service().set_saved(investigation_id, payload.saved)
    return _public_record(updated)


@router.post("/investigations/{investigation_id}/ask")
def ask_question(investigation_id: str, payload: AskRequest):
    if not llm_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "Ask OpenCredit needs an LLM API key (OPENAI_API_KEY), which is "
                "not configured on this server."
            ),
        )
    record = _require(investigation_id)
    if record["status"] not in TERMINAL_STATUSES or record["result"] is None:
        raise HTTPException(
            status_code=409,
            detail="Ask OpenCredit is available once the report is complete.",
        )
    try:
        answer = ask(record, payload.question)
    except Exception:
        logger.exception("Ask failed for %s", investigation_id)
        raise HTTPException(
            status_code=502,
            detail="Ask OpenCredit could not reach the model. Please try again.",
        )
    return {"question": payload.question, "answer": answer}


@router.get("/health")
def health():
    return {"status": "ok", "llm_configured": llm_configured()}


# ── Helpers ────────────────────────────────────────────────────────────────


def _require(investigation_id: str) -> dict:
    record = get_service().get(investigation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    return record


def _public_record(record: dict) -> dict:
    """Full record minus the heavy result payload (fetched via /result)."""
    public = dict(record)
    public.pop("result", None)
    return public
