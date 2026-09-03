"""End-to-end HTTP API tests.

The agent is faked; the ML adapter is real (deterministic, offline), so the
aggregation path exercised here is the genuine one.
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.services import ask
from backend.app.services.adapters.agent_adapter import AgentAdapter
from backend.app.services.adapters.ml_adapter import MLAdapter
from backend.app.services.investigations import (
    InvestigationService,
    get_service,
    set_service,
)
from backend.app.services.storage import InvestigationStorage

from tests.backend.helpers import make_agent_result, wait_terminal

app = create_app()


def _post_create(client, payload, files=None):
    data = {"payload": json.dumps(payload)}
    return client.post("/api/investigations", data=data, files=files)


@pytest.fixture()
def client(tmp_path):
    storage = InvestigationStorage(str(tmp_path / "investigations"))
    service = InvestigationService(
        agent_adapter=AgentAdapter(investigate_fn=lambda bi: make_agent_result()),
        ml_adapter=MLAdapter(),
        storage=storage,
    )
    set_service(service)
    yield TestClient(app)
    set_service(InvestigationService(
        agent_adapter=AgentAdapter(investigate_fn=lambda bi: make_agent_result()),
        ml_adapter=MLAdapter(),
        storage=InvestigationStorage(str(tmp_path / "cleanup")),
    ))


def _run_to_completion(client, service):
    payload = {
        "name": "Karachi Threads",
        "location": "Karachi",
        "website": "karachithreads.com",
        "social_links": ["instagram.com/karachithreads"],
        "additional_info": "10 years in business",
    }
    response = _post_create(client, payload)
    assert response.status_code == 202
    record = response.json()
    final = wait_terminal(service, record["id"])
    return record["id"], final


class TestHealth:
    def test_health(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert isinstance(body["llm_configured"], bool)


class TestCreate:
    def test_create_returns_202_with_record(self, client):
        response = _post_create(client, {"name": "Karachi Threads"})
        assert response.status_code == 202
        record = response.json()
        assert record["id"].startswith("inv_")
        assert record["status"] in ("queued", "investigating")
        assert record["business"]["name"] == "Karachi Threads"
        assert "result" not in record

    def test_create_normalises_urls(self, client):
        response = _post_create(
            client, {"name": "B", "website": "example.com"}
        )
        assert response.status_code == 202
        assert response.json()["business"]["website"] == "https://example.com/"

    def test_blank_name_is_friendly_422(self, client):
        response = _post_create(client, {"name": "  "})
        assert response.status_code == 422
        assert "Business name is required" in response.json()["detail"]

    def test_bad_website_is_friendly_422(self, client):
        response = _post_create(
            client, {"name": "B", "website": "not a url"}
        )
        assert response.status_code == 422
        assert "valid website URL" in response.json()["detail"]

    def test_bad_social_link_is_friendly_422(self, client):
        response = _post_create(
            client,
            {"name": "B", "social_links": ["instagram.com/x", "oops"]},
        )
        assert response.status_code == 422
        assert "valid social media URL" in response.json()["detail"]

    def test_validation_error_has_no_pydantic_jargon(self, client):
        response = _post_create(client, {"name": "  "})
        detail = response.json()["detail"]
        assert "Input should be" not in detail
        assert "validation error" not in detail.lower()


class TestDocuments:
    def test_uploaded_txt_appears_as_document_source(self, client):
        service = get_service()
        files = {
            "documents": (
                "revenue.txt",
                io.BytesIO(b"Monthly revenue Rs 500,000. 8 employees."),
                "text/plain",
            )
        }
        response = _post_create(
            client, {"name": "Doc Biz", "location": "Lahore"}, files=files
        )
        assert response.status_code == 202
        record = response.json()
        assert len(record["documents"]) == 1
        assert record["documents"][0]["filename"] == "revenue.txt"

        final = wait_terminal(service, record["id"])
        assert final["status"] == "completed"

        result = client.get(f"/api/investigations/{record['id']}/result").json()
        doc_sources = [s for s in result["sources_detail"] if s["type"] == "document"]
        assert doc_sources
        assert doc_sources[0]["name"] == "revenue.txt"
        assert any(
            ev["source_name"] == "uploaded_document:revenue.txt"
            for ev in result["evidence"]
        )


class TestStatusAndResult:
    def test_unknown_id_is_404(self, client):
        assert client.get("/api/investigations/inv_no/status").status_code == 404
        assert client.get("/api/investigations/inv_no").status_code == 404

    def test_result_409_while_running(self, client):
        # Hand-write a queued record so no race with the executor is possible.
        service = get_service()
        service._storage.save({
            "id": "inv_running", "created_at": "2026-08-31T00:00:00+00:00",
            "started_at": None, "completed_at": None, "status": "queued",
            "phase_label": "Queued", "error": None, "saved": False,
            "business": {"name": "Running Biz"}, "result": None,
        })
        response = client.get("/api/investigations/inv_running/result")
        assert response.status_code == 409
        assert "still running" in response.json()["detail"]

    def test_status_endpoint_shape(self, client):
        service = get_service()
        service._storage.save({
            "id": "inv_q", "created_at": "2026-08-31T00:00:00+00:00",
            "started_at": None, "completed_at": None, "status": "queued",
            "phase_label": "Queued", "error": None, "saved": False,
            "business": {"name": "Q Biz"}, "result": None,
        })
        body = client.get("/api/investigations/inv_q/status").json()
        assert body == {
            "id": "inv_q", "status": "queued", "phase_label": "Queued",
            "error": None,
        }

    def test_full_lifecycle(self, client):
        service = get_service()
        inv_id, final = _run_to_completion(client, service)
        assert final["status"] == "completed"

        status = client.get(f"/api/investigations/{inv_id}/status").json()
        assert status["status"] == "completed"
        assert status["error"] is None

        result = client.get(f"/api/investigations/{inv_id}/result")
        assert result.status_code == 200
        body = result.json()
        assert body["agent_status"] == "complete"
        assert body["recommendation"]
        assert body["analysis_context"]["credibility"]
        assert body["sources_detail"]

    def test_failed_investigation_result_is_422_with_message(self, tmp_path):
        def explode(bi):
            raise RuntimeError("boom")

        storage = InvestigationStorage(str(tmp_path / "inv"))
        service = InvestigationService(
            agent_adapter=AgentAdapter(investigate_fn=explode),
            ml_adapter=MLAdapter(),
            storage=storage,
        )
        set_service(service)
        try:
            client = TestClient(app)
            record = _post_create(
                client, {"name": "Fail Biz"}
            ).json()
            final = wait_terminal(service, record["id"])
            assert final["status"] == "failed"

            result = client.get(f"/api/investigations/{record['id']}/result")
            assert result.status_code == 422
            assert "temporarily unavailable" in result.json()["detail"]
        finally:
            set_service(InvestigationService(
                agent_adapter=AgentAdapter(investigate_fn=lambda bi: make_agent_result()),
                ml_adapter=MLAdapter(),
                storage=InvestigationStorage(str(tmp_path / "cleanup")),
            ))


class TestListAndSave:
    def test_list_contains_completed_summary(self, client):
        service = get_service()
        inv_id, _ = _run_to_completion(client, service)
        response = client.get("/api/investigations")
        assert response.status_code == 200
        summaries = response.json()
        assert any(s["id"] == inv_id for s in summaries)
        match = next(s for s in summaries if s["id"] == inv_id)
        assert match["status"] == "completed"
        assert match["trustworthiness"]["level"]
        assert match["justification"]

    def test_save_toggle(self, client):
        service = get_service()
        inv_id, _ = _run_to_completion(client, service)
        response = client.post(
            f"/api/investigations/{inv_id}/save", json={"saved": True}
        )
        assert response.status_code == 200
        assert response.json()["saved"] is True
        listed = client.get("/api/investigations").json()
        match = next(s for s in listed if s["id"] == inv_id)
        assert match["saved"] is True

    def test_save_unknown_404(self, client):
        response = client.post(
            "/api/investigations/inv_none/save", json={"saved": True}
        )
        assert response.status_code == 404


class TestAsk:
    def test_ask_returns_503_without_llm_key(self, client, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        service = get_service()
        inv_id, _ = _run_to_completion(client, service)
        response = client.post(
            f"/api/investigations/{inv_id}/ask", json={"question": "Why moderate?"}
        )
        assert response.status_code == 503
        assert "API key" in response.json()["detail"]

    def test_ask_answers_with_mocked_llm(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        ask.set_call_fn(lambda prompt, model, temp: "Because reviews are positive.")
        try:
            service = get_service()
            inv_id, _ = _run_to_completion(client, service)
            response = client.post(
                f"/api/investigations/{inv_id}/ask",
                json={"question": "Why is trust moderate?"},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["question"] == "Why is trust moderate?"
            assert body["answer"] == "Because reviews are positive."
        finally:
            ask.set_call_fn(None)

    def test_ask_409_when_not_complete(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        service = get_service()
        service._storage.save({
            "id": "inv_q2", "created_at": "2026-08-31T00:00:00+00:00",
            "started_at": None, "completed_at": None, "status": "queued",
            "phase_label": "Queued", "error": None, "saved": False,
            "business": {"name": "Q2 Biz"}, "result": None,
        })
        response = client.post(
            "/api/investigations/inv_q2/ask", json={"question": "Anything?"}
        )
        assert response.status_code == 409

    def test_ask_blank_question_422(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        service = get_service()
        inv_id, _ = _run_to_completion(client, service)
        response = client.post(
            f"/api/investigations/{inv_id}/ask", json={"question": "  "}
        )
        assert response.status_code == 422
        assert "type a question" in response.json()["detail"]

    def test_ask_llm_failure_is_502(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        def explode(prompt, model, temp):
            raise RuntimeError("api down")

        ask.set_call_fn(explode)
        try:
            service = get_service()
            inv_id, _ = _run_to_completion(client, service)
            response = client.post(
                f"/api/investigations/{inv_id}/ask",
                json={"question": "Why?"},
            )
            assert response.status_code == 502
            assert "could not reach the model" in response.json()["detail"]
        finally:
            ask.set_call_fn(None)


class TestNoStackTraces:
    def test_unhandled_error_returns_friendly_500(self, client, monkeypatch):
        service = get_service()

        def explode(investigation_id):
            raise RuntimeError("secret internal details")

        monkeypatch.setattr(service, "get", explode)
        tolerant = TestClient(app, raise_server_exceptions=False)
        response = tolerant.get("/api/investigations/inv_x/status")
        assert response.status_code == 500
        assert "secret internal details" not in response.json()["detail"]
        assert response.json()["detail"] == (
            "Something went wrong on our side. Please try again."
        )
