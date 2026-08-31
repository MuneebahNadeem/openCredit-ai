"""Tests for the investigation orchestration service."""

from __future__ import annotations

import pytest

from agent.schemas.result import InvestigationStatus
from backend.app.services.adapters.agent_adapter import AgentAdapter
from backend.app.services.adapters.ml_adapter import MLAdapter
from backend.app.services.investigations import InvestigationService
from backend.app.services.storage import InvestigationStorage

from tests.backend.helpers import make_agent_result, wait_terminal


@pytest.fixture()
def service(tmp_path):
    storage = InvestigationStorage(str(tmp_path / "investigations"))
    return InvestigationService(
        agent_adapter=AgentAdapter(investigate_fn=lambda bi: make_agent_result()),
        ml_adapter=MLAdapter(),
        storage=storage,
    )


@pytest.fixture()
def business_input():
    from agent.schemas.input import BusinessInput

    return BusinessInput(name="Karachi Threads", location="Karachi")


class TestHappyPath:
    def test_create_returns_queued_record(self, service, business_input):
        record = service.create(business_input)
        assert record["id"].startswith("inv_")
        assert record["status"] == "queued"
        assert record["phase_label"] == "Queued"
        assert record["result"] is None
        assert record["business"]["name"] == "Karachi Threads"
        assert record["saved"] is False

    def test_pipeline_reaches_completed(self, service, business_input):
        record = service.create(business_input)
        final = wait_terminal(service, record["id"])
        assert final["status"] == "completed"
        assert final["error"] is None
        assert final["completed_at"] is not None
        assert final["started_at"] is not None

    def test_aggregated_result_shape(self, service, business_input):
        record = service.create(business_input)
        final = wait_terminal(service, record["id"])
        result = final["result"]

        # Person 1/2 schema fields preserved
        assert result["status"] == "complete"
        assert result["agent_status"] == "complete"
        assert len(result["evidence"]) == 4
        assert result["trustworthiness"]["level"] in (
            "high", "moderate", "low", "insufficient_evidence",
        )
        assert result["business_potential"]["level"] in (
            "high", "moderate", "low", "insufficient_evidence",
        )
        assert result["justification"]
        assert result["positive_signals"]
        assert result["risk_signals"]

        # Person 3 aggregations
        assert result["recommendation"] in (
            "approve", "approve_with_conditions", "decline",
            "further_review", "insufficient_data",
        )
        assert "credibility" in result["analysis_context"]
        assert "sentiment" in result["analysis_context"]
        assert "business_input" not in result  # echoed separately

    def test_sources_detail_groups_evidence(self, service, business_input):
        record = service.create(business_input)
        final = wait_terminal(service, record["id"])
        detail = final["result"]["sources_detail"]

        by_url = {s["url"]: s for s in detail}
        assert by_url["https://instagram.com/karachithreads"]["type"] == "social"
        assert by_url["https://google.com/maps"]["type"] == "review"
        assert by_url["https://daraz.pk/shop/karachi-threads"]["type"] == "marketplace"
        # Visited but empty source still listed
        assert by_url["https://example.com/empty-page"]["evidence_count"] == 0
        # Sorted by evidence count, descending
        counts = [s["evidence_count"] for s in detail]
        assert counts == sorted(counts, reverse=True)

    def test_result_json_serializable(self, service, business_input):
        import json

        record = service.create(business_input)
        final = wait_terminal(service, record["id"])
        json.dumps(final["result"])  # must not raise

    def test_summary_fields(self, service, business_input):
        record = service.create(business_input)
        wait_terminal(service, record["id"])
        summaries = service.list_summaries()
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary["id"] == record["id"]
        assert summary["status"] == "completed"
        assert summary["trustworthiness"]["level"]
        assert summary["justification"]
        assert "result" not in summary
        assert "evidence" not in summary


class TestStatusMapping:
    @pytest.mark.parametrize(
        "agent_status, expected",
        [
            (InvestigationStatus.COMPLETE, "completed"),
            (InvestigationStatus.LIMIT_REACHED, "completed"),
            (InvestigationStatus.PARTIAL, "partial"),
        ],
    )
    def test_final_status(self, agent_status, expected):
        assert InvestigationService._final_status(agent_status) == expected

    def test_partial_agent_status_flows_through(self, tmp_path, business_input):
        storage = InvestigationStorage(str(tmp_path / "inv"))
        svc = InvestigationService(
            agent_adapter=AgentAdapter(
                investigate_fn=lambda bi: make_agent_result(
                    status=InvestigationStatus.PARTIAL
                )
            ),
            ml_adapter=MLAdapter(),
            storage=storage,
        )
        record = svc.create(business_input)
        final = wait_terminal(svc, record["id"])
        assert final["status"] == "partial"
        assert "limited evidence" in final["phase_label"].lower()


class TestFailures:
    def test_agent_failure_is_friendly(self, tmp_path, business_input):
        def explode(bi):
            raise RuntimeError("openai.APIConnectionError: connection refused")

        storage = InvestigationStorage(str(tmp_path / "inv"))
        svc = InvestigationService(
            agent_adapter=AgentAdapter(investigate_fn=explode),
            ml_adapter=MLAdapter(),
            storage=storage,
        )
        record = svc.create(business_input)
        final = wait_terminal(svc, record["id"])
        assert final["status"] == "failed"
        assert "openai" not in (final["error"] or "").lower()
        assert "Traceback" not in (final["error"] or "")
        assert final["result"] is None

    def test_ml_failure_is_friendly(self, tmp_path, business_input):
        def explode(result):
            raise ValueError("numpy.core._exceptions._ArrayMemoryError")

        storage = InvestigationStorage(str(tmp_path / "inv"))
        svc = InvestigationService(
            agent_adapter=AgentAdapter(investigate_fn=lambda bi: make_agent_result()),
            ml_adapter=MLAdapter(assess_fn=explode),
            storage=storage,
        )
        record = svc.create(business_input)
        final = wait_terminal(svc, record["id"])
        assert final["status"] == "failed"
        assert "risk assessment could not be completed" in final["error"]


class TestSave:
    def test_set_saved_persists(self, service, business_input):
        record = service.create(business_input)
        wait_terminal(service, record["id"])
        updated = service.set_saved(record["id"], True)
        assert updated["saved"] is True
        assert service.get(record["id"])["saved"] is True

    def test_set_saved_unknown_id(self, service):
        assert service.set_saved("inv_missing", True) is None
