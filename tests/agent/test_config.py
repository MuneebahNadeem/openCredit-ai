"""
Tests for agent/config.py — InvestigationConfig defaults, env overrides, validation.

Run with:  python -m pytest tests/agent/test_config.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from agent.config import InvestigationConfig


# ── Defaults ──────────────────────────────────────────────────────────────────

class TestDefaults:

    def test_max_searches_default(self):
        c = InvestigationConfig()
        assert c.max_searches == 15

    def test_max_sources_default(self):
        c = InvestigationConfig()
        assert c.max_sources == 20

    def test_max_iterations_default(self):
        c = InvestigationConfig()
        assert c.max_iterations == 30

    def test_max_evidence_items_default(self):
        c = InvestigationConfig()
        assert c.max_evidence_items == 50

    def test_min_evidence_to_conclude_default(self):
        c = InvestigationConfig()
        assert c.min_evidence_to_conclude == 3

    def test_request_timeout_default(self):
        c = InvestigationConfig()
        assert c.request_timeout_s == pytest.approx(10.0)

    def test_llm_model_default(self):
        c = InvestigationConfig()
        assert c.llm_model == "gpt-4o-mini"

    def test_llm_temperature_default(self):
        c = InvestigationConfig()
        assert c.llm_temperature == pytest.approx(0.2)

    def test_llm_max_tokens_default(self):
        c = InvestigationConfig()
        assert c.llm_max_tokens == 2048

    def test_search_engine_default(self):
        c = InvestigationConfig()
        assert c.search_engine == "duckduckgo"

    def test_parse_self_reported_default(self):
        c = InvestigationConfig()
        assert c.parse_self_reported is True

    def test_respect_robots_default(self):
        c = InvestigationConfig()
        assert c.respect_robots_txt is True


# ── Direct overrides ──────────────────────────────────────────────────────────

class TestDirectOverrides:

    def test_override_max_searches(self):
        c = InvestigationConfig(max_searches=5)
        assert c.max_searches == 5

    def test_override_llm_model(self):
        c = InvestigationConfig(llm_model="gpt-4")
        assert c.llm_model == "gpt-4"

    def test_override_temperature(self):
        c = InvestigationConfig(llm_temperature=0.7)
        assert c.llm_temperature == pytest.approx(0.7)

    def test_override_parse_self_reported_false(self):
        c = InvestigationConfig(parse_self_reported=False)
        assert c.parse_self_reported is False

    def test_multiple_overrides(self):
        c = InvestigationConfig(max_searches=3, max_sources=5, max_iterations=10)
        assert c.max_searches == 3
        assert c.max_sources == 5
        assert c.max_iterations == 10


# ── Env var overrides ─────────────────────────────────────────────────────────

class TestEnvVarOverrides:

    def test_env_max_searches(self, monkeypatch):
        monkeypatch.setenv("AGENT_MAX_SEARCHES", "7")
        c = InvestigationConfig()
        assert c.max_searches == 7

    def test_env_llm_model(self, monkeypatch):
        monkeypatch.setenv("AGENT_LLM_MODEL", "gpt-4-turbo")
        c = InvestigationConfig()
        assert c.llm_model == "gpt-4-turbo"

    def test_env_timeout(self, monkeypatch):
        monkeypatch.setenv("AGENT_REQUEST_TIMEOUT", "30.0")
        c = InvestigationConfig()
        assert c.request_timeout_s == pytest.approx(30.0)

    def test_env_bool_true(self, monkeypatch):
        monkeypatch.setenv("AGENT_PARSE_SELF_REPORTED", "false")
        c = InvestigationConfig()
        assert c.parse_self_reported is False

    def test_env_bool_yes(self, monkeypatch):
        monkeypatch.setenv("AGENT_PARSE_SELF_REPORTED", "yes")
        c = InvestigationConfig()
        assert c.parse_self_reported is True

    def test_env_invalid_int_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("AGENT_MAX_SEARCHES", "notanumber")
        c = InvestigationConfig()
        assert c.max_searches == 15  # default


# ── Validation ────────────────────────────────────────────────────────────────

class TestValidation:

    def test_zero_max_searches_raises(self):
        with pytest.raises(ValueError, match="max_searches"):
            InvestigationConfig(max_searches=0)

    def test_negative_max_sources_raises(self):
        with pytest.raises(ValueError, match="max_sources"):
            InvestigationConfig(max_sources=-1)

    def test_zero_max_iterations_raises(self):
        with pytest.raises(ValueError, match="max_iterations"):
            InvestigationConfig(max_iterations=0)

    def test_zero_timeout_raises(self):
        with pytest.raises(ValueError, match="request_timeout_s"):
            InvestigationConfig(request_timeout_s=0.0)

    def test_negative_timeout_raises(self):
        with pytest.raises(ValueError, match="request_timeout_s"):
            InvestigationConfig(request_timeout_s=-5.0)

    def test_temperature_too_high_raises(self):
        with pytest.raises(ValueError, match="llm_temperature"):
            InvestigationConfig(llm_temperature=2.5)

    def test_temperature_negative_raises(self):
        with pytest.raises(ValueError, match="llm_temperature"):
            InvestigationConfig(llm_temperature=-0.1)

    def test_temperature_boundary_ok(self):
        c = InvestigationConfig(llm_temperature=0.0)
        assert c.llm_temperature == 0.0
        c2 = InvestigationConfig(llm_temperature=2.0)
        assert c2.llm_temperature == 2.0

    def test_llm_max_tokens_too_low_raises(self):
        with pytest.raises(ValueError, match="llm_max_tokens"):
            InvestigationConfig(llm_max_tokens=32)

    def test_min_evidence_zero_raises(self):
        with pytest.raises(ValueError, match="min_evidence_to_conclude"):
            InvestigationConfig(min_evidence_to_conclude=0)


# ── summary() ─────────────────────────────────────────────────────────────────

class TestSummary:

    def test_summary_contains_model(self):
        c = InvestigationConfig(llm_model="gpt-4")
        assert "gpt-4" in c.summary()

    def test_summary_contains_max_searches(self):
        c = InvestigationConfig(max_searches=8)
        assert "8" in c.summary()

    def test_summary_is_string(self):
        c = InvestigationConfig()
        assert isinstance(c.summary(), str)
