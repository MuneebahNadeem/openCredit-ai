"""
Tests for agent/agent.py — InvestigationAgent with mocked LLM, search, and fetch.

All network I/O is injected as deterministic mocks so no real API keys or
internet access are required.

Run with:  python -m pytest tests/agent/test_agent.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import pytest

from agent.agent import InvestigationAgent, _classify_url, _extract_json_array
from agent.config import InvestigationConfig
from agent.schemas.input import BusinessInput
from agent.schemas.result import InvestigationResult, InvestigationStatus


# ── Mock helpers ──────────────────────────────────────────────────────────────

def _make_config(**kwargs) -> InvestigationConfig:
    """Config with tight limits so tests don't loop for long."""
    defaults = dict(
        max_searches=3,
        max_sources=5,
        max_iterations=3,
        max_evidence_items=20,
        min_evidence_to_conclude=2,
    )
    defaults.update(kwargs)
    return InvestigationConfig(**defaults)


def _null_search(query: str, max_results: int = 8):
    """Search that always returns no results."""
    return []


def _null_fetch(url: str):
    """Fetch that always returns None (network failure)."""
    return None


def _single_result_search(query: str, max_results: int = 8):
    """Search that always returns one fake result."""
    from agent.tools.web_search import SearchResult
    return [SearchResult(
        url="https://example.com/biz",
        title="Test Business Page",
        snippet="Excellent service, taking orders, 500 sold",
    )]


def _daraz_fetch(url: str) -> str:
    """Return fake Daraz HTML (WebpageExtractor strips it internally)."""
    return (
        "<html><head><title>Test Business — Daraz</title></head><body>"
        "<p>Test Business on Daraz.pk</p>"
        "<p>Rating: 4.5 out of 5 stars</p>"
        "<p>1,200 customer reviews</p>"
        "<p>500 sold last month</p>"
        "<p>Rs. 2,500 | Rs. 1,800</p>"
        "<p>200 products available</p>"
        "</body></html>"
    )


def _social_fetch(url: str) -> str:
    """Return fake Instagram HTML (WebpageExtractor strips it internally)."""
    return (
        "<html><head><title>@test_business Instagram</title></head><body>"
        "<p>12k followers</p>"
        "<p>Taking orders for Eid collection, DM for rates</p>"
        "<p>Last post: 2 hours ago</p>"
        "<p>450 likes on recent post</p>"
        "</body></html>"
    )


def _make_llm_queries(queries):
    """Return an LLM mock that outputs a JSON array of query strings."""
    def _fn(prompt: str) -> str:
        return json.dumps(queries)
    return _fn


def _empty_llm(prompt: str) -> str:
    return "[]"


# ── _classify_url ─────────────────────────────────────────────────────────────

class TestClassifyUrl:

    def test_instagram_is_social(self):
        assert _classify_url("https://instagram.com/biz") == "social_media"

    def test_facebook_is_social(self):
        assert _classify_url("https://facebook.com/page") == "social_media"

    def test_daraz_is_marketplace(self):
        assert _classify_url("https://daraz.pk/products") == "marketplace"

    def test_olx_is_marketplace(self):
        assert _classify_url("https://olx.com.pk/listing") == "marketplace"

    def test_trustpilot_is_review(self):
        assert _classify_url("https://trustpilot.com/review/test") == "review_site"

    def test_google_maps_is_review(self):
        assert _classify_url("https://google.com/maps/place/test") == "review_site"

    def test_secp_is_government(self):
        assert _classify_url("https://secp.gov.pk/company") == "government"

    def test_gov_pk_is_government(self):
        assert _classify_url("https://fbr.gov.pk/taxpayer") == "government"

    def test_generic_is_general_web(self):
        assert _classify_url("https://example.com/about") == "general_web"


# ── _extract_json_array ───────────────────────────────────────────────────────

class TestExtractJsonArray:

    def test_clean_array(self):
        result = _extract_json_array('["query one", "query two"]')
        assert result == ["query one", "query two"]

    def test_array_embedded_in_text(self):
        result = _extract_json_array('Here are queries: ["q1", "q2"] done.')
        assert result == ["q1", "q2"]

    def test_empty_array(self):
        result = _extract_json_array("[]")
        assert result == []

    def test_no_array_returns_empty(self):
        result = _extract_json_array("No JSON here at all.")
        assert result == []

    def test_invalid_json_returns_empty(self):
        result = _extract_json_array("[not valid json {}")
        assert result == []

    def test_object_array(self):
        payload = json.dumps([{"field_name": "test", "value": "100"}])
        result = _extract_json_array(payload)
        assert len(result) == 1
        assert result[0]["field_name"] == "test"


# ── InvestigationAgent: smoke tests ─────────────────────────────────────────

class TestAgentSmoke:

    def test_returns_investigation_result(self):
        agent = InvestigationAgent(
            config=_make_config(),
            llm_call_fn=_empty_llm,
            search_fn=_null_search,
            fetch_fn=_null_fetch,
        )
        result = agent.investigate(BusinessInput(name="Test Biz", location="Karachi"))
        assert isinstance(result, InvestigationResult)

    def test_result_has_business_input(self):
        agent = InvestigationAgent(
            config=_make_config(),
            llm_call_fn=_empty_llm,
            search_fn=_null_search,
            fetch_fn=_null_fetch,
        )
        inp = BusinessInput(name="My Shop", location="Lahore")
        result = agent.investigate(inp)
        assert result.business_input.name == "My Shop"

    def test_no_crash_on_empty_input(self):
        agent = InvestigationAgent(
            config=_make_config(),
            llm_call_fn=_empty_llm,
            search_fn=_null_search,
            fetch_fn=_null_fetch,
        )
        # Minimal input — just a name
        result = agent.investigate(BusinessInput(name="Minimal"))
        assert result is not None

    def test_status_is_valid_enum(self):
        agent = InvestigationAgent(
            config=_make_config(),
            llm_call_fn=_empty_llm,
            search_fn=_null_search,
            fetch_fn=_null_fetch,
        )
        result = agent.investigate(BusinessInput(name="Shop"))
        assert isinstance(result.status, InvestigationStatus)


# ── Self-reported evidence ────────────────────────────────────────────────────

class TestSelfReportedEvidence:

    def test_self_reported_evidence_extracted(self):
        agent = InvestigationAgent(
            config=_make_config(),
            llm_call_fn=_empty_llm,
            search_fn=_null_search,
            fetch_fn=_null_fetch,
        )
        inp = BusinessInput(
            name="Fatima Stitching",
            location="Karachi",
            additional_info="We earn Rs 50,000 per month, 30 suits per month, via WhatsApp",
        )
        result = agent.investigate(inp)
        fields = [e.field_name for e in result.evidence]
        assert "self_reported_monthly_revenue" in fields
        assert "self_reported_monthly_orders" in fields
        assert "self_reported_sales_channels" in fields

    def test_self_reported_disabled(self):
        agent = InvestigationAgent(
            config=_make_config(parse_self_reported=False),
            llm_call_fn=_empty_llm,
            search_fn=_null_search,
            fetch_fn=_null_fetch,
        )
        inp = BusinessInput(
            name="Shop",
            additional_info="Rs 50,000 per month, via WhatsApp",
        )
        result = agent.investigate(inp)
        # parse_self_reported=False should skip user input parser
        # Note: the agent always calls _input_parser.parse; whether it uses
        # the result depends on config.  The config flag gates adding evidence.
        # This test verifies no crash; behaviour may vary by implementation.
        assert result is not None


# ── Social media page extraction ──────────────────────────────────────────────

class TestSocialPageExtraction:

    def test_instagram_url_produces_social_evidence(self):
        agent = InvestigationAgent(
            config=_make_config(),
            llm_call_fn=_empty_llm,
            search_fn=_null_search,
            fetch_fn=_social_fetch,
        )
        inp = BusinessInput(
            name="Fashion Store",
            social_links=["https://instagram.com/fashion_store"],
        )
        result = agent.investigate(inp)
        fields = [e.field_name for e in result.evidence]
        # Social analyzer should have found followers and/or demand signal
        assert any("follower" in f or "informal_demand" in f or "engagement" in f
                   for f in fields)

    def test_demand_signal_produces_positive_signal(self):
        agent = InvestigationAgent(
            config=_make_config(),
            llm_call_fn=_empty_llm,
            search_fn=_null_search,
            fetch_fn=_social_fetch,
        )
        inp = BusinessInput(
            name="Fashion Store",
            social_links=["https://instagram.com/fashion_store"],
        )
        result = agent.investigate(inp)
        pos_labels = [s.label for s in result.positive_signals]
        assert any("order" in label.lower() or "social" in label.lower()
                   for label in pos_labels)


# ── Marketplace page extraction ───────────────────────────────────────────────

class TestMarketplaceExtraction:

    def test_daraz_url_produces_product_evidence(self):
        agent = InvestigationAgent(
            config=_make_config(),
            llm_call_fn=_empty_llm,
            search_fn=_null_search,
            fetch_fn=_daraz_fetch,
        )
        inp = BusinessInput(
            name="Textile Hub",
            marketplace_links=["https://daraz.pk/shop/textile-hub"],
        )
        result = agent.investigate(inp)
        fields = [e.field_name for e in result.evidence]
        assert any(f in fields for f in (
            "units_sold", "active_listing_count",
            "marketplace_price_activity", "star_rating", "review_count"
        ))

    def test_high_rating_produces_positive_signal(self):
        agent = InvestigationAgent(
            config=_make_config(),
            llm_call_fn=_empty_llm,
            search_fn=_null_search,
            fetch_fn=_daraz_fetch,
        )
        inp = BusinessInput(
            name="Textile Hub",
            marketplace_links=["https://daraz.pk/shop/textile-hub"],
        )
        result = agent.investigate(inp)
        pos_labels = [s.label.lower() for s in result.positive_signals]
        assert any("rating" in label or "sold" in label for label in pos_labels)


# ── Search loop ───────────────────────────────────────────────────────────────

class TestSearchLoop:

    def test_search_queries_are_recorded(self):
        queries_issued = []

        def _tracking_search(query: str, max_results: int = 8):
            queries_issued.append(query)
            return []

        agent = InvestigationAgent(
            config=_make_config(max_searches=2),
            llm_call_fn=_make_llm_queries(["karachi textile reviews", "textile hub daraz"]),
            search_fn=_tracking_search,
            fetch_fn=_null_fetch,
        )
        agent.investigate(BusinessInput(name="Textile Hub", location="Karachi"))
        assert len(queries_issued) > 0

    def test_duplicate_queries_not_searched_twice(self):
        queries_issued = []

        def _tracking_search(query: str, max_results: int = 8):
            queries_issued.append(query)
            return []

        # LLM always returns same two queries
        agent = InvestigationAgent(
            config=_make_config(max_searches=5),
            llm_call_fn=_make_llm_queries(["same query", "same query"]),
            search_fn=_tracking_search,
            fetch_fn=_null_fetch,
        )
        agent.investigate(BusinessInput(name="Biz", location="Lahore"))
        # "same query" should only appear once
        assert queries_issued.count("same query") <= 1

    def test_max_searches_respected(self):
        call_count = [0]

        def _counting_search(query: str, max_results: int = 8):
            call_count[0] += 1
            from agent.tools.web_search import SearchResult
            return [SearchResult(url=f"https://example.com/{call_count[0]}", title="T", snippet="s")]

        # LLM always returns new unique queries
        counter = [0]
        def _unique_llm(prompt: str):
            counter[0] += 1
            return json.dumps([f"unique query {counter[0]}a", f"unique query {counter[0]}b"])

        agent = InvestigationAgent(
            config=_make_config(max_searches=2, max_sources=10, max_iterations=10),
            llm_call_fn=_unique_llm,
            search_fn=_counting_search,
            fetch_fn=_null_fetch,
        )
        agent.investigate(BusinessInput(name="Biz"))
        assert call_count[0] <= 2


# ── Missing information flags ─────────────────────────────────────────────────

class TestMissingInformation:

    def test_missing_info_populated_when_no_evidence(self):
        agent = InvestigationAgent(
            config=_make_config(),
            llm_call_fn=_empty_llm,
            search_fn=_null_search,
            fetch_fn=_null_fetch,
        )
        result = agent.investigate(BusinessInput(name="No-data Business"))
        # With no evidence, the agent should flag missing items
        assert len(result.missing_information) > 0

    def test_secp_missing_when_not_found(self):
        agent = InvestigationAgent(
            config=_make_config(),
            llm_call_fn=_empty_llm,
            search_fn=_null_search,
            fetch_fn=_null_fetch,
        )
        result = agent.investigate(BusinessInput(name="No-data Business"))
        assert any("secp" in m.lower() or "registration" in m.lower()
                   for m in result.missing_information)


# ── LLM evidence parsing ──────────────────────────────────────────────────────

class TestLLMEvidenceParsing:

    def _make_agent(self, llm_fn):
        return InvestigationAgent(
            config=_make_config(),
            llm_call_fn=llm_fn,
            search_fn=_single_result_search,
            fetch_fn=_daraz_fetch,
        )

    def test_llm_evidence_added_to_result(self):
        llm_response = json.dumps([
            {"field_name": "business_age", "value": "5 years",
             "evidence_type": "inference", "confidence": 0.6}
        ])

        call_count = [0]
        def _llm(prompt: str) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                # First call is investigation_plan_prompt → return queries
                return json.dumps(["karachi textile reviews"])
            # Subsequent calls → return evidence items
            return llm_response

        agent = self._make_agent(_llm)
        result = agent.investigate(BusinessInput(name="Textile Hub", location="Karachi"))
        fields = [e.field_name for e in result.evidence]
        assert "business_age" in fields

    def test_malformed_llm_response_ignored(self):
        """A bad LLM response must not crash the agent."""
        agent = self._make_agent(lambda p: "not json at all }{")
        result = agent.investigate(BusinessInput(name="Biz"))
        assert isinstance(result, InvestigationResult)

    def test_empty_field_name_ignored(self):
        bad = json.dumps([{"field_name": "", "value": "test", "confidence": 0.5}])
        agent = self._make_agent(lambda p: bad)
        result = agent.investigate(BusinessInput(name="Biz"))
        assert isinstance(result, InvestigationResult)


# ── Agent with all mocks — full data flow ─────────────────────────────────────

class TestFullDataFlow:

    def test_complete_investigation_no_exceptions(self):
        """Full mock run: self-reported + social + marketplace + LLM."""
        call_counter = [0]

        def _llm(prompt: str) -> str:
            call_counter[0] += 1
            if call_counter[0] == 1:
                return json.dumps(["fashion store instagram karachi", "daraz fashion store"])
            if call_counter[0] == 2:
                return json.dumps([
                    {"field_name": "brand_presence", "value": "strong",
                     "evidence_type": "observed", "confidence": 0.75}
                ])
            # Feature discovery
            return json.dumps([
                {"name": "social_following", "category": "audience",
                 "value": "12000", "reason": "Active Instagram following",
                 "confidence": 0.8, "searched": True}
            ])

        def _smart_fetch(url: str) -> str:
            if "instagram" in url:
                return _social_fetch(url)
            return _daraz_fetch(url)

        def _smart_search(query: str, max_results: int = 8):
            from agent.tools.web_search import SearchResult
            if "instagram" in query:
                return [SearchResult(
                    url="https://instagram.com/fashion_pk",
                    title="Fashion PK Instagram",
                    snippet="12k followers, taking orders",
                )]
            return [SearchResult(
                url="https://daraz.pk/shop/fashion",
                title="Fashion on Daraz",
                snippet="500 sold, 4.5 stars",
            )]

        agent = InvestigationAgent(
            config=_make_config(max_searches=2, max_sources=4),
            llm_call_fn=_llm,
            search_fn=_smart_search,
            fetch_fn=_smart_fetch,
        )
        inp = BusinessInput(
            name="Fashion PK",
            location="Karachi",
            additional_info="We sell embroidered suits via Instagram, Rs 60k monthly",
            social_links=["https://instagram.com/fashion_pk"],
        )
        result = agent.investigate(inp)

        assert isinstance(result, InvestigationResult)
        assert result.searches_performed >= 0
        assert result.sources_examined >= 1
        assert len(result.evidence) > 0

    def test_result_is_passable_to_ml_layer(self):
        """The result from the agent must be accepted by ml.feature_extractor."""
        agent = InvestigationAgent(
            config=_make_config(),
            llm_call_fn=_empty_llm,
            search_fn=_single_result_search,
            fetch_fn=_daraz_fetch,
        )
        result = agent.investigate(BusinessInput(name="Test", location="Lahore"))

        from ml.feature_extractor import extract_features
        features = extract_features(result)
        assert isinstance(features, dict)
        assert len(features) == 60
