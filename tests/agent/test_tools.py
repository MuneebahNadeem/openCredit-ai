"""
Tests for agent/tools/ — SocialAnalyzer, ReviewAnalyzer, ProductAnalyzer,
UserInputParser.  WebSearchTool and WebpageExtractor are tested at the
integration level via mocked network functions.

Run with:  python -m pytest tests/agent/test_tools.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from agent.schemas.evidence import EvidenceType, SourceReliability
from agent.schemas.input import BusinessInput
from agent.tools.social_analyzer import SocialAnalyzer
from agent.tools.review_analyzer import ReviewAnalyzer
from agent.tools.product_analyzer import ProductAnalyzer
from agent.tools.user_input_parser import UserInputParser


# ═══════════════════════════════════════════════════════════════════════════════
# SocialAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════

class TestSocialAnalyzer:

    def setup_method(self):
        self.analyzer = SocialAnalyzer()

    def test_empty_text_returns_empty(self):
        items = self.analyzer.analyze("", platform="instagram")
        assert items == []

    def test_follower_count_extracted(self):
        text = "25,000 followers on our page"
        items = self.analyzer.analyze(text, platform="instagram")
        assert any("follower" in i.field_name for i in items)
        follower = next(i for i in items if "follower" in i.field_name)
        assert follower.value == "25000"

    def test_follower_count_k_suffix(self):
        text = "12k followers"
        items = self.analyzer.analyze(text, platform="instagram")
        assert any("follower" in i.field_name for i in items)
        follower = next(i for i in items if "follower" in i.field_name)
        assert follower.value == "12000"

    def test_engagement_extracted(self):
        text = "500 likes and 120 comments on the latest post"
        items = self.analyzer.analyze(text, platform="facebook")
        assert any(i.field_name == "facebook_engagement" for i in items)

    def test_active_indicator_extracted(self):
        text = "Last post: 2 hours ago"
        items = self.analyzer.analyze(text, platform="instagram")
        assert any("recently_active" in i.field_name for i in items)

    def test_informal_demand_taking_orders(self):
        text = "Taking orders for Eid suits, DM for rates"
        items = self.analyzer.analyze(text, platform="whatsapp")
        assert any(i.field_name == "informal_demand_signal" for i in items)

    def test_informal_demand_slots_available(self):
        text = "Limited slots available this month, booking open"
        items = self.analyzer.analyze(text, platform="instagram")
        assert any(i.field_name == "informal_demand_signal" for i in items)

    def test_informal_demand_reliability_is_low(self):
        text = "Taking orders for bridal suits, DM to order"
        items = self.analyzer.analyze(text, platform="instagram")
        demand = next((i for i in items if i.field_name == "informal_demand_signal"), None)
        assert demand is not None
        assert demand.source_reliability == SourceReliability.LOW

    def test_demand_signal_value(self):
        text = "taking orders now!"
        items = self.analyzer.analyze(text, platform="instagram")
        demand = next((i for i in items if i.field_name == "informal_demand_signal"), None)
        assert demand is not None
        assert demand.value == "active_ordering"

    def test_all_items_are_observed(self):
        text = "10k followers, 2 hours ago, taking orders"
        items = self.analyzer.analyze(text, platform="instagram")
        for item in items:
            assert item.evidence_type == EvidenceType.OBSERVED

    def test_platform_reliability_instagram(self):
        text = "5k followers"
        items = self.analyzer.analyze(text, platform="instagram")
        for item in items:
            assert item.source_reliability in (
                SourceReliability.MEDIUM, SourceReliability.LOW
            )

    def test_platform_reliability_tiktok_low(self):
        text = "50k followers"
        items = self.analyzer.analyze(text, platform="tiktok")
        followers = [i for i in items if "follower" in i.field_name]
        assert all(i.source_reliability == SourceReliability.LOW for i in followers)

    def test_detect_platform_instagram(self):
        assert SocialAnalyzer.detect_platform("https://instagram.com/test") == "instagram"

    def test_detect_platform_facebook(self):
        assert SocialAnalyzer.detect_platform("https://facebook.com/page") == "facebook"

    def test_detect_platform_unknown(self):
        assert SocialAnalyzer.detect_platform("https://example.com") == "unknown"

    def test_detect_platform_youtube(self):
        assert SocialAnalyzer.detect_platform("https://youtube.com/channel") == "youtube"

    def test_no_text_no_items(self):
        items = self.analyzer.analyze("   ", platform="instagram")
        assert items == []


# ═══════════════════════════════════════════════════════════════════════════════
# ReviewAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════

class TestReviewAnalyzer:

    def setup_method(self):
        self.analyzer = ReviewAnalyzer()

    def test_empty_text_returns_empty(self):
        items = self.analyzer.analyze("", source_name="Google")
        assert items == []

    def test_star_rating_extracted(self):
        text = "Rating: 4.5 out of 5 stars"
        items = self.analyzer.analyze(text, source_name="Google")
        rating = next((i for i in items if i.field_name == "star_rating"), None)
        assert rating is not None
        assert float(rating.value) == pytest.approx(4.5)

    def test_rating_slash_format(self):
        text = "Rated 4.2/5"
        items = self.analyzer.analyze(text, source_name="Daraz")
        rating = next((i for i in items if i.field_name == "star_rating"), None)
        assert rating is not None
        assert float(rating.value) == pytest.approx(4.2)

    def test_rating_out_of_range_ignored(self):
        # A "rating" of 99 is not a valid star rating
        text = "Score 99 out of 5"
        items = self.analyzer.analyze(text, source_name="Google")
        rating = next((i for i in items if i.field_name == "star_rating"), None)
        assert rating is None

    def test_review_count_extracted(self):
        text = "1,234 customer reviews"
        items = self.analyzer.analyze(text, source_name="Daraz")
        count = next((i for i in items if i.field_name == "review_count"), None)
        assert count is not None
        assert int(count.value) == 1234

    def test_complaint_signal_detected(self):
        text = "Several complaints filed. User reported fraud on this listing."
        items = self.analyzer.analyze(text, source_name="Google")
        complaint = next((i for i in items if i.field_name == "complaint_signals"), None)
        assert complaint is not None

    def test_review_snippets_extracted(self):
        text = "The product is excellent. Delivery was great. Everything was amazing."
        items = self.analyzer.analyze(text, source_name="Daraz")
        snippets = next((i for i in items if i.field_name == "review_snippets"), None)
        assert snippets is not None

    def test_google_gets_high_reliability(self):
        text = "Rating: 4.0"
        items = self.analyzer.analyze(text, source_name="Google Reviews")
        for item in items:
            assert item.source_reliability == SourceReliability.HIGH

    def test_olx_gets_low_reliability(self):
        text = "Rating: 3.5"
        items = self.analyzer.analyze(text, source_name="OLX")
        for item in items:
            assert item.source_reliability == SourceReliability.LOW

    def test_all_items_observed_type(self):
        text = "Rating: 4.0 | 100 reviews | great product!"
        items = self.analyzer.analyze(text, source_name="Daraz")
        for item in items:
            assert item.evidence_type == EvidenceType.OBSERVED

    def test_no_signals_in_generic_text(self):
        text = "This is a simple page with no review information at all."
        items = self.analyzer.analyze(text, source_name="Unknown")
        assert items == []


# ═══════════════════════════════════════════════════════════════════════════════
# ProductAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════

class TestProductAnalyzer:

    def setup_method(self):
        self.analyzer = ProductAnalyzer()

    def test_empty_text_returns_empty(self):
        items = self.analyzer.analyze("")
        assert items == []

    def test_units_sold_extracted(self):
        text = "1,500 sold in last 30 days"
        items = self.analyzer.analyze(text, source_name="Daraz")
        sold = next((i for i in items if i.field_name == "units_sold"), None)
        assert sold is not None
        assert int(sold.value) == 1500

    def test_units_sold_with_plus(self):
        text = "500+ sold"
        items = self.analyzer.analyze(text, source_name="Daraz")
        sold = next((i for i in items if i.field_name == "units_sold"), None)
        assert sold is not None
        assert int(sold.value) == 500

    def test_active_listing_count_extracted(self):
        text = "234 products available"
        items = self.analyzer.analyze(text, source_name="Daraz")
        listings = next((i for i in items if i.field_name == "active_listing_count"), None)
        assert listings is not None
        assert int(listings.value) == 234

    def test_price_activity_detected(self):
        text = "Rs. 2,500 | Rs. 1,800 | PKR 3,000"
        items = self.analyzer.analyze(text, source_name="Daraz")
        price_act = next((i for i in items if i.field_name == "marketplace_price_activity"), None)
        assert price_act is not None
        assert int(price_act.value) == 3

    def test_listing_freshness_today(self):
        text = "Posted today — new arrival"
        items = self.analyzer.analyze(text, source_name="OLX")
        fresh = next((i for i in items if i.field_name == "listing_freshness"), None)
        assert fresh is not None

    def test_listing_freshness_hours_ago(self):
        text = "Listed 3 hours ago"
        items = self.analyzer.analyze(text, source_name="OLX")
        fresh = next((i for i in items if i.field_name == "listing_freshness"), None)
        assert fresh is not None

    def test_daraz_gets_medium_reliability(self):
        text = "100 sold"
        items = self.analyzer.analyze(text, source_name="Daraz Store")
        for item in items:
            assert item.source_reliability == SourceReliability.MEDIUM

    def test_unknown_platform_gets_low_reliability(self):
        text = "100 sold"
        items = self.analyzer.analyze(text, source_name="some_marketplace")
        for item in items:
            assert item.source_reliability == SourceReliability.LOW

    def test_all_items_observed_type(self):
        text = "500 sold | 100 products | Rs. 3000 | posted today"
        items = self.analyzer.analyze(text, source_name="Daraz")
        for item in items:
            assert item.evidence_type == EvidenceType.OBSERVED


# ═══════════════════════════════════════════════════════════════════════════════
# UserInputParser
# ═══════════════════════════════════════════════════════════════════════════════

class TestUserInputParser:

    def setup_method(self):
        self.parser = UserInputParser()

    def _make_input(self, additional_info=None, description=None) -> BusinessInput:
        return BusinessInput(
            name="Test Business",
            additional_info=additional_info,
            description=description,
        )

    def test_empty_input_returns_empty(self):
        inp = BusinessInput(name="Test")
        items = self.parser.parse(inp)
        assert items == []

    def test_monthly_revenue_extracted(self):
        inp = self._make_input(additional_info="We earn Rs 50,000 per month from sales")
        items = self.parser.parse(inp)
        rev = next((i for i in items if i.field_name == "self_reported_monthly_revenue"), None)
        assert rev is not None
        assert int(rev.value) == 50000

    def test_revenue_k_suffix(self):
        inp = self._make_input(additional_info="PKR 30k monthly income")
        items = self.parser.parse(inp)
        rev = next((i for i in items if i.field_name == "self_reported_monthly_revenue"), None)
        assert rev is not None
        assert int(rev.value) == 30000

    def test_monthly_orders_extracted(self):
        inp = self._make_input(additional_info="We do 40 suits per month")
        items = self.parser.parse(inp)
        orders = next((i for i in items if i.field_name == "self_reported_monthly_orders"), None)
        assert orders is not None
        assert orders.value == "40"

    def test_years_direct_format(self):
        inp = self._make_input(additional_info="We have been operating for 5 years")
        items = self.parser.parse(inp)
        years = next((i for i in items if i.field_name == "self_reported_years_in_business"), None)
        assert years is not None
        assert years.value == "5"

    def test_years_since_format(self):
        inp = self._make_input(additional_info="established since 2018")
        items = self.parser.parse(inp)
        years = next((i for i in items if i.field_name == "self_reported_years_in_business"), None)
        assert years is not None
        # years = current_year - 2018; accept any positive integer
        assert int(years.value) > 0

    def test_staff_count_extracted(self):
        inp = self._make_input(additional_info="We have 5 employees in our team")
        items = self.parser.parse(inp)
        staff = next((i for i in items if i.field_name == "self_reported_staff_count"), None)
        assert staff is not None
        assert staff.value == "5"

    def test_tailors_counted_as_staff(self):
        inp = self._make_input(additional_info="3 tailors work full time")
        items = self.parser.parse(inp)
        staff = next((i for i in items if i.field_name == "self_reported_staff_count"), None)
        assert staff is not None

    def test_sales_channels_extracted(self):
        inp = self._make_input(additional_info="We sell via WhatsApp and Facebook")
        items = self.parser.parse(inp)
        channels = next((i for i in items if i.field_name == "self_reported_sales_channels"), None)
        assert channels is not None
        assert "whatsapp" in channels.value.lower()

    def test_multiple_channels_deduplicated(self):
        inp = self._make_input(additional_info="WhatsApp WhatsApp Instagram Daraz")
        items = self.parser.parse(inp)
        channels = next((i for i in items if i.field_name == "self_reported_sales_channels"), None)
        assert channels is not None
        # Values should be comma-separated unique entries
        vals = [v.strip() for v in channels.value.split(",")]
        assert len(vals) == len(set(vals))

    def test_product_types_extracted(self):
        inp = self._make_input(additional_info="We make embroidered kurta and bridal suits")
        items = self.parser.parse(inp)
        products = next((i for i in items if i.field_name == "self_reported_product_types"), None)
        assert products is not None
        assert "bridal" in products.value or "kurta" in products.value

    def test_all_items_are_inference(self):
        inp = self._make_input(
            additional_info="Rs 40k monthly, 30 suits per month, 5 years, via WhatsApp"
        )
        items = self.parser.parse(inp)
        for item in items:
            assert item.evidence_type == EvidenceType.INFERENCE

    def test_all_items_low_reliability(self):
        inp = self._make_input(additional_info="Rs 40k monthly, 30 suits per month")
        items = self.parser.parse(inp)
        for item in items:
            assert item.source_reliability == SourceReliability.LOW

    def test_confidence_below_threshold(self):
        inp = self._make_input(additional_info="Rs 40k monthly, 30 suits per month, 5 employees")
        items = self.parser.parse(inp)
        for item in items:
            assert item.confidence <= 0.50

    def test_description_field_also_parsed(self):
        inp = self._make_input(description="5 tailors, 20 suits per month via Instagram")
        items = self.parser.parse(inp)
        assert len(items) > 0

    def test_combined_additional_and_description(self):
        inp = self._make_input(
            additional_info="Rs 30k monthly",
            description="selling via WhatsApp",
        )
        items = self.parser.parse(inp)
        fields = [i.field_name for i in items]
        assert "self_reported_monthly_revenue" in fields
        assert "self_reported_sales_channels" in fields
