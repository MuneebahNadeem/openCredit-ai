"""Tests for API request validation (friendly error messages)."""

from __future__ import annotations

import pytest

from backend.app.schemas import InvestigationCreateRequest


class TestNameValidation:
    def test_blank_name_rejected_with_friendly_message(self):
        with pytest.raises(ValueError) as err:
            InvestigationCreateRequest(name="   ")
        assert "Business name is required" in str(err.value)

    def test_missing_name_rejected(self):
        with pytest.raises(ValueError):
            InvestigationCreateRequest(name=None)

    def test_name_stripped(self):
        req = InvestigationCreateRequest(name="  Karachi Threads  ")
        assert req.name == "Karachi Threads"


class TestUrlValidation:
    def test_website_gets_https_prefix(self):
        req = InvestigationCreateRequest(name="B", website="instagram.com/biz")
        assert req.website == "https://instagram.com/biz"

    def test_website_with_scheme_untouched(self):
        req = InvestigationCreateRequest(name="B", website="http://example.com")
        assert req.website == "http://example.com"

    def test_website_with_spaces_rejected(self):
        with pytest.raises(ValueError) as err:
            InvestigationCreateRequest(name="B", website="not a url")
        assert "valid website URL" in str(err.value)

    def test_website_without_dot_rejected(self):
        with pytest.raises(ValueError):
            InvestigationCreateRequest(name="B", website="localhost")

    def test_social_links_each_validated(self):
        with pytest.raises(ValueError) as err:
            InvestigationCreateRequest(name="B", social_links=["ok.com/a", "bad link"])
        assert "valid social media URL" in str(err.value)

    def test_marketplace_links_each_validated(self):
        with pytest.raises(ValueError):
            InvestigationCreateRequest(name="B", marketplace_links=[""])

    def test_empty_link_lists_allowed(self):
        req = InvestigationCreateRequest(name="B")
        assert req.social_links == []
        assert req.marketplace_links == []


class TestOptionalFields:
    def test_whitespace_optionals_become_none(self):
        req = InvestigationCreateRequest(
            name="B", location="   ", category="", description="  x  "
        )
        assert req.location is None
        assert req.category is None
        assert req.description == "x"

    def test_to_business_input_produces_person1_schema(self):
        from agent.schemas.input import BusinessInput

        req = InvestigationCreateRequest(
            name="Karachi Threads",
            location="Karachi",
            website="karachithreads.com",
            social_links=["instagram.com/karachithreads"],
            additional_info="10 years in business",
        )
        bi = req.to_business_input()
        assert isinstance(bi, BusinessInput)
        assert bi.name == "Karachi Threads"
        assert str(bi.website) == "https://karachithreads.com/"
        assert str(bi.social_links[0]) == "https://instagram.com/karachithreads"
        assert bi.additional_info == "10 years in business"
        assert bi.location == "Karachi"
