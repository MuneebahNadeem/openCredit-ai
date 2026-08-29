"""
Tests for agent/schemas/input.py — BusinessInput model.

Run with:  pytest tests/agent/test_input.py -v
"""

import sys
import os

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from pydantic import ValidationError
from agent.schemas.input import BusinessInput


# ── Helpers ───────────────────────────────────────────────────────────────────

MINIMAL = {"name": "Sunshine Bakery"}

FULL = {
    "name": "Sunshine Bakery",
    "location": "Lagos, Nigeria",
    "category": "Food & Beverage",
    "website": "https://sunshinebakery.com",
    "social_links": ["https://instagram.com/sunshinebakery"],
    "marketplace_links": ["https://jumia.com.ng/sunshine-bakery"],
    "description": "A family-owned artisan bakery established in 2015.",
    "additional_info": "Monthly revenue approx ₦2M. 8 employees.",
}


# ── Valid input ───────────────────────────────────────────────────────────────

def test_minimal_valid_input():
    b = BusinessInput(**MINIMAL)
    assert b.name == "Sunshine Bakery"
    assert b.location is None
    assert b.website is None
    assert b.social_links == []
    assert b.marketplace_links == []


def test_full_valid_input():
    b = BusinessInput(**FULL)
    assert b.name == "Sunshine Bakery"
    assert b.location == "Lagos, Nigeria"
    assert b.category == "Food & Beverage"
    assert str(b.website) == "https://sunshinebakery.com/"
    assert len(b.social_links) == 1
    assert len(b.marketplace_links) == 1


def test_name_is_stripped():
    b = BusinessInput(name="  Sunshine Bakery  ")
    assert b.name == "Sunshine Bakery"


def test_optional_strings_stripped_to_none():
    b = BusinessInput(name="Test Biz", location="   ", category="   ")
    assert b.location is None
    assert b.category is None


# ── Missing / optional fields ─────────────────────────────────────────────────

def test_all_optional_fields_absent():
    b = BusinessInput(name="Mystery Shop")
    assert b.category is None
    assert b.description is None
    assert b.additional_info is None
    assert b.has_online_presence() is False


def test_has_online_presence_with_website():
    b = BusinessInput(name="Biz A", website="https://example.com")
    assert b.has_online_presence() is True


def test_has_online_presence_with_social_only():
    b = BusinessInput(name="Biz B", social_links=["https://twitter.com/bizb"])
    assert b.has_online_presence() is True


def test_has_online_presence_with_marketplace_only():
    b = BusinessInput(name="Biz C", marketplace_links=["https://amazon.com/shop/bizc"])
    assert b.has_online_presence() is True


# ── Invalid input ─────────────────────────────────────────────────────────────

def test_missing_name_raises():
    with pytest.raises(ValidationError):
        BusinessInput()


def test_blank_name_raises():
    with pytest.raises(ValidationError):
        BusinessInput(name="   ")


def test_invalid_website_url_raises():
    with pytest.raises(ValidationError):
        BusinessInput(name="Bad URL Biz", website="not-a-url")


def test_invalid_social_link_raises():
    with pytest.raises(ValidationError):
        BusinessInput(name="Bad Social", social_links=["not-a-url"])


# ── summary() helper ─────────────────────────────────────────────────────────

def test_summary_name_only():
    b = BusinessInput(name="Solo Biz")
    assert b.summary() == "Solo Biz"


def test_summary_with_location_and_category():
    b = BusinessInput(name="Sunshine Bakery", location="Lagos", category="Food")
    assert b.summary() == "Sunshine Bakery | Lagos | (Food)"


def test_summary_with_location_no_category():
    b = BusinessInput(name="Sunshine Bakery", location="Lagos")
    assert b.summary() == "Sunshine Bakery | Lagos"
