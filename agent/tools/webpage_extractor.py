"""
Webpage Extractor — fetches a URL and extracts clean text content.

Downloads a web page, strips HTML tags, and returns structured text
suitable for the LLM to read and extract evidence from.

In production this makes real HTTP requests.  Inject a custom ``fetch_fn``
in tests to avoid network calls.

Usage::

    from agent.tools.webpage_extractor import WebpageExtractor

    extractor = WebpageExtractor()
    page = extractor.fetch("https://example.com")
    if page:
        print(page.title)
        print(page.text[:500])
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class PageContent:
    """Cleaned content extracted from one web page."""
    url: str
    title: str
    text: str          # Plain text, HTML stripped, whitespace normalised
    word_count: int

    @property
    def is_empty(self) -> bool:
        return self.word_count == 0

    def truncated(self, max_chars: int = 4000) -> str:
        """Return text truncated to max_chars for LLM consumption."""
        return self.text[:max_chars]


# ── HTML cleaning helpers ──────────────────────────────────────────────────────

def _strip_html(html: str) -> str:
    """Remove HTML tags and decode common entities."""
    # Remove script and style blocks entirely
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html,
                     flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    # Decode common entities
    entities = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&nbsp;": " ",
    }
    for entity, char in entities.items():
        cleaned = cleaned.replace(entity, char)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_title(html: str) -> str:
    """Extract the <title> text from HTML."""
    m = re.search(r"<title[^>]*>(.*?)</title>", html,
                  re.IGNORECASE | re.DOTALL)
    if m:
        return _strip_html(m.group(1))[:200]
    return ""


# ── Default fetch function ────────────────────────────────────────────────────

def _default_fetch(url: str, timeout: float, user_agent: str) -> Optional[str]:
    """Fetch a URL and return raw HTML, or None on failure."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text" not in content_type and "html" not in content_type:
                return None
            raw = resp.read(500_000)  # cap at 500 KB
            return raw.decode("utf-8", errors="ignore")
    except Exception:
        return None


# ── Tool class ────────────────────────────────────────────────────────────────

class WebpageExtractor:
    """
    Fetches and cleans web page content for evidence extraction.

    Parameters
    ----------
    config:
        InvestigationConfig — uses ``request_timeout_s`` and ``user_agent``.
    fetch_fn:
        Optional override for the HTTP fetch function.  Inject a mock in tests.
    """

    def __init__(self, config=None, fetch_fn=None) -> None:
        self._timeout = config.request_timeout_s if config else 10.0
        self._user_agent = (
            config.user_agent if config
            else "OpenCreditAI/1.0 (business investigation agent)"
        )
        self._fetch_fn = fetch_fn or (
            lambda url: _default_fetch(url, self._timeout, self._user_agent)
        )

    def fetch(self, url: str) -> Optional[PageContent]:
        """
        Fetch a URL and return a ``PageContent`` object, or None on failure.

        Never raises — returns None if the page cannot be fetched or parsed.
        """
        if not url or not url.strip():
            return None
        try:
            html = self._fetch_fn(url.strip())
            if not html:
                return None
            title = _extract_title(html)
            text = _strip_html(html)
            word_count = len(text.split())
            return PageContent(
                url=url.strip(),
                title=title,
                text=text,
                word_count=word_count,
            )
        except Exception:
            return None

    def fetch_multiple(self, urls: list) -> list:
        """Fetch multiple URLs, skipping failures. Returns list of PageContent."""
        results = []
        for url in urls:
            page = self.fetch(url)
            if page and not page.is_empty:
                results.append(page)
        return results
