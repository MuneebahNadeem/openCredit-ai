"""
Web Search Tool — performs keyword searches and returns result URLs + snippets.

In production this calls a real search API (DuckDuckGo by default, or Google
if configured).  The tool is designed to be mockable in tests — inject a
custom ``search_fn`` to avoid real HTTP calls.

Usage::

    from agent.tools.web_search import WebSearchTool
    from agent.config import InvestigationConfig

    tool = WebSearchTool(config=InvestigationConfig())
    results = tool.search("Karachi Textile Hub reviews Pakistan")
    for r in results:
        print(r.url, r.snippet)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class SearchResult:
    """One result returned by a web search."""
    url: str
    title: str
    snippet: str


# ── DuckDuckGo search (no API key required) ───────────────────────────────────

def _duckduckgo_search(query: str, max_results: int) -> List[SearchResult]:
    """
    Run a DuckDuckGo search using the public HTML endpoint.

    Falls back gracefully if the request fails — returns an empty list
    rather than raising so the agent can continue with what it has.
    """
    try:
        import urllib.parse
        import urllib.request

        encoded = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; OpenCreditAI/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        results = []
        # Extract result blocks — each result has a title link and a snippet
        blocks = re.findall(
            r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL,
        )
        for href, title_html, snippet_html in blocks[:max_results]:
            title = re.sub(r"<[^>]+>", "", title_html).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet_html).strip()
            # DuckDuckGo wraps URLs in a redirect; extract the actual URL
            m = re.search(r"uddg=([^&]+)", href)
            actual_url = urllib.parse.unquote(m.group(1)) if m else href
            if actual_url and title:
                results.append(SearchResult(url=actual_url, title=title, snippet=snippet))

        return results

    except Exception:
        return []


# ── Tool class ────────────────────────────────────────────────────────────────

class WebSearchTool:
    """
    Executes web searches and returns a list of SearchResults.

    Parameters
    ----------
    config:
        InvestigationConfig — uses ``search_engine`` and ``request_timeout_s``.
    max_results_per_search:
        How many results to return per query.
    search_fn:
        Optional override for the search function.  Inject a mock in tests.
    """

    def __init__(
        self,
        config=None,
        max_results_per_search: int = 8,
        search_fn: Optional[Callable[[str, int], List[SearchResult]]] = None,
    ) -> None:
        self._config = config
        self._max_results = max_results_per_search
        self._search_fn = search_fn or _duckduckgo_search

    def search(self, query: str) -> List[SearchResult]:
        """
        Run a search for the given query string.

        Returns up to ``max_results_per_search`` results.
        Returns an empty list on failure — never raises.
        """
        if not query or not query.strip():
            return []
        try:
            return self._search_fn(query.strip(), self._max_results)
        except Exception:
            return []

    def build_query(self, business_name: str, location: Optional[str], topic: str) -> str:
        """
        Construct a focused search query for a specific investigation topic.

        Examples
        --------
        >>> tool.build_query("Karachi Textile Hub", "Karachi", "reviews")
        'Karachi Textile Hub Karachi reviews'
        """
        parts = [business_name]
        if location:
            parts.append(location)
        parts.append(topic)
        return " ".join(parts)
