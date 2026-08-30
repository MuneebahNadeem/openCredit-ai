"""
Investigation Agent — the main orchestrator that drives the investigation loop.

This is the top-level entry point for Person 1's work.  It:

1. Parses self-reported information from BusinessInput immediately.
2. Builds initial search queries from the business's known details.
3. Runs a search → fetch → extract → discover loop until stop conditions are met.
4. Collects EvidenceItems and DiscoveredFeatures into InvestigationState.
5. Returns a fully populated InvestigationResult ready for ml.assessment.

The agent does NOT score or assess the business — that is Person 2's job.
It only investigates, extracts, and structures evidence.

Usage::

    from agent.agent import InvestigationAgent
    from agent.schemas.input import BusinessInput

    agent = InvestigationAgent()
    result = agent.investigate(BusinessInput(name="Karachi Textile Hub", location="Karachi"))

    # Pass to ML layer
    from ml.assessment import generate_assessment
    enriched = generate_assessment(result)
    print(enriched.trustworthiness.level)
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from agent.config import InvestigationConfig
from agent.prompts.templates import (
    extraction_prompt,
    feature_discovery_prompt,
    investigation_plan_prompt,
)
from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability
from agent.schemas.feature import DiscoveredFeature, FeatureCategory
from agent.schemas.input import BusinessInput
from agent.schemas.result import InvestigationResult, Signal
from agent.state import InvestigationState
from agent.tools.product_analyzer import ProductAnalyzer
from agent.tools.review_analyzer import ReviewAnalyzer
from agent.tools.social_analyzer import SocialAnalyzer
from agent.tools.user_input_parser import UserInputParser
from agent.tools.web_search import SearchResult, WebSearchTool
from agent.tools.webpage_extractor import PageContent, WebpageExtractor


# ── Source type detection ──────────────────────────────────────────────────────

def _classify_url(url: str) -> str:
    """Return a source type label for a URL."""
    url_lower = url.lower()
    if any(x in url_lower for x in ("instagram", "facebook", "twitter",
                                     "tiktok", "youtube", "linkedin")):
        return "social_media"
    if any(x in url_lower for x in ("daraz", "olx", "amazon", "shopify",
                                     "alibaba")):
        return "marketplace"
    if any(x in url_lower for x in ("review", "rating", "feedback",
                                     "trustpilot", "yelp", "google.com/maps")):
        return "review_site"
    if any(x in url_lower for x in ("secp.gov", "fbr.gov", "nadra",
                                     "pakistan.gov", ".gov.pk")):
        return "government"
    return "general_web"


# ── JSON parsing helpers ───────────────────────────────────────────────────────

def _extract_json_array(text: str) -> list:
    """
    Extract the first JSON array from an LLM response string.
    Returns an empty list if nothing valid is found.
    """
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


# ── LLM client wrapper ────────────────────────────────────────────────────────

class _LLMClient:
    """
    Thin wrapper around the LLM API.

    In production this calls the OpenAI / Anthropic API.
    The ``call_fn`` parameter can be replaced in tests with a mock.
    """

    def __init__(self, config: InvestigationConfig, call_fn=None) -> None:
        self._config = config
        self._call_fn = call_fn  # None means use real API

    def complete(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and return the response text.

        Falls back to an empty string on any error so the agent can
        continue with partial information rather than crashing.
        """
        if self._call_fn is not None:
            # Test / mock path
            try:
                return self._call_fn(prompt)
            except Exception:
                return ""

        # Production path — requires OPENAI_API_KEY in environment
        try:
            import openai  # type: ignore
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self._config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._config.llm_temperature,
                max_tokens=self._config.llm_max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception:
            return ""


# ── Main agent class ───────────────────────────────────────────────────────────

class InvestigationAgent:
    """
    Orchestrates the full business investigation loop.

    Parameters
    ----------
    config:
        Investigation configuration.  Defaults to ``InvestigationConfig()``
        which reads from environment variables.
    llm_call_fn:
        Optional mock for the LLM call.  Inject in tests to avoid API calls.
    search_fn:
        Optional mock for the web search function.
    fetch_fn:
        Optional mock for the page fetch function.
    """

    def __init__(
        self,
        config: Optional[InvestigationConfig] = None,
        llm_call_fn=None,
        search_fn=None,
        fetch_fn=None,
    ) -> None:
        self._config = config or InvestigationConfig()
        self._llm = _LLMClient(self._config, call_fn=llm_call_fn)
        self._search = WebSearchTool(config=self._config, search_fn=search_fn)
        self._extractor = WebpageExtractor(config=self._config, fetch_fn=fetch_fn)
        self._social = SocialAnalyzer()
        self._review = ReviewAnalyzer()
        self._product = ProductAnalyzer()
        self._input_parser = UserInputParser()

    # ── Public entry point ────────────────────────────────────────────────

    def investigate(self, business_input: BusinessInput) -> InvestigationResult:
        """
        Run a full investigation for the given business.

        Steps:
        1. Parse self-reported information from additional_info / description.
        2. Search known URLs provided by the user.
        3. Run LLM-planned search → fetch → extract loop.
        4. Discover features from accumulated evidence.
        5. Return a populated InvestigationResult.

        The result has no assessments filled — call
        ``ml.assessment.generate_assessment(result)`` next.
        """
        state = InvestigationState(config=self._config)

        # Step 1: self-reported evidence (always run first)
        self_evidence = self._input_parser.parse(business_input)
        state.add_evidence_batch(self_evidence)

        # Step 2: process any URLs the user explicitly provided
        self._process_known_urls(business_input, state)

        # Step 3: main search-fetch-extract loop
        self._run_investigation_loop(business_input, state)

        # Step 4: feature discovery from all accumulated evidence
        self._discover_features(business_input, state)

        # Step 5: build signals from evidence
        self._build_signals(state)

        return state.build_result(business_input)

    # ── Step 2: process user-provided URLs ────────────────────────────────

    def _process_known_urls(
        self, business_input: BusinessInput, state: InvestigationState
    ) -> None:
        """Fetch and extract evidence from URLs the user explicitly provided."""
        urls = []
        if business_input.website:
            urls.append(str(business_input.website))
        for link in business_input.social_links:
            urls.append(str(link))
        for link in business_input.marketplace_links:
            urls.append(str(link))

        for url in urls:
            if state.should_stop():
                break
            if not state.add_source(url):
                continue
            page = self._extractor.fetch(url)
            if page:
                self._extract_from_page(page, business_input.name, state)

    # ── Step 3: main investigation loop ───────────────────────────────────

    def _run_investigation_loop(
        self, business_input: BusinessInput, state: InvestigationState
    ) -> None:
        """
        Iteratively ask the LLM for search queries, execute them,
        fetch result pages, and extract evidence until stop conditions are met.
        """
        while not state.should_stop():
            state.increment_iteration()

            # Ask LLM for next queries
            queries = self._plan_searches(business_input, state)
            if not queries:
                state.stop("no_more_queries")
                break

            any_new = False
            for query in queries:
                if state.should_stop():
                    break
                if not state.record_search(query):
                    continue  # already searched this
                any_new = True

                results = self._search.search(query)
                for result in results:
                    if state.should_stop():
                        break
                    if not state.add_source(result.url):
                        continue  # already read this URL

                    # Add snippet as quick low-confidence evidence
                    if result.snippet:
                        state.add_evidence(EvidenceItem(
                            field_name="search_snippet",
                            value=result.snippet[:200],
                            evidence_type=EvidenceType.INFERENCE,
                            source_url=result.url,
                            source_name=result.title[:80] if result.title else "web",
                            source_reliability=SourceReliability.UNKNOWN,
                            confidence=0.30,
                            raw_snippet=result.snippet[:200],
                        ))

                    # Fetch and deeply extract the page
                    page = self._extractor.fetch(result.url)
                    if page and not page.is_empty:
                        self._extract_from_page(page, business_input.name, state)

            if not any_new:
                state.stop("no_new_searches")
                break

            # Stop early if we have enough reliable evidence
            if state.has_sufficient_evidence() and state.iteration >= 3:
                state.stop("sufficient_evidence")
                break

    # ── LLM: plan next searches ───────────────────────────────────────────

    def _plan_searches(
        self, business_input: BusinessInput, state: InvestigationState
    ) -> List[str]:
        """Ask the LLM what to search for next. Returns up to 3 query strings."""
        evidence_summary = self._summarise_evidence(state)
        prompt = investigation_plan_prompt(
            business_name=business_input.name,
            location=business_input.location,
            category=business_input.category,
            website=str(business_input.website) if business_input.website else None,
            social_links=[str(l) for l in business_input.social_links],
            marketplace_links=[str(l) for l in business_input.marketplace_links],
            description=business_input.description,
            additional_info=business_input.additional_info,
            already_searched=list(state.search_queries_issued),
            evidence_summary=evidence_summary,
        )
        response = self._llm.complete(prompt)
        queries = _extract_json_array(response)
        # Filter to strings only, deduplicate against already-searched
        return [
            q for q in queries
            if isinstance(q, str) and q.strip()
            and not state.has_searched(q)
        ][:3]

    # ── Extraction from a page ────────────────────────────────────────────

    def _extract_from_page(
        self,
        page: PageContent,
        business_name: str,
        state: InvestigationState,
    ) -> None:
        """
        Extract evidence from a fetched page using both regex tools and the LLM.
        """
        url = page.url
        source_type = _classify_url(url)
        platform = SocialAnalyzer.detect_platform(url)

        # ── Regex-based extraction by source type ─────────────────────────
        if source_type == "social_media":
            items = self._social.analyze(page.text, url=url, platform=platform)
            state.add_evidence_batch(items)

        elif source_type == "review_site":
            source_name = re.sub(r"https?://(?:www\.)?", "", url).split("/")[0]
            items = self._review.analyze(page.text, url=url, source_name=source_name)
            state.add_evidence_batch(items)

        elif source_type == "marketplace":
            source_name = re.sub(r"https?://(?:www\.)?", "", url).split("/")[0]
            items = self._product.analyze(page.text, url=url, source_name=source_name)
            state.add_evidence_batch(items)
            # Also check for reviews on marketplace pages
            review_items = self._review.analyze(page.text, url=url, source_name=source_name)
            state.add_evidence_batch(review_items)

        # ── LLM-based deep extraction ──────────────────────────────────────
        prompt = extraction_prompt(
            business_name=business_name,
            url=url,
            source_type=source_type,
            page_text=page.truncated(3000),
        )
        response = self._llm.complete(prompt)
        llm_items = self._parse_evidence_from_llm(response, url, source_type)
        state.add_evidence_batch(llm_items)

    # ── LLM response → EvidenceItems ──────────────────────────────────────

    def _parse_evidence_from_llm(
        self, response: str, url: str, source_type: str
    ) -> List[EvidenceItem]:
        """Parse the LLM extraction response into EvidenceItems."""
        raw_list = _extract_json_array(response)
        items: List[EvidenceItem] = []

        reliability_map = {
            "government": SourceReliability.HIGH,
            "review_site": SourceReliability.MEDIUM,
            "marketplace": SourceReliability.MEDIUM,
            "social_media": SourceReliability.MEDIUM,
            "general_web": SourceReliability.LOW,
        }
        reliability = reliability_map.get(source_type, SourceReliability.UNKNOWN)

        for raw in raw_list:
            if not isinstance(raw, dict):
                continue
            try:
                field_name = str(raw.get("field_name", "")).strip()
                value = str(raw.get("value", "")).strip()
                if not field_name or not value:
                    continue

                etype_raw = str(raw.get("evidence_type", "observed")).lower()
                etype = {
                    "observed": EvidenceType.OBSERVED,
                    "corroborated": EvidenceType.CORROBORATED,
                    "inference": EvidenceType.INFERENCE,
                }.get(etype_raw, EvidenceType.OBSERVED)

                confidence = float(raw.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))

                items.append(EvidenceItem(
                    field_name=field_name,
                    value=value,
                    unit=raw.get("unit") or None,
                    evidence_type=etype,
                    source_url=url or None,
                    source_name=re.sub(r"https?://(?:www\.)?", "", url).split("/")[0],
                    source_reliability=reliability,
                    confidence=confidence,
                    raw_snippet=str(raw.get("raw_snippet", ""))[:200] or None,
                ))
            except Exception:
                continue

        return items

    # ── Step 4: feature discovery ─────────────────────────────────────────

    def _discover_features(
        self, business_input: BusinessInput, state: InvestigationState
    ) -> None:
        """Ask the LLM to identify business signals from all collected evidence."""
        if not state.evidence:
            return

        evidence_list = "\n".join(
            f"- {e.field_name}: {e.value} [{e.evidence_type.value}, "
            f"confidence={e.confidence:.2f}, source={e.source_name or 'unknown'}]"
            for e in state.evidence[:30]  # cap to avoid exceeding context
        )

        prompt = feature_discovery_prompt(
            business_name=business_input.name,
            location=business_input.location,
            category=business_input.category,
            evidence_list=evidence_list,
        )
        response = self._llm.complete(prompt)
        features = self._parse_features_from_llm(response, state)
        for feature in features:
            state.add_feature(feature)

    def _parse_features_from_llm(
        self, response: str, state: InvestigationState
    ) -> List[DiscoveredFeature]:
        """Parse the LLM feature discovery response into DiscoveredFeature objects."""
        raw_list = _extract_json_array(response)
        features: List[DiscoveredFeature] = []

        for raw in raw_list:
            if not isinstance(raw, dict):
                continue
            try:
                name = str(raw.get("name", "")).strip()
                reason = str(raw.get("reason", "")).strip()
                if not name or not reason:
                    continue

                cat_raw = str(raw.get("category", "unknown")).lower()
                try:
                    category = FeatureCategory(cat_raw)
                except ValueError:
                    category = FeatureCategory.UNKNOWN

                confidence = float(raw.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))

                # Link evidence items whose field_name appears in the feature name
                backing = [
                    e for e in state.evidence
                    if any(part in e.field_name for part in name.split("_"))
                ][:3]

                features.append(DiscoveredFeature(
                    name=name,
                    category=category,
                    value=str(raw["value"]) if raw.get("value") is not None else None,
                    reason=reason,
                    evidence=backing,
                    confidence=confidence,
                    searched=bool(raw.get("searched", True)),
                ))
            except Exception:
                continue

        return features

    # ── Step 5: build signals ─────────────────────────────────────────────

    def _build_signals(self, state: InvestigationState) -> None:
        """
        Derive positive and risk signals from collected evidence.

        This is a rule-based pass — the LLM step produced features,
        and now we translate high/low values into Signal objects.
        """
        for e in state.evidence:
            fn = e.field_name.lower()

            # Positive signals
            if fn in ("star_rating",):
                try:
                    rating = float(e.value)
                    if rating >= 4.0:
                        state.add_positive_signal(Signal(
                            label=f"High rating on {e.source_name or 'review site'}",
                            detail=f"Rated {e.value} stars — positive customer experience.",
                            evidence_refs=[e.field_name],
                        ))
                except ValueError:
                    pass

            if "follower" in fn or "subscriber" in fn:
                try:
                    count = int(e.value)
                    if count >= 1000:
                        state.add_positive_signal(Signal(
                            label=f"Social media following on {e.source_name or 'social'}",
                            detail=f"{count:,} {fn.replace('_', ' ')} indicates market reach.",
                            evidence_refs=[e.field_name],
                        ))
                except ValueError:
                    pass

            if fn == "informal_demand_signal":
                state.add_positive_signal(Signal(
                    label="Active order intake",
                    detail="Business is actively accepting orders through social channels.",
                    evidence_refs=[e.field_name],
                ))

            if fn == "units_sold":
                try:
                    count = int(e.value)
                    if count >= 10:
                        state.add_positive_signal(Signal(
                            label=f"Marketplace sales on {e.source_name or 'marketplace'}",
                            detail=f"{count:,} units sold — demonstrated demand.",
                            evidence_refs=[e.field_name],
                        ))
                except ValueError:
                    pass

            # Risk signals
            if fn == "complaint_signals":
                state.add_risk_signal(Signal(
                    label="Complaints detected",
                    detail=f"Complaint-related terms found: {e.value}",
                    evidence_refs=[e.field_name],
                ))

            if fn in ("star_rating",):
                try:
                    rating = float(e.value)
                    if rating < 2.5:
                        state.add_risk_signal(Signal(
                            label=f"Low rating on {e.source_name or 'review site'}",
                            detail=f"Rated only {e.value} stars — potential quality issues.",
                            evidence_refs=[e.field_name],
                        ))
                except ValueError:
                    pass

        # Missing info flags
        has_field = {e.field_name for e in state.evidence}
        if "registration_number" not in has_field and "secp" not in " ".join(has_field):
            state.add_missing_info("SECP/business registration not found")
        if not any("review" in f or "rating" in f for f in has_field):
            state.add_missing_info("No customer reviews found")
        if not any("follower" in f or "social" in f for f in has_field):
            state.add_missing_info("Social media presence not verified")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _summarise_evidence(self, state: InvestigationState) -> str:
        """Produce a brief text summary of collected evidence for the LLM."""
        if not state.evidence:
            return "No evidence collected yet."
        lines = []
        for e in state.evidence[-10:]:  # most recent 10
            lines.append(
                f"- {e.field_name}: {e.value[:60]} "
                f"[{e.evidence_type.value}, confidence={e.confidence:.2f}]"
            )
        return "\n".join(lines)
