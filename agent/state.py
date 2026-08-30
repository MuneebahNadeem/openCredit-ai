"""
InvestigationState — mutable state for one investigation run.

Tracks everything the agent has done so far: which URLs were searched,
which sources were read, what evidence was collected, what features were
found, and whether the stop conditions have been met.

The state is created fresh at the start of each investigation and mutated
by the agent and its tools as the investigation progresses.  It is never
shared between investigations.

Usage::

    from agent.config import InvestigationConfig
    from agent.state import InvestigationState

    config = InvestigationConfig()
    state = InvestigationState(config=config)

    state.record_search("karachi textile hub Pakistan")
    state.add_evidence(some_evidence_item)
    state.add_source("https://example.com")

    if state.should_stop():
        result = state.build_result(business_input)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from agent.config import InvestigationConfig
from agent.schemas.evidence import EvidenceItem
from agent.schemas.feature import DiscoveredFeature
from agent.schemas.input import BusinessInput
from agent.schemas.result import (
    InvestigationResult,
    InvestigationStatus,
    Signal,
)


@dataclass
class InvestigationState:
    """
    Mutable state for one investigation run.

    Attributes
    ----------
    config:
        The investigation configuration (limits, model, timeouts).
    searches_performed:
        Counter of web searches executed so far.
    sources_read:
        Set of URLs already fetched — prevents re-reading the same page.
    search_queries_issued:
        Set of query strings already searched — prevents duplicate searches.
    evidence:
        All EvidenceItems collected so far.
    features:
        All DiscoveredFeatures identified so far.
    positive_signals:
        Positive signals surfaced during investigation.
    risk_signals:
        Risk signals surfaced during investigation.
    missing_information:
        Things the agent looked for but could not find.
    stop_reason:
        Set when the agent decides to stop; explains why.
    iteration:
        Current reasoning loop counter.
    """

    config: InvestigationConfig

    # ── Counters and deduplication sets ──────────────────────────────────
    searches_performed: int = 0
    sources_read: Set[str] = field(default_factory=set)
    search_queries_issued: Set[str] = field(default_factory=set)

    # ── Collected data ────────────────────────────────────────────────────
    evidence: List[EvidenceItem] = field(default_factory=list)
    features: List[DiscoveredFeature] = field(default_factory=list)
    positive_signals: List[Signal] = field(default_factory=list)
    risk_signals: List[Signal] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)

    # ── Control ───────────────────────────────────────────────────────────
    iteration: int = 0
    stop_reason: Optional[str] = None

    # ── Search recording ──────────────────────────────────────────────────

    def record_search(self, query: str) -> bool:
        """
        Record a search query.  Returns True if it is new, False if duplicate.

        The agent should only proceed with a search if this returns True.
        """
        normalised = query.strip().lower()
        if normalised in self.search_queries_issued:
            return False
        self.search_queries_issued.add(normalised)
        self.searches_performed += 1
        return True

    def has_searched(self, query: str) -> bool:
        """True if this query (normalised) has already been issued."""
        return query.strip().lower() in self.search_queries_issued

    # ── Source recording ──────────────────────────────────────────────────

    def add_source(self, url: str) -> bool:
        """
        Record a source URL.  Returns True if new, False if already visited.
        """
        normalised = url.strip().rstrip("/")
        if normalised in self.sources_read:
            return False
        self.sources_read.add(normalised)
        return True

    def has_visited(self, url: str) -> bool:
        """True if this URL has already been read."""
        return url.strip().rstrip("/") in self.sources_read

    # ── Evidence accumulation ─────────────────────────────────────────────

    def add_evidence(self, item: EvidenceItem) -> None:
        """Add one EvidenceItem to the collected set."""
        self.evidence.append(item)

    def add_evidence_batch(self, items: List[EvidenceItem]) -> None:
        """Add multiple EvidenceItems at once."""
        self.evidence.extend(items)

    @property
    def evidence_count(self) -> int:
        return len(self.evidence)

    @property
    def reliable_evidence_count(self) -> int:
        return sum(1 for e in self.evidence if e.is_reliable())

    # ── Feature accumulation ──────────────────────────────────────────────

    def add_feature(self, feature: DiscoveredFeature) -> None:
        """Add or replace a DiscoveredFeature (replaces by name)."""
        existing_names = {f.name for f in self.features}
        if feature.name in existing_names:
            self.features = [
                feature if f.name == feature.name else f
                for f in self.features
            ]
        else:
            self.features.append(feature)

    def get_feature(self, name: str) -> Optional[DiscoveredFeature]:
        """Return the feature with the given name, or None if not found."""
        for f in self.features:
            if f.name == name:
                return f
        return None

    # ── Signal accumulation ───────────────────────────────────────────────

    def add_positive_signal(self, signal: Signal) -> None:
        self.positive_signals.append(signal)

    def add_risk_signal(self, signal: Signal) -> None:
        self.risk_signals.append(signal)

    def add_missing_info(self, item: str) -> None:
        if item not in self.missing_information:
            self.missing_information.append(item)

    # ── Stop condition checks ─────────────────────────────────────────────

    def should_stop(self) -> bool:
        """
        Return True if the investigation should stop.

        Checks all configured limits and sets ``stop_reason`` if any are hit.
        Also returns True if ``stop_reason`` was already set explicitly.
        """
        if self.stop_reason is not None:
            return True

        if self.searches_performed >= self.config.max_searches:
            self.stop_reason = "max_searches_reached"
            return True

        if len(self.sources_read) >= self.config.max_sources:
            self.stop_reason = "max_sources_reached"
            return True

        if self.iteration >= self.config.max_iterations:
            self.stop_reason = "max_iterations_reached"
            return True

        if len(self.evidence) >= self.config.max_evidence_items:
            self.stop_reason = "max_evidence_reached"
            return True

        return False

    def stop(self, reason: str = "agent_decided") -> None:
        """Explicitly signal the agent to stop."""
        self.stop_reason = reason

    def has_sufficient_evidence(self) -> bool:
        """
        True when reliable evidence meets the configured minimum threshold.
        Used by the agent to decide if it can produce a confident assessment.
        """
        return self.reliable_evidence_count >= self.config.min_evidence_to_conclude

    def increment_iteration(self) -> None:
        """Advance the iteration counter by one."""
        self.iteration += 1

    # ── Final status derivation ───────────────────────────────────────────

    def _derive_status(self) -> InvestigationStatus:
        """Map the stop reason to an InvestigationStatus enum value."""
        if self.stop_reason is None:
            return InvestigationStatus.COMPLETE
        if self.stop_reason in (
            "max_searches_reached",
            "max_sources_reached",
            "max_iterations_reached",
            "max_evidence_reached",
        ):
            return InvestigationStatus.LIMIT_REACHED
        if self.stop_reason == "partial_failure":
            return InvestigationStatus.PARTIAL
        if self.stop_reason == "failed":
            return InvestigationStatus.FAILED
        # agent_decided or any custom reason → complete
        return InvestigationStatus.COMPLETE

    # ── Build final result ────────────────────────────────────────────────

    def build_result(self, business_input: BusinessInput) -> InvestigationResult:
        """
        Convert the current state into an ``InvestigationResult``.

        This is called at the end of the investigation loop.  The result
        contains all evidence and features collected so far plus investigation
        metadata.  Assessments are left at INSUFFICIENT_EVIDENCE — the ML
        layer (``ml.assessment.generate_assessment()``) populates them.
        """
        return InvestigationResult(
            business_input=business_input,
            status=self._derive_status(),
            searches_performed=self.searches_performed,
            sources_examined=len(self.sources_read),
            evidence=list(self.evidence),
            features=list(self.features),
            positive_signals=list(self.positive_signals),
            risk_signals=list(self.risk_signals),
            missing_information=list(self.missing_information),
            sources=list(self.sources_read),
            justification="",
        )

    # ── Debug helper ──────────────────────────────────────────────────────

    def summary(self) -> str:
        return (
            f"InvestigationState("
            f"iter={self.iteration}, "
            f"searches={self.searches_performed}/{self.config.max_searches}, "
            f"sources={len(self.sources_read)}/{self.config.max_sources}, "
            f"evidence={self.evidence_count}, "
            f"reliable={self.reliable_evidence_count}, "
            f"stop={self.stop_reason or 'running'})"
        )
