"""
Investigation configuration — all runtime limits and model settings.

Every value has a sensible default and can be overridden via environment
variables, so nothing is hardcoded and deployments can tune behaviour
without touching source code.

Usage::

    from agent.config import InvestigationConfig
    config = InvestigationConfig()          # all defaults
    config = InvestigationConfig(max_searches=20)  # override one value
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(key: str, default: int) -> int:
    """Read an integer from an env var, falling back to default."""
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "").lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


@dataclass
class InvestigationConfig:
    """
    All tuneable parameters for one investigation run.

    Attributes
    ----------
    max_searches:
        Maximum number of web searches the agent may perform.
        Prevents runaway investigations on businesses with huge online footprints.
    max_sources:
        Maximum number of individual web pages / sources to read.
    max_iterations:
        Maximum number of agent reasoning loops (think → act → observe cycles).
    max_evidence_items:
        Cap on how many EvidenceItems to collect.  Keeps memory bounded.
    min_evidence_to_conclude:
        Minimum reliable evidence items before the agent is allowed to finalise
        an assessment.  Below this threshold the result is INSUFFICIENT_EVIDENCE.
    request_timeout_s:
        Seconds to wait for a single HTTP request before giving up.
    llm_model:
        The LLM model identifier passed to the LLM client.
    llm_temperature:
        Sampling temperature for the LLM.  Lower = more deterministic.
    llm_max_tokens:
        Maximum tokens the LLM may produce in a single response.
    search_engine:
        Which search backend to use: ``"duckduckgo"`` (default, free) or
        ``"google"`` (requires API key).
    user_agent:
        HTTP User-Agent string sent when fetching web pages.
    respect_robots_txt:
        Whether to honour robots.txt directives.
    parse_self_reported:
        Whether to extract evidence from the user's own ``additional_info``
        and ``description`` fields (self-reported, low-confidence evidence).
    """

    # ── Investigation limits ───────────────────────────────────────────────
    max_searches: int = field(
        default_factory=lambda: _env_int("AGENT_MAX_SEARCHES", 15)
    )
    max_sources: int = field(
        default_factory=lambda: _env_int("AGENT_MAX_SOURCES", 20)
    )
    max_iterations: int = field(
        default_factory=lambda: _env_int("AGENT_MAX_ITERATIONS", 30)
    )
    max_evidence_items: int = field(
        default_factory=lambda: _env_int("AGENT_MAX_EVIDENCE", 50)
    )
    min_evidence_to_conclude: int = field(
        default_factory=lambda: _env_int("AGENT_MIN_EVIDENCE", 3)
    )

    # ── HTTP / network ────────────────────────────────────────────────────
    request_timeout_s: float = field(
        default_factory=lambda: _env_float("AGENT_REQUEST_TIMEOUT", 10.0)
    )
    user_agent: str = field(
        default_factory=lambda: _env_str(
            "AGENT_USER_AGENT",
            "OpenCreditAI/1.0 (business investigation agent)",
        )
    )
    respect_robots_txt: bool = field(
        default_factory=lambda: _env_bool("AGENT_RESPECT_ROBOTS", True)
    )

    # ── LLM ───────────────────────────────────────────────────────────────
    llm_model: str = field(
        default_factory=lambda: _env_str("AGENT_LLM_MODEL", "gpt-4o-mini")
    )
    llm_temperature: float = field(
        default_factory=lambda: _env_float("AGENT_LLM_TEMPERATURE", 0.2)
    )
    llm_max_tokens: int = field(
        default_factory=lambda: _env_int("AGENT_LLM_MAX_TOKENS", 2048)
    )

    # ── Search ────────────────────────────────────────────────────────────
    search_engine: str = field(
        default_factory=lambda: _env_str("AGENT_SEARCH_ENGINE", "duckduckgo")
    )

    # ── Self-reported evidence ────────────────────────────────────────────
    parse_self_reported: bool = field(
        default_factory=lambda: _env_bool("AGENT_PARSE_SELF_REPORTED", True)
    )

    def __post_init__(self) -> None:
        if self.max_searches < 1:
            raise ValueError("max_searches must be at least 1.")
        if self.max_sources < 1:
            raise ValueError("max_sources must be at least 1.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        if self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive.")
        if not (0.0 <= self.llm_temperature <= 2.0):
            raise ValueError("llm_temperature must be between 0.0 and 2.0.")
        if self.llm_max_tokens < 64:
            raise ValueError("llm_max_tokens must be at least 64.")
        if self.min_evidence_to_conclude < 1:
            raise ValueError("min_evidence_to_conclude must be at least 1.")

    def summary(self) -> str:
        return (
            f"InvestigationConfig("
            f"max_searches={self.max_searches}, "
            f"max_sources={self.max_sources}, "
            f"max_iterations={self.max_iterations}, "
            f"model={self.llm_model})"
        )
