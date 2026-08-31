"""Thin integration boundary around Person 1's investigation agent."""

from __future__ import annotations

import os
from typing import Callable, Optional

from agent.schemas.input import BusinessInput
from agent.schemas.result import InvestigationResult


class AgentAdapter:
    """Wraps Person 1's ``InvestigationAgent.investigate`` entry point.

    ``investigate_fn`` is injectable so tests (and any future transport such
    as a remote agent service) can replace the direct in-process call.
    """

    def __init__(self, investigate_fn: Optional[Callable] = None) -> None:
        self._investigate_fn = investigate_fn

    def run(self, business_input: BusinessInput) -> InvestigationResult:
        if self._investigate_fn is not None:
            return self._investigate_fn(business_input)
        # Imported lazily so the API layer can boot without the agent's
        # optional LLM dependency being installed.
        from agent.agent import InvestigationAgent

        return InvestigationAgent().investigate(business_input)


def llm_configured() -> bool:
    """True when an LLM API key is present for agent reasoning / Ask OpenCredit."""
    return bool(os.environ.get("OPENAI_API_KEY"))
