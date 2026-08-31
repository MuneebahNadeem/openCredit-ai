"""Thin integration boundary around Person 2's ML / risk assessment layer."""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Tuple

from agent.schemas.result import InvestigationResult


class MLAdapter:
    """Wraps Person 2's assessment entry points.

    Returns the fully assessed ``InvestigationResult`` (produced by
    ``ml.assessment.generate_assessment``) plus an analysis context dict
    with the credibility / sentiment details and recommendation that the
    wrapper computes but does not store on the result itself.

    ``assess_fn`` is injectable for tests.
    """

    def __init__(self, assess_fn: Optional[Callable] = None) -> None:
        self._assess_fn = assess_fn

    def run(
        self, result: InvestigationResult
    ) -> Tuple[InvestigationResult, dict]:
        if self._assess_fn is not None:
            return self._assess_fn(result)

        from ml.assessment import generate_assessment, generate_recommendation
        from ml.risk_engine import assess_risk

        risk = assess_risk(result)
        enriched = generate_assessment(result)

        context = {
            "recommendation": generate_recommendation(risk),
            "credibility": dataclasses.asdict(risk.credibility),
            "sentiment": dataclasses.asdict(risk.sentiment),
        }
        return enriched, context
