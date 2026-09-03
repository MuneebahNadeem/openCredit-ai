"""
Shared fixtures for the ML test suite.

``ml.risk_engine.assess_risk()`` defaults to the process-wide model
predictor, which loads whatever is saved in ``data/models/``.  Left
unchecked that would make every existing risk-engine / assessment test
environment-dependent (blended scores on machines with trained models,
pure rule scores on machines without).

This conftest patches the default predictor to a stub whose models are
unavailable, so every test that calls ``assess_risk()`` without an
explicit predictor exercises the deterministic pure-rule path — exactly
the behaviour those tests were written against.

Hybrid-blend behaviour is covered explicitly with injected predictors in
``test_risk_engine.py::TestHybridBlend`` and end-to-end in
``test_model_predictor.py::TestEndToEnd``.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from ml.model_predictor import ModelPrediction


class NullPredictor:
    """Stand-in for ``ModelPredictor`` whose models never load."""

    def predict(self, features: dict) -> ModelPrediction:
        return ModelPrediction(
            trust_score=None,
            potential_score=None,
            trust_model=None,
            potential_model=None,
            available=False,
            reason="null predictor (test stub)",
        )


@pytest.fixture(autouse=True)
def _no_default_predictor(monkeypatch):
    """Keep default ``assess_risk()`` calls deterministic (pure rules)."""
    monkeypatch.setattr("ml.risk_engine.get_predictor", lambda: NullPredictor())
