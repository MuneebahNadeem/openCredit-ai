"""
Model Predictor — production inference over the saved trained models.

Loads the winning saved artifacts from ``data/models/`` (by default
``random_forest``, the evaluation winner for both targets — see
``documents/MODEL_EVALUATION.md``) and turns the 60 numeric investigation
features into calibrated probability scores for trustworthiness and
business potential.

The predictor is deliberately defensive: if the model files are missing,
corrupt, or unloadable for any reason, ``predict()`` returns an
unavailable ``ModelPrediction`` instead of raising.  The rule-based risk
engine remains the safety net — the hybrid blend in ``ml/risk_engine.py``
falls back to pure rule scores when no model prediction is available.

Usage::

    from ml.model_predictor import get_predictor

    predictor = get_predictor()               # process-wide singleton
    prediction = predictor.predict(features)  # dict[str, float]

    if prediction.available:
        print(prediction.trust_score)      # e.g. 0.87
        print(prediction.potential_score)  # e.g. 0.64

The models were trained on synthetic data (see ``ml/dataset.py``); scores
are one input to the hybrid assessment, never the sole basis of a
conclusion.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from agent.schemas.input import BusinessInput
from agent.schemas.result import InvestigationResult, InvestigationStatus
from ml.feature_extractor import extract_features

# Default production models — evaluation winners for both targets
# (documents/MODEL_EVALUATION.md, 2026-09-03).
_DEFAULT_TRUST_MODEL = "random_forest"
_DEFAULT_POTENTIAL_MODEL = "random_forest"


# ── Prediction result ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelPrediction:
    """Outcome of running the saved models over one feature vector."""

    # Calibrated probability of the positive class (0.0–1.0);
    # None when no model prediction is available.
    trust_score: Optional[float]
    potential_score: Optional[float]
    # Which saved artifact produced each score (None when unavailable).
    trust_model: Optional[str]
    potential_model: Optional[str]
    # False when the models could not be loaded — callers must fall back.
    available: bool
    # Why the models are unavailable (empty when available).
    reason: str = ""


# ── Canonical feature schema ─────────────────────────────────────────────────

def _minimal_result() -> InvestigationResult:
    """An empty investigation result — extract_features() on it yields the
    canonical 60-column schema (every key is emitted unconditionally)."""
    return InvestigationResult(
        business_input=BusinessInput(name="schema"),
        status=InvestigationStatus.COMPLETE,
        searches_performed=0,
        sources_examined=0,
        evidence=[],
        features=[],
    )


def canonical_feature_columns() -> List[str]:
    """
    The 60 training feature columns, in training order.

    ``build_dataset()`` builds training rows directly from
    ``extract_features()`` dicts, so the key order of ``extract_features()``
    is exactly the column order the models were fitted on.
    """
    return list(extract_features(_minimal_result()).keys())


# ── Predictor ────────────────────────────────────────────────────────────────

class ModelPredictor:
    """
    Lazy-loading inference wrapper around the saved model artifacts.

    One instance per process is enough — use ``get_predictor()``.  Loading
    is thread-safe (the backend runs investigations on a thread pool).
    """

    def __init__(
        self,
        trust_model: str = _DEFAULT_TRUST_MODEL,
        potential_model: str = _DEFAULT_POTENTIAL_MODEL,
    ) -> None:
        self._trust_model_name = trust_model
        self._potential_model_name = potential_model

        self._lock = threading.Lock()
        self._loaded = False
        self._trust_clf = None
        self._potential_clf = None
        self._feature_columns: List[str] = []
        self._load_error: Optional[str] = None

    # ── Loading ─────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load the saved artifacts once; never raises."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                from ml.model_trainer import load_model

                self._trust_clf = load_model(
                    self._trust_model_name, "trust_label"
                )
                self._potential_clf = load_model(
                    self._potential_model_name, "potential_label"
                )
                self._feature_columns = canonical_feature_columns()
            except Exception as exc:  # missing file, corrupt pickle, ...
                self._trust_clf = None
                self._potential_clf = None
                self._load_error = (
                    f"{type(exc).__name__}: {exc}"
                )
            finally:
                self._loaded = True

    def reset(self) -> None:
        """Forget loaded models so the next ``predict()`` reloads from disk."""
        with self._lock:
            self._loaded = False
            self._trust_clf = None
            self._potential_clf = None
            self._feature_columns = []
            self._load_error = None

    # ── Introspection ───────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True when the saved models are loadable."""
        self._ensure_loaded()
        return self._trust_clf is not None and self._potential_clf is not None

    def load_error(self) -> Optional[str]:
        """The load failure reason, or None when models are available."""
        self._ensure_loaded()
        return self._load_error

    # ── Inference ───────────────────────────────────────────────────────

    def predict(self, features: dict) -> ModelPrediction:
        """
        Score one feature vector with the saved models.

        ``features`` is a ``dict[str, float]`` as produced by
        ``ml.feature_extractor.extract_features()``.  Missing keys default
        to 0.0; extra keys are ignored — the vector is aligned to the
        canonical 60-column training schema either way.
        """
        self._ensure_loaded()

        if not self.is_available():
            return ModelPrediction(
                trust_score=None,
                potential_score=None,
                trust_model=None,
                potential_model=None,
                available=False,
                reason=self._load_error or "models not loaded",
            )

        # Align to the training schema (names and order).
        row = {
            col: float(features.get(col, 0.0))
            for col in self._feature_columns
        }
        X = pd.DataFrame([row])

        trust_score = float(self._trust_clf.predict_proba(X)[0, 1])
        potential_score = float(self._potential_clf.predict_proba(X)[0, 1])

        return ModelPrediction(
            trust_score=round(trust_score, 4),
            potential_score=round(potential_score, 4),
            trust_model=self._trust_model_name,
            potential_model=self._potential_model_name,
            available=True,
        )


# ── Process-wide singleton ───────────────────────────────────────────────────

_DEFAULT_PREDICTOR: Optional[ModelPredictor] = None
_SINGLETON_LOCK = threading.Lock()


def get_predictor() -> ModelPredictor:
    """Return the shared default predictor (created on first use)."""
    global _DEFAULT_PREDICTOR
    with _SINGLETON_LOCK:
        if _DEFAULT_PREDICTOR is None:
            _DEFAULT_PREDICTOR = ModelPredictor()
        return _DEFAULT_PREDICTOR
