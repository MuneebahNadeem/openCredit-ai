"""
Tests for ml/model_predictor.py — production inference over saved models.

Run with:  python -m pytest tests/ml/test_model_predictor.py -v

The predictor tests are self-contained: a module-scoped fixture trains small
saved models first, so they never depend on whatever ``data/models/``
happens to hold on this machine.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from agent.schemas.input import BusinessInput
from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability
from agent.schemas.feature import DiscoveredFeature, FeatureCategory
from agent.schemas.result import (
    AssessmentLevel,
    InvestigationResult,
    InvestigationStatus,
    Signal,
)
from ml.feature_extractor import extract_features
from ml.model_predictor import (
    ModelPrediction,
    ModelPredictor,
    canonical_feature_columns,
    get_predictor,
)
from ml.risk_engine import assess_risk


# ── Fixtures / helpers ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained():
    """Train small saved models so the predictor has artifacts to load."""
    from ml.model_trainer import train_all_models

    return train_all_models(n_synthetic=120, seed=0)


def make_business() -> BusinessInput:
    return BusinessInput(name="Test Business")


def make_evidence(
    field="field_a",
    value="100",
    etype=EvidenceType.OBSERVED,
    reliability=SourceReliability.MEDIUM,
    confidence=0.8,
    source_name="SourceA",
    raw_snippet=None,
) -> EvidenceItem:
    return EvidenceItem(
        field_name=field,
        value=value,
        evidence_type=etype,
        source_name=source_name,
        source_reliability=reliability,
        confidence=confidence,
        raw_snippet=raw_snippet,
    )


def make_feature(
    name="feat_a",
    category=FeatureCategory.AUDIENCE,
    value="500",
    reason="Relevant signal.",
    confidence=0.85,
    searched=True,
) -> DiscoveredFeature:
    return DiscoveredFeature(
        name=name, category=category, value=value,
        reason=reason, confidence=confidence, searched=searched,
    )


def make_signal(label="Signal A") -> Signal:
    return Signal(label=label, detail="Explanation.", evidence_refs=["field_a"])


def make_result(**kwargs) -> InvestigationResult:
    defaults = dict(
        business_input=make_business(),
        status=InvestigationStatus.COMPLETE,
        searches_performed=5,
        sources_examined=4,
        evidence=[
            make_evidence("ev1", "excellent quality",
                          EvidenceType.CORROBORATED, SourceReliability.HIGH,
                          0.9, "GovRegistry", "The business provides excellent quality products."),
            make_evidence("ev2", "25000",
                          EvidenceType.OBSERVED, SourceReliability.MEDIUM,
                          0.8, "Instagram", "Business has 25000 followers on Instagram."),
            make_evidence("ev3", "4.5",
                          EvidenceType.OBSERVED, SourceReliability.MEDIUM,
                          0.75, "ReviewSite", "Average rating is 4.5 stars."),
        ],
        features=[
            make_feature("audience", FeatureCategory.AUDIENCE, "25000"),
            make_feature("engagement", FeatureCategory.ENGAGEMENT, "high"),
            make_feature("reputation", FeatureCategory.REPUTATION, "4.5"),
        ],
        positive_signals=[make_signal("Active social presence"), make_signal("Good reviews")],
        risk_signals=[],
        missing_information=[],
        sources=["https://gov.registry.com", "https://instagram.com/biz"],
        justification="Test justification.",
    )
    defaults.update(kwargs)
    return InvestigationResult(**defaults)


def make_risky_result() -> InvestigationResult:
    """A low-quality, risk-heavy investigation result."""
    return make_result(
        evidence=[
            make_evidence("ev1", "complaints found",
                          EvidenceType.INFERENCE, SourceReliability.LOW,
                          0.3, "Forum"),
            make_evidence("ev2", "no website",
                          EvidenceType.OBSERVED, SourceReliability.UNKNOWN,
                          0.4, "Search"),
        ],
        features=[make_feature("risk_feat", FeatureCategory.RISK, "high")],
        positive_signals=[],
        risk_signals=[make_signal("Multiple complaints"),
                      make_signal("No online presence"),
                      make_signal("Suspicious activity")],
    )


# ── ModelPrediction ──────────────────────────────────────────────────────────

class TestModelPrediction:

    def test_defaults(self):
        p = ModelPrediction(
            trust_score=None, potential_score=None,
            trust_model=None, potential_model=None,
            available=False,
        )
        assert p.reason == ""

    def test_available_prediction_fields(self):
        p = ModelPrediction(
            trust_score=0.87, potential_score=0.64,
            trust_model="random_forest", potential_model="random_forest",
            available=True, reason="",
        )
        assert p.available is True
        assert p.trust_score == 0.87
        assert p.potential_score == 0.64

    def test_frozen(self):
        p = ModelPrediction(
            trust_score=None, potential_score=None,
            trust_model=None, potential_model=None,
            available=False,
        )
        with pytest.raises(AttributeError):
            p.available = True


# ── Canonical feature schema ─────────────────────────────────────────────────

class TestCanonicalColumns:

    def test_sixty_columns(self):
        assert len(canonical_feature_columns()) == 60

    def test_matches_extractor_keys(self):
        """The schema must be exactly the key order of extract_features()."""
        empty = InvestigationResult(
            business_input=BusinessInput(name="schema"),
            status=InvestigationStatus.COMPLETE,
            searches_performed=0,
            sources_examined=0,
            evidence=[],
            features=[],
        )
        assert canonical_feature_columns() == list(extract_features(empty).keys())

    def test_order_stable(self):
        assert canonical_feature_columns() == canonical_feature_columns()

    def test_no_label_columns(self):
        cols = canonical_feature_columns()
        assert "trust_label" not in cols
        assert "potential_label" not in cols

    def test_contains_known_features(self):
        cols = canonical_feature_columns()
        assert "evidence_count_total" in cols


# ── Loading / availability ────────────────────────────────────────────────────

class TestModelPredictorLoading:

    def test_is_available_with_models(self, trained):
        assert ModelPredictor().is_available() is True

    def test_load_error_none_when_available(self, trained):
        assert ModelPredictor().load_error() is None

    def test_missing_model_unavailable(self):
        p = ModelPredictor(trust_model="no_such_model",
                           potential_model="no_such_model")
        assert p.is_available() is False
        assert p.load_error() is not None

    def test_missing_model_predict_reports_reason(self):
        p = ModelPredictor(trust_model="no_such_model",
                           potential_model="no_such_model")
        prediction = p.predict({})
        assert prediction.available is False
        assert prediction.trust_score is None
        assert prediction.potential_score is None
        assert prediction.reason == p.load_error()

    def test_both_models_required(self, trained):
        """One loadable + one missing artifact → overall unavailable."""
        p = ModelPredictor(trust_model="random_forest",
                           potential_model="no_such_model")
        assert p.is_available() is False

    def test_reset_reloads_from_disk(self, trained):
        p = ModelPredictor()
        first = p.predict(extract_features(make_result()))
        p.reset()
        assert p.is_available() is True
        second = p.predict(extract_features(make_result()))
        assert first.trust_score == second.trust_score


# ── Inference ─────────────────────────────────────────────────────────────────

class TestPredict:

    def test_available_prediction(self, trained):
        prediction = ModelPredictor().predict(extract_features(make_result()))
        assert prediction.available is True
        assert 0.0 <= prediction.trust_score <= 1.0
        assert 0.0 <= prediction.potential_score <= 1.0

    def test_scores_rounded_to_four_decimals(self, trained):
        prediction = ModelPredictor().predict(extract_features(make_result()))
        assert prediction.trust_score == round(prediction.trust_score, 4)
        assert prediction.potential_score == round(prediction.potential_score, 4)

    def test_default_model_names(self, trained):
        prediction = ModelPredictor().predict(extract_features(make_result()))
        assert prediction.trust_model == "random_forest"
        assert prediction.potential_model == "random_forest"

    def test_custom_model_names(self, trained):
        p = ModelPredictor(trust_model="gradient_boosting",
                           potential_model="xgboost")
        prediction = p.predict(extract_features(make_result()))
        assert prediction.trust_model == "gradient_boosting"
        assert prediction.potential_model == "xgboost"

    def test_extra_keys_ignored(self, trained):
        p = ModelPredictor()
        features = extract_features(make_result())
        with_junk = dict(features, junk_key_that_is_not_a_feature=5.0)
        assert p.predict(with_junk) == p.predict(features)

    def test_missing_keys_default_to_zero(self, trained):
        """An empty dict must not raise — keys align to the training schema."""
        prediction = ModelPredictor().predict({})
        assert prediction.available is True
        assert prediction.trust_score is not None

    def test_deterministic(self, trained):
        p = ModelPredictor()
        features = extract_features(make_result())
        assert p.predict(features) == p.predict(features)

    def test_good_business_scores_higher_than_risky(self, trained):
        """The models learned structural evidence patterns in training, so a
        corroborated high-reliability result should out-score a low-quality
        risk-heavy one on trustworthiness."""
        p = ModelPredictor()
        good = p.predict(extract_features(make_result()))
        risky = p.predict(extract_features(make_risky_result()))
        assert good.trust_score > risky.trust_score


# ── Singleton ─────────────────────────────────────────────────────────────────

class TestGetPredictor:

    def test_returns_model_predictor(self):
        assert isinstance(get_predictor(), ModelPredictor)

    def test_singleton_identity(self):
        assert get_predictor() is get_predictor()


# ── End-to-end with the risk engine ──────────────────────────────────────────

class TestEndToEnd:

    def test_assess_risk_with_real_predictor(self, trained):
        r = make_result()
        a = assess_risk(r, predictor=ModelPredictor())
        assert a.model_prediction is not None
        assert a.model_prediction.available is True
        assert a.model_prediction.trust_model == "random_forest"
        assert 0.0 <= a.trustworthiness.score <= 1.0
        assert 0.0 <= a.business_potential.score <= 1.0
        assert "trained ML model" in a.trustworthiness.explanation

    def test_assess_risk_level_matches_score(self, trained):
        r = make_result()
        a = assess_risk(r, predictor=ModelPredictor())
        if a.trustworthiness.score >= 0.70:
            assert a.trustworthiness.level == AssessmentLevel.HIGH
        elif a.trustworthiness.score >= 0.45:
            assert a.trustworthiness.level == AssessmentLevel.MODERATE
        else:
            assert a.trustworthiness.level == AssessmentLevel.LOW

    def test_missing_models_fall_back_to_rules(self):
        """Two different broken predictors must agree — both degrade to the
        same pure rule scores."""
        r = make_result()
        a1 = assess_risk(r, predictor=ModelPredictor(
            trust_model="no_such_a", potential_model="no_such_a"))
        a2 = assess_risk(r, predictor=ModelPredictor(
            trust_model="no_such_b", potential_model="no_such_b"))
        assert a1.model_prediction.available is False
        assert a1.trustworthiness.score == a2.trustworthiness.score
        assert a1.business_potential.score == a2.business_potential.score
        assert "trained ML model" not in a1.trustworthiness.explanation

    def test_generate_assessment_full_production_path(self, trained, monkeypatch):
        """Person 3's actual entry point: generate_assessment() with the
        default predictor wired to the real saved models."""
        from ml.assessment import generate_assessment, generate_recommendation

        monkeypatch.setattr(
            "ml.risk_engine.get_predictor", lambda: ModelPredictor(),
        )
        enriched = generate_assessment(make_result())
        assert enriched.trustworthiness.score is not None
        assert 0.0 <= enriched.trustworthiness.score <= 1.0
        assert 0.0 <= enriched.business_potential.score <= 1.0
        assert enriched.justification
        # The backend derives its recommendation from the same assessment.
        risk = assess_risk(make_result(), predictor=ModelPredictor())
        assert generate_recommendation(risk) in (
            "approve", "approve_with_conditions", "decline",
            "further_review", "insufficient_data",
        )
