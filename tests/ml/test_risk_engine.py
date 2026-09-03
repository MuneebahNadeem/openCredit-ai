"""
Tests for ml/risk_engine.py — central risk scoring module.

Run with:  python -m pytest tests/ml/test_risk_engine.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from agent.schemas.input import BusinessInput
from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability
from agent.schemas.feature import DiscoveredFeature, FeatureCategory
from agent.schemas.result import (
    AssessmentLevel,
    AssessmentScore,
    InvestigationResult,
    InvestigationStatus,
    Signal,
)
from ml.risk_engine import (
    RiskAssessment,
    assess_risk,
    _MODEL_WEIGHT,
    _blend_scores,
    _score_to_level,
    _score_trustworthiness,
    _score_business_potential,
    _explain_trustworthiness,
    _explain_business_potential,
)
from ml.model_predictor import ModelPrediction
from ml.credibility_scorer import score_credibility, CredibilityScore
from ml.feature_extractor import extract_features
from ml.sentiment import analyze_sentiment, score_evidence_texts, SentimentScore


# ── Fixtures / helpers ────────────────────────────────────────────────────────

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


# ── RiskAssessment ────────────────────────────────────────────────────────────

class TestRiskAssessment:

    def test_creation(self):
        r = make_result()
        a = assess_risk(r)
        assert isinstance(a, RiskAssessment)
        assert isinstance(a.trustworthiness, AssessmentScore)
        assert isinstance(a.business_potential, AssessmentScore)
        assert isinstance(a.credibility, CredibilityScore)
        assert isinstance(a.sentiment, SentimentScore)
        assert isinstance(a.features, dict)

    def test_frozen(self):
        r = make_result()
        a = assess_risk(r)
        with pytest.raises(AttributeError):
            a.trustworthiness = None


# ── _score_to_level ──────────────────────────────────────────────────────────

class TestScoreToLevel:

    def test_high(self):
        assert _score_to_level(0.80) == AssessmentLevel.HIGH

    def test_high_at_threshold(self):
        assert _score_to_level(0.70) == AssessmentLevel.HIGH

    def test_moderate(self):
        assert _score_to_level(0.55) == AssessmentLevel.MODERATE

    def test_moderate_at_threshold(self):
        assert _score_to_level(0.45) == AssessmentLevel.MODERATE

    def test_low(self):
        assert _score_to_level(0.20) == AssessmentLevel.LOW

    def test_zero(self):
        assert _score_to_level(0.0) == AssessmentLevel.LOW

    def test_one(self):
        assert _score_to_level(1.0) == AssessmentLevel.HIGH


# ── _score_trustworthiness ───────────────────────────────────────────────────

class TestScoreTrustworthiness:

    def test_perfect_evidence(self):
        features = {
            "positive_signal_ratio": 1.0,
            "risk_signal_ratio": 0.0,
            "evidence_reliable_ratio": 1.0,
            "source_reliability_high_ratio": 1.0,
        }
        sentiment = SentimentScore(
            positive=1.0, negative=0.0, neutral=0.0,
            compound=1.0, label="positive",
        )
        credibility = CredibilityScore(
            source_reliability_score=1.0, evidence_quality_score=1.0,
            confidence_score=1.0, reliable_ratio=1.0,
            source_diversity_score=1.0, corroboration_score=1.0,
            evidence_depth_score=1.0, overall_score=1.0, level="high",
        )
        score = _score_trustworthiness(features, sentiment, credibility)
        assert score == pytest.approx(1.0)

    def test_worst_evidence(self):
        features = {
            "positive_signal_ratio": 0.0,
            "risk_signal_ratio": 1.0,
            "evidence_reliable_ratio": 0.0,
            "source_reliability_high_ratio": 0.0,
        }
        sentiment = SentimentScore(
            positive=0.0, negative=1.0, neutral=0.0,
            compound=-1.0, label="negative",
        )
        credibility = CredibilityScore(
            source_reliability_score=0.0, evidence_quality_score=0.0,
            confidence_score=0.0, reliable_ratio=0.0,
            source_diversity_score=0.0, corroboration_score=0.0,
            evidence_depth_score=0.0, overall_score=0.0, level="low",
        )
        score = _score_trustworthiness(features, sentiment, credibility)
        assert score == pytest.approx(0.0)

    def test_score_bounded(self):
        """Score should always be in [0, 1]."""
        features = {
            "positive_signal_ratio": 0.5,
            "risk_signal_ratio": 0.5,
            "evidence_reliable_ratio": 0.5,
            "source_reliability_high_ratio": 0.5,
        }
        sentiment = SentimentScore(
            positive=0.5, negative=0.5, neutral=0.0,
            compound=0.0, label="mixed",
        )
        credibility = CredibilityScore(
            source_reliability_score=0.5, evidence_quality_score=0.5,
            confidence_score=0.5, reliable_ratio=0.5,
            source_diversity_score=0.5, corroboration_score=0.5,
            evidence_depth_score=0.5, overall_score=0.5, level="moderate",
        )
        score = _score_trustworthiness(features, sentiment, credibility)
        assert 0.0 <= score <= 1.0


# ── _score_business_potential ────────────────────────────────────────────────

class TestScoreBusinessPotential:

    def test_strong_potential(self):
        features = {
            "positive_signal_ratio": 1.0,
            "evidence_reliable_ratio": 1.0,
            "features_total": 5.0,
            "features_cat_audience": 2.0,
            "features_cat_engagement": 1.0,
            "features_cat_demand": 1.0,
            "features_cat_growth": 1.0,
            "features_cat_market_presence": 0.0,
        }
        sentiment = SentimentScore(
            positive=1.0, negative=0.0, neutral=0.0,
            compound=1.0, label="positive",
        )
        credibility = CredibilityScore(
            source_reliability_score=1.0, evidence_quality_score=1.0,
            confidence_score=1.0, reliable_ratio=1.0,
            source_diversity_score=1.0, corroboration_score=1.0,
            evidence_depth_score=1.0, overall_score=1.0, level="high",
        )
        score = _score_business_potential(features, sentiment, credibility)
        assert score >= 0.70

    def test_no_features(self):
        features = {
            "positive_signal_ratio": 0.0,
            "evidence_reliable_ratio": 0.0,
            "features_total": 0.0,
            "features_cat_audience": 0.0,
            "features_cat_engagement": 0.0,
            "features_cat_demand": 0.0,
            "features_cat_growth": 0.0,
            "features_cat_market_presence": 0.0,
        }
        sentiment = SentimentScore(
            positive=0.0, negative=0.0, neutral=1.0,
            compound=0.0, label="neutral",
        )
        credibility = CredibilityScore(
            source_reliability_score=0.0, evidence_quality_score=0.0,
            confidence_score=0.0, reliable_ratio=0.0,
            source_diversity_score=0.0, corroboration_score=0.0,
            evidence_depth_score=0.0, overall_score=0.0, level="low",
        )
        score = _score_business_potential(features, sentiment, credibility)
        assert score == pytest.approx(0.0)

    def test_score_bounded(self):
        features = {
            "positive_signal_ratio": 0.5,
            "evidence_reliable_ratio": 0.5,
            "features_total": 3.0,
            "features_cat_audience": 1.0,
            "features_cat_engagement": 1.0,
            "features_cat_demand": 0.0,
            "features_cat_growth": 0.0,
            "features_cat_market_presence": 0.0,
        }
        sentiment = SentimentScore(
            positive=0.5, negative=0.5, neutral=0.0,
            compound=0.0, label="mixed",
        )
        credibility = CredibilityScore(
            source_reliability_score=0.5, evidence_quality_score=0.5,
            confidence_score=0.5, reliable_ratio=0.5,
            source_diversity_score=0.5, corroboration_score=0.5,
            evidence_depth_score=0.5, overall_score=0.5, level="moderate",
        )
        score = _score_business_potential(features, sentiment, credibility)
        assert 0.0 <= score <= 1.0


# ── Explanation generators ───────────────────────────────────────────────────

class TestExplanations:

    def test_trustworthiness_explanation_high(self):
        cred = CredibilityScore(
            source_reliability_score=0.8, evidence_quality_score=0.8,
            confidence_score=0.8, reliable_ratio=0.8,
            source_diversity_score=0.8, corroboration_score=0.8,
            evidence_depth_score=0.8, overall_score=0.8, level="high",
        )
        features = {"evidence_reliable_ratio": 0.8, "risk_signal_ratio": 0.1}
        explanation = _explain_trustworthiness(0.8, cred, features)
        assert "credible" in explanation.lower() or "reliable" in explanation.lower()

    def test_trustworthiness_explanation_risk(self):
        cred = CredibilityScore(
            source_reliability_score=0.3, evidence_quality_score=0.3,
            confidence_score=0.3, reliable_ratio=0.3,
            source_diversity_score=0.3, corroboration_score=0.3,
            evidence_depth_score=0.3, overall_score=0.3, level="low",
        )
        features = {"evidence_reliable_ratio": 0.2, "risk_signal_ratio": 0.7}
        explanation = _explain_trustworthiness(0.3, cred, features)
        assert "risk" in explanation.lower()

    def test_business_potential_explanation_positive(self):
        sent = SentimentScore(
            positive=0.8, negative=0.0, neutral=0.2,
            compound=0.8, label="positive",
        )
        features = {"positive_signal_ratio": 0.8}
        explanation = _explain_business_potential(0.8, sent, features)
        assert "positive" in explanation.lower() or "strong" in explanation.lower()

    def test_business_potential_explanation_negative_sentiment(self):
        sent = SentimentScore(
            positive=0.0, negative=0.8, neutral=0.2,
            compound=-0.8, label="negative",
        )
        features = {"positive_signal_ratio": 0.1}
        explanation = _explain_business_potential(0.2, sent, features)
        assert "negative" in explanation.lower()


# ── assess_risk (integration) ────────────────────────────────────────────────

class TestAssessRisk:

    def test_no_evidence(self):
        r = make_result(evidence=[], features=[],
                        positive_signals=[], risk_signals=[])
        a = assess_risk(r)
        assert a.trustworthiness.level == AssessmentLevel.INSUFFICIENT_EVIDENCE
        assert a.business_potential.level == AssessmentLevel.INSUFFICIENT_EVIDENCE
        assert a.trustworthiness.score is None
        assert a.business_potential.score is None

    def test_assessment_scores_are_valid_pydantic(self):
        """AssessmentScore objects should be valid Pydantic models."""
        r = make_result()
        a = assess_risk(r)
        # Should not raise
        assert 0.0 <= a.trustworthiness.score <= 1.0
        assert 0.0 <= a.business_potential.score <= 1.0
        assert isinstance(a.trustworthiness.evidence_count, int)
        assert isinstance(a.trustworthiness.explanation, str)
        assert len(a.trustworthiness.explanation) > 0

    def test_good_business(self):
        """A business with strong positive evidence should score well."""
        r = make_result()
        a = assess_risk(r)
        assert a.trustworthiness.score > 0.0
        assert a.business_potential.score > 0.0
        # Should have explanations
        assert len(a.trustworthiness.explanation) > 0
        assert len(a.business_potential.explanation) > 0

    def test_risky_business(self):
        """A business with risk signals and low-quality evidence."""
        evidence = [
            make_evidence("ev1", "complaints found",
                          EvidenceType.INFERENCE, SourceReliability.LOW,
                          0.3, "Forum"),
            make_evidence("ev2", "no website",
                          EvidenceType.OBSERVED, SourceReliability.UNKNOWN,
                          0.4, "Search"),
        ]
        r = make_result(
            evidence=evidence,
            features=[make_feature("risk_feat", FeatureCategory.RISK, "high")],
            positive_signals=[],
            risk_signals=[make_signal("Multiple complaints"),
                          make_signal("No online presence"),
                          make_signal("Suspicious activity")],
        )
        a = assess_risk(r)
        # Trust should be lower than a good business
        assert a.trustworthiness.score < 0.5

    def test_trustworthiness_separate_from_potential(self):
        """A business can have different trust and potential levels."""
        # High trust evidence but no business potential signals
        evidence = [
            make_evidence("ev1", "registered business",
                          EvidenceType.CORROBORATED, SourceReliability.HIGH,
                          0.95, "GovRegistry"),
            make_evidence("ev1", "registered business",
                          EvidenceType.CORROBORATED, SourceReliability.HIGH,
                          0.90, "BusinessBureau"),
        ]
        r = make_result(
            evidence=evidence,
            features=[],
            positive_signals=[],
            risk_signals=[],
        )
        a = assess_risk(r)
        # Trust should be decent (corroborated, high reliability)
        # Potential may be lower (no audience/demand signals)
        assert a.trustworthiness.score is not None
        assert a.business_potential.score is not None

    def test_features_populated(self):
        r = make_result()
        a = assess_risk(r)
        assert len(a.features) > 0
        assert "evidence_count_total" in a.features

    def test_credibility_populated(self):
        r = make_result()
        a = assess_risk(r)
        assert a.credibility.overall_score >= 0.0

    def test_sentiment_populated(self):
        r = make_result()
        a = assess_risk(r)
        # The evidence has raw_snippets with text, so sentiment should be computed
        assert isinstance(a.sentiment, SentimentScore)

    def test_assessment_level_matches_score(self):
        """The level should be consistent with the numeric score."""
        r = make_result()
        a = assess_risk(r)

        if a.trustworthiness.score >= 0.70:
            assert a.trustworthiness.level == AssessmentLevel.HIGH
        elif a.trustworthiness.score >= 0.45:
            assert a.trustworthiness.level == AssessmentLevel.MODERATE
        else:
            assert a.trustworthiness.level == AssessmentLevel.LOW

    def test_evidence_count_matches_reliable(self):
        r = make_result()
        a = assess_risk(r)
        expected = len(r.reliable_evidence())
        assert a.trustworthiness.evidence_count == expected
        assert a.business_potential.evidence_count == expected


# ── Hybrid blend (trained-model integration) ─────────────────────────────────

class _StubPredictor:
    """Configurable stand-in for ModelPredictor; records predict() calls."""

    def __init__(self, trust=0.8, potential=0.8,
                 available=True, raise_on_predict=False):
        self.trust = trust
        self.potential = potential
        self.available = available
        self.raise_on_predict = raise_on_predict
        self.calls = 0

    def predict(self, features):
        self.calls += 1
        if self.raise_on_predict:
            raise RuntimeError("stub predictor exploded")
        return ModelPrediction(
            trust_score=self.trust if self.available else None,
            potential_score=self.potential if self.available else None,
            trust_model="stub_trust" if self.available else None,
            potential_model="stub_potential" if self.available else None,
            available=self.available,
            reason="" if self.available else "stub unavailable",
        )


def _rule_scores(r: InvestigationResult):
    """Pure rule-engine scores for a result (no model involvement)."""
    features = extract_features(r)
    sentiment = score_evidence_texts(r.evidence)
    credibility = score_credibility(r)
    return (
        _score_trustworthiness(features, sentiment, credibility),
        _score_business_potential(features, sentiment, credibility),
    )


class TestBlendScores:

    def test_no_model_score_returns_rule_score(self):
        assert _blend_scores(0.72, None, 0.5) == pytest.approx(0.72)

    def test_equal_weights_average(self):
        assert _blend_scores(0.6, 0.8, 0.5) == pytest.approx(0.7)

    def test_zero_weight_is_pure_rules(self):
        assert _blend_scores(0.6, 0.99, 0.0) == pytest.approx(0.6)

    def test_one_weight_is_pure_model(self):
        assert _blend_scores(0.6, 0.99, 1.0) == pytest.approx(0.99)

    def test_quarter_weight(self):
        assert _blend_scores(0.4, 0.8, 0.25) == pytest.approx(0.5)

    def test_result_clamped_to_unit_interval(self):
        assert _blend_scores(1.0, 1.0, 0.5) == 1.0
        assert _blend_scores(0.0, 0.0, 0.5) == 0.0


class TestHybridBlend:

    def test_default_model_weight_is_half(self):
        assert _MODEL_WEIGHT == 0.5

    def test_default_predictor_used_when_none_passed(self):
        """The conftest patches get_predictor() to a null predictor, so the
        default path must yield the same scores as an explicitly
        unavailable predictor."""
        r = make_result()
        default = assess_risk(r)
        explicit = assess_risk(r, predictor=_StubPredictor(available=False))
        assert default.trustworthiness.score == explicit.trustworthiness.score
        assert default.business_potential.score == explicit.business_potential.score

    def test_unavailable_model_falls_back_to_rule_scores(self):
        r = make_result()
        trust_rule, potential_rule = _rule_scores(r)
        a = assess_risk(r, predictor=_StubPredictor(available=False))
        assert a.trustworthiness.score == pytest.approx(round(trust_rule, 4))
        assert a.business_potential.score == pytest.approx(round(potential_rule, 4))

    def test_blended_score_math(self):
        r = make_result()
        trust_rule, potential_rule = _rule_scores(r)
        a = assess_risk(r, predictor=_StubPredictor(trust=0.9, potential=0.1))
        assert a.trustworthiness.score == pytest.approx(
            round(0.5 * 0.9 + 0.5 * trust_rule, 4)
        )
        assert a.business_potential.score == pytest.approx(
            round(0.5 * 0.1 + 0.5 * potential_rule, 4)
        )

    def test_model_weight_zero_is_pure_rules(self):
        r = make_result()
        trust_rule, potential_rule = _rule_scores(r)
        a = assess_risk(
            r, predictor=_StubPredictor(trust=1.0, potential=1.0),
            model_weight=0.0,
        )
        assert a.trustworthiness.score == pytest.approx(round(trust_rule, 4))
        assert a.business_potential.score == pytest.approx(round(potential_rule, 4))

    def test_model_weight_one_is_pure_model(self):
        r = make_result()
        a = assess_risk(
            r, predictor=_StubPredictor(trust=0.83, potential=0.27),
            model_weight=1.0,
        )
        assert a.trustworthiness.score == pytest.approx(0.83)
        assert a.business_potential.score == pytest.approx(0.27)

    def test_model_weight_below_zero_raises(self):
        with pytest.raises(ValueError):
            assess_risk(make_result(), model_weight=-0.01)

    def test_model_weight_above_one_raises(self):
        with pytest.raises(ValueError):
            assess_risk(make_result(), model_weight=1.01)

    def test_boundary_weights_accepted(self):
        r = make_result()
        assess_risk(r, model_weight=0.0)
        assess_risk(r, model_weight=1.0)

    def test_model_prediction_attached(self):
        r = make_result()
        a = assess_risk(r, predictor=_StubPredictor(trust=0.9, potential=0.6))
        assert a.model_prediction is not None
        assert a.model_prediction.available is True
        assert a.model_prediction.trust_model == "stub_trust"
        assert a.model_prediction.potential_model == "stub_potential"

    def test_unavailable_prediction_attached_with_reason(self):
        r = make_result()
        a = assess_risk(r, predictor=_StubPredictor(available=False))
        assert a.model_prediction is not None
        assert a.model_prediction.available is False
        assert a.model_prediction.reason == "stub unavailable"

    def test_raising_predictor_falls_back_to_rules(self):
        r = make_result()
        trust_rule, potential_rule = _rule_scores(r)
        a = assess_risk(r, predictor=_StubPredictor(raise_on_predict=True))
        assert a.model_prediction.available is False
        assert "raised" in a.model_prediction.reason
        assert a.trustworthiness.score == pytest.approx(round(trust_rule, 4))
        assert a.business_potential.score == pytest.approx(round(potential_rule, 4))

    def test_no_evidence_never_calls_predictor(self):
        r = make_result(evidence=[], features=[],
                        positive_signals=[], risk_signals=[])
        stub = _StubPredictor()
        a = assess_risk(r, predictor=stub)
        assert stub.calls == 0
        assert a.trustworthiness.level == AssessmentLevel.INSUFFICIENT_EVIDENCE
        assert a.model_prediction is None

    def test_explanation_discloses_model_use(self):
        r = make_result()
        used = assess_risk(r, predictor=_StubPredictor())
        assert "trained ML model" in used.trustworthiness.explanation
        assert "trained ML model" in used.business_potential.explanation

    def test_explanation_silent_without_model(self):
        r = make_result()
        unused = assess_risk(r, predictor=_StubPredictor(available=False))
        assert "trained ML model" not in unused.trustworthiness.explanation
        assert "trained ML model" not in unused.business_potential.explanation

    def test_blended_scores_stay_bounded(self):
        """Extreme model scores must never push the blend out of [0, 1]."""
        r = make_result()
        for extreme in (0.0, 1.0):
            a = assess_risk(
                r, predictor=_StubPredictor(trust=extreme, potential=extreme),
            )
            assert 0.0 <= a.trustworthiness.score <= 1.0
            assert 0.0 <= a.business_potential.score <= 1.0

    def test_model_moves_score_in_both_directions(self):
        """A confident model should move the blended score up and down —
        that is the point of the hybrid."""
        r = make_result()
        rule = assess_risk(r, predictor=_StubPredictor(available=False))
        up = assess_risk(r, predictor=_StubPredictor(trust=1.0, potential=1.0))
        down = assess_risk(r, predictor=_StubPredictor(trust=0.0, potential=0.0))
        assert up.trustworthiness.score > rule.trustworthiness.score
        assert down.trustworthiness.score < rule.trustworthiness.score
