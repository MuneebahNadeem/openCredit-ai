"""
Tests for ml/sentiment.py — lexicon-based sentiment analysis.

Run with:  python -m pytest tests/ml/test_sentiment.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability
from ml.sentiment import (
    SentimentScore,
    analyze_sentiment,
    analyze_batch,
    aggregate_sentiment,
    score_evidence_texts,
)


# ── SentimentScore ────────────────────────────────────────────────────────────

class TestSentimentScore:

    def test_creation(self):
        s = SentimentScore(
            positive=0.7, negative=0.1, neutral=0.2,
            compound=0.6, label="positive",
        )
        assert s.positive == 0.7
        assert s.negative == 0.1
        assert s.neutral == 0.2
        assert s.compound == 0.6
        assert s.label == "positive"

    def test_frozen(self):
        s = SentimentScore(
            positive=0.5, negative=0.5, neutral=0.0,
            compound=0.0, label="mixed",
        )
        with pytest.raises(AttributeError):
            s.compound = 1.0

    def test_clamping_positive_above_one(self):
        s = SentimentScore(
            positive=1.5, negative=0.0, neutral=0.0,
            compound=2.0, label="positive",
        )
        assert s.positive == 1.0
        assert s.compound == 1.0

    def test_clamping_negative_below_zero(self):
        s = SentimentScore(
            positive=-0.5, negative=0.0, neutral=0.0,
            compound=-2.0, label="negative",
        )
        assert s.positive == 0.0
        assert s.compound == -1.0


# ── analyze_sentiment ─────────────────────────────────────────────────────────

class TestAnalyzeSentiment:

    # ── Positive text ────────────────────────────────────────────────────

    def test_positive_text(self):
        s = analyze_sentiment("The quality is excellent and the service is great.")
        assert s.label == "positive"
        assert s.compound > 0.0
        assert s.positive > s.negative

    def test_strongly_positive(self):
        s = analyze_sentiment("Outstanding product, exceptional and perfect quality.")
        assert s.compound > 0.5
        assert s.label == "positive"

    def test_mildly_positive(self):
        s = analyze_sentiment("The product is good and decent.")
        assert s.label == "positive"
        assert s.compound > 0.0

    # ── Negative text ────────────────────────────────────────────────────

    def test_negative_text(self):
        s = analyze_sentiment("Terrible experience, the product was defective.")
        assert s.label == "negative"
        assert s.compound < 0.0
        assert s.negative > s.positive

    def test_strongly_negative(self):
        s = analyze_sentiment("This is a scam, fraudulent and dishonest business.")
        assert s.compound < -0.5
        assert s.label == "negative"

    def test_mildly_negative(self):
        s = analyze_sentiment("The service was slow and disappointing.")
        assert s.label == "negative"
        assert s.compound < 0.0

    # ── Neutral text ─────────────────────────────────────────────────────

    def test_neutral_text(self):
        s = analyze_sentiment("The business is located in Lagos.")
        assert s.label == "neutral"
        assert s.compound == 0.0
        assert s.neutral == 1.0

    def test_empty_string(self):
        s = analyze_sentiment("")
        assert s.label == "neutral"
        assert s.compound == 0.0
        assert s.neutral == 1.0

    def test_whitespace_only(self):
        s = analyze_sentiment("   \t\n  ")
        assert s.label == "neutral"
        assert s.compound == 0.0

    def test_no_lexicon_words(self):
        s = analyze_sentiment("they sell shoes and bags online")
        assert s.label == "neutral"
        assert s.compound == 0.0
        assert s.neutral == 1.0

    # ── Mixed text ───────────────────────────────────────────────────────

    def test_mixed_text(self):
        s = analyze_sentiment("The quality is great but delivery was terrible.")
        assert s.positive > 0.0
        assert s.negative > 0.0
        assert s.label == "mixed"

    # ── Negation handling ────────────────────────────────────────────────

    def test_negation_flips_positive(self):
        without_neg = analyze_sentiment("good")
        with_neg = analyze_sentiment("not good")
        assert without_neg.compound > 0.0
        assert with_neg.compound < 0.0

    def test_negation_flips_negative(self):
        without_neg = analyze_sentiment("bad")
        with_neg = analyze_sentiment("not bad")
        assert without_neg.compound < 0.0
        assert with_neg.compound > 0.0

    def test_negation_with_dont(self):
        s = analyze_sentiment("dont recommend this")
        # "dont" is a negator, so "recommend" (positive) should flip to negative.
        assert s.compound < 0.0

    def test_negation_dampening(self):
        """Negated positive should be weaker than a direct negative word."""
        direct_neg = analyze_sentiment("terrible")
        negated_pos = analyze_sentiment("not excellent")
        # "terrible" is -2, negated "excellent" is -2 * 0.75 = -1.5
        assert direct_neg.compound < negated_pos.compound

    # ── Case insensitivity ───────────────────────────────────────────────

    def test_case_insensitive(self):
        lower = analyze_sentiment("excellent quality")
        upper = analyze_sentiment("EXCELLENT QUALITY")
        mixed = analyze_sentiment("Excellent Quality")
        assert lower.compound == upper.compound == mixed.compound

    # ── Punctuation handling ─────────────────────────────────────────────

    def test_punctuation_stripped(self):
        clean = analyze_sentiment("great product")
        punctuated = analyze_sentiment("great! product!!!")
        assert clean.compound == punctuated.compound


# ── analyze_batch ─────────────────────────────────────────────────────────────

class TestAnalyzeBatch:

    def test_multiple_texts(self):
        scores = analyze_batch([
            "The service is excellent.",
            "Terrible quality and slow delivery.",
            "Located in downtown area.",
        ])
        assert len(scores) == 3
        assert scores[0].label == "positive"
        assert scores[1].label == "negative"
        assert scores[2].label == "neutral"

    def test_empty_list(self):
        scores = analyze_batch([])
        assert scores == []

    def test_single_item(self):
        scores = analyze_batch(["great product"])
        assert len(scores) == 1
        assert scores[0].compound > 0.0


# ── aggregate_sentiment ──────────────────────────────────────────────────────

class TestAggregateSentiment:

    def test_all_positive(self):
        scores = [
            SentimentScore(positive=0.8, negative=0.0, neutral=0.2, compound=0.8, label="positive"),
            SentimentScore(positive=0.6, negative=0.0, neutral=0.4, compound=0.6, label="positive"),
        ]
        agg = aggregate_sentiment(scores)
        assert agg.positive == pytest.approx(0.7)
        assert agg.negative == pytest.approx(0.0)
        assert agg.compound == pytest.approx(0.7)
        assert agg.label == "positive"

    def test_mixed_scores(self):
        scores = [
            SentimentScore(positive=0.8, negative=0.0, neutral=0.2, compound=0.8, label="positive"),
            SentimentScore(positive=0.0, negative=0.8, neutral=0.2, compound=-0.8, label="negative"),
        ]
        agg = aggregate_sentiment(scores)
        assert agg.positive == pytest.approx(0.4)
        assert agg.negative == pytest.approx(0.4)
        assert agg.compound == pytest.approx(0.0)

    def test_empty_list(self):
        agg = aggregate_sentiment([])
        assert agg.label == "neutral"
        assert agg.compound == 0.0
        assert agg.neutral == 1.0

    def test_single_score(self):
        s = SentimentScore(positive=0.5, negative=0.3, neutral=0.2, compound=0.2, label="mixed")
        agg = aggregate_sentiment([s])
        assert agg.positive == pytest.approx(0.5)
        assert agg.compound == pytest.approx(0.2)

    def test_aggregate_label_matches_compound(self):
        scores = [
            SentimentScore(positive=0.9, negative=0.0, neutral=0.1, compound=0.9, label="positive"),
            SentimentScore(positive=0.7, negative=0.0, neutral=0.3, compound=0.7, label="positive"),
        ]
        agg = aggregate_sentiment(scores)
        assert agg.label == "positive"


# ── score_evidence_texts ─────────────────────────────────────────────────────

def _make_evidence(
    value="good product",
    raw_snippet=None,
) -> EvidenceItem:
    return EvidenceItem(
        field_name="review",
        value=value,
        evidence_type=EvidenceType.OBSERVED,
        source_name="TestSource",
        source_reliability=SourceReliability.MEDIUM,
        confidence=0.8,
        raw_snippet=raw_snippet,
    )


class TestScoreEvidenceTexts:

    def test_uses_raw_snippet_when_available(self):
        item = _make_evidence(value="bad", raw_snippet="excellent quality product")
        s = score_evidence_texts([item])
        # Should use raw_snippet ("excellent") not value ("bad").
        assert s.compound > 0.0

    def test_falls_back_to_value(self):
        item = _make_evidence(value="terrible service")
        s = score_evidence_texts([item])
        assert s.compound < 0.0

    def test_multiple_evidence_items(self):
        items = [
            _make_evidence(value="great quality"),
            _make_evidence(value="slow delivery"),
        ]
        s = score_evidence_texts(items)
        # One positive + one negative → mixed or near-neutral.
        assert s.positive > 0.0
        assert s.negative > 0.0

    def test_empty_evidence_list(self):
        s = score_evidence_texts([])
        assert s.label == "neutral"
        assert s.compound == 0.0

    def test_skips_empty_text(self):
        items = [
            _make_evidence(value="valid value", raw_snippet=""),
            _make_evidence(value="great product"),
        ]
        s = score_evidence_texts(items)

        # Empty raw_snippet should be skipped; "great product" counts.
        assert s.compound > 0.0

    def test_all_empty_texts(self):
        items = [
            _make_evidence(value="valid value", raw_snippet=""),
            _make_evidence(value="valid value", raw_snippet="  "),
        ]
        s = score_evidence_texts(items)

        assert s.label == "neutral"
        assert s.compound == 0.0

