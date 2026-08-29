"""
Sentiment analysis for text-based evidence.

Analyses review text, social mentions, and raw snippets from the agent's
evidence to produce numeric sentiment scores.  These scores become ML
features that help the model distinguish businesses with positive public
reputation from those with negative sentiment.

The scorer is **lexicon-based** — it uses a curated word list with known
positive / negative weights.  No external API or heavy ML dependency is
required, keeping the module lightweight and fully testable.

Usage::

    from ml.sentiment import analyze_sentiment, SentimentScore

    score = analyze_sentiment("The product quality is excellent and delivery was fast.")
    print(score.compound)   # 0.68
    print(score.label)      # "positive"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


# ── Sentiment score ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SentimentScore:
    """Sentiment result for a single text or an aggregated batch."""

    positive: float      # 0.0 – 1.0  proportion of positive signal
    negative: float      # 0.0 – 1.0  proportion of negative signal
    neutral: float       # 0.0 – 1.0  proportion of neutral signal
    compound: float      # -1.0 – 1.0 overall sentiment (-1 = very negative)
    label: str           # "positive" | "negative" | "neutral" | "mixed"

    def __post_init__(self) -> None:
        # Clamp to valid ranges (defensive, not a hard error).
        object.__setattr__(self, "positive", max(0.0, min(1.0, self.positive)))
        object.__setattr__(self, "negative", max(0.0, min(1.0, self.negative)))
        object.__setattr__(self, "neutral", max(0.0, min(1.0, self.neutral)))
        object.__setattr__(self, "compound", max(-1.0, min(1.0, self.compound)))


# ── Lexicon ──────────────────────────────────────────────────────────────────

# Each word maps to a weight:  positive values = positive sentiment,
# negative values = negative sentiment.  Magnitude encodes strength.
#
# Weights range from -3 (very negative) to +3 (very positive).

_POSITIVE_WORDS: dict[str, float] = {
    # +1  mild positive
    "good": 1.0, "nice": 1.0, "decent": 1.0, "fine": 1.0,
    "satisfied": 1.0, "helpful": 1.0, "easy": 1.0, "fair": 1.0,
    "okay": 1.0, "ok": 1.0, "reliable": 1.0, "recommend": 1.0,
    "active": 1.0, "responsive": 1.0, "friendly": 1.0,
    # +2  strong positive
    "great": 2.0, "excellent": 2.0, "trustworthy": 2.0,
    "professional": 2.0, "quality": 2.0, "consistent": 2.0,
    "impressive": 2.0, "credible": 2.0, "popular": 2.0,
    "established": 2.0, "verified": 2.0, "legitimate": 2.0,
    "transparent": 2.0, "genuine": 2.0, "authentic": 2.0,
    "praised": 2.0, "positive": 2.0, "reputable": 2.0,
    # +3  very strong positive
    "outstanding": 3.0, "exceptional": 3.0, "superb": 3.0,
    "fantastic": 3.0, "perfect": 3.0, "amazing": 3.0,
    "brilliant": 3.0, "remarkable": 3.0,
}

_NEGATIVE_WORDS: dict[str, float] = {
    # -1  mild negative
    "bad": -1.0, "poor": -1.0, "slow": -1.0, "late": -1.0,
    "issue": -1.0, "problem": -1.0, "complaint": -1.0,
    "disappointed": -1.0, "disappointing": -1.0, "lacking": -1.0, "limited": -1.0,
    "overpriced": -1.0, "confusing": -1.0, "unclear": -1.0,
    "average": -1.0, "mediocre": -1.0,
    # -2  strong negative
    "scam": -2.0, "fraud": -2.0, "fraudulent": -2.0,
    "dishonest": -2.0, "unreliable": -2.0, "unprofessional": -2.0,
    "misleading": -2.0, "terrible": -2.0, "horrible": -2.0,
    "awful": -2.0, "worst": -2.0, "angry": -2.0,
    "broken": -2.0, "defective": -2.0, "fake": -2.0,
    "negative": -2.0, "suspicious": -2.0, "risky": -2.0,
    # -3  very strong negative
    "scamming": -3.0, "stolen": -3.0, "theft": -3.0,
    "criminal": -3.0, "devastating": -3.0, "catastrophic": -3.0,
}

# Merge into a single lookup.
_LEXICON: dict[str, float] = {**_POSITIVE_WORDS, **_NEGATIVE_WORDS}

# Words that negate the sentiment of the following word.
_NEGATORS: set[str] = {
    "not", "no", "never", "neither", "nobody", "nothing",
    "nowhere", "nor", "cannot", "cant", "dont", "doesnt",
    "didnt", "wont", "wouldnt", "shouldnt", "couldnt", "isnt",
    "arent", "wasnt", "werent",
}

# Pattern: split on non-alphanumeric characters, keep words.
_TOKEN_RE = re.compile(r"[a-z']+")


# ── Internal helpers ─────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase and split text into word tokens."""
    return _TOKEN_RE.findall(text.lower())

def _classify_sentiment(raw_pos: float, raw_neg: float) -> str:
    """Classify sentiment using polarity strength and balance."""
    if raw_pos == 0.0 and raw_neg == 0.0:
        return "neutral"

    if raw_pos > 0.0 and raw_neg > 0.0:
        # A strong negative signal can dominate weaker positive wording.
        if raw_neg > raw_pos:
            return "negative"

        # Otherwise, when both sides contain substantial sentiment,
        # classify as mixed.
        return "mixed"

    if raw_pos > 0.0:
        return "positive"

    return "negative"
    
def _classify_compound(compound: float) -> str:
    """Classify sentiment based on the compound score."""
    if compound > 0.0:
        return "positive"
    if compound < 0.0:
        return "negative"
    return "neutral"

# ── Core analysis ────────────────────────────────────────────────────────────

def analyze_sentiment(text: str) -> SentimentScore:
    """
    Analyse a single piece of text and return a sentiment score.

    The scorer tokenises the text, looks up each word in the lexicon,
    handles simple negation ("not good" flips the polarity), and
    normalises the result to a -1.0 … +1.0 compound score.

    Returns a neutral score for empty or unrecognised text.
    """
    if not text or not text.strip():
        return SentimentScore(
            positive=0.0, negative=0.0, neutral=1.0,
            compound=0.0, label="neutral",
        )

    tokens = _tokenize(text)
    if not tokens:
        return SentimentScore(
            positive=0.0, negative=0.0, neutral=1.0,
            compound=0.0, label="neutral",
        )

    raw_pos = 0.0
    raw_neg = 0.0
    matched = 0

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Check for negation: if the current token is a negator,
        # flip the polarity of the *next* token's score.
        negate = False
        if token in _NEGATORS and i + 1 < len(tokens):
            negate = True
            i += 1
            token = tokens[i]

        weight = _LEXICON.get(token, 0.0)
        if weight != 0.0:
            if negate:
                weight = -weight * 0.75  # negation dampens slightly
            if weight > 0:
                raw_pos += weight
            else:
                raw_neg += abs(weight)
            matched += 1

        i += 1

    # If no lexicon words matched, the text is neutral.
    if matched == 0:
        return SentimentScore(
            positive=0.0, negative=0.0, neutral=1.0,
            compound=0.0, label="neutral",
        )

    # Normalise proportions to 0.0 – 1.0.
    total = raw_pos + raw_neg
    pos_ratio = raw_pos / total if total > 0 else 0.0
    neg_ratio = raw_neg / total if total > 0 else 0.0
    neu_ratio = 0.0  # when words matched, neutral ratio is zero

    # Compound: 
    compound = (raw_pos - raw_neg) / 2.0
    compound = max(-1.0, min(1.0, compound))

    return SentimentScore(
        positive=pos_ratio,
        negative=neg_ratio,
        neutral=neu_ratio,
        compound=compound,
        label=_classify_sentiment(raw_pos, raw_neg),
    )


# ── Batch analysis ───────────────────────────────────────────────────────────

def analyze_batch(texts: List[str]) -> List[SentimentScore]:
    """Analyse a list of texts and return one SentimentScore per text."""
    return [analyze_sentiment(t) for t in texts]


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate_sentiment(scores: List[SentimentScore]) -> SentimentScore:
    """
    Aggregate multiple SentimentScores into a single summary score.

    Averages the positive, negative, neutral, and compound values.
    Returns a neutral score for an empty list.
    """
    if not scores:
        return SentimentScore(
            positive=0.0, negative=0.0, neutral=1.0,
            compound=0.0, label="neutral",
        )

    n = len(scores)
    avg_pos = sum(s.positive for s in scores) / n
    avg_neg = sum(s.negative for s in scores) / n
    avg_neu = sum(s.neutral for s in scores) / n
    avg_compound = sum(s.compound for s in scores) / n

    return SentimentScore(
        positive=avg_pos,
        negative=avg_neg,
        neutral=avg_neu,
        compound=avg_compound,
        label=_classify_compound(avg_compound),
    )


# ── Evidence integration ─────────────────────────────────────────────────────

def score_evidence_texts(evidence_items: list) -> SentimentScore:
    """
    Extract text from a list of EvidenceItems and return an aggregate sentiment.

    Uses each item's ``raw_snippet`` if available, otherwise its ``value``.
    Items with empty text are skipped.
    """
    texts: List[str] = []
    for item in evidence_items:
        text = getattr(item, "raw_snippet", None) or getattr(item, "value", None)
        if text and text.strip():
            texts.append(text)

    scores = analyze_batch(texts)
    return aggregate_sentiment(scores)
