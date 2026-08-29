# OpenCredit AI — Development Log

---

## Step 1 — BusinessInput Schema

**Status:** Complete

**What was built:**
- `BusinessInput` Pydantic model — the structured entry point for all agent investigation.
- Covers all user-supplied fields: `name` (required), `location`, `category`, `website`, `social_links`, `marketplace_links`, `description`, `additional_info`.
- Validation: blank name rejected, whitespace-only optional strings coerced to `None`, invalid URLs rejected.
- Helper methods: `has_online_presence()`, `summary()`.

**Files created:**
- `agent/schemas/__init__.py`
- `agent/schemas/input.py`
- `tests/__init__.py`
- `tests/agent/__init__.py`
- `tests/agent/test_input.py`
- `documents/DEVELOPMENT_LOG.md` (this file)

**Dependencies installed:** `pydantic==2.13.5`, `pytest==9.1.1`

**Test result:** 15 / 15 PASSED

---

## Step 2 — Evidence and Feature Schemas

**Status:** Complete

**What was built:**
- `EvidenceItem` — a single collected fact with `field_name`, `value`, `evidence_type` (observed / corroborated / inference / unknown), `source_url`, `source_reliability`, and `confidence`.
- `EvidenceType` and `SourceReliability` enums enforce the project rule that inferences must never be silently promoted to verified facts.
- `DiscoveredFeature` — a business signal the agent identifies, with `name`, `category` (identity / reputation / audience / demand / risk / …), `value`, `reason` (mandatory), a list of backing `EvidenceItem`s, and `confidence`.
- `FeatureCategory` enum covers both trustworthiness and business-potential signal types.
- Helper methods: `is_reliable()`, `is_found()`, `evidence_type()`, `summary()`.

**Files created:**
- `agent/schemas/evidence.py`
- `agent/schemas/feature.py`
- `tests/agent/test_evidence.py`
- `tests/agent/test_feature.py`

**Test result:** 30 / 30 PASSED (0 warnings)

---

## Step 3 — InvestigationResult Schema

**Status:** Complete

**What was built:**
- `Signal` — a single positive or risk signal with label, detail, and evidence references.
- `AssessmentScore` — one assessment (trustworthiness OR business potential) with level enum, numeric score (0–1), evidence count, and explanation.
- `AssessmentLevel` enum: `high / moderate / low / insufficient_evidence`.
- `InvestigationStatus` enum: `complete / limit_reached / partial / failed`.
- `InvestigationResult` — the complete agent output envelope: echoes the input, records investigation metadata (searches, sources, timestamp), holds all evidence and features, carries both assessments separately, positive/risk signals, missing information list, source URLs, and a ~2-line justification.
- Updated `agent/schemas/__init__.py` to export all schema types from one import.
- Helpers: `has_sufficient_evidence`, `evidence_count_total`, `reliable_evidence()`, `get_features_by_category()`, `summary()`.

**Files created/modified:**
- `agent/schemas/result.py` (created)
- `agent/schemas/__init__.py` (updated — now exports all types)
- `tests/agent/test_result.py` (created)

**Test result:** 19 / 19 PASSED — full suite 64 / 64 PASSED (0 warnings)

---

## Step 4 — ML Feature Extractor (Person 2)

**Status:** Complete

**What was built:**
- `ml/feature_extractor.py` — converts `InvestigationResult` (from Person 1) into a flat `dict[str, float]` for tabular ML models.
- Six sub-extractors, each returning a `FeatureDict`:
  - `extract_evidence_counts()` — total/reliable counts, reliable ratio, EvidenceType breakdown (7 features)
  - `extract_confidence_stats()` — mean, min, max, std deviation of confidence scores (4 features)
  - `extract_signal_ratios()` — positive/risk signal counts and ratios, missing info count (6 features)
  - `extract_source_reliability()` — normalised HIGH/MEDIUM/LOW/UNKNOWN distribution (4 features)
  - `extract_feature_categories()` — total/found/searched features, count per FeatureCategory (16 features)
  - `extract_investigation_meta()` — searches, sources, one-hot InvestigationStatus flags (7 features)
- `extract_features()` — main entry point merging all sub-extractors. **Total: 44 numeric features.**

**Files created:**
- `ml/feature_extractor.py` (rewritten from empty)
- `tests/ml/__init__.py` (created)
- `tests/ml/test_feature_extractor.py` (created)

**Test result:** 44 / 44 PASSED — full suite 108 / 108 PASSED (0 warnings)

---

## Step 5 — Sentiment Analysis (Person 2)

**Status:** Complete

**What was built:**
- `ml/sentiment.py` — lexicon-based sentiment analysis for text evidence (reviews, social mentions, raw snippets).
- `SentimentScore` dataclass: positive, negative, neutral proportions (0–1), compound score (-1 to +1), human-readable label.
- `analyze_sentiment()` — tokenises text, looks up words in a curated lexicon (~65 words with weights -3 to +3), handles negation ("not good" flips polarity with 0.75x dampening), returns structured score.
- `_classify_sentiment()` — uses polarity strength and balance (not just compound direction) to distinguish "mixed" from purely positive/negative text.
- Compound formula: `(raw_pos - raw_neg) / 2.0` with clamping — preserves magnitude differences between strong and weak sentiment words.
- `analyze_batch()` — analyse multiple texts at once.
- `aggregate_sentiment()` — average multiple scores into a single summary.
- `score_evidence_texts()` — extracts text from `EvidenceItem.raw_snippet` or `.value`, analyses each, and returns an aggregate score.

**Files created:**
- `ml/sentiment.py` (rewritten from empty)
- `tests/ml/test_sentiment.py` (created)

**Test result:** 40 / 40 PASSED — full suite 148 / 148 PASSED (0 warnings)

---

## Step 6 — Credibility Scorer (Person 2)

**Status:** Complete

**What was built:**
- `ml/credibility_scorer.py` — evaluates the quality and reliability of an investigation's evidence.
- `CredibilityScore` dataclass: 7 sub-scores + weighted overall score + level ("high" / "moderate" / "low" / "insufficient_evidence").
- Sub-scorers (each 0.0–1.0):
  - `_score_source_reliability()` — weighted avg of HIGH/MEDIUM/LOW/UNKNOWN source reliability
  - `_score_evidence_quality()` — weighted avg of CORROBORATED/OBSERVED/INFERENCE evidence types
  - `_score_confidence()` — average agent confidence across evidence
  - `_score_reliable_ratio()` — proportion passing `is_reliable()`
  - `_score_source_diversity()` — unique named sources / total sources
  - `_score_corroboration()` — proportion of field_names backed by 2+ distinct sources
  - `_score_evidence_depth()` — sqrt-curve measuring evidence sufficiency (threshold: 10 items)
- `score_credibility()` — main entry point, weighted combination with level classification.

**Files created:**
- `ml/credibility_scorer.py` (rewritten from empty)
- `tests/ml/test_credibility_scorer.py` (created)

**Test result:** 47 / 47 PASSED — full suite 195 / 195 PASSED (0 warnings)

---

## Step 7b: Risk Engine (Person 2)
**Date:** 2026-08-29
**Status:** ✅ Complete — 29 tests passing

### What was built
- `ml/risk_engine.py` (227 lines) — central risk scoring engine
- `tests/ml/test_risk_engine.py` (316 lines) — 29 tests

### Module overview
The risk engine orchestrates all ML components to produce the two main assessments:

1. **Runs sub-modules** — calls `extract_features()`, `score_evidence_texts()`, `score_credibility()` on the `InvestigationResult`.
2. **Trustworthiness scoring** (6 weighted factors): credibility (35%), sentiment (15%), positive signals (15%), risk signals inverted (15%), reliable evidence (10%), source quality (10%).
3. **Business potential scoring** (5 weighted factors): positive signals (25%), sentiment (20%), credibility (15%), reliable evidence (15%), business features — audience/engagement/demand/growth/market (25%).
4. **Level mapping** — scores are converted to AssessmentLevel enum values (HIGH ≥ 0.70, MODERATE ≥ 0.45, LOW < 0.45) compatible with `InvestigationResult`.
5. **Explanation generation** — plain-English explanations for each assessment.
6. **Output** — `RiskAssessment` dataclass containing two `AssessmentScore` objects (directly compatible with the result schema), plus `CredibilityScore`, `SentimentScore`, and raw features for transparency.
7. **Edge cases** — no evidence returns `INSUFFICIENT_EVIDENCE` with `score=None`.

### Test summary (29 tests)
- RiskAssessment structure: 2 tests
- Score-to-level mapping: 7 tests
- Trustworthiness scoring: 3 tests (perfect, worst, bounded)
- Business potential scoring: 3 tests (strong, no features, bounded)
- Explanation generators: 4 tests
- Integration tests: 10 tests (no evidence, Pydantic validity, good/risky business, trust vs potential separation, features/credibility/sentiment populated, level consistency, evidence count)

### How to run tests
```bash
python -m pytest tests/ml/test_risk_engine.py -v
# All ML tests:
python -m pytest tests/ -v
```

### Running the risk engine
```python
from ml.risk_engine import assess_risk
assessment = assess_risk(investigation_result)
print(assessment.trustworthiness.level)      # e.g. "moderate"
print(assessment.business_potential.score)   # e.g. 0.72
print(assessment.trustworthiness.explanation)
```

---

## Step 8b: Assessment Wrapper (Person 2)
**Date:** 2026-08-29
**Status:** ✅ Complete — 34 tests passing

### What was built
- `ml/assessment.py` (179 lines) — final assessment wrapper
- `tests/ml/test_assessment.py` (320 lines) — 34 tests

### Module overview
The assessment wrapper is the single entry point for Person 3's backend:

1. **`generate_assessment(result)`** — takes a Person 1 `InvestigationResult` (with evidence, features, signals), runs the risk engine, generates a justification, and returns a new `InvestigationResult` with ML-generated assessments overlaid. All original data is preserved via `model_copy()`.
2. **`generate_justification(result, assessment)`** — produces an evidence-based ~2-line justification combining trustworthiness level, business potential level, and evidence counts. Never speculative.
3. **`generate_recommendation(assessment)`** — produces one of 5 recommendation strings based on the two assessment levels:
   - `"approve"` — both HIGH
   - `"approve_with_conditions"` — at least one MODERATE, none LOW
   - `"decline"` — trustworthiness LOW
   - `"further_review"` — potential LOW but trust adequate
   - `"insufficient_data"` — either INSUFFICIENT_EVIDENCE

### Test summary (34 tests)
- Justification generation: 7 tests (high, moderate, low, insufficient, format, evidence count)
- Recommendation logic: 11 tests (all 5 outcomes, edge cases)
- Integration tests: 16 tests (result type, assessments populated, justification generated, original data preserved — evidence/features/signals/business_input/status/sources, no evidence case, scores bounded, level-score consistency, JSON serialisation, summary method)

### How to run tests
```bash
python -m pytest tests/ml/test_assessment.py -v
# All ML tests:
python -m pytest tests/ml/ -v
```

### Using the assessment module
```python
from ml.assessment import generate_assessment

result = agent.invest(business_input)    # Person 1's output
enriched = generate_assessment(result)   # Person 2's ML overlay
print(enriched.trustworthiness.level)    # "moderate"
print(enriched.justification)            # evidence-based summary
json_output = enriched.model_dump_json() # ready for backend
```

---

## Next steps (pending approval)

**Person 1 — Step 7a: agent/config.py and agent/state.py**

- `config.py`: investigation limits (max searches, max sources, max iterations), LLM model name, timeouts — all configurable via environment variables.
- `state.py`: the mutable investigation state object — tracks what has been searched, what evidence has been collected, what features have been found, and when to stop.

---

