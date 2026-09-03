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

## Step 7a: agent/config.py and agent/state.py (Person 1)
**Date:** 2026-08-30
**Status:** ✅ Complete — 47 tests passing

### What was built
- `agent/config.py` — `InvestigationConfig` dataclass with all runtime limits configurable via env vars
- `agent/state.py` — `InvestigationState` mutable dataclass tracking everything during one investigation run
- `tests/agent/test_config.py` — 25 tests (defaults, env overrides, validation, summary)
- `tests/agent/test_state.py` — 47 tests (search recording, source dedup, evidence/feature accumulation, stop conditions, status derivation, build_result)

### Module overview

**`agent/config.py`**
`InvestigationConfig` dataclass with:
- Investigation limits: `max_searches=15`, `max_sources=20`, `max_iterations=30`, `max_evidence_items=50`, `min_evidence_to_conclude=3`
- HTTP: `request_timeout_s=10.0`, `user_agent`, `respect_robots_txt`
- LLM: `llm_model="gpt-4o-mini"`, `llm_temperature=0.2`, `llm_max_tokens=2048`
- Search: `search_engine="duckduckgo"`, `parse_self_reported=True`
- All values overridable via env vars (`AGENT_MAX_SEARCHES`, `AGENT_LLM_MODEL`, etc.)
- `__post_init__` validates ranges; `summary()` for logging

**`agent/state.py`**
`InvestigationState` dataclass with:
- `record_search(query) → bool` — deduplicates by normalised lowercase; returns False if already searched
- `add_source(url) → bool` — deduplicates by normalised URL; returns False if already visited
- `add_evidence()`, `add_evidence_batch()`, `add_feature()` (replaces by name), `add_positive_signal()`, `add_risk_signal()`, `add_missing_info()` (dedup)
- `should_stop() → bool` — checks all 4 limits, sets `stop_reason`
- `has_sufficient_evidence()` — `reliable_evidence_count >= min_evidence_to_conclude`
- `_derive_status()` — maps stop_reason to `InvestigationStatus` enum
- `build_result(business_input) → InvestigationResult` — assembles final output (assessments left empty for ML layer)

### How to run tests
```bash
python -m pytest tests/agent/test_config.py tests/agent/test_state.py -v
```

---

## Step 7b: agent/tools/ — 6 investigation tools (Person 1)
**Date:** 2026-08-30
**Status:** ✅ Complete — 97 tests passing

### What was built
- `agent/tools/web_search.py` — DuckDuckGo HTML scraping (no API key)
- `agent/tools/webpage_extractor.py` — HTML fetch + clean text extraction
- `agent/tools/social_analyzer.py` — social media signals + Pakistani informal market demand phrases
- `agent/tools/review_analyzer.py` — star ratings, review counts, complaint signals
- `agent/tools/product_analyzer.py` — marketplace units sold, listing counts, price activity
- `agent/tools/user_input_parser.py` — self-reported business info → low-confidence evidence
- `tests/agent/test_tools.py` — 97 tests covering all 4 analysers and user input parser

### Module overview

**`web_search.py`** — `WebSearchTool(config, search_fn=None)`
- `search(query) → List[SearchResult]` — runs DuckDuckGo HTML search; injectable `search_fn` for tests
- `build_query(business_name, location, topic)` — formats search strings
- `SearchResult(url, title, snippet)` dataclass

**`webpage_extractor.py`** — `WebpageExtractor(config, fetch_fn=None)`
- `fetch(url) → Optional[PageContent]` — fetches URL, strips HTML, returns clean text; injectable `fetch_fn` for tests
- `fetch_multiple(urls) → list`
- `PageContent(url, title, text, word_count)` with `is_empty`, `truncated(max_chars)`

**`social_analyzer.py`** — `SocialAnalyzer`
- `analyze(text, url, platform) → List[EvidenceItem]` — extracts follower counts (k/M suffix parsed), engagement, recent-activity indicator, and informal demand signals (Pakistani market phrases: "taking orders", "DM for rates", "limited slots", "booking open", etc.)
- `detect_platform(url) → str` — guesses platform from URL
- Demand signal items always get `SourceReliability.LOW`; TikTok/WhatsApp get `LOW`; others `MEDIUM`

**`review_analyzer.py`** — `ReviewAnalyzer`
- `analyze(text, url, source_name) → List[EvidenceItem]` — extracts star rating, review count, review snippets (sentences with strong sentiment words), complaint signals
- `_reliability_for(source_name)` — Google → HIGH, Daraz → MEDIUM, OLX → LOW

**`product_analyzer.py`** — `ProductAnalyzer`
- `analyze(text, url, source_name) → List[EvidenceItem]` — extracts units sold, active listing count, price activity count (Rs./PKR patterns), listing freshness (today/hours ago/etc.)

**`user_input_parser.py`** — `UserInputParser` ← **Key for informal Pakistani micro-businesses**
- `parse(business_input) → List[EvidenceItem]` — processes `additional_info` + `description` fields
- Extracts: monthly revenue (Rs/PKR/k suffix), monthly orders (suits/pieces/items), years in business (direct or "since YYYY"), staff count (employees/tailors/stitchers), sales channels (WhatsApp/Instagram/Daraz/home-based/etc.), product types (lawn/kurta/bridal/embroidered/etc.)
- ALL output: `EvidenceType.INFERENCE`, `SourceReliability.LOW`, `confidence ≤ 0.50`

### How to run tests
```bash
python -m pytest tests/agent/test_tools.py -v
```

---

## Step 7c: agent/prompts/ — LLM prompt templates (Person 1)
**Date:** 2026-08-30
**Status:** ✅ Complete

### What was built
- `agent/prompts/templates.py` — 4 LLM prompt templates
- `agent/prompts/__init__.py` — package init

### Module overview

**`agent/prompts/templates.py`**
1. `investigation_plan_prompt(...)` — given business details and what has already been searched, asks LLM for 3 new search queries. Returns: JSON array of strings.
2. `extraction_prompt(...)` — given a URL, source type, and page text, asks LLM to extract structured evidence. Returns: JSON array of objects with `field_name/value/unit/evidence_type/confidence/raw_snippet`.
3. `feature_discovery_prompt(...)` — given business details and collected evidence, asks LLM to identify business signals. Returns: JSON array with `name/category/value/reason/confidence/searched`.
4. `assessment_prompt(...)` — asks LLM for a 2-sentence evidence-based justification.

---

## Step 7d: agent/agent.py — main orchestrator (Person 1)
**Date:** 2026-08-30
**Status:** ✅ Complete — 35 integration tests passing

### What was built
- `agent/agent.py` — `InvestigationAgent` class, the top-level entry point for all investigation logic
- `tests/agent/test_agent.py` — 35 tests covering URL classification, JSON parsing, smoke tests, self-reported evidence, social/marketplace extraction, search loop, missing info flagging, LLM evidence parsing, and full end-to-end flow

### Module overview

**`_classify_url(url) → str`** — classifies a URL as `social_media`, `marketplace`, `review_site`, `government`, or `general_web`

**`_extract_json_array(text) → list`** — safely parses the first JSON array from any LLM response string; returns `[]` on any failure

**`_LLMClient(config, call_fn=None)`** — wraps the LLM API. `call_fn=None` hits real OpenAI; inject any callable in tests to avoid API calls.

**`InvestigationAgent(config, llm_call_fn, search_fn, fetch_fn)`** — all dependencies injectable:
- `investigate(business_input) → InvestigationResult` — 5-step entry point:
  1. Parse self-reported info via `UserInputParser` (produces INFERENCE/LOW evidence)
  2. Fetch all user-provided URLs (website, social_links, marketplace_links)
  3. LLM-planned search → fetch → extract loop until stop condition
  4. LLM feature discovery from all accumulated evidence
  5. Rule-based signal building from evidence values
- `_build_signals()` — positive signals: rating ≥ 4.0, followers ≥ 1000, informal demand, units_sold ≥ 10; risk signals: complaint_signals present, rating < 2.5; missing info: SECP registration, reviews, social presence
- Result has no ML assessments filled — pass to `ml.assessment.generate_assessment()` next

### Full investigation pipeline
```
BusinessInput
     ↓ (UserInputParser)
 self-reported evidence (INFERENCE/LOW)
     ↓ (known URL fetch)
 social/marketplace/review evidence (OBSERVED/MEDIUM)
     ↓ (search → fetch → LLM extract loop)
 web evidence (OBSERVED/INFERENCE, various reliability)
     ↓ (LLM feature discovery)
 DiscoveredFeatures
     ↓ (rule-based signal building)
 InvestigationResult  →  ml.assessment.generate_assessment()
```

### How to run tests
```bash
python -m pytest tests/agent/test_agent.py -v
# All agent tests:
python -m pytest tests/agent/ -v
```

---

## Step 9b: SHAP Explainability Layer (Person 2)
**Date:** 2026-08-30
**Status:** ✅ Complete — 42 tests passing

### What was built
- `ml/explainability.py` (220 lines) — SHAP-based feature importance for both assessments
- `tests/ml/test_explainability.py` (260 lines) — 42 tests

### Module overview
Uses SHAP's `KernelExplainer` to approximate per-feature Shapley values for the trustworthiness and business potential scoring functions, with no trained model required — it wraps the deterministic risk engine functions directly.

1. **`FeatureImportance`** — dataclass holding one feature's name, its signed SHAP value (positive = raises score, negative = lowers score), and its actual value in the investigation.
2. **`ExplainabilityReport`** — holds two sorted lists of `FeatureImportance` (one per assessment), plus baseline scores (expected value over background) and predicted scores. Helper methods `top_trust_drivers(n)` and `top_potential_drivers(n)` return the top-n drivers.
3. **`_make_background()`** — builds a 20-row synthetic background dataset used as the SHAP reference distribution (mid-range "neutral" investigation values, seeded for reproducibility).
4. **`explain_assessment(result, n_samples=100)`** — main entry point. Extracts features, runs sentiment + credibility, wraps the two scoring functions for SHAP, and returns a fully populated `ExplainabilityReport`. Results are sorted by absolute SHAP value descending (biggest driver first).
5. **`format_report(report, top_n=5)`** — convenience function that returns a human-readable string of the top drivers in each assessment for logging or display.

### Test summary (42 tests)
- `FeatureImportance` structure and repr: 4 tests
- Background dataset shape, finiteness, range: 3 tests
- `ExplainabilityReport` structure (types, lengths, field types): 10 tests
- Score bounds (all scores 0–1): 4 tests
- Sort order (descending abs SHAP): 2 tests
- Top drivers helpers: 5 tests
- Feature names match extractor: 3 tests
- Empty evidence edge case: 4 tests
- `format_report` output: 7 tests

### How to run tests
```bash
python -m pytest tests/ml/test_explainability.py -v
# All ML tests:
python -m pytest tests/ml/ -v
```

### Using the explainability module
```python
from ml.explainability import explain_assessment, format_report

result = agent.invest(business_input)          # Person 1's output
enriched = generate_assessment(result)          # Person 2's ML overlay
report = explain_assessment(result, n_samples=100)

# Top 5 trust drivers
for fi in report.top_trust_drivers(5):
    print(fi.feature_name, fi.shap_value)

# Human-readable summary
print(format_report(report))
```

**Dependency added:** `shap==0.52.0`

---

## Step 10b: Feature Extractor Expanded to 60 Features (Person 2)
**Date:** 2026-08-30
**Status:** ✅ Complete — 56 tests passing (243 total ML tests passing)

### What was built
- `ml/feature_extractor.py` expanded — 3 new sub-extractors, 16 new features (44 → 60)
- `tests/ml/test_feature_extractor.py` updated — 12 new tests added (44 → 56 tests)

### New sub-extractors added

**`extract_evidence_type_ratios(evidence)`** — 4 features
Normalised ratios (0–1) of evidence by type: observed, corroborated, inference, unknown.
Complements the existing raw counts — comparisons across investigations of different sizes are now meaningful.

**`extract_sentiment_features(result)`** — 4 features
Runs the sentiment module over all evidence text and exposes `sentiment_positive`, `sentiment_negative`, `sentiment_neutral`, `sentiment_compound` as flat numeric features. Previously the risk engine used sentiment internally but it was invisible to the model.

**`extract_credibility_features(result)`** — 8 features
Exposes all 7 credibility sub-scores plus the overall credibility score: `credibility_source_reliability`, `credibility_evidence_quality`, `credibility_confidence`, `credibility_reliable_ratio`, `credibility_source_diversity`, `credibility_corroboration`, `credibility_evidence_depth`, `credibility_overall`. Previously also only used internally by the risk engine.

### Complete feature list (60 features)
| Group | Features | Count |
|---|---|---|
| Evidence counts | total, reliable, reliable_ratio, observed/corroborated/inference/unknown counts | 7 |
| Confidence stats | mean, min, max, std | 4 |
| Signal ratios | positive/risk counts, total, ratios, missing_info | 6 |
| Source reliability | high/medium/low/unknown ratios | 4 |
| Feature categories | total, found, found_ratio, searched, searched_ratio, 11 category counts | 16 |
| Investigation meta | searches, sources, unique_sources, 4 status flags | 7 |
| Evidence type ratios | observed/corroborated/inference/unknown ratios | 4 |
| Sentiment | positive, negative, neutral, compound | 4 |
| Credibility sub-scores | 7 sub-scores + overall | 8 |
| **Total** | | **60** |

### How to run tests
```bash
python -m pytest tests/ml/test_feature_extractor.py -v
# Full ML suite:
python -m pytest tests/ml/ -v
```

---

## Step 11b: Dataset Preparation + Model Training (Person 2)
**Status:** ✅ Complete — 52 tests passing
*(Entry backfilled 2026-09-03 — the step was completed earlier but its log entry was never written.)*

### What was built
- `ml/dataset.py` (495 lines) — converts `InvestigationResult` objects into a training-ready pandas DataFrame:
  - `build_dataset(labeled_samples)` — real-data path: list of `(result, trust_label, potential_label)` pairs → DataFrame of 60 features + both labels + `business_name`
  - `generate_synthetic_dataset(n_samples)` — synthetic Pakistani business profiles, structurally identical to real data; swap the generator for real data with zero other code changes
- `ml/model_trainer.py` (299 lines) — trains and persists 6 models:
  - Random Forest, Gradient Boosting, XGBoost × `trust_label` / `potential_label`
  - Per-model accuracy, F1, AUC-ROC and cross-validation in a training report; models saved to `data/models/`
- `tests/ml/test_model_trainer.py` — 52 tests

**Trained artifacts:** `data/models/` — 6 `.pkl` files (3 algorithms × 2 targets)

### How to run
```bash
python -m pytest tests/ml/test_model_trainer.py -v
```

**Next for Person 2:** Step 12b — model comparison and evaluation (`ml/model_evaluator.py`).

---

## Step 1c: Backend API Layer (Person 3)
**Date:** 2026-08-31
**Status:** ✅ Complete

### What was built
FastAPI service that connects the frontend to Person 1's agent and Person 2's ML layer through thin adapters — no investigation or scoring logic is duplicated in the backend.

**Endpoints** (`backend/app/api/routes.py`):
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/investigations` | Start an investigation (202 Accepted) |
| GET | `/api/investigations` | List investigation summaries |
| GET | `/api/investigations/{id}` | Full record (minus heavy result payload) |
| GET | `/api/investigations/{id}/status` | Poll status — `{id, status, phase_label, error}` |
| GET | `/api/investigations/{id}/result` | Aggregated report (409 while running, 422 if failed) |
| POST | `/api/investigations/{id}/save` | Save / unsave a report |
| POST | `/api/investigations/{id}/ask` | Ask OpenCredit — Q&A over a completed report |
| GET | `/api/health` | Service status + `llm_configured` flag |

**Architecture:**
- `backend/app/main.py` — app factory, CORS, router mounting, exception handlers
- `backend/app/services/investigations.py` — orchestration service: `ThreadPoolExecutor` runs agent + ML off the request thread; status flow `queued → investigating → analyzing → completed | partial | failed`. Phase labels come from the orchestrator only — no endpoint fabricates progress.
- `backend/app/services/adapters/agent_adapter.py` — wraps `InvestigationAgent.investigate()`; `investigate_fn` injectable for tests
- `backend/app/services/adapters/ml_adapter.py` — wraps `ml.assessment.generate_assessment()`; captures `generate_recommendation()`; injectable for tests
- `backend/app/services/storage.py` — JSON-file persistence under `data/investigations/` with atomic writes
- `backend/app/services/ask.py` — Ask OpenCredit: answers grounded in the stored report's evidence only, via OpenAI
- `backend/app/schemas.py` — request models with friendly validation ("Please enter a valid website URL." instead of raw Pydantic errors); converts once to Person 1's `BusinessInput` at the service boundary
- `backend/app/config.py` — pydantic-settings, `.env` support (CORS origins, storage dir, concurrency, Ask model)

**Files created:**
- `backend/app/main.py`, `backend/app/config.py`, `backend/app/schemas.py`
- `backend/app/api/routes.py`
- `backend/app/services/investigations.py`, `storage.py`, `ask.py`
- `backend/app/services/adapters/agent_adapter.py`, `ml_adapter.py`
- `requirements.txt` (repo-wide, includes Person 3 runtime deps), `.env.example`
- All package `__init__.py` files

**Dependencies added:** `fastapi==0.141.1`, `uvicorn==0.52.0`, `pydantic==2.13.4`, `pydantic-settings==2.14.2`, `python-dotenv==1.2.2`, `openai>=1.0.0`, dev: `httpx>=0.27.0`

---

## Step 2c: Backend Tests (Person 3)
**Date:** 2026-08-31
**Status:** ✅ Complete — 54 tests passing

### What was built
- `tests/backend/helpers.py` — shared fixtures: injected fake agent/ML functions, temp storage isolation, `wait_terminal()` poll helper
- `tests/backend/test_schemas.py` — validation behaviour (name required, URL cleaning, optional stripping, `to_business_input` conversion)
- `tests/backend/test_storage.py` — create/get/list/save round-trips, concurrent-write safety, corrupt-file tolerance
- `tests/backend/test_investigations.py` — full lifecycle: queued→investigating→analyzing→completed, agent failure → `failed`, ML failure → `partial`, recommendation capture, Ask gating
- `tests/backend/test_api.py` — all 8 endpoints over FastAPI TestClient: status codes (202/409/422/404/503), response shapes, no fabricated progress

All backend tests run against injected fakes — no network, no LLM key, no real agent run.

### How to run
```bash
python -m pytest tests/backend/ -v
```

---

## Step 3c: Frontend — Landing, Form, Live Report (Person 3)
**Date:** 2026-08-31
**Status:** ✅ Complete — production build passes (55 modules)

### What was built
React 18 + Vite + React Router SPA. Plain CSS with a design-token system (Ink `#071A21`, Deep Teal `#0C3435`, Accent Lime `#C7F36B`, Light `#F5F7F3`) and the OC monogram logo.

**Routes** (`frontend/src/App.jsx`):
| Path | Page | Purpose |
|---|---|---|
| `/` | `Landing` | Marketing page + demo report |
| `/new` | `NewInvestigation` | Business input form |
| `/investigation/:id` | `InvestigationRoom` | Live status — polls `/status` every 1.5s |
| `/report/:id` | `ReportPage` | Live report view |
| `*` | `Landing` | Fallback |

**Components** (12): `ReportView` (the full report renderer), `LevelGauge`, `EvidenceTable`, `EvidenceGraph` (SVG source→evidence→assessment map), `SignalCard`, `SourceList`, `CredibilityPanel`, `SentimentBar`, `AskPanel`, `Badges`, `Nav`, `Logo`.

**Key behaviours:**
- `ReportView` renders both the landing-page demo (from `data/exampleReport.js`, badged "Example — demo data") and live API results through one code path — the `demo` prop only controls the badge and Ask-panel disabled state, so demo and live can never visually diverge
- Investigation room shows only real phases from the status endpoint — nothing simulated; auto-navigates to the report on `completed`/`partial`
- Form validation mirrors backend messages; social/marketplace link fields are dynamic (add/remove)
- Save toggle (☆/★) and Ask OpenCredit panel (graceful 503 "no LLM key" note)
- Accessible: gauge/sentiment values exposed via aria-labels, focus-visible styles, skip-link

**Files created:** `frontend/src/main.jsx`, `App.jsx`, 4 pages, 12 components, `lib/api.js`, `lib/format.js`, `data/exampleReport.js`, `styles/tokens.css`, `styles/global.css`, `styles/app.css`; `frontend/index.html`, `frontend/package.json`, `vite.config.js` (dev proxy `/api` → `127.0.0.1:8000`)

### How to run
```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
npm run build      # production build check
```

---

## Step 4c: End-to-End Integration Test (Person 3)
**Date:** 2026-08-31
**Status:** ✅ Complete — full golden path verified in a real browser

### What was verified
Run with backend (uvicorn :8000) + frontend (Vite :5173, dev proxy) live, `llm_configured: false`:

1. **Landing page** — hero, sections, and demo report render; demo badge "Example — demo data" present; gauges have correct aria-labels; 6 evidence rows with Type/Reliability badges; evidence graph, missing-info, sources, credibility sub-scores, sentiment, disabled demo Ask panel all render
2. **Form validation** — empty submit → "Business name is required…"; invalid URL → "Please enter a valid website URL."
3. **Golden path** — submitted "Karachi Threads Test" (Karachi, Clothing & fashion, example.com, self-reported revenue/years/channels) → 202 → investigation room → auto-navigated to `/report/:id`
4. **Live report** — real Person 1 + Person 2 output: recommendation, both gauges, 2 inference evidence items derived from the self-reported text (correctly typed `Inference`, low reliability), example.com source "Visited — no extractable evidence", credibility sub-scores, sentiment, missing-info list
5. **Save toggle** — ☆ → ★ Saved (`aria-pressed` true) → back to ☆
6. **Ask 503 path** — question submitted → graceful "needs an LLM API key" note
7. **404 path** — unknown report id → "We couldn't find that investigation" with recovery links
8. **Slow investigation / room phases** — created via API with 3 non-routable URLs (~30s investigating): room showed real phases (done checkmark, active phase + spinner, todo phases), then completed and auto-navigated; all-unreachable URLs still yield an honest "Insufficient data" recommendation, not a failure
9. **Console** — clean; only Vite/React-DevTools info and React Router v7 future-flag warnings

Test data (`inv_1ac5027819`, `inv_2e758d4286`) was deleted after verification; dev servers stopped.

### Full-suite test result (this date)
`python -m pytest tests/` → **492 passed** (1 deprecation warning from FastAPI's TestClient).
Note: `tests/ml/test_explainability.py` and `tests/ml/test_model_trainer.py` fail collection in this environment (`ModuleNotFoundError: No module named 'numpy'` — Person 2's shap/sklearn stack is not installed on this Python 3.14 machine). Unrelated to Person 3 code; both modules are excluded from the 492 count.

---

## Step 12c: ML Stack Added to requirements.txt (Person 2)
**Date:** 2026-09-03
**Status:** ✅ Complete — full suite 587/587 passing

### What was done
- Restored the repo venv (Python 3.14) with the full dependency stack installed
- Added the ML/risk-assessment dependencies to `requirements.txt` with exact pins verified on Python 3.14:
  - `numpy==2.5.2`, `pandas==3.0.5`, `scikit-learn==1.9.0`, `xgboost==3.4.1`, `shap==0.52.0`
- A fresh `pip install -r requirements.txt` now installs everything needed to run the backend, the agent, the ML layer, **and** the full test suite — previously the ML stack was installed only ad hoc in a local venv

### Test result
`python -m pytest tests/` → **587 passed** (ml 295 · agent 237 · backend 55)

The previously uncollectable `tests/ml/test_explainability.py` (42 tests) and `tests/ml/test_model_trainer.py` (52 tests) are now part of the passing suite.

### Next steps (Person 2)
1. **Step 12b — `ml/model_evaluator.py`:** load the 6 saved models, compare accuracy / F1 / AUC-ROC / cross-validation, produce an evaluation report
2. **Wire trained models into live inference:** the backend currently uses only the rule-based risk engine; the 6 trained models in `data/models/` are not yet used in production scoring

---

## Step 12b: Model Evaluator (Person 2)
**Date:** 2026-09-03
**Status:** ✅ Complete — 59 tests passing (full suite 646/646)

### What was built
- `ml/model_evaluator.py` (496 lines) — head-to-head comparison of the 6 saved models
- `tests/ml/test_model_evaluator.py` (396 lines) — 59 tests
- `documents/MODEL_EVALUATION.md` — generated evaluation report artifact

### Module overview
`evaluate_saved_models()` loads the **saved production artifacts** from `data/models/` (not fresh fits) and evaluates each on a held-out test set, using the same split protocol and seed defaults as `train_all_models()` so metrics are directly comparable with the training report.

Metrics per model × target:
- Classification: accuracy, precision, recall, F1, ROC-AUC
- Probabilistic calibration: log loss, Brier score
- Stability: 5-fold cross-validated accuracy (mean ± std) of a cloned estimator
- Confusion counts (TP/TN/FP/FN) on the held-out set

`EvaluationReport` provides `ranking(target, metric)`, `best(target, metric)`, a console `summary()`, and a `markdown()` document. `feature_importance(model, target, top_n)` ranks features by tree importance. `write_markdown_report(report, path=None)` writes the full artifact (default `documents/MODEL_EVALUATION.md`) including the winning models' top features.

Supports real-data evaluation: pass a `build_dataset()` DataFrame to `evaluate_saved_models(df=...)`.

### Evaluation result (synthetic data, 300 samples, seed 42)
- Winner both targets: `random_forest` (ROC-AUC 1.000)
- All 3 algorithms reach perfect metrics — **expected**: the synthetic generator derives labels as deterministic functions of the 60 features, so this only confirms the algorithms learn the generator's rules. The generated report carries an explicit caveat; re-evaluate on real labeled data before trusting any model in production decisions.

### How to run
```bash
python -m pytest tests/ml/test_model_evaluator.py -v

# Regenerate the evaluation report:
python -c "from ml.model_evaluator import evaluate_saved_models, write_markdown_report; write_markdown_report(evaluate_saved_models())"
```

### Next for Person 2
Wire the trained models into live inference (blend model predictions with the rule-based risk engine inside `ml/assessment.py`, so Person 3's backend needs no changes).

---

## Step 13: Hybrid Inference — Trained Models Wired into the Risk Engine (Person 2)
**Date:** 2026-09-03
**Status:** ✅ Complete — 51 new tests (full suite 697/697)

### What was built
- `ml/model_predictor.py` (224 lines) — production inference over the saved models
- `ml/risk_engine.py` — hybrid blend of rule scores + model probabilities in `assess_risk()`
- `tests/ml/test_model_predictor.py` (366 lines) — 28 tests
- `tests/ml/test_risk_engine.py` — +23 hybrid-blend tests (52 total in file)
- `tests/ml/conftest.py` — determinism guard for the ML suite

### Module overview
`ModelPredictor` lazy-loads the saved artifacts from `data/models/` (default `random_forest` for both targets — the evaluation winners from Step 12b), thread-safe via a process-wide singleton `get_predictor()`. `canonical_feature_columns()` derives the 60-column training schema directly from `extract_features()` key order, so predictions always align with what the models were fitted on. `predict(features)` returns a frozen `ModelPrediction` (trust/potential probabilities, model names, availability, reason).

### Hybrid design (decision: hybrid with rules as safety net)
`assess_risk(result, predictor=None, model_weight=0.5)` — each final score = `model_weight × model probability + (1 − model_weight) × rule score`. Graceful degradation at every level:

- Models missing/corrupt on disk → predictor returns `available=False` → pure rule scores
- Predictor raises mid-prediction → caught, pure rule scores (an ML failure must never fail an investigation)
- No evidence → early `INSUFFICIENT_EVIDENCE` return, predictor never called

Transparency: `RiskAssessment.model_prediction` carries the full prediction; explanations append "score blends evidence-based rules with a trained ML model" whenever the models actually contributed. **Person 3's backend needed zero changes** — `generate_assessment()` flows through `assess_risk()` and picks the hybrid up automatically.

### Blend numbers (production models, 300 samples, seed 42)
| Fixture | Rule-only trust | Model trust | Blended trust | Rule-only potential | Model potential | Blended potential | Recommendation |
|---|---|---|---|---|---|---|---|
| Good business | 0.7164 (high) | 0.9400 | 0.8282 (high) | 0.7106 (high) | 0.5200 | 0.6153 (moderate) | approve_with_conditions |
| Risky business | 0.1030 (low) | 0.2600 | 0.1815 (low) | 0.0441 (low) | 0.1500 | 0.0971 (low) | decline |

Note: the model is conservative on potential for feature-light fixtures, pulling borderline-high potentials into "moderate" (→ conditional approval) — defensible for a lending product; the rules remain the evidence-grounded floor.

### Test determinism
`tests/ml/conftest.py` patches the default predictor to an unavailable stub, so every pre-existing `assess_risk()` test keeps asserting pure rule scores regardless of whether `data/models/*.pkl` exist on the machine. Hybrid behaviour is covered explicitly with injected stub predictors plus end-to-end tests against real trained models.

### How to run
```bash
python -m pytest tests/ml/test_model_predictor.py tests/ml/test_risk_engine.py -v

# Sanity-check the live hybrid path:
python -c "from ml.model_predictor import get_predictor; print(get_predictor().predict({}))"
```

### Next for Person 2
Documentation close-out: reflect the full ML pipeline (extractor → trainer → evaluator → predictor → hybrid engine) in `README.md` / `documents/CODEBASE_GUIDE.md`. Long-term: retrain on real labeled investigations and re-run the evaluator before trusting model weights in production.

---

## Step 14: Documentation Close-Out — ML Pipeline Documented (Person 2)
**Date:** 2026-09-03
**Status:** ✅ Complete — docs only, no code changes (suite still 697/697)

### What was updated
- `documents/CODEBASE_GUIDE.md` — added a full "Person 2 ML Files" section (all 10 `ml/` modules, in pipeline order), refreshed the libraries table to match `requirements.txt` (backend + document-processing + ML stack, pinned versions), removed the stale "will be needed in future" table, updated the Complete Data Flow to show the hybrid path with a backend-integration note, and refreshed the test section (697 tests, per-module commands, conftest determinism note, model-restore note)
- `README.md` — updated the repository-structure entry for `ml/` (full pipeline listing), fixed per-person test counts (backend 55 · agent 237 · ml 405 = 697), added "4. ML models (Person 2)" run section: hybrid blend description, model-optional fallback note, retrain + evaluation-report commands, synthetic-data caveat pointer
- `requirements.txt` — updated the ML-stack comment (587 → 697 tests passing)

### Key facts now documented for the team
- The hybrid: rules 50% + models 50%, models optional, graceful fallback to pure rules
- `data/models/` is restorable with one command after test runs overwrite it
- `tests/ml/conftest.py` keeps ordinary ML tests deterministic
- Person 3 needs zero backend changes to benefit from the trained models

### Person 2 status against the architecture
**100%.** All ML responsibilities are implemented, tested, wired into production, and documented: feature engineering (60 features), sentiment, credibility, hybrid risk engine, assessment wrapper with justification/recommendation, SHAP explainability, dataset + training (6 models), evaluation (with honest synthetic caveat), and live inference. Only long-term item remains: retrain on real labeled data when available.

---

