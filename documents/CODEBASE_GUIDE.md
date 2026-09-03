# OpenCredit AI — Codebase Guide

This document explains every non-test source file that exists so far,
what it does, why it exists, and what lives inside it.
It also lists every library the project currently uses and why each one is needed.

**Coverage:** `agent/` (Person 1) and `ml/` (Person 2) are documented in full
below. Person 3's `backend/` and `frontend/` sources are not yet documented
in this guide — see `documents/DEVELOPMENT_LOG.md` for their build history
and `README.md` for how to run them.

---

## Libraries Required

All runtime dependencies are pinned in `requirements.txt`. Everything the
project uses today:

| Library | Version | Why it is needed |
|---|---|---|
| `pydantic` | 2.13.4 | Data validation and modelling. Every schema in the project is a Pydantic `BaseModel`. It automatically validates types, runs custom validators, and rejects bad input before it reaches agent logic. |
| `fastapi` | 0.141.1 | The backend API server (Person 3). |
| `uvicorn` | 0.52.0 | ASGI server that runs the FastAPI app (Person 3). |
| `pydantic-settings` | 2.14.2 | Typed settings (API keys, config) for the backend (Person 3). |
| `python-dotenv` | 1.2.2 | Loading API keys and config from a `.env` file so secrets are never hardcoded. |
| `openai` | ≥1.0.0 | LLM provider used by the investigation agent and "Ask OpenCredit". Optional — the system runs in limited mode without a key. |
| `python-multipart` | — | File uploads for the document-evidence endpoint (Person 3). |
| `pypdf`, `python-docx`, `openpyxl`, `pillow`, `pytesseract` | — | Document processing for uploaded evidence (PDF / DOCX / XLSX / images + OCR) (Person 3). |
| `numpy` | 2.5.2 | Numerical operations for feature extraction and model training. |
| `pandas` | 3.0.5 | DataFrame-based dataset building and ML feature pipeline. |
| `scikit-learn` | 1.9.0 | ML model training (RandomForest, GradientBoosting), evaluation metrics, and cross-validation. |
| `xgboost` | 3.4.1 | Gradient-boosted tree model used alongside scikit-learn models. |
| `shap` | 0.52.0 | SHAP KernelExplainer — feature importance for trustworthiness and business potential scores. |
| `pytest` | 9.1.1 | Test runner. Used to run all files under `tests/`. |
| `httpx` | ≥0.27.0 | HTTP client used by the backend test suite. |

---

## Source Files

### `agent/schemas/__init__.py`

**What it is:** Package initialiser and central export hub.

**What it does:** Re-exports every class and enum from the four schema files so that any other part of the codebase can import everything from one place:

```python
from agent.schemas import BusinessInput, InvestigationResult, EvidenceType
```

Without this file, every importer would have to know which sub-file each class lives in. With it, the internal file layout can change without breaking imports elsewhere.

---

### `agent/schemas/input.py`

**What it is:** The front door — defines what the user tells the system about a business.

**What it does:** Contains the `BusinessInput` Pydantic model. This is the very first object created when a user submits a business for investigation. It holds:

- `name` — the only required field; everything else is optional
- `location`, `category` — where the business is and what it does
- `website`, `social_links`, `marketplace_links` — online pointers the agent will visit
- `description`, `additional_info` — anything the user wants to add in plain text (revenue, staff count, etc.)

**Validation rules built in:**
- Blank name is rejected outright
- Whitespace-only optional strings are silently converted to `None` (treated as "not provided")
- Any URL that is not a valid HTTP/HTTPS URL is rejected

**Helper methods:**
- `has_online_presence()` — returns `True` if the user gave at least one URL; tells the agent whether it has starting points or must search from scratch
- `summary()` — one-line string of the most important fields, used in agent prompts and logs

---

### `agent/schemas/evidence.py`

**What it is:** The atomic unit of collected information — one fact the agent found.

**What it does:** Contains three things:

**`EvidenceType` enum** — enforces the project's core rule that inferences must never be treated as verified facts. Every piece of evidence must be labelled as one of:
- `observed` — the agent read this directly from a source
- `corroborated` — two or more independent sources confirm the same fact
- `inference` — the agent concluded this from other evidence; it was not stated anywhere
- `unknown` — insufficient information found

**`SourceReliability` enum** — grades where the information came from:
- `high` — official or first-party source (the business's own site, a government registry)
- `medium` — independent third-party (a news article, a review platform)
- `low` — user-generated or unverified (a forum post, a social comment)
- `unknown` — source quality could not be assessed

**`EvidenceItem` model** — stores a single collected fact with:
- `field_name` + `value` + optional `unit` — what was found and in what units
- `evidence_type` + `confidence` (0.0–1.0) — how certain we are
- `source_url`, `source_name`, `source_reliability` — where it came from
- `raw_snippet` — the exact text extracted, preserved for auditability
- `collected_at` — UTC timestamp set automatically when the item is created

**Helper methods:**
- `is_reliable()` — returns `True` only when confidence ≥ 0.7, source is HIGH or MEDIUM, and type is OBSERVED or CORROBORATED. Used to filter evidence before scoring.
- `summary()` — one-line string for logging

---

### `agent/schemas/feature.py`

**What it is:** A business signal — something meaningful the agent identified about the business.

**What it does:** Contains two things:

**`FeatureCategory` enum** — classifies what dimension of the business a signal relates to:
- Trustworthiness signals: `identity`, `reputation`, `transparency`, `history`
- Business potential signals: `audience`, `engagement`, `demand`, `growth`, `market_presence`
- Cross-cutting: `risk`, `unknown`

**`DiscoveredFeature` model** — stores one business signal with:
- `name` — a machine-readable identifier, e.g. `instagram_follower_count`
- `category` — which of the above categories it belongs to
- `value` + `unit` — what was found; `None` means the agent looked and found nothing
- `reason` — **mandatory** — the agent must explain why this signal is relevant to this business; prevents the LLM from inventing arbitrary signals
- `evidence` — the list of `EvidenceItem`s that back this feature; no unsupported claims
- `confidence` (0.0–1.0) — how certain the agent is about the feature's value
- `searched` — distinguishes "agent looked and found nothing" from "agent never looked"

**Helper methods:**
- `is_found()` — `True` if a value was found
- `evidence_type()` — returns the strongest evidence type across all backing items (corroborated beats observed, which beats inference, which beats unknown)
- `summary()` — one-line string for logging

---

### `agent/schemas/result.py`

**What it is:** The complete output of one investigation — the contract between Person 1 and everyone else.

**What it does:** Contains five things:

**`AssessmentLevel` enum** — the four possible verdicts for each assessment:
`high`, `moderate`, `low`, `insufficient_evidence`

**`InvestigationStatus` enum** — how the investigation ended:
- `complete` — agent finished normally
- `limit_reached` — hit the configured maximum searches or iterations
- `partial` — some sources failed but others succeeded
- `failed` — agent could not complete the investigation at all

**`Signal` model** — one positive or risk signal surfaced for the user:
- `label` — short display label, e.g. "Active Instagram presence"
- `detail` — one sentence explaining why this matters
- `evidence_refs` — list of `field_name` values from the evidence that support this signal

**`AssessmentScore` model** — one of the two assessments (trustworthiness or business potential):
- `level` — the `AssessmentLevel` verdict
- `score` — optional 0.0–1.0 number; `None` when evidence is insufficient to produce a number
- `evidence_count` — how many reliable items backed this assessment
- `explanation` — plain-English description of how the level was reached

**`InvestigationResult` model** — the full envelope returned by the agent:
- Echoes `business_input` so the result is fully self-contained
- Investigation metadata: `status`, `searches_performed`, `sources_examined`, `investigated_at`
- All raw data: `evidence` list, `features` list
- Two independent assessments: `trustworthiness` and `business_potential` (kept separate by design — a business can score high on one and low on the other)
- `positive_signals` and `risk_signals` lists
- `missing_information` — things the agent looked for but could not find
- `sources` — all unique URLs the agent visited
- `justification` — approximately 2-line evidence-based summary for the user

**Helper properties/methods:**
- `evidence_count_total` — total number of evidence items collected
- `has_sufficient_evidence` — `True` if at least one assessment has a real score
- `reliable_evidence()` — filters evidence to only items that pass `is_reliable()`
- `get_features_by_category(category)` — returns all features of a given type
- `summary()` — one-line string showing name, both assessment levels, evidence count, and status

---

## How the Files Relate to Each Other

```
User submits a business
        ↓
   BusinessInput          ← agent/schemas/input.py
   (validated entry)
        ↓
   Agent investigates
        ↓
   EvidenceItem ×N        ← agent/schemas/evidence.py
   (one fact each)
        ↓
   DiscoveredFeature ×N   ← agent/schemas/feature.py
   (one signal each, backed by evidence)
        ↓
   InvestigationResult    ← agent/schemas/result.py
   (full output — passed to ML layer and backend)
```

The `__init__.py` lets any file in the project import all of these types from:

```python
from agent.schemas import BusinessInput, InvestigationResult, EvidenceType
```

---

## How to Run the Tests

```bash
# From the repo root

# All agent tests (schemas + config + state + tools + agent):
python -m pytest tests/agent/ -v

# Individual schema test files:
python -m pytest tests/agent/test_input.py -v
python -m pytest tests/agent/test_evidence.py -v
python -m pytest tests/agent/test_feature.py -v
python -m pytest tests/agent/test_result.py -v

# Person 1 agent files:
python -m pytest tests/agent/test_config.py -v
python -m pytest tests/agent/test_state.py -v
python -m pytest tests/agent/test_tools.py -v
python -m pytest tests/agent/test_agent.py -v

# All ML tests (Person 2):
python -m pytest tests/ml/ -v

# ML per module:
python -m pytest tests/ml/test_feature_extractor.py -v
python -m pytest tests/ml/test_sentiment.py -v
python -m pytest tests/ml/test_credibility_scorer.py -v
python -m pytest tests/ml/test_risk_engine.py -v            # incl. hybrid blend
python -m pytest tests/ml/test_assessment.py -v
python -m pytest tests/ml/test_explainability.py -v
python -m pytest tests/ml/test_model_trainer.py -v
python -m pytest tests/ml/test_model_evaluator.py -v
python -m pytest tests/ml/test_model_predictor.py -v        # trains its own models

# Everything:
python -m pytest tests/ -v
```

Expected result: **697 passed, 0 failures** (ml 405 · agent 237 · backend 55).

Notes:
- `tests/ml/conftest.py` pins the default predictor to an unavailable stub,
  so ordinary ML tests are deterministic (pure rule scores) regardless of
  whether `data/models/` exists on the machine; hybrid behaviour has its own
  explicit tests.
- `test_model_predictor.py` and `test_model_evaluator.py` train small models
  into `data/models/` while running. To restore the production artifacts
  afterwards: `python -c "from ml.model_trainer import train_all_models; train_all_models()"`.

---

## Person 1 Agent Files

### `agent/config.py`

**What it is:** All runtime limits and model settings in one place.

**What it does:** Defines `InvestigationConfig`, a Python dataclass where every field has a sensible default and can be overridden by environment variable. No hardcoded values — deployments tune behaviour without touching code.

**Key fields:** `max_searches` (15), `max_sources` (20), `max_iterations` (30), `max_evidence_items` (50), `min_evidence_to_conclude` (3), `request_timeout_s` (10s), `llm_model` ("gpt-4o-mini"), `llm_temperature` (0.2), `search_engine` ("duckduckgo"), `parse_self_reported` (True).

**Validation:** `__post_init__` rejects impossible values (e.g. `max_searches=0`, `temperature=3.0`). `summary()` prints a one-line log-friendly string.

**Passes its data to:** `InvestigationState` (limits), `WebSearchTool`, `WebpageExtractor` (timeout/user-agent), `InvestigationAgent` (LLM settings).

---

### `agent/state.py`

**What it is:** The mutable investigation notebook — everything the agent has done and found so far.

**What it does:** `InvestigationState` is created fresh per investigation. It tracks search queries issued (deduplicated), URLs visited (deduplicated), all `EvidenceItem`s and `DiscoveredFeature`s collected, positive and risk signals, and missing information items.

**Key methods:**
- `record_search(query) → bool` — normalises query (lowercase, stripped), returns `False` if already searched; increments counter only on new queries.
- `add_source(url) → bool` — normalises URL (strip trailing `/`), returns `False` if already visited.
- `add_feature(feature)` — adds or **replaces** a feature by name (prevents duplicates when LLM re-discovers the same signal).
- `should_stop() → bool` — checks all 4 configured limits; sets `stop_reason` when any is hit.
- `build_result(business_input) → InvestigationResult` — assembles the final output; assessments are left at `INSUFFICIENT_EVIDENCE` for the ML layer to fill.

**Passes its data to:** `InvestigationAgent._run_investigation_loop()` reads and mutates it; `build_result()` produces the `InvestigationResult` consumed by `ml.assessment`.

---

### `agent/tools/web_search.py`

**What it is:** Keyword search — given a query, returns URLs and snippets.

**What it does:** `WebSearchTool` wraps DuckDuckGo's public HTML endpoint (no API key). In tests, inject `search_fn(query, max_results) → List[SearchResult]` to avoid network calls.

**Key method:** `search(query) → List[SearchResult]` — never raises; returns `[]` on any failure.

**Passes its data to:** `InvestigationAgent._run_investigation_loop()` uses the returned URLs to call `WebpageExtractor.fetch()`.

---

### `agent/tools/webpage_extractor.py`

**What it is:** Page fetcher and HTML cleaner.

**What it does:** `WebpageExtractor` fetches a URL (caps at 500 KB), strips all HTML tags, removes `<script>` and `<style>` blocks, decodes HTML entities, and collapses whitespace. Returns a `PageContent(url, title, text, word_count)` object. In tests, inject `fetch_fn(url) → str` returning raw HTML.

**Key method:** `fetch(url) → Optional[PageContent]` — never raises; returns `None` on failure.

**Passes its data to:** `InvestigationAgent._extract_from_page()` passes `page.text` to the analyser tools and `page.truncated(3000)` to the LLM extraction prompt.

---

### `agent/tools/social_analyzer.py`

**What it is:** Social media signal extractor with Pakistani informal market awareness.

**What it does:** `SocialAnalyzer.analyze(text, url, platform)` extracts:
- **Follower/subscriber counts** — parses k/K/M suffix correctly (12k → 12000)
- **Engagement signals** — likes, comments, shares, views
- **Recent activity** — "posted 2 hours ago", "last post", etc.
- **Informal demand signals** — critical for Pakistani micro-businesses: "taking orders", "DM for rates", "limited slots", "booking open", "custom stitching", etc. → field: `informal_demand_signal`

Demand signals always get `SourceReliability.LOW`; TikTok/WhatsApp get `LOW`; Instagram/Facebook/Twitter get `MEDIUM`.

**Passes its data to:** `InvestigationState.add_evidence_batch()` → eventually `InvestigationResult.evidence`.

---

### `agent/tools/review_analyzer.py`

**What it is:** Review and rating extractor.

**What it does:** `ReviewAnalyzer.analyze(text, url, source_name)` extracts:
- `star_rating` — "4.5 out of 5", "4.5/5", "Rated 4.5 stars"
- `review_count` — "1,234 customer reviews"
- `review_snippets` — sentences containing strong positive/negative sentiment words (for the sentiment module)
- `complaint_signals` — complaint, fraud, scam, dispute, refund issue

`_reliability_for(source_name)` maps Google → HIGH, Daraz/Trustpilot → MEDIUM, OLX → LOW.

**Passes its data to:** `InvestigationState.add_evidence_batch()`. `review_snippets` is also consumed by `ml.sentiment.score_evidence_texts()`.

---

### `agent/tools/product_analyzer.py`

**What it is:** Marketplace product and demand evidence extractor.

**What it does:** `ProductAnalyzer.analyze(text, url, source_name)` extracts:
- `units_sold` — "1,234 sold", "500+ sold"
- `active_listing_count` — "234 products available"
- `marketplace_price_activity` — counts Rs./PKR price occurrences (indicates active commerce)
- `listing_freshness` — "today", "2 hours ago", "yesterday"

Daraz → MEDIUM reliability; unknown platforms → LOW.

**Passes its data to:** `InvestigationState.add_evidence_batch()`.

---

### `agent/tools/user_input_parser.py`

**What it is:** Self-reported information extractor — critical for informal Pakistani micro-businesses with no online presence.

**What it does:** `UserInputParser.parse(business_input)` processes the `additional_info` and `description` text fields of `BusinessInput` and extracts:
- `self_reported_monthly_revenue` — "Rs 50,000 per month", "PKR 30k monthly"
- `self_reported_monthly_orders` — "40 suits per month", "50 pieces monthly"
- `self_reported_years_in_business` — "5 years" or "since 2018"
- `self_reported_staff_count` — "5 employees", "3 tailors"
- `self_reported_sales_channels` — WhatsApp, Instagram, Daraz, home-based, etc.
- `self_reported_product_types` — lawn, kurta, bridal, embroidered, etc.

**All output is marked:** `EvidenceType.INFERENCE`, `SourceReliability.LOW`, `confidence ≤ 0.50`. The scoring layer weights these appropriately — they inform the model but cannot anchor a high-confidence assessment on their own.

**Passes its data to:** `InvestigationState.add_evidence_batch()`.

---

### `agent/prompts/templates.py`

**What it is:** The words the agent uses when talking to the LLM.

**What it does:** Four functions that return formatted prompt strings:
1. `investigation_plan_prompt(...)` → asks LLM for 3 new search queries; LLM returns a JSON array of strings.
2. `extraction_prompt(...)` → asks LLM to extract structured evidence from a page; LLM returns a JSON array of `{field_name, value, unit, evidence_type, confidence, raw_snippet}` objects.
3. `feature_discovery_prompt(...)` → asks LLM to identify business signals from all collected evidence; LLM returns a JSON array of `{name, category, value, reason, confidence, searched}` objects.
4. `assessment_prompt(...)` → asks LLM for a 2-sentence evidence-based justification.

**Passes its data to:** `InvestigationAgent._plan_searches()`, `_extract_from_page()`, `_discover_features()`.

---

### `agent/agent.py`

**What it is:** The main orchestrator — the single entry point that runs a complete business investigation.

**What it does:** `InvestigationAgent.investigate(business_input) → InvestigationResult` runs 5 steps:
1. **Self-reported parse** — calls `UserInputParser` immediately; even informal verbal-only businesses get evidence.
2. **Known URL fetch** — visits `website`, `social_links`, `marketplace_links` from `BusinessInput`; dispatches to the right analyser by URL type.
3. **Search loop** — calls `_plan_searches()` (LLM), executes each query via `WebSearchTool`, fetches each result via `WebpageExtractor`, extracts evidence via analysers + LLM extraction prompt. Stops when any limit is hit or all new queries are exhausted.
4. **Feature discovery** — calls LLM with all collected evidence to produce `DiscoveredFeature` objects; added to state by name (dedup).
5. **Signal building** — rule-based pass: high rating → positive signal; followers ≥ 1000 → positive; informal demand → positive; units sold ≥ 10 → positive; complaint signals → risk; low rating → risk. Missing info flags: SECP registration, reviews, social presence.

All 3 external dependencies (LLM, search, fetch) are injectable — pass mock functions to `llm_call_fn`, `search_fn`, `fetch_fn` for testing with zero real API calls.

**Passes its data to:** Returns `InvestigationResult` → pass to `ml.assessment.generate_assessment()`.

---

## Person 2 ML Files

The ML layer turns Person 1's `InvestigationResult` into the two final
assessments. It is a **hybrid**: a transparent rule engine (the safety net,
always available) blended 50/50 with trained classifiers (loaded from
`data/models/` when present). All modules degrade gracefully — an ML failure
never fails an investigation.

```text
InvestigationResult
        ↓
  feature_extractor     60 numeric features
  sentiment             lexicon sentiment of evidence text
  credibility_scorer    7-dimension evidence credibility
        ↓
  risk_engine           hybrid: rule scores + model probabilities
        ↓
  assessment            justification + recommendation (Person 3 entry point)
        ↓
  explainability        SHAP drivers (optional, backend-side)
```

Training/evaluation (offline, not in the request path):
`dataset` → `model_trainer` → `model_evaluator` → `data/models/*.pkl` →
`model_predictor` (loaded at inference time).

---

### `ml/feature_extractor.py`

**What it is:** The bridge between structured evidence and numbers.

**What it does:** `extract_features(result) → dict[str, float]` converts an
`InvestigationResult` into exactly **60 numeric features** — the training
schema for the models and the input for the rule engine. Covers:

- Evidence counts and ratios (total, reliable, corroborated, per reliability level, per evidence type)
- Signal ratios (positive/risk/missing, normalised)
- Confidence statistics (mean/weighted by reliability)
- Feature-category counts (audience, engagement, demand, growth, market presence, reputation, …)
- Investigation metadata (searches performed, sources examined, unique sources)
- One-hot investigation-status flags
- Self-reported fields parsed by `UserInputParser` (revenue, orders, years in business, staff)

**Why it matters:** the extractor's key order *is* the training column order —
`ml.model_predictor.canonical_feature_columns()` relies on this to align live
predictions with what the models were fitted on.

---

### `ml/sentiment.py`

**What it is:** Lexicon-based sentiment analysis of evidence text.

**What it does:** `analyze_sentiment(text) → SentimentScore` scores one text;
`score_evidence_texts(evidence) → SentimentScore` aggregates every
`raw_snippet` on the evidence list. Returns positive/negative/neutral
proportions, a `compound` score in [-1, +1], and a label.

**Passes its data to:** the risk engine uses `compound` (clamped to [0, 1]) as
one weighted factor in both assessments; `explainability` uses the same score.

---

### `ml/credibility_scorer.py`

**What it is:** How trustworthy the *evidence itself* is.

**What it does:** `score_credibility(result) → CredibilityScore` computes 7
sub-scores — source reliability, evidence quality, confidence, reliable ratio,
source diversity, corroboration, evidence depth — and combines them into an
`overall_score` (0.0–1.0) with a `high`/`moderate`/`low` level.

**Passes its data to:** the risk engine weights `overall_score` at 35% of
trustworthiness and 15% of business potential.

---

### `ml/risk_engine.py`

**What it is:** The central scoring module — the hybrid brain.

**What it does:** `assess_risk(result, predictor=None, model_weight=0.5) →
RiskAssessment` orchestrates the three modules above and produces the two
`AssessmentScore` objects directly compatible with `InvestigationResult`:

1. **Rule scores** (deterministic, always available):
   - *Trustworthiness* — credibility 35%, sentiment 15%, positive signals 15%, inverted risk signals 15%, reliable evidence 10%, source quality 10%
   - *Business potential* — positive signals 25%, sentiment 20%, credibility 15%, reliable evidence 15%, business features 25%
2. **Model scores** — `ModelPredictor.predict(features)` returns
   `predict_proba` scores for both targets (see below).
3. **Blend** — `final = model_weight × model + (1 − model_weight) × rule`,
   clamped to [0, 1], then mapped to a level (high ≥ 0.70, moderate ≥ 0.45,
   otherwise low).

**Graceful degradation:** models missing/corrupt → pure rule scores;
predictor raises → caught, pure rule scores; no evidence at all → early
`INSUFFICIENT_EVIDENCE` return (the predictor is never even called).

**Transparency:** the returned `RiskAssessment.model_prediction` carries the
full prediction (scores, model names, availability, reason), and both
explanations disclose when a trained model contributed.

---

### `ml/assessment.py`

**What it is:** The final wrapper — Person 3's single entry point.

**What it does:** `generate_assessment(result) → InvestigationResult` runs the
risk engine, then generates a two-line evidence-based `justification` and a
`recommendation` (`approve` / `approve_with_conditions` / `decline` /
`further_review` / `insufficient_data`). Returns a **new** result via
`model_copy(update={...})` — the input is never mutated.

`generate_recommendation(risk)` is also exported separately; the backend calls
it to build its response payload.

---

### `ml/explainability.py`

**What it is:** SHAP-based "why did it score this way?" report.

**What it does:** `explain_assessment(result) → ExplainabilityReport` wraps
the *rule* scoring functions in SHAP's `KernelExplainer` (they are
deterministic functions, so Shapley values are exact) and returns per-feature
signed contributions, sorted by impact, plus baselines and predicted scores
for both assessments. `format_report()` renders a human-readable summary.

---

### `ml/dataset.py`

**What it is:** Synthetic training data generator.

**What it does:** `generate_synthetic_dataset(n_samples, seed)` builds a
DataFrame of plausible investigations from profile archetypes
(trustworthy / medium / risky) and derives `trust_label` and
`potential_label` as deterministic functions of the 60 features.
`build_dataset(df=None, ...)` uses a real `InvestigationResult` list when
available, falling back to synthetic. `split_features_labels(df, target)` and
`get_feature_columns(df)` feed the trainer.

**Important:** the labels are generator rules, not real-world outcomes — see
the caveat in `documents/MODEL_EVALUATION.md`.

---

### `ml/model_trainer.py`

**What it is:** Trains and persists the six candidate models.

**What it does:** `train_all_models(...) → TrainingReport` fits
`random_forest`, `gradient_boosting`, and `xgboost` against both targets with
a fixed protocol (seed 42, stratified 80/20 split), reports accuracy /
ROC-AUC / log loss, and saves each classifier to
`data/models/<name>__<target>.pkl`. `load_model(name, target)` restores a
saved artifact; `predict` / `predict_proba` are convenience wrappers.

---

### `ml/model_evaluator.py`

**What it is:** Head-to-head comparison of the saved artifacts.

**What it does:** `evaluate_saved_models(...)` loads the six `.pkl` files
(what production actually uses), scores accuracy / precision / recall / F1 /
ROC-AUC / log loss / Brier, runs 5-fold CV on cloned estimators, extracts
confusion counts and tree feature importances, and produces an
`EvaluationReport` with rankings. `write_markdown_report(report)` writes
`documents/MODEL_EVALUATION.md`. Winner (synthetic data): `random_forest`
for both targets — with an explicit synthetic-data caveat, because all three
algorithms reach perfect metrics when labels are deterministic.

---

### `ml/model_predictor.py`

**What it is:** Production inference — the trained models in the live path.

**What it does:** `ModelPredictor` lazy-loads the saved artifacts (default
`random_forest` for both targets, the evaluation winners), thread-safe via
the process-wide singleton `get_predictor()`. `predict(features) →
ModelPrediction` aligns any feature dict to the canonical 60-column training
schema (missing keys → 0.0, extras ignored) and returns calibrated
probabilities for both targets plus availability info. If the artifacts are
missing or corrupt, `predict()` returns `available=False` with the reason —
it never raises, so the hybrid blend in the risk engine falls back to pure
rule scores.

---

## Complete Data Flow

```
User submits BusinessInput
        ↓
   [UserInputParser]
   self-reported evidence (INFERENCE/LOW/≤0.50 confidence)
        ↓
   [WebpageExtractor → SocialAnalyzer / ReviewAnalyzer / ProductAnalyzer]
   website + social + marketplace evidence (OBSERVED/MEDIUM)
        ↓
   [LLM → WebSearchTool → WebpageExtractor → all analysers → LLM extraction]
   web evidence (mixed types and reliability)
        ↓
   [InvestigationState.build_result()]
   InvestigationResult (evidence + features, no assessments yet)
        ↓
   [ml.assessment.generate_assessment()]
   [ml.risk_engine.assess_risk()  ← hybrid]
        ├─ feature_extractor → 60 numeric features
        ├─ sentiment + credibility_scorer → rule scores (safety net)
        └─ model_predictor → saved random_forest probabilities
   InvestigationResult with trustworthiness + business_potential assessments
        ↓
   [ml.explainability.explain_assessment()]
   ExplainabilityReport (top SHAP drivers per assessment)
        ↓
   JSON output → backend / frontend
```

**Backend integration note:** `backend/app/services/adapters/ml_adapter.py`
calls `generate_assessment()`, `assess_risk()`, and `generate_recommendation()`
— the hybrid is picked up automatically. No backend changes are required to
use the trained models, and the system keeps working if `data/models/` is
absent (pure rule scores).



