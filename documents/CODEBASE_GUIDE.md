# OpenCredit AI — Codebase Guide

This document explains every non-test source file that exists so far,
what it does, why it exists, and what lives inside it.
It also lists every library the project currently uses and why each one is needed.

---

## Libraries Required

### Currently installed

| Library | Version | Why it is needed |
|---|---|---|
| `pydantic` | 2.13.5 | Data validation and modelling. Every schema in the project is a Pydantic `BaseModel`. It automatically validates types, runs custom validators, and rejects bad input before it reaches agent logic. |
| `pytest` | 9.1.1 | Test runner. Used to run all files under `tests/`. |
| `scikit-learn` | 1.6.1 | ML model training (RandomForest, GradientBoosting) and evaluation. |
| `xgboost` | latest | Gradient-boosted tree model used alongside scikit-learn models. |
| `shap` | 0.52.0 | SHAP KernelExplainer — feature importance for trustworthiness and business potential scores. |
| `pandas` | 3.0.2 | DataFrame-based dataset building and ML feature pipeline. |
| `numpy` | 2.4.4 | Numerical operations for feature extraction and model training. |

### Will be needed in future steps (not yet installed)

| Library | Why it will be needed |
|---|---|
| `openai` / `anthropic` | The LLM provider the agent will call in production (all LLM calls are mocked in tests today). |
| `python-dotenv` | Loading API keys and config from a `.env` file so secrets are never hardcoded. |
| `fastapi` | The backend API server (Person 3's responsibility). |
| `uvicorn` | ASGI server that runs the FastAPI app. |
| `sqlalchemy` or `asyncpg` | Database access layer (Person 3). |

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
# From the repo root: C:\Users\HP\Desktop\openCredit-ai

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

# All ML tests:
python -m pytest tests/ml/ -v

# Everything:
python -m pytest tests/ -v
```

Expected result: **532 passed, 0 failures**

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
   InvestigationResult with trustworthiness + business_potential assessments
        ↓
   [ml.explainability.explain_assessment()]
   ExplainabilityReport (top SHAP drivers per assessment)
        ↓
   JSON output → backend / frontend
```

