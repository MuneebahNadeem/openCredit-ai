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

### Will be needed in future steps (not yet installed)

| Library | Why it will be needed |
|---|---|
| `langchain` or `langgraph` | Orchestrating the investigation agent — managing LLM calls, tool use, and the search loop. |
| `openai` / `anthropic` | The LLM provider the agent will use to reason about businesses and decide what to search for next. |
| `httpx` or `requests` | Making HTTP requests to fetch web pages during investigation. |
| `beautifulsoup4` | Parsing HTML from fetched web pages to extract useful text. |
| `python-dotenv` | Loading API keys and config from a `.env` file so secrets are never hardcoded. |
| `fastapi` | The backend API server (Person 3's responsibility). |
| `uvicorn` | ASGI server that runs the FastAPI app. |
| `sqlalchemy` or `asyncpg` | Database access layer (Person 3). |
| `scikit-learn` / `xgboost` | ML model training and evaluation (Person 2). |
| `shap` | Explainability for ML model outputs (Person 2). |
| `pandas` / `numpy` | Data manipulation for the ML pipeline (Person 2). |

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

# All agent tests:
python -m pytest tests/agent/ -v

# Individual test files:
python -m pytest tests/agent/test_input.py -v
python -m pytest tests/agent/test_evidence.py -v
python -m pytest tests/agent/test_feature.py -v
python -m pytest tests/agent/test_result.py -v
```

Expected result: **64 passed, 0 warnings**
