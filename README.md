# OpenCredit AI

OpenCredit AI is an AI-powered business investigation and decision-support system designed to help evaluate whether a business or brand is trustworthy and potentially beneficial to work with or invest in.

The system combines **user-provided business information, publicly available internet data, AI-based investigation, feature discovery, evidence analysis, and machine learning** to produce an explainable assessment.

The system does not rely only on a fixed set of features. It has predefined business signals, while the investigation agent can also discover additional relevant signals based on the specific business being investigated.

---

## Project Workflow

```text
User Business Information
          │
          ▼
┌─────────────────────────┐
│   Investigation Agent   │
│        Person 1         │
└────────────┬────────────┘
             │
             ▼
     Search Public Web
             │
             ▼
      Extract Information
             │
             ▼
     Discover New Features
             │
             ▼
       Cross-Check Data
             │
             ▼
      Business Evidence
             │
             ▼
┌─────────────────────────┐
│   ML / Risk Analysis    │
│        Person 2         │
└────────────┬────────────┘
             │
             ▼
     Risk / Feature Analysis
             │
             ▼
┌─────────────────────────┐
│   Backend & Frontend    │
│        Person 3         │
└────────────┬────────────┘
             │
             ▼
       Final Assessment
             │
      ┌──────┴──────┐
      ▼             ▼
Trustworthiness   Business /
Assessment        Investment
                  Potential
```

### Overall Process

1. The user provides whatever business information is available.
2. The Investigation Agent uses this information as a starting point.
3. The agent searches publicly available sources on the internet.
4. Relevant business information is extracted and structured.
5. The agent identifies predefined and business-specific features.
6. If additional information is needed, the agent performs further targeted investigation.
7. Information is cross-checked across multiple sources.
8. The resulting evidence is passed to the ML/risk assessment component.
9. The system evaluates trustworthiness and potential business/investment benefit.
10. The final result contains scores/assessment, positive signals, risks, missing information, evidence, sources, and a concise justification.

---

# Team Responsibilities

## Person 1 — Maheen Fatima

### Business Investigation Agent

Person 1 is responsible for developing the **AI-powered business investigation and evidence-gathering agent**.

### Responsibilities

* Receive and validate business information.
* Search publicly available internet sources.
* Find relevant websites, social profiles, marketplaces, reviews, and other public sources.
* Extract useful business information from discovered sources.
* Identify and verify the business across different sources.
* Cross-check information between sources.
* Track the source and confidence of important evidence.
* Identify positive business signals.
* Identify potential risk signals.
* Identify missing information.
* Use predefined business features as an initial investigation checklist.
* Dynamically discover additional relevant features based on the specific business.
* Investigate discovered features through additional targeted searches.
* Investigate signals such as:

  * Social media presence
  * Followers and engagement
  * Reviews and sentiment
  * Product activity
  * Product demand signals
  * Marketplace presence
  * Business age
  * Public mentions
  * Website activity
  * Customer interest
  * Other business-specific signals
* Prevent duplicate and unnecessary searches.
* Stop investigation when sufficient evidence has been collected or investigation limits are reached.
* Produce structured `BusinessEvidence` for the ML component.
* Provide evidence-backed reasoning without presenting unsupported assumptions as facts.

### Main Output

```text
BusinessEvidence
├── Business Identity
├── Online Presence
├── Products / Services
├── Reviews
├── Discovered Features
├── Positive Signals
├── Risk Signals
├── Missing Information
└── Sources / Evidence
```

---

## Person 2 — Muneebah Nadeem

### ML / Risk Assessment

Person 2 is responsible for the **machine learning and risk assessment component**.

### Responsibilities

* Receive structured business evidence from Person 1.
* Design and prepare ML features.
* Perform data preprocessing and feature engineering.
* Prepare and evaluate relevant datasets.
* Train and compare appropriate ML models.
* Generate business/risk assessments.
* Convert relevant agent-derived evidence into usable model features.
* Evaluate model performance.
* Implement model explainability where appropriate.
* Use SHAP or similar techniques to explain important model features.
* Produce interpretable risk/credit-related outputs.

### Main Input

```text
BusinessEvidence
        ↓
Feature Engineering
        ↓
ML Model
        ↓
Risk / Business Assessment
```

Person 2 should not duplicate the web investigation performed by Person 1.

---

## Person 3 — Esha Ashfaq

### Backend, Frontend & System Integration

Person 3 is responsible for connecting the system components and building the user-facing application.

### Backend Responsibilities

* Develop the backend API.
* Connect the frontend with the backend.
* Connect the backend with the Investigation Agent.
* Connect the backend with the ML component.
* Handle request/response flow.
* Integrate database functionality where required.
* Manage communication between system components.
* Ensure the agreed schemas/interfaces are followed.

### Frontend Responsibilities

Build the interface for:

* Business information input
* Investigation status/results
* Trustworthiness assessment
* Business/investment potential
* Positive signals
* Risk signals
* Discovered features
* Missing information
* Evidence and sources
* Final two-line justification

### Main Role

```text
Frontend
    ↓
Backend API
    ↓
Investigation Agent
    ↓
Business Evidence
    ↓
ML / Risk Analysis
    ↓
Backend
    ↓
Frontend Result
```

---

# Agent Design

The Investigation Agent is designed to be **adaptive rather than completely rule-based**.

It begins with known features but can discover additional features that may help evaluate a particular business.

For example:

```text
Business
   ↓
Initial Investigation
   ↓
Large social media presence discovered
   ↓
Agent identifies audience size as relevant
   ↓
Investigates followers + engagement
   ↓
Product activity discovered
   ↓
Agent investigates product demand signals
   ↓
Customer reviews discovered
   ↓
Agent investigates reputation
   ↓
Evidence is cross-checked
```

Different businesses may result in different investigation paths.

The agent should therefore answer:

> **What evidence do I have, what does it indicate, and what additional information would help me make a better assessment?**

---

# Evidence Principles

The system distinguishes between:

### Verified Fact

Information directly supported by a source.

### Corroborated Fact

Information supported by multiple independent sources.

### Inference

A conclusion drawn from available evidence.

### Unknown

Information that could not be sufficiently verified.

The agent must not turn an inference into a verified fact.

For example:

```text
Observed:
250,000 social media followers

Possible inference:
Large online audience

Not automatically valid:
The business is trustworthy because it has 250,000 followers.
```

Similarly, missing information should not automatically be interpreted as negative evidence.

When evidence is insufficient, the system should be able to report:

```text
Insufficient Evidence
```

---

# Final Assessment

The system evaluates two separate dimensions:

## Trustworthiness

Considers evidence such as:

* Business identity consistency
* Public presence
* Source reliability
* Customer reputation
* Reviews
* Complaints
* Business history
* Transparency
* Consistency across sources

## Business / Investment Potential

Considers evidence such as:

* Market interest
* Product demand signals
* Customer activity
* Business activity
* Audience
* Growth signals
* Market presence
* Product/service attractiveness
* User-provided business information
* Other relevant discovered signals

The system provides **decision support**, not a guaranteed investment or lending decision.

---

# Repository Structure

```text
opencredit-ai/
├── frontend/
├── backend/
├── ml/
├── agent/
├── data/
├── documents/
├── tests/
├── docker/
└── README.md
```

### Component Ownership

```text
frontend/  → Person 3
backend/   → Person 3
ml/        → Person 2
agent/     → Person 1
data/      → Shared
documents/ → Shared
tests/     → Shared
docker/    → Shared
README.md  → Shared
```

The top-level repository structure should remain stable unless a technical change is proposed and agreed upon by the team.
