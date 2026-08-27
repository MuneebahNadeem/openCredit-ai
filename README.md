# OpenCredit AI

> **OpenCredit AI is an AI-powered alternative-data business assessment platform that helps lending organizations evaluate small and informal businesses that may lack complete traditional financial records.**

The five-day MVP focuses on an **agentic public-information investigation workflow**. Given a business name and available public identifiers—such as a website, social media URL, marketplace URL, or location—the system searches permitted public sources, extracts relevant evidence, cross-checks business identity and activity, analyzes reputation and customer signals, identifies missing information, and generates an explainable assessment.

OpenCredit AI is designed for **banks, microfinance institutions, fintech lenders, and other lending organizations**. It is a **decision-support system, not an autonomous loan approval system**; the final credit decision remains the responsibility of the lender or loan officer.

---

## Five-Day Development Plan

### DAY 1 — Foundation

**Goal:** Build the complete application skeleton and establish the core workflow.

**Person 1**
- Set up the LLM and agent architecture
- Integrate web search and webpage extraction tools
- Build the initial investigation workflow

**Person 2**
- Finalize assessment features and requirements
- Obtain and prepare the dataset
- Create the baseline ML pipeline

**Person 3**
- Set up React, FastAPI, and PostgreSQL
- Build the business input form
- Create the initial API and application skeleton

**End-of-day target:**

`Frontend → Backend → Agent → Dummy Response`

---

### DAY 2 — Intelligence

**Goal:** Enable the system to investigate businesses and generate an initial assessment.

**Person 1**
- Implement web search and source extraction
- Extract business identity information
- Classify sources
- Store collected evidence

**Person 2**
- Implement sentiment analysis
- Extract relevant features
- Build the initial ML model
- Generate preliminary scoring

**Person 3**
- Build the dashboard and results page
- Add loading states
- Create evidence cards
- Integrate frontend with APIs

**End-of-day target:**

`Business → Web Search → Evidence → Basic Score → Dashboard`

---

### DAY 3 — Integration

**Goal:** Connect all components into a complete end-to-end prototype.

**Person 1**
- Complete the `Search → Extraction → Cross-check → Evidence` pipeline

**Person 2**
- Complete the `Evidence → Features → ML → Credibility/Risk` pipeline

**Person 3**
- Connect the complete system:

`React → FastAPI → Agent → ML → Database → React`

**End-of-day target:**

A fully functional end-to-end prototype, even if the UI is not yet polished.

---

### DAY 4 — Polish & Explainability

**Goal:** Improve reliability, transparency, and user experience.

**Person 1**
- Improve agent reasoning
- Strengthen source verification
- Detect missing information
- Improve evidence confidence scoring

**Person 2**
- Add SHAP explainability
- Evaluate model performance and metrics
- Identify feature importance
- Generate understandable risk explanations

**Person 3**
- Improve the UI and overall user experience
- Add charts and visualizations
- Build an evidence timeline
- Create the report page
- Improve error handling

**End-of-day target:**

An explainable and presentation-ready prototype.

---

### DAY 5 — Deployment & Demo

#### Morning
- Fix bugs
- Test the complete workflow
- Clean up the UI
- Test multiple business cases

#### Afternoon
- Dockerize the application
- Deploy the MVP
- Test the deployed version
- Prepare screenshots and presentation materials

#### Final Hours
- Stop adding new features
- Prepare the live demo
- Finalize the presentation
- Explain the system architecture
- Document limitations
- Define the future roadmap

---

## MVP Principle

> **Build the full workflow first. Polish second. Explain everything clearly.**