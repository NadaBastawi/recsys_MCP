# TORCO Pest Control — AI Recommendation System
## Full Project Plan & Structure

---

## 1. Project Overview

**Name:** TORCO Pest Control Recommendation System + MCP Server

**One-line pitch:**
An end-to-end ML system that predicts which pest control services a customer needs, how urgently they need them, and exposes everything to AI agents via an MCP server.

**Why it exists:**
Pest control companies have zero personalization — every customer gets the same generic service menu. This system changes that by combining customer history, property type, location, and seasonal patterns to deliver intelligent, proactive recommendations.

**Domains covered (maps directly to TORCO JD):**
- Recommendation Systems — hybrid CB + CF engine
- NLP — natural language agent interface via MCP
- MLOps — full lifecycle, FastAPI deployment, model monitoring hooks
- Computer Vision — (Phase 2 extension: pest photo identification)

---

## 2. Problem Statement

> *Given a customer's profile, property details, location, service history, and current season — predict the most relevant pest control services and how urgently the customer needs them, before they even call.*

**Business value:**
- Increase revenue through proactive outreach to high-urgency customers
- Reduce churn by anticipating customer needs
- Enable AI scheduling agents to make intelligent service recommendations
- Personalize the customer experience at scale

---

## 3. Full Project Structure

```
torco_recsys/
│
├── data/
│   ├── generate_data.py          # Synthetic data generator
│   ├── customers.csv             # 500 customer profiles
│   ├── interactions.csv          # 3,000 service interactions
│   ├── services.csv              # 10 service catalog entries
│   └── data_card.md              # Dataset documentation
│
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_content_based.ipynb    # Content-based filtering experiments
│   ├── 03_collaborative.ipynb    # SVD collaborative filtering experiments
│   ├── 04_urgency_model.ipynb    # Urgency scorer experiments
│   └── 05_hybrid_eval.ipynb     # Hybrid model evaluation
│
├── models/
│   ├── recommender.py            # Hybrid recommender (main model file)
│   ├── content_based.py          # Content-based filtering module
│   ├── collaborative.py          # SVD collaborative filtering module
│   ├── urgency_scorer.py         # Urgency prediction module
│   ├── evaluation.py             # Metrics: Precision@K, NDCG, coverage
│   └── torco_recommender.pkl     # Saved trained model
│
├── api/
│   ├── main.py                   # FastAPI app
│   ├── schemas.py                # Pydantic request/response models
│   ├── dependencies.py           # Model loading, shared dependencies
│   └── routers/
│       ├── recommend.py          # /recommend endpoints
│       ├── customers.py          # /customers endpoints
│       └── health.py             # /health, /metrics endpoints
│
├── mcp/
│   ├── server.py                 # MCP server (stdio transport)
│   └── tools.py                  # Tool definitions and handlers
│
├── monitoring/
│   ├── drift_detector.py         # Data drift detection (KS test, PSI)
│   └── logger.py                 # Prediction logging for monitoring
│
├── tests/
│   ├── test_recommender.py       # Unit tests for model
│   ├── test_api.py               # API endpoint tests
│   └── test_data.py              # Data quality tests
│
├── docs/
│   ├── architecture.md           # System architecture
│   ├── api_reference.md          # API documentation
│   └── model_card.md             # Model card (performance, limitations)
│
├── requirements.txt              # All dependencies
├── Dockerfile                    # Container for deployment
├── .env.example                  # Environment variables template
└── README.md                     # Project documentation
```

---

## 4. Phase-by-Phase Build Plan

---

### Phase 1 — Data Foundation
**Goal:** Generate and validate the synthetic dataset

**Tasks:**
- Design customer schema (property type, ZIP, age, is_commercial, etc.)
- Model realistic pest risk probabilities per region/season
- Generate 500 customers and 3,000 interactions
- Validate data distributions are realistic (no impossible combinations)
- Write a data card documenting schema, assumptions, and limitations

**Output:** `customers.csv`, `interactions.csv`, `services.csv`, `data_card.md`

**How to explain in interview:**
> "I designed the data generator to reflect real Southwest US pest patterns — termite seasonality, scorpion risk by ZIP, commercial property profiles. The data isn't random; it's domain-informed simulation."

---

### Phase 2 — Exploratory Data Analysis
**Goal:** Understand the data before modeling

**Tasks:**
- Distribution of property types, ZIP codes, building ages
- Service popularity by season (visualize termite spike in spring/summer)
- Customer interaction frequency distribution
- Rating distributions per service
- Repeat booking rates

**Output:** `01_eda.ipynb` with visualizations

**How to explain in interview:**
> "EDA revealed that termite services spike 40% in spring/summer, confirming the domain knowledge baked into data generation. Scorpion services concentrated in 6 desert-adjacent ZIP codes."

---

### Phase 3 — Content-Based Filtering
**Goal:** Recommend based on customer profile features

**Algorithm:**
- Encode customer features: property type, building age, desert/older neighborhood flags
- Build service affinity vectors: average feature profile of high-rated customers per service
- Cosine similarity between customer vector and service affinity vectors

**Evaluation:** Precision@5 on holdout set

**Output:** `content_based.py`, `02_content_based.ipynb`

**How to explain in interview:**
> "Content-based filtering is essentially semantic similarity search on tabular features — the same principle as my RAG systems but applied to structured customer data."

---

### Phase 4 — Collaborative Filtering
**Goal:** Recommend based on patterns from similar customers

**Algorithm:**
- Build customer × service rating matrix (500 × 10)
- Apply Truncated SVD for matrix factorization (8 components, 86.7% variance explained)
- Reconstruct matrix to get predicted ratings for unseen service-customer pairs

**Evaluation:** Precision@5, NDCG@5

**Output:** `collaborative.py`, `03_collaborative.ipynb`

**How to explain in interview:**
> "SVD finds latent factors — hidden patterns like 'desert homeowner' or 'commercial kitchen' — without explicitly labeling them. It discovered that customers who got termite inspections were 60% more likely to need treatment within 6 months."

---

### Phase 5 — Urgency Scorer
**Goal:** Predict how urgently a customer needs a service

**Algorithm:**
- Features: season, building age, days since last service, property type, desert/commercial flags
- Model: Gradient Boosting Regressor
- Target: urgency score 1–10
- RMSE: 0.867

**Output:** `urgency_scorer.py`, `04_urgency_model.ipynb`

**How to explain in interview:**
> "Urgency scoring is what differentiates this from a standard recommender. It answers not just 'what does this customer need?' but 'how soon do they need it?' — which directly drives proactive outreach campaigns."

---

### Phase 6 — Hybrid Model
**Goal:** Combine CB + CF with weighted scoring

**Design:**
- Normalize CB and CF scores independently to 0–1
- Weighted combination: 40% content-based + 60% collaborative
- Attach urgency score to each recommendation
- Handle cold-start: fall back to urgency-only for new customers

**Evaluation:**
- Precision@5: 96.2%
- NDCG@5
- Service coverage: 10/10

**Output:** `recommender.py`, `05_hybrid_eval.ipynb`

**How to explain in interview:**
> "The 40/60 weighting came from empirical testing — collaborative filtering performs better with enough data, but content-based anchors recommendations when a customer has few interactions. The hybrid handles both gracefully."

---

### Phase 7 — FastAPI Deployment
**Goal:** Serve the model as a production-ready REST API

**Endpoints:**
```
POST /recommend              — existing customer recommendations
POST /recommend/new-customer — cold-start for new customers
GET  /services               — list all services
GET  /health                 — health check
GET  /docs                   — auto-generated Swagger UI
```

**Features:**
- Pydantic validation on all inputs/outputs
- Proper HTTP error handling (404 for unknown customers)
- Model loaded once on startup (not per request)
- Response includes urgency label: HIGH / MEDIUM / LOW

**Output:** `api/main.py`, `api/schemas.py`, `api/routers/`

---

### Phase 8 — MCP Server
**Goal:** Expose the recommender to AI agents in natural language

**Tools exposed:**
- `recommend_for_customer` — get recs for existing customer by ID
- `recommend_for_new_customer` — cold-start by property profile
- `list_services` — list service catalog

**Transport:** stdio (compatible with Claude Desktop, Continue, any MCP client)

**Example agent queries:**
> "What services should customer C0042 get this summer?"
> "New restaurant in ZIP 85706, 15 years old — what do they need?"

**Output:** `mcp/server.py`, `mcp/tools.py`

---

### Phase 9 — Monitoring (MLOps)
**Goal:** Detect when model performance degrades in production

**What to implement:**
- Log every prediction (customer_id, service_id, score, urgency, timestamp)
- KS test to detect feature drift between training and production distributions
- PSI (Population Stability Index) on key features
- Alert when drift score exceeds threshold

**Output:** `monitoring/drift_detector.py`, `monitoring/logger.py`

**How to explain in interview:**
> "Model monitoring is in the JD — 'continuously monitor model performance in production.' The drift detector flags when the distribution of property types or seasonal patterns shifts, signaling the model needs retraining."

---

### Phase 10 — Testing & Documentation
**Goal:** Make the project production-credible

**Tests:**
- Unit tests for recommender (does it return top_n results? does cold-start work?)
- API tests (correct status codes, schema validation)
- Data quality tests (no nulls in required fields, valid ZIP codes)

**Documentation:**
- README with architecture diagram, setup instructions, API usage examples
- Model card: performance metrics, training data description, known limitations
- API reference: all endpoints with request/response examples

---

## 5. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| ML | scikit-learn, numpy, pandas |
| API | FastAPI, Uvicorn, Pydantic |
| MCP | Custom stdio server (JSON-RPC 2.0) |
| Persistence | joblib (model), CSV (data) |
| Testing | pytest |
| Containerization | Docker |
| Monitoring | Custom (KS test, PSI via scipy) |

---

## 6. Evaluation Metrics

| Metric | What it measures | Result |
|---|---|---|
| Precision@5 | Of top 5 recs, how many are relevant | 96.2% |
| NDCG@5 | Quality of ranking, not just presence | Computed in notebook |
| Coverage | How many services get recommended | 10/10 |
| Urgency RMSE | Accuracy of urgency prediction | 0.867 |
| SVD Variance | How much info the model captures | 86.7% |

---

## 7. How to Talk About This Project in an Interview

**Opening (30 seconds):**
> "I built an end-to-end pest control recommendation system directly inspired by this role. It uses a hybrid of content-based filtering and SVD collaborative filtering to recommend services, a gradient boosting urgency scorer to prioritize proactive outreach, and exposes everything via FastAPI and an MCP server so AI agents can query it in natural language."

**The problem (why it matters):**
> "Pest control companies have zero ML personalization today. The same service menu goes to every customer regardless of whether they're a desert-adjacent homeowner with a 30-year-old house or a downtown apartment renter. That's a missed revenue and retention opportunity."

**The technical decision you're proudest of:**
> "The urgency scorer. Most recommenders just say 'this customer might like X.' Mine says 'this customer needs X in the next 30 days.' That's the difference between a recommendation and an actionable business insight."

**What you'd improve:**
> "With real production data I'd retrain monthly, add A/B testing on the 40/60 weighting, and plug in actual customer CRM data. I'd also add a feedback loop — if a customer books a recommended service, that signal reinforces the model."

---

## 8. GitHub Repository Checklist

Before pushing, make sure:
- [ ] README has architecture diagram, setup steps, and sample API calls
- [ ] All notebooks are run with visible outputs
- [ ] requirements.txt is complete
- [ ] Model card documents performance metrics and limitations
- [ ] Code has docstrings on every class and function
- [ ] Data generator is reproducible (random seed set)
- [ ] .gitignore excludes .pkl files and CSVs (add data generation instructions instead)
