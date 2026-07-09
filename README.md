# TORCO Pest Control — AI Recommendation System + MCP Server

[![codecov](https://codecov.io/gh/NadaBastawi/recsys_MCP/branch/main/graph/badge.svg)](https://codecov.io/gh/NadaBastawi/recsys_MCP)

> **End-to-end ML project:** Hybrid recommendation engine for pest control service personalization, exposed as a REST API and MCP server for AI agent integration.

---

## Problem Statement

Pest control companies treat every customer the same — same service menu, same outreach, zero personalization. Yet a residential house in a desert-adjacent Tucson ZIP code has fundamentally different pest risks than a downtown apartment or a commercial restaurant.

This system uses machine learning to:
1. **Recommend** the most relevant pest control services per customer
2. **Score urgency** — predict how soon a customer needs a service
3. **Handle cold-start** — recommend for brand new customers with no history
4. **Expose via MCP** — so AI agents can query it in natural language

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   MCP Server (stdio)                 │
│         AI agents query in natural language          │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│              FastAPI REST API (:8000)                │
│   /recommend   /recommend/new-customer   /services   │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│            Hybrid Recommendation Engine              │
│                                                      │
│  ┌─────────────────┐     ┌────────────────────────┐ │
│  │ Content-Based   │     │ Collaborative Filtering │ │
│  │ (cosine sim on  │ 40% │ (SVD matrix             │ │
│  │  customer       │  +  │  factorization,         │ │
│  │  features)      │ 60% │  86.7% var explained)   │ │
│  └────────┬────────┘     └───────────┬─────────────┘ │
│           └──────────┬───────────────┘               │
│                      ▼                               │
│         ┌────────────────────────┐                   │
│         │    Urgency Scorer      │                   │
│         │ (GradientBoosting,     │                   │
│         │  RMSE: 0.867)          │                   │
│         └────────────────────────┘                   │
│                      ▼                               │
│         ┌────────────────────────┐                   │
│         │ Drift Monitoring       │                   │
│         │ (KS test + PSI)        │                   │
│         └────────────────────────┘                   │
└─────────────────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│              Synthetic Dataset                       │
│  500 customers | 3,000 interactions | 10 services    │
│  Modeled on real Tucson pest patterns:               │
│  - Termites peak spring/summer                       │
│  - Scorpions in desert-adjacent ZIPs                 │
│  - Commercial properties → rodent/ant risk           │
│  - Older neighborhoods → higher termite risk         │
└─────────────────────────────────────────────────────┘
```

---

## Results

| Metric | Value |
|---|---|
| Precision@5 | 96.2% |
| SVD Variance Explained | 86.7% |
| Urgency Scorer RMSE | 0.867 |
| Service Coverage | 10/10 |
| Cold-Start | ✅ Handled |

---

## Project Structure

```
torco_recsys/
├── data/
│   ├── generate_data.py      # Synthetic data generator
│   ├── customers.csv         # 500 customer profiles
│   ├── interactions.csv      # 3,000 service interactions
│   └── services.csv          # 10 service catalog entries
├── models/
│   ├── recommender.py        # Hybrid recommender (CB + CF + Urgency)
│   └── torco_recommender.pkl # Saved model
├── api/
│   └── main.py               # FastAPI REST API
├── mcp/
│   └── server.py             # MCP server (stdio transport)
└── README.md
```

---

## Setup & Run

CI runs automatically for pushes and pull requests to the main branch via GitHub Actions, and the test suite is exercised with pytest in the workflow.

```bash
# Install dependencies
pip install pandas numpy scikit-learn scipy faker fastapi uvicorn joblib httpx

# Generate data
cd data && python generate_data.py

# Train model
cd models && python recommender.py

# Start API
cd api && uvicorn main:app --reload --port 8000

# Start MCP server (in separate terminal)
cd mcp && python server.py
```

---

## API Usage

**Existing customer:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"customer_id": "C0001", "season": "summer", "days_since_last_service": 200}'
```

**New customer (cold-start):**
```bash
curl -X POST http://localhost:8000/recommend/new-customer \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"property_type": "residential_house", "zip_code": "85743", "building_age_years": 25, "season": "spring"}'
```

---

## MCP Agent Queries

Once connected to an MCP client (Claude Desktop, Continue, etc.):

> *"What services should I recommend for customer C0042?"*
> *"A new homeowner in ZIP 85743 with a 30-year-old house — what do they need?"*
> *"List all available services"*

---

## Key Design Decisions

**Why hybrid (CB + CF)?**
Content-based alone misses collaborative patterns. CF alone can't handle new customers (cold-start problem). Hybrid gets the best of both: personalization from history + profile-based fallback.

**Why 40/60 weighting?**
Collaborative filtering captures richer behavioral patterns with enough interaction data (3,000 interactions). Content-based anchors recommendations for sparse users.

**Why synthetic data?**
No real TORCO data is publicly available. The synthetic generator models real Southwest US pest patterns: termite seasonality, desert proximity risks, commercial property profiles — making it a realistic proxy for production data.

**Why MCP?**
Exposes the recommender to AI agents natively. A TORCO support agent or scheduling AI can query "what does this customer need?" without writing API calls.

**Why monitoring?**
The monitoring module tracks prediction drift with KS tests and PSI to surface shifts in customer behavior or service demand over time, helping keep the recommender reliable in production.

---

## Tech Stack

`Python` `scikit-learn` `FastAPI` `pandas` `numpy` `joblib` `httpx` `MCP Protocol`

## License

This project is licensed under the MIT License — see the LICENSE file for details.
