# KeaBuilder AI Engineer Assessment
**Dream Reflection Media — AI Engineer Role**

---

## Overview

Two working APIs built to demonstrate AI system thinking and practical
implementation for the KeaBuilder platform.

---

## Live Demos

### Demo 1: Lead Classifier
Classifies incoming leads as HOT / WARM / COLD using Claude `claude-sonnet-4-6`.
Generates personalized, human-sounding responses. Handles incomplete inputs intelligently.
Uses prompt caching (90% input token savings on repeated calls) and streaming (no HTTP timeouts).

- **Swagger UI**: https://keabuilder-ai-assessment-production.up.railway.app/docs
- **ReDoc**: https://keabuilder-ai-assessment-production.up.railway.app/redoc

### Demo 2: Similarity Search
Finds similar user inputs using TF-IDF cosine similarity.
Production-ready upgrade path to sentence-transformers + pgvector documented.

- **Swagger UI**: *(deploying)*
- **ReDoc**: *(deploying)*

---

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/YOUR_USERNAME/keabuilder-ai-assessment
cd keabuilder-ai-assessment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 4. Run Demo 1 — Lead Classifier
cd demo1_lead_classifier
python -m uvicorn app:app --reload --port 8000

# 5. Run Demo 2 — Similarity Search (new terminal)
cd ../demo2_similarity_search
python -m uvicorn app:app --reload --port 8001
```

---

## Test the APIs

```bash
# HOT Lead — pricing asked, deadline next month
curl -X POST http://localhost:8000/classify-lead \
  -H "Content-Type: application/json" \
  -d '{"name":"Ravi Kumar","email":"ravi@startup.com","business_type":"SaaS","message":"Need funnels for product launch next month. Pricing?","source":"landing_page"}'

# WARM Lead — exploring, no urgency
curl -X POST http://localhost:8000/classify-lead \
  -H "Content-Type: application/json" \
  -d '{"name":"Priya","email":"priya@coach.com","business_type":"Coaching","message":"I run a coaching business and exploring tools for lead capture.","source":"blog"}'

# COLD Lead — no context
curl -X POST http://localhost:8000/classify-lead \
  -H "Content-Type: application/json" \
  -d '{"message":"hi"}'

# Similarity Search — coaching program
curl -X POST http://localhost:8001/find-similar \
  -H "Content-Type: application/json" \
  -d '{"query":"I want to sell my coaching program online","top_k":3}'

# Similarity Search — automation
curl -X POST http://localhost:8001/find-similar \
  -H "Content-Type: application/json" \
  -d '{"query":"automate my email follow-ups","top_k":2}'
```

---

## Interactive API Docs

| Service | URL |
|---------|-----|
| Lead Classifier Swagger (live) | https://keabuilder-ai-assessment-production.up.railway.app/docs |
| Lead Classifier ReDoc (live) | https://keabuilder-ai-assessment-production.up.railway.app/redoc |
| Similarity Search Swagger | *(deploying)* |
| Similarity Search ReDoc | *(deploying)* |

---

## System Design

All 7 architectural answers: [`docs/system_design_answers.md`](docs/system_design_answers.md)

End-to-end project explanation: [`docs/project_explainer.md`](docs/project_explainer.md)

Covers:
1. Lead classification system design
2. Multi-provider content routing architecture
3. LoRA integration for personalised AI images
4. Face/text similarity search with pgvector
5. Multi-AI fallback strategy (3-layer)
6. High-volume AI request handling
7. Tools and frameworks

---

## Project Structure

```
keabuilder-ai-assessment/
├── .env.example                        # API key template
├── .gitignore
├── README.md
├── requirements.txt                    # Root dependencies
├── demo1_lead_classifier/
│   ├── app.py                          # FastAPI app + endpoints
│   ├── prompts.py                      # Claude prompt + system design data
│   ├── models.py                       # Pydantic input/output models
│   ├── requirements.txt
│   └── sample_output.json              # 3 test cases (HOT/WARM/COLD)
├── demo2_similarity_search/
│   ├── app.py                          # FastAPI app + TF-IDF search
│   ├── models.py                       # Pydantic input/output models
│   ├── requirements.txt
│   └── sample_output.json              # 3 test cases with scores
└── docs/
    ├── system_design_answers.md        # Full architectural answers (Q1–Q7)
    └── project_explainer.md            # End-to-end project explanation (how it works)
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| AI Model | Claude `claude-sonnet-4-6` (Anthropic) |
| NLP | scikit-learn TF-IDF + cosine similarity |
| Runtime | Python 3.11+ |
| Data Validation | Pydantic v2 |
| Production DB | PostgreSQL + pgvector |
| Production ML | sentence-transformers |
| Production Queue | BullMQ / SQS |
| Production Cache | Redis (ElastiCache) |
