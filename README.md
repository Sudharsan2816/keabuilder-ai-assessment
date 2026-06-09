# AI Lead Intelligence API

FastAPI project for lead qualification and lightweight text similarity search.

This repository was previously named `keabuilder-ai-assessment`. The recruiter-friendly name should be `ai-lead-intelligence-api` because the project is not just an assessment: it demonstrates two practical AI API patterns for a sales/marketing product.

## What It Builds

### 1. Lead Classifier

Classifies inbound leads as `HOT`, `WARM`, or `COLD` and generates a personalized response.

Key implementation details:

- FastAPI endpoint for structured lead input.
- Claude-powered classification and response generation.
- Prompt caching for repeated static system instructions.
- Streaming response handling to reduce timeout risk.
- Pydantic request/response models.

Live API docs:

- Swagger: https://keabuilder-ai-assessment-production.up.railway.app/docs
- ReDoc: https://keabuilder-ai-assessment-production.up.railway.app/redoc

### 2. Similarity Search

Finds similar user inputs from a small knowledge base using TF-IDF cosine similarity.

Key implementation details:

- FastAPI search endpoint.
- scikit-learn `TfidfVectorizer`.
- Cosine similarity ranking.
- Documented production path to sentence-transformers plus pgvector.

## Architecture

```text
Client
  -> FastAPI
  -> Lead Classifier
      -> Claude prompt
      -> structured lead score + response
  -> Similarity Search
      -> TF-IDF vectors
      -> cosine similarity
      -> ranked matches
```

## Tech Stack

- Python, FastAPI, Pydantic
- Anthropic Claude
- scikit-learn TF-IDF and cosine similarity
- Railway deployment for the lead classifier demo

## Run Locally

```bash
git clone https://github.com/Sudharsan2816/ai-lead-intelligence-api
cd ai-lead-intelligence-api
pip install -r requirements.txt
cp .env.example .env
```

Run the lead classifier:

```bash
cd demo1_lead_classifier
python -m uvicorn app:app --reload --port 8000
```

Run similarity search in another terminal:

```bash
cd demo2_similarity_search
python -m uvicorn app:app --reload --port 8001
```

## Example Requests

```bash
curl -X POST http://localhost:8000/classify-lead \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Ravi Kumar",
    "email": "ravi@startup.com",
    "business_type": "SaaS",
    "message": "Need funnels for product launch next month. Pricing?",
    "source": "landing_page"
  }'
```

```bash
curl -X POST http://localhost:8001/find-similar \
  -H "Content-Type: application/json" \
  -d '{"query": "I want to sell my coaching program online", "top_k": 3}'
```

## Portfolio Value

This repo demonstrates:

- Applied LLM API development.
- Prompt engineering tied to business classification logic.
- Structured API contracts with Pydantic.
- Baseline ML similarity search.
- Clear production upgrade thinking for semantic search and vector databases.

## Current Production Gaps

- Add unit tests for lead classification validation and similarity search.
- Add Docker Compose for both services.
- Replace hardcoded in-memory similarity data with a persistent store.
- Add request authentication and rate limiting.
- Add evaluation data for lead classification quality.

## Recommended GitHub Metadata

- Repository name: `ai-lead-intelligence-api`
- Description: `FastAPI lead intelligence system with Claude-powered lead scoring, personalized responses, TF-IDF similarity search, and a documented pgvector upgrade path.`
- Topics: `python`, `fastapi`, `llm`, `anthropic`, `lead-scoring`, `similarity-search`, `scikit-learn`, `backend`, `ai-engineering`
