from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from models import QueryInput, SimilarityResult, SingleMatch

app = FastAPI(
    title="KeaBuilder Similarity Search",
    description="""
    Text similarity search for KeaBuilder user inputs.

    Finds the most similar entries from a knowledge base
    using TF-IDF cosine similarity.

    Production upgrade: sentence-transformers + pgvector/Pinecone
    """,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

KNOWLEDGE_BASE = [
    {"id": 1, "text": "I want to build a sales funnel for my e-commerce store", "category": "funnel"},
    {"id": 2, "text": "Looking for lead capture tools for my coaching business", "category": "lead_capture"},
    {"id": 3, "text": "Need automation for follow-up emails after form submission", "category": "automation"},
    {"id": 4, "text": "Want to create landing pages that convert visitors to leads", "category": "landing_page"},
    {"id": 5, "text": "Building a webinar funnel to sell my online course", "category": "funnel"},
    {"id": 6, "text": "How do I integrate payment gateway into my funnel", "category": "payment"},
    {"id": 7, "text": "Looking for CRM to manage all my leads in one place", "category": "crm"},
    {"id": 8, "text": "Need to automate WhatsApp follow-up messages for leads", "category": "automation"},
    {"id": 9, "text": "Want to A/B test my landing page headlines", "category": "landing_page"},
    {"id": 10, "text": "How to track conversion rates across my funnels", "category": "analytics"},
]

PRODUCTION_NOTE = (
    "Production upgrade path: "
    "(1) Replace TfidfVectorizer with sentence-transformers/all-MiniLM-L6-v2 for semantic search. "
    "(2) Store embeddings in pgvector (PostgreSQL) or Pinecone. "
    "(3) For face similarity: use InsightFace/DeepFace embeddings with same pgvector approach. "
    "(4) Set thresholds: score > 0.7 = strong match, > 0.4 = partial, < 0.4 = no match."
)


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "KeaBuilder Similarity Search",
        "version": "1.0.0",
        "status": "running",
        "corpus_size": len(KNOWLEDGE_BASE),
        "endpoints": {
            "search": "POST /find-similar",
            "corpus": "GET /corpus",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "service": "similarity-search",
        "corpus_size": len(KNOWLEDGE_BASE)
    }


@app.get("/corpus", tags=["Data"])
def get_corpus():
    """Returns all entries in the current knowledge base"""
    return {
        "entries": KNOWLEDGE_BASE,
        "count": len(KNOWLEDGE_BASE),
        "categories": list(set(item["category"] for item in KNOWLEDGE_BASE))
    }


@app.post("/find-similar", response_model=SimilarityResult, tags=["Similarity Search"])
def find_similar(query_input: QueryInput):
    """
    Find the most similar entries to a user query.

    Uses TF-IDF vectorization + cosine similarity.
    Returns top-k matches ranked by similarity score.
    """
    if not query_input.query or len(query_input.query.strip()) < 3:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 3 characters long"
        )

    top_k = min(query_input.top_k, len(KNOWLEDGE_BASE))

    corpus = [item["text"] for item in KNOWLEDGE_BASE]
    all_texts = corpus + [query_input.query]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=10000
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vectorization failed: {str(e)}")

    query_vector = tfidf_matrix[-1]
    corpus_matrix = tfidf_matrix[:-1]

    scores = cosine_similarity(query_vector, corpus_matrix)[0]

    results = []
    for i, score in enumerate(scores):
        results.append(SingleMatch(
            id=KNOWLEDGE_BASE[i]["id"],
            text=KNOWLEDGE_BASE[i]["text"],
            category=KNOWLEDGE_BASE[i]["category"],
            similarity_score=round(float(score), 4)
        ))

    results.sort(key=lambda x: x.similarity_score, reverse=True)

    return SimilarityResult(
        query=query_input.query,
        top_match=results[0],
        all_matches=results[:top_k],
        method="TF-IDF Cosine Similarity (ngram_range=1-2)",
        total_corpus_size=len(KNOWLEDGE_BASE),
        production_note=PRODUCTION_NOTE
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
