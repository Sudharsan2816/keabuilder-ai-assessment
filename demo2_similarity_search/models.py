from pydantic import BaseModel
from typing import Optional


class QueryInput(BaseModel):
    query: str
    top_k: Optional[int] = 3

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "I want to sell my coaching program online",
                    "top_k": 3
                }
            ]
        }
    }


class SingleMatch(BaseModel):
    id: int
    text: str
    category: str
    similarity_score: float


class SimilarityResult(BaseModel):
    query: str
    top_match: SingleMatch
    all_matches: list[SingleMatch]
    method: str
    total_corpus_size: int
    production_note: str
