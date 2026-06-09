from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_health_returns_loaded_corpus_size():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["corpus_size"] == 10


def test_find_similar_returns_ranked_matches():
    response = client.post(
        "/find-similar",
        json={"query": "automate email follow-ups", "top_k": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["top_match"]["category"] == "automation"
    assert len(body["all_matches"]) == 2
    scores = [match["similarity_score"] for match in body["all_matches"]]
    assert scores == sorted(scores, reverse=True)


def test_short_query_is_rejected():
    response = client.post("/find-similar", json={"query": "hi"})

    assert response.status_code == 400


def test_top_k_less_than_one_is_rejected():
    response = client.post(
        "/find-similar",
        json={"query": "sales funnel automation", "top_k": 0},
    )

    assert response.status_code == 400
