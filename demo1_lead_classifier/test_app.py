import app as app_module
from fastapi.testclient import TestClient


client = TestClient(app_module.app)


def test_root_and_docs_are_available(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    root = client.get("/")
    docs = client.get("/docs")

    assert root.status_code == 200
    assert root.json()["status"] == "running"
    assert docs.status_code == 200


def test_health_does_not_require_llm_key(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "lead-classifier",
        "version": "1.2.0",
        "llm_provider": "unconfigured",
        "llm_configured": False,
    }


def test_classification_returns_503_without_provider_key(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    response = client.post(
        "/classify-lead", json={"message": "I need pricing for a launch next week"}
    )

    assert response.status_code == 503
    assert "does not have a configured API key" in response.json()["detail"]


def test_classification_validates_structured_output(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)

    async def fake_completion(_user_message: str) -> str:
        return """{
            "classification": "HOT",
            "confidence": 0.94,
            "reasoning": "The lead requested pricing and has a deadline.",
            "signals": ["pricing", "deadline"],
            "response": "I can help you prepare for next week's launch.",
            "follow_up_question": null
        }"""

    monkeypatch.setattr(app_module, "_generate_completion", fake_completion)

    response = client.post(
        "/classify-lead", json={"message": "I need pricing for a launch next week"}
    )

    assert response.status_code == 200
    assert response.json()["classification"] == "HOT"
    assert response.json()["confidence"] == 0.94


def test_malformed_model_output_returns_502(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)

    async def fake_completion(_user_message: str) -> str:
        return "not-json"

    monkeypatch.setattr(app_module, "_generate_completion", fake_completion)

    response = client.post(
        "/classify-lead", json={"message": "Tell me more about your product"}
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The LLM returned an invalid structured response"
    )


def test_classification_requires_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "test-deployment-token")

    unauthorized = client.post(
        "/classify-lead", json={"message": "I need pricing next week"}
    )

    assert unauthorized.status_code == 401

    async def fake_completion(_user_message: str) -> str:
        return """{
            "classification": "HOT",
            "confidence": 0.9,
            "reasoning": "The lead requested pricing with a deadline.",
            "signals": ["pricing", "deadline"],
            "response": "Let us discuss your launch requirements.",
            "follow_up_question": null
        }"""

    monkeypatch.setattr(app_module, "_generate_completion", fake_completion)
    authorized = client.post(
        "/classify-lead",
        headers={"X-API-Key": "test-deployment-token"},
        json={"message": "I need pricing next week"},
    )

    assert authorized.status_code == 200
