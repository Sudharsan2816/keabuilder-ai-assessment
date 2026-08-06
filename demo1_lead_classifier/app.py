import json
import os
import secrets
from typing import Any

import anthropic
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import ValidationError

from models import HealthResponse, LeadInput, LeadOutput
from prompts import CLASSIFY_PROMPT, SYSTEM_DESIGN

load_dotenv()

NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
).rstrip("/")
NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1"
)
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

app = FastAPI(
    title="AI Lead Intelligence API",
    description=(
        "Classifies leads as HOT, WARM, or COLD and generates a "
        "personalized response using a configured LLM provider."
    ),
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class LLMNotConfiguredError(RuntimeError):
    pass


class LLMProviderError(RuntimeError):
    pass


def _require_api_key(provided: str | None = Security(api_key_header)) -> None:
    expected = os.getenv("API_TOKEN")
    if not expected:
        return
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _provider_configuration() -> tuple[str, str | None]:
    requested = os.getenv("LLM_PROVIDER", "").strip().lower()

    if requested and requested not in {"nvidia", "anthropic"}:
        return requested, None
    if requested == "nvidia":
        return requested, os.getenv("NVIDIA_API_KEY")
    if requested == "anthropic":
        return requested, os.getenv("ANTHROPIC_API_KEY")
    if os.getenv("NVIDIA_API_KEY"):
        return "nvidia", os.getenv("NVIDIA_API_KEY")
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic", os.getenv("ANTHROPIC_API_KEY")
    return "unconfigured", None


def _build_user_message(lead: LeadInput) -> str:
    return f"""
Lead Form Submission Details:
------------------------------
Name: {lead.name or 'Not provided'}
Email: {lead.email or 'Not provided'}
Business Type: {lead.business_type or 'Not provided'}
Message: {lead.message}
Lead Source: {lead.source or 'form'}
------------------------------

Analyze this lead and return the classification and personalized response.
"""


async def _nvidia_completion(api_key: str, user_message: str) -> str:
    payload = {
        "model": NVIDIA_MODEL,
        "messages": [
            {"role": "system", "content": CLASSIFY_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.2,
        "max_tokens": 1000,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        timeout = httpx.Timeout(60.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            body: dict[str, Any] = response.json()
        return body["choices"][0]["message"]["content"].strip()
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMProviderError("NVIDIA generation failed") from exc


async def _anthropic_completion(api_key: str, user_message: str) -> str:
    cached_system = [
        {
            "type": "text",
            "text": CLASSIFY_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1000,
            system=cached_system,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text.strip()
    except (anthropic.APIError, AttributeError, IndexError) as exc:
        raise LLMProviderError("Anthropic generation failed") from exc


async def _generate_completion(user_message: str) -> str:
    provider, api_key = _provider_configuration()
    if not api_key:
        raise LLMNotConfiguredError(
            f"LLM provider '{provider}' does not have a configured API key"
        )
    if provider == "nvidia":
        return await _nvidia_completion(api_key, user_message)
    if provider == "anthropic":
        return await _anthropic_completion(api_key, user_message)
    raise LLMNotConfiguredError(f"Unsupported LLM provider '{provider}'")


def _parse_model_output(raw_text: str) -> LeadOutput:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()
    result = json.loads(cleaned)
    return LeadOutput.model_validate(result)


@app.get("/", tags=["Root"])
def root():
    provider, api_key = _provider_configuration()
    return {
        "service": "AI Lead Intelligence API",
        "version": "1.2.0",
        "status": "running",
        "llm_provider": provider,
        "llm_configured": bool(api_key),
        "endpoints": {
            "classify": "POST /classify-lead",
            "health": "GET /health",
            "architecture": "GET /architecture",
            "docs": "GET /docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    provider, api_key = _provider_configuration()
    return HealthResponse(
        status="ok",
        service="lead-classifier",
        version="1.2.0",
        llm_provider=provider,
        llm_configured=bool(api_key),
    )


@app.get("/architecture", tags=["System Design"])
def get_architecture():
    return SYSTEM_DESIGN


@app.post(
    "/classify-lead",
    response_model=LeadOutput,
    tags=["Lead Classification"],
    dependencies=[Security(_require_api_key)],
)
async def classify_lead(lead: LeadInput):
    if not lead.message.strip():
        raise HTTPException(status_code=400, detail="Message field cannot be empty")

    try:
        raw_text = await _generate_completion(_build_user_message(lead))
        return _parse_model_output(raw_text)
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=502, detail="The configured LLM provider is unavailable"
        ) from exc
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise HTTPException(
            status_code=502, detail="The LLM returned an invalid structured response"
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
