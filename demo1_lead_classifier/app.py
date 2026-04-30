from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import anthropic
import json
import os
from dotenv import load_dotenv
from prompts import CLASSIFY_PROMPT, SYSTEM_DESIGN
from models import LeadInput, LeadOutput, HealthResponse

load_dotenv()

app = FastAPI(
    title="KeaBuilder Lead Classifier",
    description="""
    AI-powered lead qualification system for KeaBuilder.

    Classifies leads as HOT / WARM / COLD and generates
    personalized, human-sounding responses using Claude claude-sonnet-4-6.
    """,
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# System prompt with cache_control so Anthropic caches it after the first request.
# CLASSIFY_PROMPT is static — every subsequent call reads from cache at ~10% of
# normal input token cost instead of re-tokenising the full prompt each time.
CACHED_SYSTEM = [
    {
        "type": "text",
        "text": CLASSIFY_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "KeaBuilder Lead Classifier",
        "version": "1.1.0",
        "status": "running",
        "model": "claude-sonnet-4-6",
        "features": ["prompt_caching", "streaming"],
        "endpoints": {
            "classify": "POST /classify-lead",
            "health": "GET /health",
            "architecture": "GET /architecture",
            "docs": "GET /docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="ok",
        service="lead-classifier",
        version="1.1.0"
    )


@app.get("/architecture", tags=["System Design"])
def get_architecture():
    """
    Returns the full system design for KeaBuilder AI features.
    Covers: multi-provider routing, LoRA integration,
    fallback strategy, high-volume architecture.
    """
    return SYSTEM_DESIGN


@app.post("/classify-lead", response_model=LeadOutput, tags=["Lead Classification"])
async def classify_lead(lead: LeadInput):
    """
    Classify an incoming lead and generate a personalized response.

    - **HOT**: Clear intent, urgency, specific need → Immediate action response
    - **WARM**: Genuine interest, vague timeline → Nurturing response
    - **COLD**: Browsing, no clear need → Light curiosity response

    Uses streaming internally so the request never times out under load.
    The system prompt is cached — repeated calls cost ~90% less on input tokens.
    """
    if not lead.message or len(lead.message.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Message field cannot be empty"
        )

    user_message = f"""
Lead Form Submission Details:
------------------------------
Name: {lead.name if lead.name else 'Not provided'}
Email: {lead.email if lead.email else 'Not provided'}
Business Type: {lead.business_type if lead.business_type else 'Not provided'}
Message: {lead.message}
Lead Source: {lead.source if lead.source else 'form'}
------------------------------

Analyze this lead and return your classification + personalized response.
"""

    raw_text = ""
    try:
        # Stream the response so HTTP never times out regardless of output length.
        # get_final_message() waits for the full stream and returns a standard Message.
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=CACHED_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            final = stream.get_final_message()

        raw_text = final.content[0].text.strip()

        # Strip markdown code fences if the model wraps output in them
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        result = json.loads(raw_text)
        return LeadOutput(**result)

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model returned malformed JSON: {str(e)}. Raw: {raw_text[:200]}"
        )
    except anthropic.APIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Anthropic API error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
