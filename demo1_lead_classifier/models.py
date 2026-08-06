from typing import Literal, Optional

from pydantic import BaseModel, Field


class LeadInput(BaseModel):
    name: Optional[str] = Field(default=None, max_length=200)
    email: Optional[str] = Field(default=None, max_length=320)
    business_type: Optional[str] = Field(default=None, max_length=200)
    message: str = Field(min_length=1, max_length=10_000)
    source: Optional[str] = Field(default="form", max_length=200)

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Ravi Kumar",
                    "email": "ravi@startup.com",
                    "business_type": "SaaS",
                    "message": "Need funnels for product launch next month. Pricing?",
                    "source": "landing_page",
                }
            ]
        },
    }


class LeadOutput(BaseModel):
    classification: Literal["HOT", "WARM", "COLD"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1, max_length=2_000)
    signals: list[str] = Field(default_factory=list, max_length=20)
    response: str = Field(min_length=1, max_length=5_000)
    follow_up_question: Optional[str] = Field(default=None, max_length=1_000)

    model_config = {"extra": "forbid"}


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    llm_provider: str
    llm_configured: bool
