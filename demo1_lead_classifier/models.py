from pydantic import BaseModel
from typing import Optional


class LeadInput(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    business_type: Optional[str] = None
    message: str
    source: Optional[str] = "form"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Ravi Kumar",
                    "email": "ravi@startup.com",
                    "business_type": "SaaS",
                    "message": "Need funnels for product launch next month. Pricing?",
                    "source": "landing_page"
                }
            ]
        }
    }


class LeadOutput(BaseModel):
    classification: str
    confidence: float
    reasoning: str
    signals: list[str]
    response: str
    follow_up_question: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
