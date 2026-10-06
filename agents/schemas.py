from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class RiskFactor(BaseModel):
    feature: str
    raw_value: float
    shap_impact: float

class InvestigationInput(BaseModel):
    user_id: str
    transaction_id: str
    amount: float
    risk_score: float
    top_risk_drivers: List[RiskFactor]

class IncidentAuditReport(BaseModel):
    """The strict schema enforced on Gemini's output."""
    investigation_id: str = Field(description="Unique ID for this audit case")
    verdict: Literal["BLOCKED", "2FA_CHALLENGE", "ALLOW_MONITORED"]
    confidence_score: float = Field(ge=0.0, le=1.0, description="Agent confidence score")
    root_cause_analysis: str = Field(description="Clear explanation combining ML SHAP factors with user history")
    mitigation_action: str = Field(description="Concrete action taken (e.g., card frozen, SMS OTP dispatched)")
    compliance_notes: str = Field(description="Summary for legal, auditing, and fraud operations teams")