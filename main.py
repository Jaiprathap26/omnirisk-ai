import os
import sys
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Ensure local imports work cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "ml")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "agents")))

from explain import ModelExplainer
from schemas import RiskFactor, InvestigationInput, IncidentAuditReport
from investigator import FraudInvestigationAgent

app = FastAPI(
    title="OmniRisk AI — Production Fraud & Anomaly Intelligence",
    description="Real-time ML risk scoring with SHAP explainability and autonomous GenAI agent investigation.",
    version="1.0.0"
)

# Global singletons loaded once at startup (prevents re-loading model per request)
explainer: Optional[ModelExplainer] = None
agent: Optional[FraudInvestigationAgent] = None

@app.on_event("startup")
def startup_event():
    global explainer, agent
    model_path = os.path.join(os.path.dirname(__file__), "ml", "artifacts", "model.joblib")
    explainer = ModelExplainer(model_path=model_path)
    agent = FraudInvestigationAgent()
    print("✅ Model and GenAI Agent initialized successfully.")

class TransactionRequest(BaseModel):
    user_id: str = Field(..., example="USR_9821")
    amount: float = Field(..., ge=0.0, example=650.0)
    tx_velocity_1h: int = Field(..., ge=0, example=8)
    avg_amount_ratio: float = Field(..., ge=0.0, example=5.4)
    is_foreign_ip: int = Field(..., ge=0, le=1, example=1)
    is_new_device: int = Field(..., ge=0, le=1, example=1)
    is_night_transaction: int = Field(..., ge=0, le=1, example=1)

class TransactionResponse(BaseModel):
    transaction_id: str
    user_id: str
    risk_score: float
    decision: str
    audit_report: Optional[IncidentAuditReport] = None

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Health check endpoint for Cloud Run and load balancers."""
    return {"status": "healthy", "service": "omnirisk-ai"}

@app.post("/api/v1/analyze", response_model=TransactionResponse, tags=["Inference"])
async def analyze_transaction(tx: TransactionRequest):
    """
    Evaluates transaction via ML model and triggers autonomous agent if risk threshold is breached.
    """
    if not explainer or not agent:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Services not loaded.")
    
    txn_id = f"TXN_{uuid.uuid4().hex[:8].upper()}"
    
    # 1. Run low-latency ML scoring & SHAP explanation (5-15ms)
    tx_data = tx.model_dump()
    user_id = tx_data.pop("user_id")
    
    ml_result = explainer.explain_transaction(tx_data)
    risk_score = ml_result["risk_score"]
    
    # 2. Decision logic: Low risk passes immediately
    if not ml_result["is_flagged"]:
        return TransactionResponse(
            transaction_id=txn_id,
            user_id=user_id,
            risk_score=risk_score,
            decision="APPROVED",
            audit_report=None
        )
    
    # 3. High risk: Trigger autonomous multi-agent forensic investigation
    drivers = [
        RiskFactor(feature=d["feature"], raw_value=d["raw_value"], shap_impact=d["shap_impact"])
        for d in ml_result["top_risk_drivers"]
    ]
    
    investigation_input = InvestigationInput(
        user_id=user_id,
        transaction_id=txn_id,
        amount=tx.amount,
        risk_score=risk_score,
        top_risk_drivers=drivers
    )
    
    audit_report = agent.investigate(investigation_input)
    
    return TransactionResponse(
        transaction_id=txn_id,
        user_id=user_id,
        risk_score=risk_score,
        decision=audit_report.verdict,
        audit_report=audit_report
    )