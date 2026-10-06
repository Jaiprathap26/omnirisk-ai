import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

from schemas import InvestigationInput, IncidentAuditReport
from tools import get_user_account_history, check_device_telemetry, lookup_fraud_policy

load_dotenv()

class FraudInvestigationAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.tools = [get_user_account_history, check_device_telemetry, lookup_fraud_policy]
        
        self.system_instruction = (
            "You are an elite Autonomous Fraud & Security Operations Forensics Agent. "
            "You receive an anomalous transaction flagged by an ML model along with its mathematical SHAP drivers. "
            "Investigate the user's account and device history using your tools, consult fraud compliance policies, "
            "and output an authoritative IncidentAuditReport."
        )

    def investigate(self, payload: InvestigationInput) -> IncidentAuditReport:
        drivers_summary = "\n".join([
            f"- {d.feature} (Value: {d.raw_value}): Added {d.shap_impact:+.4f} to risk"
            for d in payload.top_risk_drivers
        ])

        user_prompt = f"""
        INVESTIGATION TICKET:
        - User ID: {payload.user_id}
        - Transaction ID: {payload.transaction_id}
        - Amount: ${payload.amount}
        - ML Model Risk Score: {payload.risk_score} (Threshold: 0.70)
        
        Top Mathematical SHAP Risk Drivers:
        {drivers_summary}

        Investigate this account thoroughly using your tools and produce your final audit report.
        """

        # Using client.chats.create handles multi-turn Automatic Function Calling cleanly
        chat = self.client.chats.create(
            model="gemini-3.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                tools=self.tools,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=IncidentAuditReport
            )
        )

        response = chat.send_message(user_prompt)
        return IncidentAuditReport.model_validate_json(response.text)