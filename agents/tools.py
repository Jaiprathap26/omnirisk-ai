def get_user_account_history(user_id: str) -> dict:
    """
    Fetches past account history, travel notices, and chargebacks for a user.
    """
    # Mock database lookup
    db = {
        "USR_9821": {
            "account_age_days": 420,
            "has_active_travel_notice": False,
            "disputed_chargebacks": 0,
            "kyc_verified": True
        },
        "USR_1044": {
            "account_age_days": 12,
            "has_active_travel_notice": False,
            "disputed_chargebacks": 3,
            "kyc_verified": False
        }
    }
    return db.get(user_id, {
        "account_age_days": 30,
        "has_active_travel_notice": False,
        "disputed_chargebacks": 1,
        "kyc_verified": True
    })

def check_device_telemetry(user_id: str) -> dict:
    """
    Checks device trust status and known device fingerprints for a user.
    """
    return {
        "user_id": user_id,
        "device_recognized": False,
        "is_rooted_or_emulator": True,
        "device_risk_tier": "HIGH"
    }

def lookup_fraud_policy(risk_tier: str) -> dict:
    """
    Returns the company's mandated compliance rules for a given risk tier.
    """
    policies = {
        "CRITICAL": "Mandatory immediate account freeze if device is emulator or chargebacks > 2.",
        "HIGH": "Issue step-up 2FA challenge if account is older than 90 days and KYC verified.",
        "LOW": "Allow transaction and append to daily anomaly review ledger."
    }
    return {"risk_tier": risk_tier, "policy": policies.get(risk_tier, policies["HIGH"])}