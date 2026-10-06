import os
import joblib
import numpy as np
import pandas as pd
import shap

class ModelExplainer:
    def __init__(self, model_path: str = "artifacts/model.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Run ml/train.py first.")
        
        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.feature_names = artifact["feature_names"]
        
        # Extract components from pipeline
        self.scaler = self.pipeline.named_steps["scaler"]
        self.classifier = self.pipeline.named_steps["classifier"]
        
        # Initialize TreeSHAP explainer on the trained Random Forest
        self.explainer = shap.TreeExplainer(self.classifier)

    def explain_transaction(self, transaction_data: dict) -> dict:
        """
        Takes a single transaction dictionary, computes prediction probability,
        and extracts the exact top positive and negative SHAP risk drivers.
        """
        # Convert dict to DataFrame matching training feature columns
        df = pd.DataFrame([transaction_data])[self.feature_names]
        
        # Scale inputs using the fitted scaler
        scaled_features = self.scaler.transform(df)
        
        # Predict probability of fraud (class 1)
        risk_score = float(self.classifier.predict_proba(scaled_features)[0, 1])
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(scaled_features)
        
        # Handle SHAP output format differences across versions:
        # In binary classification, shap_values can be a list [class_0, class_1] or an array
        if isinstance(shap_values, list):
            class_1_shap = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            class_1_shap = shap_values[0, :, 1]
        else:
            class_1_shap = shap_values[0]
            
        # Pair feature names with their mathematical SHAP impact
        feature_impacts = []
        for name, value, shap_val in zip(self.feature_names, df.iloc[0], class_1_shap):
            feature_impacts.append({
                "feature": name,
                "raw_value": float(value),
                "shap_impact": float(shap_val),
                "direction": "INCREASED_RISK" if shap_val > 0 else "DECREASED_RISK"
            })
            
        # Sort by absolute impact (highest magnitude first)
        feature_impacts.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
        
        # Separate top drivers
        top_risk_drivers = [f for f in feature_impacts if f["direction"] == "INCREASED_RISK"][:3]
        top_safe_drivers = [f for f in feature_impacts if f["direction"] == "DECREASED_RISK"][:2]
        
        return {
            "risk_score": round(risk_score, 4),
            "is_flagged": risk_score >= 0.70,
            "top_risk_drivers": top_risk_drivers,
            "top_safe_drivers": top_safe_drivers
        }

if __name__ == "__main__":
    explainer = ModelExplainer()
    
    # Test on an anomalous high-risk transaction
    suspicious_tx = {
        "amount": 650.0,
        "tx_velocity_1h": 8,
        "avg_amount_ratio": 5.4,
        "is_foreign_ip": 1,
        "is_new_device": 1,
        "is_night_transaction": 1
    }
    
    result = explainer.explain_transaction(suspicious_tx)
    print("\n--- SHAP EXPLANATION RESULT ---")
    print(f"Risk Score: {result['risk_score']} | Flagged: {result['is_flagged']}")
    print("\nTop Factors Driving Risk Up:")
    for f in result["top_risk_drivers"]:
        print(f"  • {f['feature']} (value: {f['raw_value']}) -> Added {f['shap_impact']:+.4f} to risk")
        
    print("\nTop Factors Keeping Risk Down:")
    for f in result["top_safe_drivers"]:
        print(f"  • {f['feature']} (value: {f['raw_value']}) -> Subtracted {f['shap_impact']:+.4f} from risk")