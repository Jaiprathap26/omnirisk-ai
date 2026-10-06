import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from dataset import generate_transaction_data

def train_risk_model(output_dir: str = "artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("1. Generating dataset...")
    df = generate_transaction_data(n_samples=15000)
    
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    
    # 80% train, 20% test split stratified by target to preserve fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("2. Building production Pipeline with class balancing...")
    # class_weight='balanced' ensures rare fraud cases receive proportional learning weight
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("3. Training model...")
    pipeline.fit(X_train, y_train)
    
    print("4. Evaluating model on unseen test set...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Save the complete pipeline and feature names
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump({
        "pipeline": pipeline,
        "feature_names": list(X.columns)
    }, model_path)
    print(f"\nModel serialized successfully to: {model_path}")

if __name__ == "__main__":
    train_risk_model()