# 🛡️ OmniRisk AI — Real-Time Predictive ML & Autonomous Agent Intelligence

[[https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi]](https://fastapi.tiangolo.com/)
[[https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white]](https://scikit-learn.org/)
[[https://img.shields.io/badge/Google%20Gemini-1.5%20Flash-8E75C4?logo=google&logoColor=white]](https://ai.google.dev/)
[[https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white]](https://www.docker.com/)

OmniRisk AI is an enterprise fraud detection and risk intelligence platform that bridges traditional machine learning with autonomous generative AI agents.

## 🏗️ Architecture
Incoming Transaction JSON
│
▼
┌─────────────────────────────────┐
│ 1. Scikit-Learn Pipeline (~10ms)│
│    (Scaling + Random Forest)    │
└────────────────┬────────────────┘
│
Risk Threshold >= 0.70?
├── NO  ──► Status: APPROVED (< 15ms total latency)
└── YES ──► Extract Top 3 SHAP Drivers
│
▼
┌───────────────────────────┐
│ 2. Gemini Forensics Agent │
│    (Automatic Tool Calls) │
└─────────────┬─────────────┘
├── Device Telemetry Tool
├── Account History DB Tool
└── Compliance Policy Tool
│
▼
┌───────────────────────────┐
│ 3. Pydantic v2 Contract   │
│    (Strict JSON Verdict)  │
└───────────────────────────┘


## 🚀 Key Technical Highlights
- **Cost-Optimized Dual Path:** Low-risk transactions are approved via local ML inference in ~10ms without calling external LLM APIs.
- **Mathematical Explainability (SHAP):** Decomposes black-box tree predictions into exact additive contributions using TreeSHAP before triggering forensic review.
- **Autonomous Multi-Turn Tool Calling:** Gemini 1.5 Flash queries account age, KYC status, and emulator flags via function calling.
- **Strict Schema Enforcement:** Enforces Pydantic v2 schemas on model output, preventing JSON deserialization errors.

## 🛠️ Local Execution
```bash
git clone [https://github.com/Jaiprathap26/omnirisk-ai.git](https://github.com/Jaiprathap26/omnirisk-ai.git)
cd omnirisk-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train model artifact
python ml/train.py

# Run API
export GEMINI_API_KEY="your-api-key"
uvicorn main:app --host 0.0.0.0 --port 8080