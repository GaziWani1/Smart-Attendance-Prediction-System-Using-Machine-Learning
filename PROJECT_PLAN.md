# Smart Attendance Prediction System - Project Plan

## 1. Project Objective and Scope

### 1.1 Objective
The **Smart Attendance Prediction System** is a machine learning-powered decision-support application designed to proactively identify university students who are at risk of finishing a semester below the mandatory 75% attendance threshold.

Unlike simple percentage calculators that merely report historical rates, this system predicts **future outcome probability** based on behavioral and engagement indicators up to the current point in time. It enables academic advisors and students to intervene before failure becomes mathematically unavoidable.

### 1.2 Scope
- **In-Scope**:
  - Synthetic dataset generation simulating realistic student semester attendance trajectories without data leakage.
  - Training and comparing scikit-learn models (Logistic Regression baseline vs. Random Forest classifier).
  - Pipeline serialization with Joblib (fitted preprocessor + model).
  - High-performance FastAPI backend serving predictions and metadata.
  - Interactive Streamlit frontend featuring an analytical dashboard, real-time prediction interface, and transparent model explainability.
  - Deterministic calculations alongside ML predictions (e.g., minimum required attendance, mathematically impossible bounds).
  - Comprehensive automated test suite with Pytest and FastAPI TestClient.
- **Out-of-Scope**:
  - Live university database integrations or student information systems (SIS).
  - Biometric, facial recognition, or QR-code hardware scanners.
  - User authentication and multi-tenant authorization (kept lightweight for portfolio learning).
  - Persistent external databases (PostgreSQL, MongoDB, Redis).

---

## 2. Functional Requirements

- **FR-1: Dataset Generation & Ingestion**: Produce a reproducible, realistic synthetic dataset with documented features and target definition.
- **FR-2: Model Training & Evaluation**: Train Logistic Regression and Random Forest models with stratified train/validation/test splits, evaluating accuracy, precision, recall, F1, and ROC-AUC.
- **FR-3: Pipeline Serialization**: Save the fitted preprocessing transformers and selected model into a single Joblib artifact.
- **FR-4: Backend REST API**:
  - `GET /`: Health & application info.
  - `GET /health`: Readiness probe checking model availability.
  - `GET /model-info`: Safe metadata, feature schemas, and test performance.
  - `POST /predict`: Real-time risk probability and classification inference.
- **FR-5: Input Validation**: Strict validation using Pydantic preventing impossible values (e.g., attended classes > conducted classes).
- **FR-6: Streamlit Multi-Page UI**:
  - *Page 1 (Dashboard)*: Dataset demographics, class distribution, interactive visualizations.
  - *Page 2 (Predict Attendance)*: Interactive prediction form, model probability, risk category, and deterministic calculations.
  - *Page 3 (Model Insights)*: Confusion matrix, metrics comparison, feature importance, and model limitations.

---

## 3. Non-Functional Requirements

- **Performance**: Sub-100ms API inference latency; single model load in-memory on backend startup (no re-loading per request).
- **Resource Efficiency**: Optimized for a MacBook Air M2 (8GB RAM), minimal memory footprint (< 300MB RAM combined).
- **Reliability & Error Handling**: Graceful degradation when model artifacts are missing; friendly error messages in Streamlit if backend is offline.
- **Separation of Concerns**: Complete isolation between UI (Streamlit) and business logic (FastAPI via HTTP). No direct ML imports in Streamlit.
- **Security & Privacy**: Zero sensitive personal data; anonymous synthetic IDs; CORS restricted to local frontend origins; input sanitization.

---

## 4. User Workflow

```
[ Academic Advisor / Student ]
              │
              ▼
   Opens Streamlit Frontend
              │
      ┌───────┴─────────────────────────┐
      ▼                                 ▼
Explore Historical Trends       Input Student Attendance Data
(Dashboard & Visualizations)    - Classes Conducted
                                - Classes Attended
                                - Classes Remaining
                                - Recent Attendance Trend
                                - Consecutive Absences
                                - Participation Score
                                        │
                                        ▼
                               Click "Predict Risk"
                                        │
                                        ▼
                               HTTP POST to FastAPI
                                        │
                            ┌───────────┴───────────┐
                            ▼                       ▼
                     Pydantic Schema          Joblib Pipeline
                       Validation               Inference
                            │                       │
                            └───────────┬───────────┘
                                        ▼
                            HTTP Response (JSON)
                            - Risk Label ("At Risk" / "Safe")
                            - Risk Probability (e.g. 78.4%)
                            - Advisory Explanation
                            - Deterministic Bounds
                                        │
                                        ▼
                            Streamlit UI Renders
                            - Color-coded Risk Badge
                            - Gauge / Probability Meter
                            - Mathematical Recovery Plan
```

---

## 5. Architecture and Request Flow

```
┌────────────────────────────────────────────────────────┐
│               STREAMLIT FRONTEND (Port 8501)           │
│  - Dashboard (Plotly Charts)                           │
│  - Predict Attendance (Forms & Sliders)                │
│  - Model Insights (Feature Importance & Confusion Mat) │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP REST (requests)
                           ▼
┌────────────────────────────────────────────────────────┐
│                FASTAPI BACKEND (Port 8000)             │
│  - CORS Middleware & Logging                           │
│  - Pydantic Request / Response Validation              │
│  - PredictionService (Singleton in memory)             │
└──────────────────────────┬─────────────────────────────┘
                           │ Joblib In-Memory Loading
                           ▼
┌────────────────────────────────────────────────────────┐
│            TRAINED ML PIPELINE & METRICS               │
│  - artifacts/attendance_pipeline.joblib                │
│  - artifacts/model_metrics.json                        │
└────────────────────────────────────────────────────────┘
```

---

## 6. Dataset Requirements & Target Definition

### 6.1 No Data Leakage Target Definition
- **Total Semester Classes** = `classes_conducted` + `classes_remaining`.
- **Target Label (`attendance_risk`)**:
  - `attendance_risk = 1` if final semester attendance rate is strictly less than 75.0% (`final_attendance_pct < 75.0`).
  - `attendance_risk = 0` if final semester attendance rate is `>= 75.0%`.
- **Leakage Prevention**:
  The target is generated from a simulated future attendance trial (`future_classes_attended`) across `classes_remaining`. At prediction time, neither `future_classes_attended` nor `final_attendance_pct` is accessible to the feature vector.

### 6.2 Features Available at Prediction Time
1. `classes_conducted` (int, > 0): Number of classes held up to the prediction checkpoint.
2. `classes_attended` (int, >= 0 and <= `classes_conducted`): Classes attended so far.
3. `classes_remaining` (int, > 0): Scheduled classes left in the semester.
4. `current_attendance_pct` (float, 0.0 - 100.0): `round((classes_attended / classes_conducted) * 100, 2)`.
5. `previous_attendance_trend` (categorical: `declining`, `stable`, `improving`): Direction of student attendance in recent weeks.
6. `consecutive_absences_recent` (int, >= 0): Consecutive missed sessions preceding the checkpoint.
7. `participation_score` (float, 1.0 - 10.0): Classroom engagement / quiz completion index.

---

## 7. ML Training and Evaluation Strategy

- **Algorithms**:
  1. *Baseline*: Logistic Regression (interpretable, calibrated probability estimates).
  2. *Challenger*: Random Forest Classifier (captures non-linear feature interactions).
- **Data Splitting**: Stratified 80/20 train-test split (and 5-fold stratified cross-validation on the train split to prevent overfitting).
- **Evaluation Metrics**:
  - **Recall on At-Risk (Positive Class)**: Crucial because failing to identify an at-risk student (False Negative) has severe academic consequences.
  - **Precision on At-Risk**: Important to avoid advisor intervention fatigue from False Positives.
  - **F1-Score**: Harmonic mean balancing precision and recall.
  - **ROC-AUC & Confusion Matrix**: Overall discrimination threshold performance.
- **Serialization**: Complete pipeline containing `ColumnTransformer` (StandardScaler for numerics, OneHotEncoder for categoricals) + final estimator saved via `joblib.dump()`.

---

## 8. API Endpoint Specifications

- **`GET /`**:
  - Status 200: Returns API name, version, status, and summary.
- **`GET /health`**:
  - Status 200: Returns `{"status": "healthy", "model_loaded": true, "timestamp": "..."}`.
  - Status 503: Returns `{"status": "unhealthy", "model_loaded": false}` if artifact missing.
- **`GET /model-info`**:
  - Status 200: Returns model name, features, target definition, and validation metrics.
- **`POST /predict`**:
  - Request Body: JSON with student features.
  - Status 200: Returns risk classification, probability, explanation, and deterministic bounds.
  - Status 422: Returns readable validation errors if counts are contradictory or out of range.

---

## 9. Frontend Page Specifications

- **`app.py`**: Entrypoint, navigation sidebar, system status banner, and global theme.
- **`pages/1_Dashboard.py`**:
  - KPI metric cards (Total students, Class conducted average, At-risk rate).
  - Distribution histograms and interactive pie/bar charts with Plotly.
  - Synthetic data transparency callout.
- **`pages/2_Predict_Attendance.py`**:
  - Interactive prediction inputs with real-time range constraints.
  - Predict action with loading indicator.
  - Risk verdict display card with risk badge and gauge chart.
  - Mathematical recovery plan table (consecutive classes needed to reach 75%).
- **`pages/3_Model_Insights.py`**:
  - Metric comparison tables (Baseline vs Random Forest).
  - Confusion Matrix visualization.
  - Feature importance / odds ratio explanation.
  - Model caveats, assumptions, and ethical ML disclaimers.

---

## 10. Folder Structure

```
Smart Student Attendence system/
├── backend/
│   ├── __init__.py
│   ├── config.py
│   ├── dependencies.py
│   ├── main.py
│   ├── schemas.py
│   └── services/
│       ├── __init__.py
│       └── prediction_service.py
├── frontend/
│   ├── app.py
│   ├── pages/
│   │   ├── 1_Dashboard.py
│   │   ├── 2_Predict_Attendance.py
│   │   └── 3_Model_Insights.py
│   └── components/
│       ├── __init__.py
│       ├── api_client.py
│       └── ui_helpers.py
├── ml/
│   ├── __init__.py
│   ├── generate_dataset.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── data/
│   └── attendance_dataset.csv
├── artifacts/
│   ├── attendance_pipeline.joblib
│   └── model_metrics.json
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_prediction.py
│   └── test_schemas.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── API.md
│   └── ML_METHODOLOGY.md
├── .gitignore
├── .env.example
├── requirements.txt
├── README.md
├── PROJECT_PLAN.md
└── TASKS.md
```

---

## 11. Dependencies and Environment Setup

- Supported Python: **Python 3.10 to 3.13**.
- Setup commands:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

---

## 12. Testing Strategy

- **Schema Tests**: Pydantic constraints, bounds, attended > conducted rejections.
- **Prediction Service Tests**: Artifact loading, inference formatting, output validation.
- **API Tests**: FastAPI `TestClient` covering all endpoints, 422 validations, 503 health states.
- **ML Integrity Tests**: Reproducible synthetic generator, pipeline transformer stability, probability ranges (0.0 to 1.0).

---

## 13. Common Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Data Leakage | High | Strictly hide future attendance trials from the training feature set. |
| Inconsistent Inputs | Medium | Enforce Pydantic root validators (`attended <= conducted`, `remaining > 0`). |
| Missing Model File | High | Graceful API fallback with 503 status code and user-friendly instruction. |
| Frontend-Backend Coupling | Medium | Pure HTTP client communication via `requests` with timeouts and retry safety. |
| Misleading Synthetic Accuracy | Low | Prominent UI and documentation badges explicitly stating data is synthetic. |

---

## 14. Development Milestones & Completion Criteria

- **Milestone 1**: Project Plan, Architecture, and Task Checklist.
- **Milestone 2**: Synthetic Dataset Generation & Validation.
- **Milestone 3**: ML Training Pipeline, Model Comparison, and Serialization.
- **Milestone 4**: FastAPI Backend, Prediction Service, and Endpoints.
- **Milestone 5**: Streamlit Multi-Page Application & HTTP Integration.
- **Milestone 6**: Automated Test Suite & Edge Case Hardening.
- **Milestone 7**: Comprehensive Documentation & Startup Verification.

