# System Architecture

## 1. Architectural Philosophy & Principles

The **Smart Attendance Prediction System** is structured following modern production ML engineering practices:

1. **Strict Separation of Concerns**: The user interface (Streamlit) and prediction engine (FastAPI) run in separate operating system processes and communicate solely over standard HTTP/REST.
2. **Zero In-UI Inference**: Streamlit never imports `sklearn`, `joblib`, or the backend `PredictionService`. If the backend is modified, upgraded, or containerized in the future, the UI remains unaffected.
3. **In-Memory Model Caching**: The trained Joblib model artifact is loaded into memory exactly once when FastAPI boots up (via FastAPI's `lifespan` handler). Subsequent predictions execute with sub-5ms latency with zero disk read overhead.
4. **Distinction Between Deterministic Calculation & ML Prediction**: The system clearly separates physical mathematical boundaries (e.g., maximum possible attendance percentage) from stochastic behavioral predictions.
5. **Lightweight Footprint**: Zero external database dependencies (no PostgreSQL, MongoDB, Redis). The entire stack operates comfortably within 250MB RAM on a base MacBook Air M2 (8GB RAM).

---

## 2. High-Level System Architecture

```
┌────────────────────────────────────────────────────────┐
│               STREAMLIT FRONTEND (Port 8501)           │
│                                                        │
│   ┌──────────────┐  ┌──────────────────┐  ┌─────────┐  │
│   │ 1_Dashboard  │  │ 2_Predict_Attend │  │3_Insights│ │
│   └──────┬───────┘  └─────────┬────────┘  └────┬────┘  │
│          │                    │                │       │
│          └────────────────────┼────────────────┘       │
│                               │                        │
│                   HTTP Requests (api_client.py)        │
└───────────────────────────────┼────────────────────────┘
                                │ JSON Over HTTP
                                ▼
┌────────────────────────────────────────────────────────┐
│                FASTAPI BACKEND (Port 8000)             │
│                                                        │
│  - FastAPI Application (backend/main.py)               │
│  - CORS Middleware & Centralized Logging               │
│  - Pydantic v2 Request Validation (schemas.py)         │
│  - Dependency Injection (dependencies.py)              │
│                                                        │
│              ┌────────────────────────────────┐        │
│              │       PredictionService        │        │
│              │ - Computes Math Bounds         │        │
│              │ - Formats Feature DataFrame    │        │
│              │ - Generates Explanations       │        │
│              └───────────────┬────────────────┘        │
└──────────────────────────────┼─────────────────────────┘
                               │ In-Memory Pipeline Call
                               ▼
┌────────────────────────────────────────────────────────┐
│           SCIKIT-LEARN ML PIPELINE (Joblib)            │
│                                                        │
│   StandardScaler + OneHotEncoder  ==> RandomForest     │
│   (Fitted Preprocessor)                Classifier      │
└────────────────────────────────────────────────────────┘
```

---

## 3. Request & Response Lifecycle

When a user submits the prediction form on the Streamlit UI:

```
[User clicks "Predict Attendance Risk"]
                  │
                  ▼
  [Streamlit UI: 2_Predict_Attendance.py]
  - Performs initial boundary checks (classes_attended <= classes_conducted)
  - Packages form data into a JSON dictionary
                  │
                  ▼
  [Frontend Client: api_client.py]
  - Issues HTTP POST request to http://127.0.0.1:8000/predict
  - Manages timeouts (5s) and connection exceptions
                  │
                  ▼
  [FastAPI Backend: backend/main.py]
  - Receives HTTP POST request
  - Pydantic schema validation: AttendancePredictionRequest
    * Enforces: attended <= conducted
    * Enforces: conducted > 0, remaining > 0
    * Enforces: participation_score between 1.0 and 10.0
                  │
                  ▼
  [Service Layer: PredictionService]
  - Computes deterministic bounds:
      max_possible_attendance_pct = (attended + remaining) / total * 100
      classes_needed_for_75 = ceil(0.75 * total - attended)
  - Structures single-row pandas.DataFrame with exact feature schema
  - Calls pipeline.predict(df) and pipeline.predict_proba(df)
  - Constructs transparent plain-English explanation
                  │
                  ▼
  [HTTP 200 Response Payload]
  - attendance_risk: 1 or 0
  - risk_label: "At Risk" or "Safe"
  - risk_probability: 0.82
  - deterministic_bounds: { ... }
  - explanation: "..."
                  │
                  ▼
  [Streamlit UI Renders]
  - Displays color-coded risk status badge
  - Visualizes risk probability gauge chart
  - Shows deterministic mathematical recovery roadmap
```

---

## 4. Deterministic vs. Probabilistic Domain Logic

A core innovation in this system is avoiding misleading ML claims when a case is mathematically determined:

1. **Case A: Mathematically Impossible to Reach 75%**
   - Condition: $\frac{\text{classes\_attended} + \text{classes\_remaining}}{\text{classes\_conducted} + \text{classes\_remaining}} \times 100 < 75.0\%$
   - Behavior: Even if the student attends every remaining lecture, they cannot reach 75%.
   - Action: The system flags this as a mathematical certainty, overriding statistical uncertainty to avoid false optimism.

2. **Case B: Mathematically Guaranteed Safe**
   - Condition: $\frac{\text{classes\_attended}}{\text{classes\_conducted} + \text{classes\_remaining}} \times 100 \ge 75.0\%$
   - Behavior: Even if the student attends zero remaining lectures, they cannot fall below 75%.
   - Action: The system flags this as a mathematical guarantee.

3. **Case C: Behavior-Dependent (The ML Domain)**
   - Condition: $\text{min\_possible} < 75.0\% \le \text{max\_possible}$.
   - Behavior: The student's outcome depends on their future choices.
   - Action: Here the Random Forest classifier provides critical insight, estimating risk probability based on attendance momentum, absence streaks, and engagement scores.

---

## 5. Storage and Artifact Design

| Artifact File | Format | Purpose | Loaded When |
|---------------|--------|---------|-------------|
| `data/attendance_dataset.csv` | CSV | Synthetic training dataset (2,500 records) | During training & Dashboard view |
| `artifacts/attendance_pipeline.joblib` | Binary (Joblib) | Serialized preprocessing pipeline + trained classifier | Once at FastAPI startup |
| `artifacts/model_metrics.json` | JSON | Model evaluation metrics, comparison, and feature importances | At FastAPI startup & Insights view |

