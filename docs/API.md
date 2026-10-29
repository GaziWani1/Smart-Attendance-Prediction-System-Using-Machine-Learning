# REST API Documentation

The **Smart Attendance Prediction System** provides a high-performance RESTful API built with **FastAPI** and **Pydantic v2**.

- **Default Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger UI**: `http://127.0.0.1:8000/docs`
- **Interactive ReDoc UI**: `http://127.0.0.1:8000/redoc`

---

## 1. Endpoints Overview

| Method | Endpoint | Description | Status Codes |
|--------|----------|-------------|--------------|
| `GET` | `/` | Application overview and version | `200` |
| `GET` | `/health` | Service health and model readiness check | `200`, `503` |
| `GET` | `/model-info` | Metadata, feature schemas, and evaluation metrics | `200`, `503` |
| `POST` | `/predict` | Predict attendance risk and calculate bounds | `200`, `422`, `503` |

---

## 2. Endpoint Details

### 2.1 Root Application Info
- **URL**: `GET /`
- **Response `200 OK`**:
```json
{
  "app_name": "Smart Attendance Prediction System",
  "version": "1.0.0",
  "status": "operational",
  "docs_url": "/docs",
  "description": "Production-ready ML API estimating semester attendance threshold risk."
}
```

---

### 2.2 Health & Readiness Check
- **URL**: `GET /health`
- **Response `200 OK` (Healthy)**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "RandomForestClassifier",
  "timestamp": "2026-10-29T12:35:00.000000Z"
}
```
- **Response `503 Service Unavailable` (Degraded - Model Not Loaded)**:
```json
{
  "detail": {
    "status": "degraded",
    "model_loaded": false,
    "timestamp": "2026-10-29T12:35:00.000000Z",
    "message": "Model artifact is not available. Please train the model using 'python -m ml.train'."
  }
}
```

---

### 2.3 Model Information & Evaluation Metrics
- **URL**: `GET /model-info`
- **Response `200 OK`**:
```json
{
  "app_name": "Smart Attendance Prediction System",
  "version": "1.0.0",
  "selected_model": "RandomForestClassifier",
  "target_definition": "attendance_risk: 1 if final semester attendance < 75.0%, else 0",
  "target_threshold": 75.0,
  "dataset_type": "synthetic (generated via behavioral stochastic simulation, zero data leakage)",
  "feature_names": [
    "classes_conducted",
    "classes_attended",
    "classes_remaining",
    "current_attendance_pct",
    "consecutive_absences_recent",
    "participation_score",
    "previous_attendance_trend"
  ],
  "numerical_features": [
    "classes_conducted",
    "classes_attended",
    "classes_remaining",
    "current_attendance_pct",
    "consecutive_absences_recent",
    "participation_score"
  ],
  "categorical_features": [
    "previous_attendance_trend"
  ],
  "metrics": {
    "accuracy": 0.964,
    "precision": 0.9076,
    "recall": 0.9391,
    "f1_score": 0.9231,
    "roc_auc": 0.9936,
    "confusion_matrix": {
      "true_negatives": 374,
      "false_positives": 11,
      "false_negatives": 7,
      "true_positives": 108
    }
  },
  "feature_importances_or_coefficients": {
    "current_attendance_pct": 0.384,
    "classes_attended": 0.215,
    "consecutive_absences_recent": 0.142,
    "participation_score": 0.118,
    "classes_conducted": 0.081,
    "previous_attendance_trend_declining": 0.035,
    "classes_remaining": 0.025
  }
}
```

---

### 2.4 Predict Attendance Risk
- **URL**: `POST /predict`
- **Request Headers**: `Content-Type: application/json`
- **Request Body**:
```json
{
  "classes_conducted": 50,
  "classes_attended": 34,
  "classes_remaining": 20,
  "previous_attendance_trend": "declining",
  "consecutive_absences_recent": 2,
  "participation_score": 6.5
}
```

#### Field Specifications & Constraints
| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `classes_conducted` | integer | Yes | $> 0, \le 200$ | Lectures held so far |
| `classes_attended` | integer | Yes | $\ge 0, \le \text{conducted}$ | Lectures attended so far |
| `classes_remaining` | integer | Yes | $> 0, \le 150$ | Scheduled classes left |
| `previous_attendance_trend` | string | No (default: `"stable"`) | `"declining"`, `"stable"`, `"improving"` | Recent trajectory |
| `consecutive_absences_recent` | integer | No (default: `0`) | $\ge 0, \le 50$ | Recent missed streak |
| `participation_score` | float | No (default: `7.0`) | $1.0 \le S \le 10.0$ | Engagement rating |

#### Success Response (`200 OK`):
```json
{
  "attendance_risk": 1,
  "risk_label": "At Risk",
  "risk_probability": 0.8124,
  "target_threshold": 75.0,
  "explanation": "The model estimates an elevated risk (81.2% probability) of finishing below 75%. Key contributing patterns: current attendance is below threshold (68.0%), attendance trend has been declining recently, student has 2 recent consecutive absences. To reach 75%, the student must attend at least 19 of the remaining 20 classes (95.0% future attendance rate).",
  "deterministic_bounds": {
    "current_attendance_pct": 68.0,
    "total_semester_classes": 70,
    "max_possible_attendance_pct": 77.14,
    "min_possible_attendance_pct": 48.57,
    "classes_needed_for_75_pct": 19,
    "mathematical_status": "behavior_dependent"
  }
}
```

#### Validation Error Response (`422 Unprocessable Entity`):
If `classes_attended > classes_conducted`:
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body"],
      "msg": "Value error, Validation Error: classes_attended (45) cannot exceed classes_conducted (40).",
      "input": {
        "classes_conducted": 40,
        "classes_attended": 45,
        "classes_remaining": 20
      }
    }
  ]
}
```

---

## 3. Example `curl` Commands

### Check Health:
```bash
curl -X GET "http://127.0.0.1:8000/health"
```

### Make Prediction:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "classes_conducted": 50,
       "classes_attended": 34,
       "classes_remaining": 20,
       "previous_attendance_trend": "declining",
       "consecutive_absences_recent": 2,
       "participation_score": 6.5
     }'
```

