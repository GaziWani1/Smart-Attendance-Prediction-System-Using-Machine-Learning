"""
API integration tests using FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.dependencies import get_prediction_service
from backend.services.prediction_service import PredictionService


@pytest.fixture
def client():
    """Provides a TestClient for the FastAPI app."""
    return TestClient(app)


def test_api_root(client):
    """Test root endpoint returns 200 and application metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Smart Attendance" in data["app_name"]
    assert data["version"] == "1.0.0"
    assert data["status"] == "operational"


def test_api_health(client):
    """Test health endpoint returns 200 when model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "timestamp" in data


def test_api_model_info(client):
    """Test model-info endpoint returns metadata and metrics."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "selected_model" in data
    assert "metrics" in data
    assert "classes_conducted" in data["feature_names"]
    assert data["target_threshold"] == 75.0


def test_api_predict_valid_request(client):
    """Test successful risk prediction with valid input."""
    payload = {
        "classes_conducted": 45,
        "classes_attended": 32,
        "classes_remaining": 20,
        "previous_attendance_trend": "declining",
        "consecutive_absences_recent": 2,
        "participation_score": 6.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["attendance_risk"] in [0, 1]
    assert data["risk_label"] in ["At Risk", "Safe"]
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert data["target_threshold"] == 75.0
    assert "explanation" in data
    assert "deterministic_bounds" in data


def test_api_predict_validation_error_attended_exceeds_conducted(client):
    """Ensure impossible values receive HTTP 422 Unprocessable Entity."""
    payload = {
        "classes_conducted": 30,
        "classes_attended": 35,  # Exceeds conducted
        "classes_remaining": 20,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_api_predict_missing_required_fields(client):
    """Ensure missing mandatory features receive HTTP 422."""
    payload = {
        "classes_conducted": 30,
        # Missing classes_attended and classes_remaining
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_unhealthy_state_when_model_missing():
    """Verify that when a model fails to load, /health returns 503."""
    dummy_service = PredictionService(
        model_path="non_existent/pipeline.joblib",
        metrics_path="non_existent/metrics.json",
    )
    # Override dependency
    app.dependency_overrides[get_prediction_service] = lambda: dummy_service

    test_client = TestClient(app, raise_server_exceptions=False)
    try:
        health_resp = test_client.get("/health")
        assert health_resp.status_code == 503

        pred_resp = test_client.post(
            "/predict",
            json={
                "classes_conducted": 40,
                "classes_attended": 30,
                "classes_remaining": 20,
            },
        )
        assert pred_resp.status_code == 503
    finally:
        # Reset overrides
        app.dependency_overrides.clear()

