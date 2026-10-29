"""
Smart Attendance Prediction System - FastAPI Application

Main backend application exposing REST API endpoints for:
- Health and model readiness checks
- Model metadata and evaluation metrics
- Attendance risk prediction
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import CORS_ORIGINS, DEBUG
from backend.schemas import (
    AttendancePredictionRequest,
    AttendancePredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from backend.services.prediction_service import PredictionService
from backend.dependencies import get_prediction_service
from ml.preprocessing import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("attendance_backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle management: Loads model artifacts into memory on startup.
    """
    logger.info("Initializing Smart Attendance Prediction Backend...")
    service = get_prediction_service()
    if service.is_loaded:
        logger.info(f"Model successfully loaded: {service.model_name}")
    else:
        logger.warning("Model failed to load on startup. Run 'python -m ml.train' to build artifacts.")
    yield
    logger.info("Shutting down Smart Attendance Prediction Backend...")


app = FastAPI(
    title="Smart Attendance Prediction System API",
    description="Machine Learning REST API to estimate university student attendance risk.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for local Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get(
    "/",
    tags=["General"],
    summary="Root application info",
)
async def root():
    """
    Returns application name, version, and documentation link.
    """
    return {
        "app_name": "Smart Attendance Prediction System",
        "version": "1.0.0",
        "status": "operational",
        "docs_url": "/docs",
        "description": "Production-ready ML API estimating semester attendance threshold risk.",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health and model readiness check",
)
async def health(service: PredictionService = Depends(get_prediction_service)):
    """
    Checks system health and ensures model pipeline is loaded in memory.
    Returns HTTP 503 if the required model is missing or failed to load.
    """
    current_time = datetime.now(timezone.utc).isoformat()
    if not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "degraded",
                "model_loaded": False,
                "timestamp": current_time,
                "message": "Model artifact is not available. Please train the model using 'python -m ml.train'.",
            },
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=service.model_name,
        timestamp=current_time,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Machine Learning"],
    summary="Retrieve model metadata and performance metrics",
)
async def model_info(service: PredictionService = Depends(get_prediction_service)):
    """
    Provides transparent metadata about the selected classifier,
    feature schemas, dataset type, and evaluation results on test data.
    """
    if not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is unavailable because the model pipeline has not been trained.",
        )

    meta = service.metrics_data
    selected_model = meta.get("selected_model", "Unknown")
    model_details = meta.get("models", {}).get(selected_model, {})
    test_metrics = model_details.get("test_metrics", {})

    importances = (
        model_details.get("feature_importances")
        or model_details.get("coefficients")
        or {}
    )

    return ModelInfoResponse(
        app_name="Smart Attendance Prediction System",
        version="1.0.0",
        selected_model=selected_model,
        target_definition=meta.get(
            "target_definition",
            "attendance_risk: 1 if final semester attendance < 75.0%, else 0",
        ),
        target_threshold=meta.get("target_threshold", 75.0),
        dataset_type=meta.get("dataset_type", "synthetic"),
        feature_names=ALL_FEATURES,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        metrics=test_metrics,
        feature_importances_or_coefficients=importances,
    )


@app.post(
    "/predict",
    response_model=AttendancePredictionResponse,
    tags=["Machine Learning"],
    summary="Predict attendance risk for a student",
)
async def predict_attendance(
    request: AttendancePredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
):
    """
    Accepts validated student attendance metrics and returns:
    - attendance_risk: 1 (At Risk) or 0 (Safe)
    - risk_probability: model calibrated risk probability (0.0 to 1.0)
    - target_threshold: 75.0
    - explanation: plain-English reasoning
    - deterministic_bounds: mathematically calculated attendance boundaries
    """
    try:
        response = service.predict(request)
        return response
    except RuntimeError as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected prediction failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during prediction.",
        )

