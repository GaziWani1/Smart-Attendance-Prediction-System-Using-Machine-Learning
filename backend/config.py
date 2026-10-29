"""
Smart Attendance Prediction System - Backend Configuration
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Server Settings
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("API_DEBUG", "False").lower() in ("true", "1", "t")

# Artifact Paths
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(ARTIFACTS_DIR / "attendance_pipeline.joblib")
)
METRICS_PATH = os.getenv(
    "METRICS_PATH",
    str(ARTIFACTS_DIR / "model_metrics.json")
)
DATASET_PATH = os.getenv(
    "DATASET_PATH",
    str(DATA_DIR / "attendance_dataset.csv")
)

# CORS Origins
CORS_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

