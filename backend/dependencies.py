"""
Smart Attendance Prediction System - Dependencies Module
"""

from typing import Optional
from backend.services.prediction_service import PredictionService
from backend.config import MODEL_PATH, METRICS_PATH

# Global instance of PredictionService
_prediction_service_instance: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """
    Dependency provider that returns the singleton PredictionService.
    Initializes on first invocation if not already loaded.
    """
    global _prediction_service_instance
    if _prediction_service_instance is None:
        _prediction_service_instance = PredictionService(
            model_path=MODEL_PATH,
            metrics_path=METRICS_PATH,
        )
    return _prediction_service_instance

