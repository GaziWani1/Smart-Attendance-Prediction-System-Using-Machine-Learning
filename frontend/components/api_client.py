"""
Smart Attendance Prediction System - Frontend API Client

Handles decoupled HTTP communication with the FastAPI backend.
"""

import os
from typing import Tuple, Dict, Any
import requests

# Base URL for the FastAPI backend, configurable via environment variable
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 5.0  # seconds


def check_backend_health() -> Tuple[bool, Dict[str, Any]]:
    """
    Checks if the FastAPI backend is running and the ML model is loaded.

    Returns:
        Tuple[bool, Dict]: (is_healthy, response_data_or_error)
    """
    url = f"{BACKEND_API_URL}/health"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return True, response.json()
        elif response.status_code == 503:
            return False, {
                "error": "Model Not Loaded",
                "message": "Backend is reachable, but the ML model is not loaded. Run 'python -m ml.train'.",
                "details": response.json(),
            }
        else:
            return False, {
                "error": f"HTTP {response.status_code}",
                "message": response.text,
            }
    except requests.exceptions.ConnectionError:
        return False, {
            "error": "Connection Failed",
            "message": f"Could not connect to FastAPI at {BACKEND_API_URL}. Is the backend server running?",
        }
    except requests.exceptions.Timeout:
        return False, {
            "error": "Timeout",
            "message": f"Connection to {BACKEND_API_URL} timed out after {REQUEST_TIMEOUT}s.",
        }
    except Exception as e:
        return False, {
            "error": "Unexpected Error",
            "message": str(e),
        }


def get_model_info() -> Tuple[bool, Dict[str, Any]]:
    """
    Fetches model metadata, training configuration, and test metrics from the backend.

    Returns:
        Tuple[bool, Dict]: (success, model_info_or_error)
    """
    url = f"{BACKEND_API_URL}/model-info"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {
                "error": f"HTTP {response.status_code}",
                "message": response.text,
            }
    except Exception as e:
        return False, {
            "error": "Request Failed",
            "message": str(e),
        }


def predict_attendance_risk(payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Sends attendance features to the FastAPI backend and retrieves prediction.

    Returns:
        Tuple[bool, Dict]: (success, prediction_response_or_error)
    """
    url = f"{BACKEND_API_URL}/predict"
    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return True, response.json()
        elif response.status_code == 422:
            return False, {
                "error": "Validation Error",
                "message": "The backend rejected the input data. Please check your input values.",
                "details": response.json().get("detail", []),
            }
        elif response.status_code == 503:
            return False, {
                "error": "Service Unavailable",
                "message": "Model is not loaded on the backend. Please train the model.",
            }
        else:
            return False, {
                "error": f"HTTP {response.status_code}",
                "message": response.text,
            }
    except requests.exceptions.ConnectionError:
        return False, {
            "error": "Backend Offline",
            "message": f"Cannot reach the prediction backend at {BACKEND_API_URL}. Ensure uvicorn is running.",
        }
    except requests.exceptions.Timeout:
        return False, {
            "error": "Timeout",
            "message": "Prediction request timed out.",
        }
    except Exception as e:
        return False, {
            "error": "Unexpected Error",
            "message": str(e),
        }

