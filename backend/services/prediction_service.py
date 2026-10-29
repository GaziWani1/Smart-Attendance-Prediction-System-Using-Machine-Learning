"""
Smart Attendance Prediction System - Prediction Service

Encapsulates model loading, input feature formatting, ML inference,
deterministic calculations, and explainability logic.
"""

import os
import math
import json
import logging
from typing import Optional, Dict, Any
import joblib
import pandas as pd

from backend.schemas import (
    AttendancePredictionRequest,
    AttendancePredictionResponse,
    DeterministicBounds,
)
from ml.preprocessing import ALL_FEATURES

logger = logging.getLogger("attendance_backend.prediction_service")


class PredictionService:
    """
    Service responsible for loading model artifacts into memory once
    and serving low-latency attendance risk predictions.
    """

    def __init__(self, model_path: str, metrics_path: str):
        self.model_path = model_path
        self.metrics_path = metrics_path
        self.pipeline = None
        self.metrics_data: Dict[str, Any] = {}
        self.is_loaded: bool = False
        self.model_name: Optional[str] = None

        # Load artifacts on instantiation
        self.load_artifacts()

    def load_artifacts(self) -> bool:
        """
        Loads the serialized pipeline and metadata from disk into memory.
        """
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found at: {self.model_path}")
                self.is_loaded = False
                return False

            logger.info(f"Loading trained pipeline from: {self.model_path}")
            self.pipeline = joblib.load(self.model_path)

            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, "r") as f:
                    self.metrics_data = json.load(f)
                self.model_name = self.metrics_data.get("selected_model", "TrainedPipeline")
            else:
                self.metrics_data = {}
                self.model_name = "TrainedPipeline"

            self.is_loaded = True
            logger.info(f"Successfully loaded model: {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}", exc_info=True)
            self.pipeline = None
            self.is_loaded = False
            return False

    def compute_deterministic_bounds(
        self, request: AttendancePredictionRequest
    ) -> DeterministicBounds:
        """
        Computes the theoretical mathematical limits of the student's attendance.
        These are mathematical certainties, completely independent of the ML model.
        """
        classes_cond = request.classes_conducted
        classes_att = request.classes_attended
        classes_rem = request.classes_remaining

        current_pct = round((classes_att / classes_cond) * 100.0, 2)
        total_semester_classes = classes_cond + classes_rem

        # Maximum possible final attendance (attending 100% of remaining sessions)
        max_possible_pct = round(
            ((classes_att + classes_rem) / total_semester_classes) * 100.0, 2
        )

        # Minimum possible final attendance (attending 0% of remaining sessions)
        min_possible_pct = round(
            (classes_att / total_semester_classes) * 100.0, 2
        )

        # Minimum classes required out of remaining to hit 75.0%
        # Target: (classes_att + needed) / total >= 0.75
        # needed >= 0.75 * total - classes_att
        raw_needed = math.ceil(0.75 * total_semester_classes - classes_att)
        classes_needed = max(0, raw_needed)

        if max_possible_pct < 75.0:
            math_status = "impossible_to_reach_75"
        elif min_possible_pct >= 75.0:
            math_status = "guaranteed_safe"
        else:
            math_status = "behavior_dependent"

        return DeterministicBounds(
            current_attendance_pct=current_pct,
            total_semester_classes=total_semester_classes,
            max_possible_attendance_pct=max_possible_pct,
            min_possible_attendance_pct=min_possible_pct,
            classes_needed_for_75_pct=classes_needed,
            mathematical_status=math_status,
        )

    def generate_explanation(
        self,
        risk: int,
        prob: Optional[float],
        request: AttendancePredictionRequest,
        bounds: DeterministicBounds,
    ) -> str:
        """
        Generates a transparent, plain-English explanation of why the model
        produced this risk assessment, emphasizing statistical associations.
        """
        current_pct = bounds.current_attendance_pct
        trend = request.previous_attendance_trend
        absences = request.consecutive_absences_recent
        needed = bounds.classes_needed_for_75_pct
        rem = request.classes_remaining

        if bounds.mathematical_status == "impossible_to_reach_75":
            return (
                f"Notice: The student's maximum possible attendance is {bounds.max_possible_attendance_pct:.1f}%. "
                f"It is mathematically impossible to reach the 75% threshold even with 100% future attendance."
            )

        if bounds.mathematical_status == "guaranteed_safe":
            return (
                f"Notice: The student already has enough attended classes ({request.classes_attended}) "
                f"that final attendance cannot drop below {bounds.min_possible_attendance_pct:.1f}%, guaranteeing safe standing."
            )

        prob_str = f"{prob * 100:.1f}%" if prob is not None else "N/A"

        if risk == 1:
            factors = []
            if current_pct < 75.0:
                factors.append(f"current attendance is below threshold ({current_pct:.1f}%)")
            if trend == "declining":
                factors.append("attendance trend has been declining recently")
            if absences >= 2:
                factors.append(f"student has {absences} recent consecutive absences")
            if request.participation_score < 6.0:
                factors.append(f"lower engagement score ({request.participation_score}/10)")

            factor_clause = ", ".join(factors) if factors else "overall feature combination"
            return (
                f"The model estimates an elevated risk ({prob_str} probability) of finishing below 75%. "
                f"Key contributing patterns: {factor_clause}. To reach 75%, the student must attend at least "
                f"{needed} of the remaining {rem} classes ({round((needed/rem)*100, 1)}% future attendance rate)."
            )
        else:
            factors = []
            if current_pct >= 75.0:
                factors.append(f"strong current attendance ({current_pct:.1f}%)")
            if trend in ("stable", "improving"):
                factors.append(f"{trend} attendance trajectory")
            if absences == 0:
                factors.append("no recent consecutive absences")

            factor_clause = ", ".join(factors) if factors else "consistent attendance patterns"
            return (
                f"The model estimates the student is on track to stay safe ({prob_str} risk probability). "
                f"Supported by {factor_clause}. Student needs to attend at least {needed} of {rem} remaining "
                f"classes to guarantee reaching the 75% threshold."
            )

    def predict(self, request: AttendancePredictionRequest) -> AttendancePredictionResponse:
        """
        Executes prediction pipeline and returns structured response.
        """
        if not self.is_loaded or self.pipeline is None:
            raise RuntimeError(
                "Prediction model is not loaded. Train the model using 'python -m ml.train' before requesting predictions."
            )

        # 1. Compute deterministic bounds
        bounds = self.compute_deterministic_bounds(request)

        # 2. Format feature DataFrame matching exact pipeline expectation
        input_data = {
            "classes_conducted": [request.classes_conducted],
            "classes_attended": [request.classes_attended],
            "classes_remaining": [request.classes_remaining],
            "current_attendance_pct": [bounds.current_attendance_pct],
            "consecutive_absences_recent": [request.consecutive_absences_recent],
            "participation_score": [request.participation_score],
            "previous_attendance_trend": [request.previous_attendance_trend],
        }
        df_input = pd.DataFrame(input_data)[ALL_FEATURES]

        # 3. Predict classification
        pred_label = int(self.pipeline.predict(df_input)[0])

        # 4. Predict probability
        prob = None
        if hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba(df_input)[0]
            # Probability of Class 1 (At Risk)
            prob = round(float(probabilities[1]), 4)

        # 5. Handle deterministic overrides if mathematically absolute
        if bounds.mathematical_status == "impossible_to_reach_75":
            pred_label = 1
            prob = max(prob or 0.99, 0.99)
        elif bounds.mathematical_status == "guaranteed_safe":
            pred_label = 0
            prob = min(prob or 0.01, 0.01)

        risk_label = "At Risk" if pred_label == 1 else "Safe"
        explanation = self.generate_explanation(pred_label, prob, request, bounds)

        return AttendancePredictionResponse(
            attendance_risk=pred_label,
            risk_label=risk_label,
            risk_probability=prob,
            target_threshold=75.0,
            explanation=explanation,
            deterministic_bounds=bounds,
        )

