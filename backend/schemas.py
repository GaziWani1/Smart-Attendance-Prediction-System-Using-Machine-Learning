"""
Smart Attendance Prediction System - Pydantic Validation Schemas
"""

from typing import Optional, Literal, Dict, Any, List
from pydantic import BaseModel, Field, model_validator


class AttendancePredictionRequest(BaseModel):
    """
    Input schema for student attendance risk prediction.
    Features reflect only information available at the prediction checkpoint.
    """
    classes_conducted: int = Field(
        ...,
        gt=0,
        le=200,
        description="Number of classes conducted in the course so far.",
        examples=[50],
    )
    classes_attended: int = Field(
        ...,
        ge=0,
        le=200,
        description="Number of classes the student has attended so far.",
        examples=[34],
    )
    classes_remaining: int = Field(
        ...,
        gt=0,
        le=150,
        description="Estimated number of classes remaining in the semester.",
        examples=[20],
    )
    previous_attendance_trend: Literal["declining", "stable", "improving"] = Field(
        default="stable",
        description="Attendance trajectory observed in recent weeks.",
        examples=["declining"],
    )
    consecutive_absences_recent: int = Field(
        default=0,
        ge=0,
        le=50,
        description="Number of consecutive sessions missed right before the checkpoint.",
        examples=[2],
    )
    participation_score: float = Field(
        default=7.0,
        ge=1.0,
        le=10.0,
        description="Engagement rating or assignment/quiz completion score (1.0 to 10.0).",
        examples=[6.5],
    )

    @model_validator(mode="after")
    def validate_attendance_consistency(self) -> "AttendancePredictionRequest":
        """
        Enforce logical physical constraints:
        Classes attended cannot exceed classes conducted.
        """
        if self.classes_attended > self.classes_conducted:
            raise ValueError(
                f"Validation Error: classes_attended ({self.classes_attended}) cannot exceed "
                f"classes_conducted ({self.classes_conducted})."
            )
        return self


class DeterministicBounds(BaseModel):
    """
    Pure mathematical attendance bounds, explicitly distinguished from ML predictions.
    """
    current_attendance_pct: float = Field(
        ...,
        description="Current attendance rate based on classes conducted so far."
    )
    total_semester_classes: int = Field(
        ...,
        description="Sum of classes conducted and classes remaining."
    )
    max_possible_attendance_pct: float = Field(
        ...,
        description="Theoretical maximum attendance if student attends 100% of remaining classes."
    )
    min_possible_attendance_pct: float = Field(
        ...,
        description="Theoretical minimum attendance if student attends 0% of remaining classes."
    )
    classes_needed_for_75_pct: int = Field(
        ...,
        description="Minimum remaining classes student must attend to reach 75.0% final attendance."
    )
    mathematical_status: Literal[
        "impossible_to_reach_75",
        "guaranteed_safe",
        "behavior_dependent"
    ] = Field(
        ...,
        description="Whether 75% is mathematically determined or dependent on future behavior."
    )


class AttendancePredictionResponse(BaseModel):
    """
    Structured prediction output returned by the FastAPI backend.
    """
    attendance_risk: int = Field(
        ...,
        description="Binary prediction: 1 = At Risk (<75% final attendance), 0 = Safe (>=75%).",
        examples=[1],
    )
    risk_label: str = Field(
        ...,
        description="Human-readable classification label ('At Risk' or 'Safe').",
        examples=["At Risk"],
    )
    risk_probability: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model-estimated probability of finishing below 75% attendance.",
        examples=[0.72],
    )
    target_threshold: float = Field(
        default=75.0,
        description="The institutional attendance threshold percentage.",
        examples=[75.0],
    )
    explanation: str = Field(
        ...,
        description="Transparent summary of learned associations driving the prediction.",
    )
    deterministic_bounds: DeterministicBounds = Field(
        ...,
        description="Mathematical certainty bounds separate from ML estimation.",
    )


class HealthResponse(BaseModel):
    """
    Application health and model readiness schema.
    """
    status: str = Field(..., examples=["healthy", "degraded"])
    model_loaded: bool = Field(..., examples=[True])
    model_name: Optional[str] = Field(None, examples=["RandomForestClassifier"])
    timestamp: str


class ModelInfoResponse(BaseModel):
    """
    Metadata describing the trained model, feature schemas, and evaluation results.
    """
    app_name: str
    version: str
    selected_model: str
    target_definition: str
    target_threshold: float
    dataset_type: str
    feature_names: List[str]
    numerical_features: List[str]
    categorical_features: List[str]
    metrics: Dict[str, Any]
    feature_importances_or_coefficients: Dict[str, float]

