"""
Smart Attendance Prediction System - Preprocessing Module

Defines the feature specifications and scikit-learn ColumnTransformer
for consistent feature encoding and scaling across training and inference.
"""

from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Explicitly defined feature schemas available at prediction time
NUMERICAL_FEATURES: List[str] = [
    "classes_conducted",
    "classes_attended",
    "classes_remaining",
    "current_attendance_pct",
    "consecutive_absences_recent",
    "participation_score",
]

CATEGORICAL_FEATURES: List[str] = [
    "previous_attendance_trend",
]

# All expected inference features in exact order
ALL_FEATURES: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Target column name
TARGET_COLUMN: str = "attendance_risk"

# Non-predictive or target-derived columns to exclude
EXCLUDED_COLUMNS: List[str] = [
    "student_id",
    "future_classes_attended",
    "final_attendance_pct",
    "attendance_risk",
    "is_synthetic",
]


def create_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that scales numerical features
    and one-hot encodes categorical trend features.

    Returns:
        ColumnTransformer: Preprocessing pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(
                    categories=[["declining", "stable", "improving"]],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return preprocessor

