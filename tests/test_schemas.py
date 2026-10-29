"""
Unit tests for Pydantic request and response schemas.
"""

import pytest
from pydantic import ValidationError
from backend.schemas import (
    AttendancePredictionRequest,
    AttendancePredictionResponse,
    DeterministicBounds,
)


def test_valid_prediction_request():
    """Test valid prediction request payload instantiation."""
    req = AttendancePredictionRequest(
        classes_conducted=50,
        classes_attended=40,
        classes_remaining=25,
        previous_attendance_trend="improving",
        consecutive_absences_recent=1,
        participation_score=8.5,
    )
    assert req.classes_conducted == 50
    assert req.classes_attended == 40
    assert req.classes_remaining == 25
    assert req.previous_attendance_trend == "improving"
    assert req.consecutive_absences_recent == 1
    assert req.participation_score == 8.5


def test_default_values_request():
    """Test default values on optional fields."""
    req = AttendancePredictionRequest(
        classes_conducted=30,
        classes_attended=20,
        classes_remaining=20,
    )
    assert req.previous_attendance_trend == "stable"
    assert req.consecutive_absences_recent == 0
    assert req.participation_score == 7.0


def test_classes_attended_exceeds_conducted_rejected():
    """Ensure impossible values (attended > conducted) trigger validation errors."""
    with pytest.raises(ValidationError) as exc_info:
        AttendancePredictionRequest(
            classes_conducted=40,
            classes_attended=45,
            classes_remaining=20,
        )
    assert "cannot exceed" in str(exc_info.value).lower()


def test_zero_or_negative_classes_conducted_rejected():
    """Ensure classes_conducted must be strictly positive."""
    with pytest.raises(ValidationError):
        AttendancePredictionRequest(
            classes_conducted=0,
            classes_attended=0,
            classes_remaining=20,
        )

    with pytest.raises(ValidationError):
        AttendancePredictionRequest(
            classes_conducted=-5,
            classes_attended=0,
            classes_remaining=20,
        )


def test_zero_or_negative_classes_remaining_rejected():
    """Ensure classes_remaining must be strictly positive."""
    with pytest.raises(ValidationError):
        AttendancePredictionRequest(
            classes_conducted=40,
            classes_attended=30,
            classes_remaining=0,
        )


def test_participation_score_bounds():
    """Ensure participation score enforces 1.0 to 10.0 boundaries."""
    # Under 1.0
    with pytest.raises(ValidationError):
        AttendancePredictionRequest(
            classes_conducted=40,
            classes_attended=30,
            classes_remaining=20,
            participation_score=0.5,
        )

    # Over 10.0
    with pytest.raises(ValidationError):
        AttendancePredictionRequest(
            classes_conducted=40,
            classes_attended=30,
            classes_remaining=20,
            participation_score=11.0,
        )


def test_invalid_trend_category_rejected():
    """Ensure trend is restricted to valid Literal values."""
    with pytest.raises(ValidationError):
        AttendancePredictionRequest(
            classes_conducted=40,
            classes_attended=30,
            classes_remaining=20,
            previous_attendance_trend="skyrocketing",  # Invalid
        )


def test_valid_deterministic_bounds():
    """Test schema instantiation of deterministic bounds."""
    bounds = DeterministicBounds(
        current_attendance_pct=75.0,
        total_semester_classes=60,
        max_possible_attendance_pct=90.0,
        min_possible_attendance_pct=50.0,
        classes_needed_for_75_pct=15,
        mathematical_status="behavior_dependent",
    )
    assert bounds.current_attendance_pct == 75.0
    assert bounds.mathematical_status == "behavior_dependent"

