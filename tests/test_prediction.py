"""
Unit and integration tests for dataset generation and prediction service.
"""

import os
import pytest
import pandas as pd
from backend.services.prediction_service import PredictionService
from backend.schemas import AttendancePredictionRequest
from ml.generate_dataset import generate_attendance_dataset
from ml.preprocessing import ALL_FEATURES, EXCLUDED_COLUMNS


def test_dataset_generation_reproducibility(tmp_path):
    """Ensure dataset generation is 100% deterministic with a fixed random seed."""
    path1 = tmp_path / "dataset1.csv"
    path2 = tmp_path / "dataset2.csv"

    df1 = generate_attendance_dataset(n_samples=100, random_seed=42, output_path=str(path1))
    df2 = generate_attendance_dataset(n_samples=100, random_seed=42, output_path=str(path2))

    pd.testing.assert_frame_equal(df1, df2)


def test_no_data_leakage_in_features():
    """Verify that future outcomes and target-derived columns are excluded from features."""
    for col in EXCLUDED_COLUMNS:
        assert col not in ALL_FEATURES, f"Target leakage detected: '{col}' found in ALL_FEATURES!"


def test_deterministic_bounds_impossible_scenario():
    """
    Test scenario where student has attended too few classes such that
    even 100% attendance in remaining classes cannot reach 75%.
    Conducted: 50, Attended: 10, Remaining: 20 -> Total: 70.
    Max possible: (10 + 20) / 70 = 30 / 70 = 42.86% < 75%.
    """
    service = PredictionService(model_path="dummy_path.joblib", metrics_path="dummy_metrics.json")
    req = AttendancePredictionRequest(
        classes_conducted=50,
        classes_attended=10,
        classes_remaining=20,
    )
    bounds = service.compute_deterministic_bounds(req)

    assert bounds.mathematical_status == "impossible_to_reach_75"
    assert bounds.max_possible_attendance_pct < 75.0
    assert bounds.classes_needed_for_75_pct > req.classes_remaining


def test_deterministic_bounds_guaranteed_safe_scenario():
    """
    Test scenario where student has attended so many classes that
    even 0% attendance in remaining classes cannot drop below 75%.
    Conducted: 80, Attended: 75, Remaining: 10 -> Total: 90.
    Min possible: 75 / 90 = 83.33% >= 75%.
    """
    service = PredictionService(model_path="dummy_path.joblib", metrics_path="dummy_metrics.json")
    req = AttendancePredictionRequest(
        classes_conducted=80,
        classes_attended=75,
        classes_remaining=10,
    )
    bounds = service.compute_deterministic_bounds(req)

    assert bounds.mathematical_status == "guaranteed_safe"
    assert bounds.min_possible_attendance_pct >= 75.0
    assert bounds.classes_needed_for_75_pct == 0


def test_deterministic_bounds_behavior_dependent_scenario():
    """
    Test normal mid-semester case where outcome depends on future attendance.
    Conducted: 50, Attended: 35, Remaining: 30 -> Total: 80.
    75% of 80 = 60.
    Classes needed = 60 - 35 = 25 out of 30 remaining.
    """
    service = PredictionService(model_path="dummy_path.joblib", metrics_path="dummy_metrics.json")
    req = AttendancePredictionRequest(
        classes_conducted=50,
        classes_attended=35,
        classes_remaining=30,
    )
    bounds = service.compute_deterministic_bounds(req)

    assert bounds.mathematical_status == "behavior_dependent"
    assert bounds.classes_needed_for_75_pct == 25
    assert bounds.min_possible_attendance_pct < 75.0
    assert bounds.max_possible_attendance_pct >= 75.0


def test_missing_model_artifact_handling():
    """Verify that attempting to predict without a loaded model raises RuntimeError."""
    service = PredictionService(
        model_path="non_existent_artifacts/model.joblib",
        metrics_path="non_existent_artifacts/metrics.json",
    )
    assert service.is_loaded is False

    req = AttendancePredictionRequest(
        classes_conducted=50,
        classes_attended=35,
        classes_remaining=20,
    )
    with pytest.raises(RuntimeError) as exc_info:
        service.predict(req)
    assert "not loaded" in str(exc_info.value).lower()


def test_real_model_inference_and_probability_range():
    """Verify that predictions from the actual trained artifact produce valid ranges."""
    model_path = "artifacts/attendance_pipeline.joblib"
    metrics_path = "artifacts/model_metrics.json"

    if not os.path.exists(model_path):
        pytest.skip("Model artifact not yet generated. Run training first.")

    service = PredictionService(model_path=model_path, metrics_path=metrics_path)
    assert service.is_loaded is True

    req = AttendancePredictionRequest(
        classes_conducted=50,
        classes_attended=38,
        classes_remaining=25,
        previous_attendance_trend="stable",
        consecutive_absences_recent=1,
        participation_score=7.5,
    )

    response = service.predict(req)
    assert response.attendance_risk in [0, 1]
    assert response.risk_label in ["At Risk", "Safe"]
    assert response.risk_probability is not None
    assert 0.0 <= response.risk_probability <= 1.0
    assert response.target_threshold == 75.0
    assert len(response.explanation) > 10

