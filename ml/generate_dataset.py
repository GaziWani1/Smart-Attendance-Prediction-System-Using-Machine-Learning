"""
Smart Attendance Prediction System - Dataset Generation Module

This module generates a reproducible, realistic synthetic dataset simulating university
student semester attendance trajectories.

CRITICAL DESIGN NOTE ON DATA LEAKAGE:
-------------------------------------
To prevent target leakage:
1. Past attendance (classes_conducted, classes_attended, current_attendance_pct, trend)
   is simulated based on initial attendance propensities.
2. Future attendance (classes_remaining) is simulated as a separate future outcome trial.
3. The target `attendance_risk` is computed strictly from the final semester total:
   attendance_risk = 1 if final_attendance_pct < 75.0 else 0
4. At training and inference time, future attendance counts and final percentages
   are excluded from the input feature set.
"""

import os
import argparse
import numpy as np
import pandas as pd


def generate_attendance_dataset(
    n_samples: int = 2500,
    random_seed: int = 42,
    output_path: str = "data/attendance_dataset.csv"
) -> pd.DataFrame:
    """
    Generates a synthetic student attendance dataset.

    Args:
        n_samples: Number of synthetic student records to generate.
        random_seed: Seed for reproducible random number generation.
        output_path: Filepath where the CSV will be saved.

    Returns:
        pd.DataFrame: The generated and validated dataset.
    """
    rng = np.random.default_rng(random_seed)

    student_ids = [f"STU_{i+1:05d}" for i in range(n_samples)]

    # 1. Total semester classes planned (typically 60 to 90 lectures)
    total_planned_classes = rng.integers(60, 91, size=n_samples)

    # 2. Checkpoint: fraction of semester conducted so far (35% to 75% of semester)
    progress_fraction = rng.uniform(0.35, 0.75, size=n_samples)
    classes_conducted = np.clip(
        np.round(total_planned_classes * progress_fraction).astype(int),
        20,
        75
    )
    classes_remaining = total_planned_classes - classes_conducted

    # 3. Latent student attendance propensity (beta distribution for realistic bimodal/skewed habits)
    # Most students attend 65-90%, some chronic absentees attend 30-50%
    cluster = rng.choice(["high", "medium", "low"], size=n_samples, p=[0.55, 0.30, 0.15])
    base_propensity = np.zeros(n_samples)

    for i in range(n_samples):
        if cluster[i] == "high":
            base_propensity[i] = rng.uniform(0.78, 0.98)
        elif cluster[i] == "medium":
            base_propensity[i] = rng.uniform(0.60, 0.79)
        else:
            base_propensity[i] = rng.uniform(0.30, 0.62)

    # 4. Simulate classes attended so far
    classes_attended = np.zeros(n_samples, dtype=int)
    trend_labels = []
    consecutive_absences_list = []

    for i in range(n_samples):
        n_cond = classes_conducted[i]
        p_base = base_propensity[i]

        # Simulate lecture-by-lecture attendance sequence
        # Add slight inertia/correlation between consecutive lectures
        attendance_sequence = []
        p_current = p_base
        for _ in range(n_cond):
            attended = 1 if rng.uniform() < p_current else 0
            attendance_sequence.append(attended)
            # Drift propensity slightly based on recent attendance
            p_current = np.clip(p_current + (0.02 if attended else -0.03), 0.10, 0.99)

        attended_count = sum(attendance_sequence)
        classes_attended[i] = attended_count

        # Compute trend comparing first half to second half of conducted classes
        mid = max(1, n_cond // 2)
        rate_first_half = sum(attendance_sequence[:mid]) / mid
        rate_second_half = sum(attendance_sequence[mid:]) / (n_cond - mid)
        diff = rate_second_half - rate_first_half

        if diff > 0.08:
            trend = "improving"
        elif diff < -0.08:
            trend = "declining"
        else:
            trend = "stable"
        trend_labels.append(trend)

        # Compute consecutive absences in the most recent classes
        consec = 0
        for val in reversed(attendance_sequence):
            if val == 0:
                consec += 1
            else:
                break
        consecutive_absences_list.append(consec)

    # 5. Engagement/Participation Score (1.0 to 10.0), correlated with attendance propensity
    participation_score = np.clip(
        base_propensity * 10.0 + rng.normal(0, 1.2, size=n_samples),
        1.0,
        10.0
    ).round(1)

    # Current attendance percentage
    current_attendance_pct = np.round((classes_attended / classes_conducted) * 100.0, 2)

    # 6. SIMULATE FUTURE ATTENDANCE (Strict separation to prevent leakage)
    future_classes_attended = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        n_rem = classes_remaining[i]
        p_future = base_propensity[i]

        # Trend impact on future attendance
        if trend_labels[i] == "improving":
            p_future = np.clip(p_future + rng.uniform(0.02, 0.07), 0.05, 0.98)
        elif trend_labels[i] == "declining":
            p_future = np.clip(p_future - rng.uniform(0.04, 0.12), 0.05, 0.98)

        # Recent absence inertia
        if consecutive_absences_list[i] >= 3:
            p_future = np.clip(p_future - 0.08, 0.05, 0.95)

        # Simulate future outcome
        future_outcome = rng.binomial(n=n_rem, p=p_future)
        future_classes_attended[i] = future_outcome

    # 7. COMPUTE FINAL OUTCOME AND TARGET LABEL
    final_classes_conducted = classes_conducted + classes_remaining
    final_classes_attended = classes_attended + future_classes_attended
    final_attendance_pct = np.round(
        (final_classes_attended / final_classes_conducted) * 100.0,
        2
    )

    # Target rule: 1 if final percentage < 75.0%, else 0
    # Exactly 75.0% is not at risk
    attendance_risk = (final_attendance_pct < 75.0).astype(int)

    # Construct DataFrame
    df = pd.DataFrame({
        "student_id": student_ids,
        "classes_conducted": classes_conducted,
        "classes_attended": classes_attended,
        "classes_remaining": classes_remaining,
        "current_attendance_pct": current_attendance_pct,
        "previous_attendance_trend": trend_labels,
        "consecutive_absences_recent": consecutive_absences_list,
        "participation_score": participation_score,
        # Future/Final attributes kept for evaluation audit & dashboard, excluded from ML features
        "future_classes_attended": future_classes_attended,
        "final_attendance_pct": final_attendance_pct,
        "attendance_risk": attendance_risk,
        "is_synthetic": True
    })

    # 8. DATA INTEGRITY CHECKS
    # Check 1: No impossible attended > conducted
    assert (df["classes_attended"] <= df["classes_conducted"]).all(), "Integrity Error: attended > conducted"
    # Check 2: classes_conducted > 0 and classes_remaining > 0
    assert (df["classes_conducted"] > 0).all() and (df["classes_remaining"] > 0).all(), "Invalid class counts"
    # Check 3: Current attendance percentage strictly matches counts
    expected_pct = np.round((df["classes_attended"] / df["classes_conducted"]) * 100.0, 2)
    assert np.allclose(df["current_attendance_pct"], expected_pct, atol=0.01), "Percentage mismatch"
    # Check 4: No null values
    assert df.isnull().sum().sum() == 0, "Null values detected"

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated synthetic dataset with {len(df)} records at: {output_path}")
    print(f"Class distribution: At Risk (1): {df['attendance_risk'].sum()} ({df['attendance_risk'].mean():.1%}), Safe (0): {(1-df['attendance_risk']).sum()} ({(1-df['attendance_risk']).mean():.1%})")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic student attendance dataset.")
    parser.add_argument("--samples", type=int, default=2500, help="Number of records to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out", type=str, default="data/attendance_dataset.csv", help="Output CSV path")
    args = parser.parse_args()

    generate_attendance_dataset(n_samples=args.samples, random_seed=args.seed, output_path=args.out)

