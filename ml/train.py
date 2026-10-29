"""
Smart Attendance Prediction System - Training Module (Lightweight Models)

Trains, evaluates, compares, and serializes lightweight machine learning models:
- Logistic Regression (ultra-fast, ~3.8 KB, linear probabilistic baseline)
- Decision Tree Classifier (compact, ~6.5 KB, transparent rule-based model)

No heavy ensembles (e.g., Random Forest) are used.
Saves the fitted pipeline to `artifacts/attendance_pipeline.joblib`
and comprehensive metrics to `artifacts/model_metrics.json`.
"""

import os
import json
import argparse
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from ml.preprocessing import (
    create_preprocessor,
    ALL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)
from ml.evaluate import evaluate_model


def train_models(
    dataset_path: str = "data/attendance_dataset.csv",
    artifacts_dir: str = "artifacts",
    random_seed: int = 42,
):
    """
    Loads dataset, trains lightweight baseline and tree models, evaluates them,
    and serializes the winning lightweight pipeline and performance metrics.
    """
    print(f"\n==================================================================")
    print(f"  Smart Attendance Prediction System - Lightweight Model Training")
    print(f"==================================================================\n")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run 'python -m ml.generate_dataset' first."
        )

    # 1. Load Dataset
    df = pd.read_csv(dataset_path)
    print(f"[1/6] Loaded dataset: {len(df)} records from {dataset_path}")

    # Check for target presence
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataset.")

    # 2. Prepare Features (X) and Target (y)
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    print(f"[2/6] Feature schema verified: {len(ALL_FEATURES)} features")
    print(f"      Numerical: {NUMERICAL_FEATURES}")
    print(f"      Categorical: {CATEGORICAL_FEATURES}")
    print(f"      Target class distribution: At Risk: {y.sum()} ({y.mean():.1%}), Safe: {(1-y).sum()} ({(1-y).mean():.1%})")

    # 3. Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_seed,
        stratify=y,
    )
    print(f"[3/6] Stratified Split: Train set = {len(X_train)} samples, Test set = {len(X_test)} samples")

    # 4. Construct Lightweight Pipelines
    # Model 1: Logistic Regression (Ultra-lightweight linear classifier)
    log_reg = LogisticRegression(
        class_weight="balanced",
        random_state=random_seed,
        max_iter=1000,
        solver="lbfgs",
    )
    pipeline_lr = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", log_reg),
    ])

    # Model 2: Compact Decision Tree (Shallow, non-ensemble, interpretable)
    dt_clf = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_seed,
    )
    pipeline_dt = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", dt_clf),
    ])

    # 5. Cross-Validation and Training
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    print("\n[4/6] Performing 5-Fold Stratified Cross-Validation on Training Split:")
    cv_scores_lr = cross_val_score(pipeline_lr, X_train, y_train, cv=cv, scoring="f1")
    cv_scores_dt = cross_val_score(pipeline_dt, X_train, y_train, cv=cv, scoring="f1")

    print(f"      Logistic Regression - Mean CV F1: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std():.4f})")
    print(f"      Decision Tree (d=4) - Mean CV F1: {cv_scores_dt.mean():.4f} (+/- {cv_scores_dt.std():.4f})")

    # Fit both models on full training set
    print("\n[5/6] Fitting lightweight models on training set and evaluating on held-out test set...")
    pipeline_lr.fit(X_train, y_train)
    pipeline_dt.fit(X_train, y_train)

    # Evaluate on held-out test set
    metrics_lr = evaluate_model(pipeline_lr, X_test, y_test)
    metrics_dt = evaluate_model(pipeline_dt, X_test, y_test)

    print("\n--- Model Evaluation Results (Held-Out Test Set) ---")
    print(f"Logistic Regression: Accuracy={metrics_lr['accuracy']:.4f}, Recall={metrics_lr['recall']:.4f}, Precision={metrics_lr['precision']:.4f}, F1={metrics_lr['f1_score']:.4f}, ROC-AUC={metrics_lr['roc_auc']:.4f}")
    print(f"Decision Tree (d=4): Accuracy={metrics_dt['accuracy']:.4f}, Recall={metrics_dt['recall']:.4f}, Precision={metrics_dt['precision']:.4f}, F1={metrics_dt['f1_score']:.4f}, ROC-AUC={metrics_dt['roc_auc']:.4f}")

    # Extract encoded feature names for explainability
    cat_encoder = pipeline_lr.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_transformed_feature_names = NUMERICAL_FEATURES + cat_feature_names

    # Coefficients for Logistic Regression
    lr_coefficients = {
        name: round(float(coef), 4)
        for name, coef in zip(
            all_transformed_feature_names,
            pipeline_lr.named_steps["classifier"].coef_[0],
        )
    }

    # Feature Importance for Decision Tree
    dt_importances = {
        name: round(float(imp), 4)
        for name, imp in zip(
            all_transformed_feature_names,
            pipeline_dt.named_steps["classifier"].feature_importances_,
        )
    }
    sorted_dt_importances = dict(sorted(dt_importances.items(), key=lambda x: x[1], reverse=True))

    # Model Selection: Pick best lightweight model based on F1
    if metrics_dt["f1_score"] >= metrics_lr["f1_score"]:
        selected_model_name = "DecisionTreeClassifier"
        selected_pipeline = pipeline_dt
        selected_metrics = metrics_dt
        selected_importances = sorted_dt_importances
    else:
        selected_model_name = "LogisticRegression"
        selected_pipeline = pipeline_lr
        selected_metrics = metrics_lr
        selected_importances = lr_coefficients

    print(f"\n>> Selected Best Lightweight Model: {selected_model_name} (F1-score: {selected_metrics['f1_score']})")

    # 6. Save Artifacts
    os.makedirs(artifacts_dir, exist_ok=True)
    pipeline_file = os.path.join(artifacts_dir, "attendance_pipeline.joblib")
    metrics_file = os.path.join(artifacts_dir, "model_metrics.json")

    joblib.dump(selected_pipeline, pipeline_file)
    artifact_size_kb = os.path.getsize(pipeline_file) / 1024
    print(f"[6/6] Saved trained pipeline artifact to: {pipeline_file} (Size: {artifact_size_kb:.2f} KB)")

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected_model_name,
        "target_definition": "attendance_risk: 1 if final semester attendance < 75.0%, else 0",
        "target_threshold": 75.0,
        "dataset_type": "synthetic (generated via behavioral stochastic simulation, zero data leakage)",
        "features": {
            "all_features": ALL_FEATURES,
            "numerical_features": NUMERICAL_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "transformed_features": all_transformed_feature_names,
        },
        "models": {
            "LogisticRegression": {
                "name": "Logistic Regression (Linear)",
                "test_metrics": metrics_lr,
                "cv_f1_mean": round(float(cv_scores_lr.mean()), 4),
                "cv_f1_std": round(float(cv_scores_lr.std()), 4),
                "coefficients": lr_coefficients,
                "feature_importances": {k: abs(v) for k, v in lr_coefficients.items()},
            },
            "DecisionTreeClassifier": {
                "name": "Decision Tree Classifier (max_depth=4)",
                "test_metrics": metrics_dt,
                "cv_f1_mean": round(float(cv_scores_dt.mean()), 4),
                "cv_f1_std": round(float(cv_scores_dt.std()), 4),
                "feature_importances": sorted_dt_importances,
            },
        },
    }

    with open(metrics_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"      Saved comprehensive metrics to: {metrics_file}")
    print("\nTraining completed successfully! (No heavy ensemble or Random Forest used)")
    return selected_model_name, selected_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train lightweight attendance risk prediction models.")
    parser.add_argument("--data", type=str, default="data/attendance_dataset.csv", help="Dataset path")
    parser.add_argument("--artifacts", type=str, default="artifacts", help="Artifacts directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_models(dataset_path=args.data, artifacts_dir=args.artifacts, random_seed=args.seed)
