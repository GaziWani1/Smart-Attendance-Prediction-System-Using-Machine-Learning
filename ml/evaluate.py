"""
Smart Attendance Prediction System - Evaluation Module

Calculates comprehensive classification metrics, confusion matrices,
and ROC-AUC for model assessment and transparency.
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(pipeline, X_test, y_test) -> Dict[str, Any]:
    """
    Evaluates a trained scikit-learn pipeline on test data.

    Args:
        pipeline: Trained Pipeline object.
        X_test: Test features.
        y_test: True target labels.

    Returns:
        dict: Evaluation metrics including accuracy, precision, recall, F1,
              confusion matrix, and ROC-AUC.
    """
    y_pred = pipeline.predict(X_test)

    # Risk class (1) probability if supported
    y_prob = None
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "support": {
            "total_test_samples": int(len(y_test)),
            "safe_count": int(tn + fp),
            "at_risk_count": int(fn + tp),
        },
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics

