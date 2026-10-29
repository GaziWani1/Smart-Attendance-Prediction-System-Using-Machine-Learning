"""
Smart Attendance Prediction System - Page 3: Model Insights & Explainability
"""

import os
import sys
from pathlib import Path

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import streamlit as st
import pandas as pd
import plotly.express as px

from frontend.components.ui_helpers import (
    inject_custom_css,
    render_sidebar_status,
    create_confusion_matrix_heatmap,
)
from frontend.components.api_client import get_model_info

st.set_page_config(page_title="Model Insights | Smart Attendance", page_icon="🧠", layout="wide")
inject_custom_css()
render_sidebar_status()

st.title("🧠 Machine Learning Model Insights")
st.markdown("Detailed model evaluation, algorithm comparison, confusion matrix, and feature importances.")

METRICS_PATH = "artifacts/model_metrics.json"

if not os.path.exists(METRICS_PATH):
    st.error(f"Evaluation metrics file `{METRICS_PATH}` not found. Run `python -m ml.train` to generate it.")
    st.stop()

with open(METRICS_PATH, "r") as f:
    meta = json.load(f)

selected_model_key = meta.get("selected_model", "DecisionTreeClassifier")
models_dict = meta.get("models", {})
selected_model_info = models_dict.get(selected_model_key, {})

# Section 1: Selected Production Model Metrics
st.subheader(f"🏆 Production Model: {selected_model_info.get('name', selected_model_key)}")
metrics = selected_model_info.get("test_metrics", {})

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{metrics.get('accuracy', 0.0) * 100:.2f}%")
col2.metric("Positive Recall", f"{metrics.get('recall', 0.0) * 100:.2f}%", help="Recall for class 1 (At Risk). Critical to catch struggling students.")
col3.metric("Positive Precision", f"{metrics.get('precision', 0.0) * 100:.2f}%", help="Precision for class 1 (At Risk). Prevents alert fatigue.")
col4.metric("F1-Score", f"{metrics.get('f1_score', 0.0):.4f}", help="Harmonic mean balancing precision and recall.")
col5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0.0):.4f}", help="Discrimination capability across all classification thresholds.")

st.markdown("---")

# Section 2: Model Comparison (Lightweight Linear vs Decision Tree)
st.subheader("⚖️ Algorithm Comparison: Linear Baseline vs. Compact Decision Tree")
st.write(
    """
    We evaluated two ultra-lightweight algorithms on the same stratified held-out test set (500 students) with zero heavy ensemble overhead:
    1. **Logistic Regression**: Interpretable linear baseline with L2 penalty and balanced class weighting (~3.8 KB).
    2. **Decision Tree Classifier (max_depth=4)**: Compact, shallow tree capturing non-linear attendance inflection points (~6.1 KB).
    """
)

comparison_rows = []
for m_key, m_val in models_dict.items():
    m_test = m_val.get("test_metrics", {})
    comparison_rows.append({
        "Algorithm": m_val.get("name", m_key),
        "Mean 5-Fold CV F1": f"{m_val.get('cv_f1_mean', 0.0):.4f} (±{m_val.get('cv_f1_std', 0.0):.4f})",
        "Test Accuracy": f"{m_test.get('accuracy', 0.0) * 100:.2f}%",
        "Test Recall (Risk)": f"{m_test.get('recall', 0.0) * 100:.2f}%",
        "Test Precision (Risk)": f"{m_test.get('precision', 0.0) * 100:.2f}%",
        "Test F1-Score": f"{m_test.get('f1_score', 0.0):.4f}",
        "ROC-AUC": f"{m_test.get('roc_auc', 0.0):.4f}",
    })

comparison_df = pd.DataFrame(comparison_rows)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Callout on Recall vs Precision
st.info(
    "🎯 **Pedagogical Note on Metrics Choice**: In academic early-warning systems, **Recall is prioritized over raw accuracy**. "
    "A False Negative (failing to identify an at-risk student) leaves a struggling student without help until it is mathematically too late. "
    "A False Positive merely results in a friendly check-in email from an academic advisor.",
    icon="💡",
)

st.markdown("---")

# Section 3: Confusion Matrix & Feature Importance
c_col1, c_col2 = st.columns(2)

with c_col1:
    st.subheader("Confusion Matrix")
    cm_dict = metrics.get("confusion_matrix", {})
    if cm_dict:
        st.plotly_chart(create_confusion_matrix_heatmap(cm_dict), use_container_width=True)
        st.caption(
            f"Evaluated on {metrics.get('support', {}).get('total_test_samples', 500)} test students. "
            f"True Positives: {cm_dict.get('true_positives')}, False Negatives: {cm_dict.get('false_negatives')}."
        )

with c_col2:
    st.subheader("Feature Importances")
    importances_dict = selected_model_info.get("feature_importances", {})
    if importances_dict:
        feat_df = pd.DataFrame({
            "Feature": list(importances_dict.keys()),
            "Importance": list(importances_dict.values()),
        }).sort_values(by="Importance", ascending=True)

        fig_bar = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig_bar.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("Feature importances indicate learned predictive associations, not causal links.")

st.markdown("---")

# Section 4: Limitations & Ethical Considerations
st.subheader("⚠️ Model Limitations & Transparent Disclaimers")
st.markdown(
    """
    1. **Synthetic Data Origin**: The training dataset was generated using a controlled behavioral simulation.
       While it accurately enforces logical constraints and avoids data leakage, real student behavior involves unobserved
       variables (e.g., medical conditions, personal emergencies, transit challenges).
    2. **Association vs. Causation**: High importance for a feature (e.g., `consecutive_absences_recent`) reflects strong
       statistical correlation with future attendance, not an absolute cause.
    3. **Intervention Decision-Support**: This model must serve solely as an advisor aid to prompt constructive check-ins,
       never as an automated academic penalty or disciplinary mechanism.
    """
)

