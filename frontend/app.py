"""
Smart Attendance Prediction System - Main Streamlit Entrypoint
"""

import sys
from pathlib import Path

# Ensure project root directory is in sys.path when running via 'streamlit run frontend/app.py'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from frontend.components.ui_helpers import inject_custom_css, render_sidebar_status

# Configure page settings
st.set_page_config(
    page_title="Smart Attendance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
render_sidebar_status()

# Hero Header
st.markdown('<div class="main-header">🎓 Smart Attendance Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Early-warning decision support using Machine Learning to predict student semester attendance risk.'
    '</div>',
    unsafe_allow_html=True,
)

# Banner explaining purpose and synthetic data caveat
st.info(
    "💡 **Academic Portfolio Showcase**: This system demonstrates end-to-end ML deployment "
    "using FastAPI and Streamlit. The underlying dataset is synthetically generated via behavioral simulation "
    "with zero target data leakage. It is designed to demonstrate prediction architecture, not replace official university systems.",
    icon="ℹ️",
)

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 📌 Why Predict Attendance Rather than Just Calculate Percentages?")
    st.write(
        """
        Conventional attendance trackers only look backward—they report the percentage of classes
        a student has attended up to today. However, an academic advisor needs to know:

        **Will this student fall below the required 75% attendance threshold by the end of the semester?**

        A student with 80% attendance today whose recent trend is sharply declining and who has missed
        three consecutive lectures is at elevated risk of failure. This application combines:
        - **Historical attendance count and percentage so far**
        - **Trajectory trend (improving, stable, declining)**
        - **Recent behavioral momentum (consecutive absences, participation score)**
        - **Remaining class count**

        ...to estimate the **probability of finishing below 75%**, enabling timely student mentorship.
        """
    )

    st.markdown("### 🚀 Application Navigation")
    st.markdown(
        """
        - **📊 [1_Dashboard](Dashboard)**: Explore the distribution of attendance, classes held, and demographic indicators.
        - **🔮 [2_Predict_Attendance](Predict_Attendance)**: Enter a student's current numbers to get real-time risk predictions and recovery calculations.
        - **🧠 [3_Model_Insights](Model_Insights)**: Inspect model evaluation metrics, confusion matrices, and feature importance.
        """
    )

with col2:
    st.markdown("### 🏗️ Technical Architecture")
    st.markdown(
        """
        ```
        ┌─────────────────────────┐
        │    Streamlit UI (Py)    │
        │   Dashboard & Forms     │
        └────────────┬────────────┘
                     │ HTTP JSON
                     ▼
        ┌─────────────────────────┐
        │     FastAPI Backend     │
        │ Pydantic Validation     │
        └────────────┬────────────┘
                     │ In-Memory Pipeline
                     ▼
        ┌─────────────────────────┐
        │     Scikit-Learn ML     │
        │ ColumnTransformer + RF  │
        └─────────────────────────┘
        ```
        """
    )
    st.markdown("### ⚙️ Quick System Checks")
    st.markdown("- **Model Framework**: Scikit-Learn (Joblib serialized)")
    st.markdown("- **Backend Framework**: FastAPI + Pydantic v2")
    st.markdown("- **Communication**: Decoupled HTTP via Python Requests")
    st.markdown("- **Data Storage**: CSV & Joblib (Zero external database required)")

