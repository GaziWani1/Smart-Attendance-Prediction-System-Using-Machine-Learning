"""
Smart Attendance Prediction System - Page 1: Dashboard
"""

import os
import sys
from pathlib import Path

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from frontend.components.ui_helpers import inject_custom_css, render_sidebar_status
from frontend.components.api_client import get_model_info

# Setup page
st.set_page_config(page_title="Dashboard | Smart Attendance", page_icon="📊", layout="wide")
inject_custom_css()
render_sidebar_status()

st.title("📊 Attendance Analytics Dashboard")
st.markdown("Overview of the student dataset, distribution of attendance rates, and risk demographics.")

# Dataset Path
DATASET_PATH = "data/attendance_dataset.csv"

if not os.path.exists(DATASET_PATH):
    st.error(f"Dataset file `{DATASET_PATH}` not found. Please run `python -m ml.generate_dataset`.")
    st.stop()

# Load Dataset
df = pd.read_csv(DATASET_PATH)

# Synthetic data banner
st.caption("🏷️ **Data Origin**: Synthetic simulation based on realistic university lecture trajectories (Zero data leakage).")

# High-Level KPI Summary
total_students = len(df)
at_risk_count = int(df["attendance_risk"].sum())
safe_count = total_students - at_risk_count
risk_percentage = round((at_risk_count / total_students) * 100, 1)
avg_current_att = round(df["current_attendance_pct"].mean(), 1)
avg_classes_cond = round(df["classes_conducted"].mean(), 1)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_students:,}</div>
            <div class="kpi-label">Total Student Records</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: #DC2626;">{at_risk_count:,} ({risk_percentage}%)</div>
            <div class="kpi-label">At-Risk (<75% Threshold)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: #16A34A;">{safe_count:,} ({100 - risk_percentage:.1f}%)</div>
            <div class="kpi-label">Safe Standing (>=75%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_current_att}%</div>
            <div class="kpi-label">Average Current Attendance</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Visualizations Row 1
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Attendance Distribution")
    fig_hist = px.histogram(
        df,
        x="current_attendance_pct",
        nbins=30,
        color="attendance_risk",
        color_discrete_map={0: "#3B82F6", 1: "#EF4444"},
        labels={"current_attendance_pct": "Current Attendance (%)", "attendance_risk": "Final Risk"},
        opacity=0.75,
    )
    # Add vertical reference line at 75% threshold
    fig_hist.add_vline(
        x=75.0,
        line_dash="dash",
        line_color="#B91C1C",
        annotation_text="75% Requirement",
        annotation_position="top left",
    )
    fig_hist.update_layout(
        legend_title_text="Risk Category (0=Safe, 1=Risk)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=340,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with chart_col2:
    st.subheader("Target Class Ratio")
    fig_pie = go.Figure(
        data=[
            go.Pie(
                labels=["Safe (>=75%)", "At Risk (<75%)"],
                values=[safe_count, at_risk_count],
                hole=0.45,
                marker=dict(colors=["#3B82F6", "#EF4444"]),
            )
        ]
    )
    fig_pie.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=340,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Visualizations Row 2: Attendance by Trend & Participation
chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.subheader("Attendance Rate by Historical Trend")
    fig_box = px.box(
        df,
        x="previous_attendance_trend",
        y="current_attendance_pct",
        color="previous_attendance_trend",
        category_orders={"previous_attendance_trend": ["declining", "stable", "improving"]},
        labels={"previous_attendance_trend": "Observed Trend", "current_attendance_pct": "Current Attendance (%)"},
    )
    fig_box.update_layout(showlegend=False, height=320, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_box, use_container_width=True)

with chart_col4:
    st.subheader("Engagement vs. Attendance")
    fig_scatter = px.scatter(
        df.sample(min(len(df), 500), random_state=42),
        x="participation_score",
        y="current_attendance_pct",
        color="attendance_risk",
        color_discrete_map={0: "#3B82F6", 1: "#EF4444"},
        labels={"participation_score": "Participation Score (1-10)", "current_attendance_pct": "Current Attendance (%)"},
        opacity=0.6,
    )
    fig_scatter.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

# Model Performance Teaser (from Backend API)
st.markdown("---")
st.subheader("🤖 Active Production Model Summary")
success, model_info = get_model_info()

if success:
    m_name = model_info.get("selected_model", "Classifier")
    metrics = model_info.get("metrics", {})
    st.write(f"The FastAPI backend currently serves **{m_name}** trained on this dataset.")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Test Accuracy", f"{metrics.get('accuracy', 0.0) * 100:.1f}%")
    m_col2.metric("Positive Recall", f"{metrics.get('recall', 0.0) * 100:.1f}%")
    m_col3.metric("Positive Precision", f"{metrics.get('precision', 0.0) * 100:.1f}%")
    m_col4.metric("F1-Score", f"{metrics.get('f1_score', 0.0):.4f}")
else:
    st.warning("Could not fetch active model metadata from FastAPI backend. Start the backend or train the pipeline.")

