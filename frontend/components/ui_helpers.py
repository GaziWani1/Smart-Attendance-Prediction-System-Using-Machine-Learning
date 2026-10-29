"""
Smart Attendance Prediction System - UI Helpers & Plotly Visualizations
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
try:
    from frontend.components.api_client import check_backend_health, BACKEND_API_URL
except ModuleNotFoundError:
    from components.api_client import check_backend_health, BACKEND_API_URL


def inject_custom_css():
    """
    Applies clean, academic, responsive CSS styles.
    """
    st.markdown(
        """
        <style>
            /* Main header styling */
            .main-header {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 2.1rem;
                font-weight: 700;
                color: #1E293B;
                margin-bottom: 0.25rem;
            }
            .sub-header {
                font-size: 1.05rem;
                color: #64748B;
                margin-bottom: 1.5rem;
            }
            /* KPI Card */
            .kpi-card {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                padding: 1.2rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                text-align: center;
            }
            .kpi-value {
                font-size: 1.8rem;
                font-weight: 700;
                color: #0F172A;
            }
            .kpi-label {
                font-size: 0.85rem;
                color: #64748B;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-top: 0.25rem;
            }
            /* Result Banner */
            .badge-safe {
                background-color: #DCFCE7;
                color: #166534;
                border: 1px solid #86EFAC;
                border-radius: 6px;
                padding: 0.8rem 1.2rem;
                font-weight: 600;
                font-size: 1.1rem;
                display: inline-block;
            }
            .badge-risk {
                background-color: #FEE2E2;
                color: #991B1B;
                border: 1px solid #FCA5A5;
                border-radius: 6px;
                padding: 0.8rem 1.2rem;
                font-weight: 600;
                font-size: 1.1rem;
                display: inline-block;
            }
            .math-notice-box {
                background-color: #F8FAFC;
                border-left: 4px solid #3B82F6;
                padding: 1rem;
                border-radius: 0 6px 6px 0;
                margin-top: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_status():
    """
    Renders system status widget in the Streamlit sidebar.
    """
    with st.sidebar:
        st.markdown("### 🔌 System Connectivity")
        healthy, info = check_backend_health()
        if healthy:
            st.success(f"● Backend Online ({info.get('model_name', 'Model Active')})")
        else:
            st.error(f"○ Backend Offline")
            st.caption(f"Target: `{BACKEND_API_URL}`")
            with st.expander("Connection Details"):
                st.write(info.get("message", "Cannot connect to FastAPI."))
                st.caption("Ensure `uvicorn backend.main:app` is running.")
        st.markdown("---")
        st.caption("Smart Attendance Predictor v1.0.0")
        st.caption("Built for macOS / Local Portfolio")


def create_probability_gauge(probability: float) -> go.Figure:
    """
    Creates an interactive gauge chart displaying the risk probability.
    """
    pct = round(probability * 100, 1)
    bar_color = "#DC2626" if pct >= 50 else "#16A34A"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Attendance Risk Probability", "font": {"size": 18, "color": "#1E293B"}},
            number={"suffix": "%", "font": {"size": 32, "color": bar_color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B"},
                "bar": {"color": bar_color},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#CBD5E1",
                "steps": [
                    {"range": [0, 35], "color": "#F0FDF4"},
                    {"range": [35, 60], "color": "#FEF9C3"},
                    {"range": [60, 100], "color": "#FEF2F2"},
                ],
                "threshold": {
                    "line": {"color": "#991B1B", "width": 3},
                    "thickness": 0.8,
                    "value": 50.0,
                },
            },
        )
    )
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def create_confusion_matrix_heatmap(cm_dict: dict) -> go.Figure:
    """
    Renders a 2x2 confusion matrix heatmap from a dictionary.
    """
    tn = cm_dict.get("true_negatives", 0)
    fp = cm_dict.get("false_positives", 0)
    fn = cm_dict.get("false_negatives", 0)
    tp = cm_dict.get("true_positives", 0)

    z = [[tn, fp], [fn, tp]]
    x = ["Predicted Safe (0)", "Predicted At Risk (1)"]
    y = ["Actual Safe (0)", "Actual At Risk (1)"]

    annotations = []
    labels = [
        [f"True Negatives (TN)<br><b>{tn}</b>", f"False Positives (FP)<br><b>{fp}</b>"],
        [f"False Negatives (FN)<br><b>{fn}</b>", f"True Positives (TP)<br><b>{tp}</b>"],
    ]

    for i in range(2):
        for j in range(2):
            annotations.append(
                dict(
                    x=x[j],
                    y=y[i],
                    text=labels[i][j],
                    font=dict(color="#0F172A", size=14),
                    showarrow=False,
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(
        title="Held-Out Test Set Confusion Matrix",
        annotations=annotations,
        height=320,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig

