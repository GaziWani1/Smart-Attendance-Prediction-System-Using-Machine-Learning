"""
Smart Attendance Prediction System - Page 2: Predict Attendance
"""

import sys
from pathlib import Path

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from frontend.components.ui_helpers import (
    inject_custom_css,
    render_sidebar_status,
    create_probability_gauge,
)
from frontend.components.api_client import predict_attendance_risk

st.set_page_config(page_title="Predict Risk | Smart Attendance", page_icon="🔮", layout="wide")
inject_custom_css()
render_sidebar_status()

st.title("🔮 Predict Attendance Risk")
st.markdown("Enter student attendance and engagement metrics to estimate semester threshold risk.")

col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.subheader("Student Metrics Input")

    with st.form("attendance_input_form"):
        # Section A: Attendance Counts
        st.markdown("**1. Course Attendance Progress**")
        classes_conducted = st.number_input(
            "Classes Conducted So Far",
            min_value=5,
            max_value=150,
            value=50,
            step=1,
            help="Total number of lectures held in the semester up to today.",
        )

        classes_attended = st.number_input(
            "Classes Attended So Far",
            min_value=0,
            max_value=classes_conducted,
            value=min(36, classes_conducted),
            step=1,
            help="Classes the student physically attended. Cannot exceed classes conducted.",
        )

        classes_remaining = st.number_input(
            "Classes Remaining in Semester",
            min_value=1,
            max_value=100,
            value=25,
            step=1,
            help="Remaining scheduled lecture sessions until the semester ends.",
        )

        # Real-time preview of current attendance
        current_rate = round((classes_attended / max(classes_conducted, 1)) * 100, 2)
        total_semester = classes_conducted + classes_remaining
        st.caption(
            f"ℹ️ Current Attendance Rate: **{current_rate}%** ({classes_attended}/{classes_conducted}). "
            f"Total planned semester classes: **{total_semester}**."
        )

        st.markdown("---")
        # Section B: Behavioral & Engagement Indicators
        st.markdown("**2. Behavioral & Trajectory Indicators**")

        trend = st.selectbox(
            "Observed Recent Attendance Trend",
            options=["declining", "stable", "improving"],
            index=1,
            help="Direction of student attendance over the most recent 2-3 weeks.",
        )

        absences = st.slider(
            "Consecutive Recent Absences",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
            help="Number of consecutive lectures missed immediately prior to this checkpoint.",
        )

        participation = st.slider(
            "Classroom Engagement / Participation Score (1-10)",
            min_value=1.0,
            max_value=10.0,
            value=7.0,
            step=0.5,
            help="Holistic score factoring in quiz completion, assignment punctuality, and lab work.",
        )

        submit_btn = st.form_submit_button("⚡ Predict Attendance Risk", use_container_width=True)

with col_result:
    st.subheader("Model Assessment & Analysis")

    if submit_btn:
        # Client-side sanity check
        if classes_attended > classes_conducted:
            st.error("Validation Error: Classes attended cannot exceed classes conducted.")
        else:
            payload = {
                "classes_conducted": int(classes_conducted),
                "classes_attended": int(classes_attended),
                "classes_remaining": int(classes_remaining),
                "previous_attendance_trend": trend,
                "consecutive_absences_recent": int(absences),
                "participation_score": float(participation),
            }

            with st.spinner("Calling FastAPI inference endpoint..."):
                success, response_data = predict_attendance_risk(payload)

            if success:
                risk_label = response_data.get("risk_label", "Unknown")
                prob = response_data.get("risk_probability")
                explanation = response_data.get("explanation", "")
                bounds = response_data.get("deterministic_bounds", {})

                # 1. Prediction Verdict Badge
                if risk_label == "At Risk":
                    st.markdown(
                        f"""
                        <div class="badge-risk">
                            ⚠️ Verdict: <b>AT RISK</b> of Falling Below 75%
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="badge-safe">
                            ✅ Verdict: <b>ON TRACK / SAFE</b> (>= 75% Projected)
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # 2. Probability Gauge
                if prob is not None:
                    st.plotly_chart(create_probability_gauge(prob), use_container_width=True)

                # 3. Model Explanation
                st.markdown("#### 🧠 Prediction Explanation")
                st.write(explanation)

                # 4. Deterministic Mathematical Calculations (Strictly separated)
                st.markdown("---")
                st.markdown("#### 📐 Deterministic Mathematical Calculation")
                st.caption("*(Note: The numbers below are exact mathematical calculations, NOT machine learning estimates)*")

                max_possible = bounds.get("max_possible_attendance_pct", 0.0)
                min_possible = bounds.get("min_possible_attendance_pct", 0.0)
                needed = bounds.get("classes_needed_for_75_pct", 0)
                rem = classes_remaining

                b_col1, b_col2, b_col3 = st.columns(3)
                b_col1.metric("Max Achievable", f"{max_possible}%", help="If student attends all remaining classes.")
                b_col2.metric("Min Achievable", f"{min_possible}%", help="If student attends 0 remaining classes.")
                b_col3.metric("Required to Reach 75%", f"{needed} of {rem}", help="Minimum classes needed to avoid falling below 75%.")

                # Advisory Action Plan
                if needed > rem:
                    st.error(
                        f"🚨 **Critical Alert**: Even with 100% attendance across the remaining {rem} classes, "
                        f"the highest achievable attendance is {max_possible}%. Student cannot mathematically reach 75% without institutional waiver."
                    )
                elif needed == 0:
                    st.success(
                        f"🎉 **Guaranteed Standing**: Even with 0% attendance across the remaining {rem} classes, "
                        f"the lowest final attendance is {min_possible}%. Safe standing is mathematically secured."
                    )
                else:
                    needed_rate = round((needed / rem) * 100, 1)
                    st.info(
                        f"📋 **Action Plan**: The student must attend at least **{needed} out of {rem}** remaining classes "
                        f"({needed_rate}% attendance rate) to conclude the semester with >= 75% attendance."
                    )

                # Disclaimer
                st.caption(
                    "⚠️ **Disclaimer**: Model predictions represent probabilistic estimates based on simulated behavioral "
                    "patterns. They should be used to support academic counseling, not for punitive decisions."
                )

            else:
                st.error(f"Prediction Failed: {response_data.get('message', 'Unknown error')}")
                if "details" in response_data:
                    st.json(response_data["details"])
    else:
        st.info("👈 Fill out the attendance details on the left and click **'Predict Attendance Risk'** to generate a prediction.")

