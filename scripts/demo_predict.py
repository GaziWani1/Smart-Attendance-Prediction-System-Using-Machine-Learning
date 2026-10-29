"""
Smart Attendance Prediction System - Client Demonstration Script

Demonstrates how to send predictions to the FastAPI backend using standard Python requests.
"""

import sys
import requests

API_URL = "http://127.0.0.1:8000"


def check_api_health():
    """Verify backend is healthy before making predictions."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            print(f"✅ Backend is Healthy! Active Model: {r.json().get('model_name')}")
            return True
        else:
            print(f"⚠️ Backend returned status {r.status_code}: {r.text}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to FastAPI at {API_URL}.")
        print("   Ensure Uvicorn is running: uvicorn backend.main:app --reload")
        return False


def run_prediction_demo():
    """Run example student scenarios through the prediction API."""
    scenarios = [
        {
            "name": "Scenario 1: High-Risk Student (Declining trend + Recent absences)",
            "payload": {
                "classes_conducted": 50,
                "classes_attended": 28,
                "classes_remaining": 25,
                "previous_attendance_trend": "declining",
                "consecutive_absences_recent": 3,
                "participation_score": 4.5,
            },
        },
        {
            "name": "Scenario 2: Borderline Student (Stable trend + Moderate engagement)",
            "payload": {
                "classes_conducted": 45,
                "classes_attended": 33,
                "classes_remaining": 20,
                "previous_attendance_trend": "stable",
                "consecutive_absences_recent": 1,
                "participation_score": 7.0,
            },
        },
        {
            "name": "Scenario 3: Strong Standing Student (Improving trend + High participation)",
            "payload": {
                "classes_conducted": 50,
                "classes_attended": 46,
                "classes_remaining": 25,
                "previous_attendance_trend": "improving",
                "consecutive_absences_recent": 0,
                "participation_score": 9.2,
            },
        },
    ]

    print("\n" + "=" * 65)
    print("  Smart Attendance Predictor - Client Inference Demonstration")
    print("=" * 65)

    for s in scenarios:
        print(f"\n--- {s['name']} ---")
        p = s["payload"]
        current_pct = round((p["classes_attended"] / p["classes_conducted"]) * 100, 1)
        print(f"Inputs: Attended {p['classes_attended']}/{p['classes_conducted']} ({current_pct}%), "
              f"Remaining: {p['classes_remaining']}, Trend: {p['previous_attendance_trend']}, "
              f"Absence Streak: {p['consecutive_absences_recent']}, Engagement: {p['participation_score']}/10")

        resp = requests.post(f"{API_URL}/predict", json=p, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            badge = "🔴 AT RISK" if data["attendance_risk"] == 1 else "🟢 SAFE"
            prob_str = f"{data['risk_probability'] * 100:.1f}%" if data["risk_probability"] is not None else "N/A"
            bounds = data["deterministic_bounds"]

            print(f"Prediction: {badge} (Risk Probability: {prob_str})")
            print(f"Explanation: {data['explanation']}")
            print(f"Mathematical Bounds: Max Possible={bounds['max_possible_attendance_pct']}%, "
                  f"Min Possible={bounds['min_possible_attendance_pct']}%, "
                  f"Needed for 75%={bounds['classes_needed_for_75_pct']} of {p['classes_remaining']} classes")
        else:
            print(f"Error {resp.status_code}: {resp.text}")


if __name__ == "__main__":
    if not check_api_health():
        sys.exit(1)
    run_prediction_demo()

