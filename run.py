"""
Smart Attendance Prediction System - Unified Application Runner

Launches both the FastAPI backend and the Streamlit frontend with a single command:
    python run.py

Automatically:
1. Verifies/generates dataset and trains model if artifacts are missing.
2. Starts FastAPI backend on port 8000.
3. Waits for backend readiness probe.
4. Starts Streamlit frontend on port 8501.
5. Opens your default web browser to the application.
6. Gracefully terminates both processes on Ctrl+C.
"""

import os
import sys
import time
import signal
import subprocess
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT_DIR / "artifacts" / "attendance_pipeline.joblib"
DATASET_PATH = ROOT_DIR / "data" / "attendance_dataset.csv"

# Global subprocess handles
backend_process = None
frontend_process = None


def shutdown_processes(signum=None, frame=None):
    """Gracefully terminates backend and frontend processes."""
    print("\n🛑 Shutting down Smart Attendance Prediction System...")
    global backend_process, frontend_process

    if frontend_process and frontend_process.poll() is None:
        frontend_process.terminate()
        try:
            frontend_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            frontend_process.kill()

    if backend_process and backend_process.poll() is None:
        backend_process.terminate()
        try:
            backend_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            backend_process.kill()

    print("✅ All processes stopped. Goodbye!")
    sys.exit(0)


def ensure_model_artifacts():
    """Checks if the dataset and trained pipeline exist, building them if absent."""
    if not DATASET_PATH.exists():
        print("📦 Dataset not found. Generating synthetic dataset (data/attendance_dataset.csv)...")
        subprocess.run([sys.executable, "-m", "ml.generate_dataset"], check=True)

    if not ARTIFACT_PATH.exists():
        print("⚙️ Model artifact not found. Training and serializing pipeline...")
        subprocess.run([sys.executable, "-m", "ml.train"], check=True)


def main():
    global backend_process, frontend_process

    # Register signal handlers for clean exit
    signal.signal(signal.SIGINT, shutdown_processes)
    signal.signal(signal.SIGTERM, shutdown_processes)

    print("=" * 65)
    print("  🎓 Smart Attendance Prediction System - One-Click Launcher")
    print("=" * 65)

    # 1. Verify ML readiness
    ensure_model_artifacts()

    # 2. Launch FastAPI Backend
    print("\n[1/3] Starting FastAPI Backend on http://127.0.0.1:8000 ...")
    backend_cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--log-level", "warning"
    ]
    backend_process = subprocess.Popen(backend_cmd)

    # Wait for backend to be ready
    import urllib.request
    import urllib.error

    backend_ready = False
    for attempt in range(20):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1) as resp:
                if resp.status == 200:
                    backend_ready = True
                    break
        except Exception:
            continue

    if not backend_ready:
        print("❌ Backend failed to start. Terminating.")
        shutdown_processes()

    print("      Backend is Healthy and Model is loaded!")

    # 3. Launch Streamlit Frontend
    print("[2/3] Starting Streamlit UI on http://localhost:8501 ...")
    frontend_cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(ROOT_DIR / "frontend" / "app.py"),
        "--server.port", "8501",
        "--server.headless", "true"
    ]
    frontend_process = subprocess.Popen(frontend_cmd)

    # Give Streamlit a moment to spin up
    time.sleep(2)

    # 4. Open in default browser
    print("[3/3] Opening application in your web browser...")
    webbrowser.open("http://localhost:8501")

    print("\n" + "=" * 65)
    print("  🚀 Application Running!")
    print("  - Streamlit UI:  http://localhost:8501")
    print("  - FastAPI Docs:  http://127.0.0.1:8000/docs")
    print("  Press Ctrl+C at any time in this terminal to stop the system.")
    print("=" * 65 + "\n")

    # Keep script alive until interrupted
    try:
        while True:
            time.sleep(1)
            # Monitor process health
            if backend_process.poll() is not None:
                print("⚠️ Backend server exited unexpectedly.")
                break
            if frontend_process.poll() is not None:
                print("⚠️ Frontend UI exited.")
                break
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_processes()


if __name__ == "__main__":
    main()

