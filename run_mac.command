#!/bin/bash
# ============================================================
# Smart Attendance Predictor - macOS One-Click Launcher
# Double-click this file in Finder to run the entire system.
# ============================================================

# Navigate to the folder containing this script
cd "$(dirname "$0")"

echo "============================================================"
echo "  🎓 Starting Smart Attendance Prediction System (macOS)"
echo "============================================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 was not found on your system."
    echo "Please download Python from https://www.python.org/downloads/"
    read -p "Press Enter to exit..."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 First-time setup: Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "📥 Installing required packages (one-time setup, please wait)..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Run the unified Python runner
python run.py

