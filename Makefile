.PHONY: help install generate-data train test run-backend run-frontend clean zip

help:
	@echo "Smart Attendance Prediction System - Developer Commands:"
	@echo "  make install        Install project dependencies"
	@echo "  make generate-data  Generate reproducible synthetic attendance dataset"
	@echo "  make train          Train ML models and save serialized pipeline"
	@echo "  make test           Run automated test suite (Pytest)"
	@echo "  make run-backend    Start FastAPI backend with Uvicorn (Port 8000)"
	@echo "  make run-frontend   Start Streamlit frontend UI (Port 8501)"
	@echo "  make clean          Clean temporary caches and pyc files"
	@echo "  make zip            Package lightweight project for sharing (excludes .venv)"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

generate-data:
	python3 -m ml.generate_dataset

train:
	python3 -m ml.train

test:
	python3 -m pytest -v

run-backend:
	uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

run-frontend:
	streamlit run frontend/app.py --server.port 8501

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

zip: clean
	@rm -f smart_attendance_project.zip
	zip -r smart_attendance_project.zip . -x ".venv/*" ".pytest_cache/*" "*/__pycache__/*" ".DS_Store" "*/.DS_Store" "smart_attendance_project.zip"
	@echo "Created lightweight shareable zip: smart_attendance_project.zip"
	@ls -lh smart_attendance_project.zip

