# Implementation Tasks & Checklist

- [x] **Milestone 1: Workspace Inspection, Project Plan, Architecture, and Task Checklist**
  - [x] Inspect workspace and existing datasets/code
  - [x] Create project directory structure (`backend`, `frontend`, `ml`, `data`, `artifacts`, `tests`, `docs`)
  - [x] Set up `.gitignore`, `.env.example`, `requirements.txt`
  - [x] Formulate comprehensive `PROJECT_PLAN.md`
  - [x] Formulate `TASKS.md`

- [x] **Milestone 2: Dataset Generation, Data Validation, and Exploratory Analysis**
  - [x] Implement `ml/generate_dataset.py` with reproducible random seed and behavioral simulation
  - [x] Implement simulation logic: separate historical attendance from future attendance trial to prevent leakage
  - [x] Enforce constraints (attended <= conducted, percentages rounded, trend categories)
  - [x] Add CLI execution entrypoint to generate and save `data/attendance_dataset.csv`
  - [x] Verify dataset summary statistics and absence of data leakage

- [x] **Milestone 3: Model Training, Evaluation, and Artifact Saving**
  - [x] Implement `ml/preprocessing.py` with `ColumnTransformer` (scaling & one-hot encoding)
  - [x] Implement `ml/train.py` comparing Logistic Regression (baseline) and Random Forest Classifier
  - [x] Implement `ml/evaluate.py` calculating accuracy, precision, recall, F1, confusion matrix, ROC-AUC
  - [x] Perform stratified data splitting and cross-validation
  - [x] Serialize best pipeline to `artifacts/attendance_pipeline.joblib` and metrics to `artifacts/model_metrics.json`
  - [x] Verify training script runs via CLI

- [x] **Milestone 4: FastAPI Application, Request Schemas, Prediction Service, and Endpoints**
  - [x] Implement `backend/config.py` for environment variables and paths
  - [x] Implement `backend/schemas.py` with Pydantic models and field/root validators
  - [x] Implement `backend/services/prediction_service.py` with singleton model loader and deterministic math calculator
  - [x] Implement `backend/dependencies.py` for dependency injection
  - [x] Implement `backend/main.py` with CORS, logging, and endpoints (`GET /`, `GET /health`, `GET /model-info`, `POST /predict`)
  - [x] Verify FastAPI endpoints with curl or python requests

- [x] **Milestone 5: Streamlit Frontend, Prediction Form, and Model Insights**
  - [x] Implement `frontend/components/api_client.py` for decoupled HTTP communication with FastAPI
  - [x] Implement `frontend/components/ui_helpers.py` for design system, badges, and Plotly charts
  - [x] Implement `frontend/app.py` home page and sidebar navigation
  - [x] Implement `frontend/pages/1_Dashboard.py` for dataset demographics and analytics
  - [x] Implement `frontend/pages/2_Predict_Attendance.py` for interactive prediction and mathematical recovery plan
  - [x] Implement `frontend/pages/3_Model_Insights.py` for performance evaluation, confusion matrix, and feature importance
  - [x] Verify Streamlit renders without runtime errors

- [x] **Milestone 6: Automated Testing and Integration Fixes**
  - [x] Implement `tests/test_schemas.py` for Pydantic validation edge cases
  - [x] Implement `tests/test_prediction.py` for prediction service and deterministic bounds
  - [x] Implement `tests/test_api.py` for FastAPI TestClient endpoints
  - [x] Run `pytest -v` and resolve any issues

- [x] **Milestone 7: Documentation, Clean Startup Commands, and Final Verification**
  - [x] Write `docs/ARCHITECTURE.md` explaining frontend-backend-ml relationship
  - [x] Write `docs/API.md` with complete endpoint schemas and examples
  - [x] Write `docs/ML_METHODOLOGY.md` explaining target definition, leakage prevention, metrics, and limitations
  - [x] Write comprehensive `README.md` with architecture diagram, setup instructions, and startup commands
  - [x] Perform end-to-end integration test with both backend and frontend running
  - [x] Final verification against all user requirements
