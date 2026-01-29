import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data Generation Parameters
RANDOM_SEED = 42
N_PERSONNEL = 1000
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"

# Service Branch Distribution
BRANCH_DISTRIBUTION = {
    "Army": 0.60,
    "Navy": 0.20,
    "Air Force": 0.20
}

# Rank Distribution
RANK_DISTRIBUTION = {
    "Junior": 0.50,
    "NCO": 0.35,
    "Officer": 0.15
}

# Military Occupational Specialties (MOS)
MOS_LIST = [
    "Infantry", "Artillery", "Armor", "Aviation", "Intelligence",
    "Signal", "Engineer", "Medical", "Logistics", "Military Police",
    "Special Forces", "Cyber Operations", "Maintenance", "Administration",
    "Human Resources"
]

# Attrition Risk Distribution (target variable)
ATTRITION_RISK_DISTRIBUTION = {
    "HIGH_RISK": 0.25,      # 25% - will leave within 12 months
    "MEDIUM_RISK": 0.30,    # 30% - will leave 12-24 months
    "LOW_RISK": 0.45        # 45% - retained beyond 24 months
}

# Readiness Score Parameters
READINESS_WEIGHTS = {
    "training_currency": 0.30,
    "health_fitness": 0.25,
    "deployment_experience": 0.20,
    "performance": 0.15,
    "skill_specialization": 0.10
}

READINESS_MEAN = 75.0
READINESS_STD = 12.0

# Missing Data Percentage
MISSING_DATA_PCT = 0.07  # 7% missing data

# Model Training Parameters
TEST_SIZE = 0.20
VAL_SIZE = 0.25  # Of remaining after test split
STRATIFY = True

# XGBoost Hyperparameters (Attrition Classification)
XGBOOST_CLASSIFIER_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "objective": "multi:softmax",
    "num_class": 3,
    "random_state": RANDOM_SEED,
    "eval_metric": "mlogloss"
}

# XGBoost Hyperparameters (Readiness Regression)
XGBOOST_REGRESSOR_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "objective": "reg:squarederror",
    "random_state": RANDOM_SEED,
    "eval_metric": "rmse"
}

# LightGBM Hyperparameters (Alternative)
LIGHTGBM_CLASSIFIER_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "objective": "multiclass",
    "num_class": 3,
    "random_state": RANDOM_SEED,
    "verbose": -1
}

LIGHTGBM_REGRESSOR_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": RANDOM_SEED,
    "verbose": -1
}

# Model Evaluation Targets
ATTRITION_RECALL_TARGET = 0.75  # High-risk class
ATTRITION_F1_TARGET = 0.67
ATTRITION_ROC_AUC_TARGET = 0.80

READINESS_RMSE_TARGET = 8.0
READINESS_R2_TARGET = 0.70
READINESS_MAE_TARGET = 6.0
READINESS_MAPE_TARGET = 10.0

# Dashboard Configuration
DASHBOARD_PORT = 8501
DASHBOARD_TITLE = "Military HR Analytics Dashboard"

# Feature Engineering Parameters
TRAINING_RECENCY_DECAY = 180  # days
DEPLOYMENT_RECENCY_DECAY = 12  # months
CONTRACT_PRESSURE_THRESHOLD = 12  # months

# Deployment Simulation Defaults
DEFAULT_MIN_READINESS = 75.0
DEFAULT_DEPLOYMENT_DURATION = 6  # months
OPTIMIZATION_STRATEGIES = ["readiness", "balanced", "low_risk"]

# File Names
PERSONNEL_DATA_FILE = "military_personnel.csv"
FEATURES_ENGINEERED_FILE = "features_engineered.csv"
DATA_DICTIONARY_FILE = "data_dictionary.csv"

ATTRITION_MODEL_FILE = "attrition_classifier.pkl"
READINESS_MODEL_FILE = "readiness_regressor.pkl"
ATTRITION_SCALER_FILE = "attrition_scaler.pkl"
READINESS_SCALER_FILE = "readiness_scaler.pkl"
ATTRITION_FEATURES_FILE = "attrition_features.txt"
READINESS_FEATURES_FILE = "readiness_features.txt"

# Visualization Parameters
FIGURE_DPI = 300
FIGURE_SIZE = (12, 6)
COLOR_PALETTE = "Set2"

ATTRITION_COLORS = {
    "LOW_RISK": "#2ecc71",      # Green
    "MEDIUM_RISK": "#f39c12",   # Orange
    "HIGH_RISK": "#e74c3c"      # Red
}

# Synthetic Data Generation - Feature Ranges
FEATURE_RANGES = {
    "age": (18, 55),
    "years_of_service": (0, 30),
    "total_training_hours": (0, 2000),
    "specialized_courses_completed": (0, 15),
    "training_score_average": (60, 100),
    "certifications_held": (0, 8),
    "health_index": (50, 100),
    "physical_fitness_score": (0, 100),
    "days_on_medical_leave": (0, 180),
    "annual_leave_taken": (0, 30),
    "emergency_leave_incidents": (0, 5),
    "unauthorized_absences": (0, 3),
    "total_deployments": (0, 8),
    "months_deployed_last_3yrs": (0, 36),
    "mission_performance_rating": (1, 5),
    "performance_review_score": (60, 100),
    "commendations": (0, 10),
    "disciplinary_actions": (0, 5),
    "leadership_potential_score": (0, 100),
    "dependents": (0, 5),
    "civilian_job_offers": (0, 3),
    "family_support_score": (0, 100),
    "peer_rating_score": (0, 100)
}

# Categorical Variables
GENDER_OPTIONS = ["Male", "Female"]
MARITAL_STATUS_OPTIONS = ["Single", "Married", "Divorced", "Widowed"]
EDUCATION_LEVELS = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
MENTAL_HEALTH_STATUS = ["Excellent", "Good", "Fair", "Concern"]
FINANCIAL_STRESS_LEVELS = ["Low", "Medium", "High"]
DEPLOYMENT_TYPES = ["Domestic", "International", "Combat"]
COMBAT_EXPOSURE_LEVELS = ["None", "Low", "Moderate", "High"]
RELOCATION_WILLINGNESS = ["Low", "Medium", "High"]

print(f"Configuration loaded successfully!")
print(f"Project root: {PROJECT_ROOT}")
print(f"Random seed: {RANDOM_SEED}")
print(f"Target sample size: {N_PERSONNEL} personnel")