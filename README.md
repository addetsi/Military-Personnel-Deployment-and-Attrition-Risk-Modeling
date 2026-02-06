# Military HR Analytics - Personnel Deployment & Attrition Risk System

**Predictive analytics system for military workforce planning**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/Status-In_Progress-yellow.svg)

---

## 📋 Project Overview

This portfolio project demonstrates end-to-end data science capabilities in HR analytics, focusing on military personnel management. The system combines classification and regression models to:

1. **Predict attrition risk** (HIGH/MEDIUM/LOW) - Early warning system for retention
2. **Forecast readiness scores** (0-100) - Support deployment planning
3. **Simulate deployment scenarios** - "What-if" analysis for workforce decisions

**Note:** All data is synthetic and does not represent actual military personnel.

---

## 🎯 Project Goals

### Business Objectives
- Identify personnel at risk of leaving before contract completion
- Predict individual and unit readiness for deployments
- Enable data-driven retention and deployment decisions
- Reduce attrition costs and optimize training investments

### Technical Objectives
- Build multi-class classification model (attrition risk)
- Build regression model (readiness forecasting)
- Engineer 25+ predictive features
- Create interactive dashboard for workforce metrics
- Develop deployment scenario simulator

---

## 🛠️ Technology Stack

**Core:**
- Python 3.8+
- Jupyter Notebook
- Pandas, NumPy

**Machine Learning:**
- Scikit-learn (preprocessing, baseline models)
- XGBoost (primary classifier/regressor)
- LightGBM (fast alternative)
- SHAP (model interpretation)

**Visualization:**
- Matplotlib, Seaborn (static plots)
- Plotly (interactive charts)
- Streamlit (dashboard)

**Deployment:**
- Docker (containerization)

---


## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/addetsi/Military-Personnel-Deployment-and-Attrition-Risk-Modeling.gut
cd directory-name

# Create virtual environment
python -m venv hr_env
source hr_env/bin/activate  # Windows: hr_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
jupyter notebook notebooks/01_data_generation.ipynb
```

Run all cells to generate 1000 synthetic personnel records.

### 3. (Coming Soon) Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📊 Current Progress

### ✅ Phase 1: Data Generation (COMPLETE)
- [x] Generated 1000 synthetic personnel records
- [x] Created 50+ features across 7 categories
- [x] Calculated 2 target variables (attrition risk, readiness score)
- [x] Introduced realistic missing data (7%)
- [x] Validated data quality and distributions

**Dataset Specifications:**
- Sample size: 1000 personnel
- Time period: 2019-2024 (5 years)
- Service branches: Army (60%), Navy (20%), Air Force (20%)
- Attrition risk: HIGH 25%, MEDIUM 30%, LOW 45%
- Readiness score: Mean ~75, Std ~12

### 🔄 Phase 2: EDA & Feature Engineering (NEXT)
- [ ] Create 15+ exploratory visualizations
- [ ] Engineer 25+ additional features
- [ ] Identify top predictors via correlation analysis
- [ ] Prepare dataset for modeling

### ⏳ Upcoming Phases
- Phase 3: Attrition Prediction Model
- Phase 4: Readiness Forecasting Model
- Phase 5: Deployment Scenario Simulator
- Phase 6: Interactive Dashboard
- Phase 7: Docker Deployment
- Phase 8: Documentation & Polish

---

## 🎯 Target Metrics

### Classification (Attrition Risk)
- Recall (HIGH_RISK): ≥ 0.75
- F1-Score: ≥ 0.67
- ROC-AUC: ≥ 0.80

### Regression (Readiness Score)
- RMSE: ≤ 8 points (on 0-100 scale)
- R²: ≥ 0.70
- MAE: ≤ 6 points

---

## 📖 Key Features Generated

**Demographics:** Age, gender, branch, rank, MOS, years of service

**Training:** Total hours, courses, certifications, scores, skills currency

**Health:** Health index, fitness score, medical leave, mental health status

**Leave:** Annual leave, emergency leave, unauthorized absences

**Deployment:** Total deployments, recent deployment history, combat exposure

**Performance:** Review scores, commendations, disciplinary actions, leadership potential

**Personal:** Marital status, dependents, education, financial stress, job offers

**Targets:**
- `attrition_risk`: HIGH_RISK / MEDIUM_RISK / LOW_RISK
- `readiness_score`: 0-100 continuous score

---


## 🙏 Acknowledgments

- Ghana Armed Forces (internship opportunity - synthetic data used)

---

