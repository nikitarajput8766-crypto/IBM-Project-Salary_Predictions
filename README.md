# SalaryAI — Intelligent Salary & Career Analytics System

## Project Overview

SalaryAI is an end-to-end machine learning web application that predicts salaries,
analyses salary patterns, benchmarks compensation, and provides career insights using
a real dataset. Built with Python, Scikit-learn, and Streamlit.

---

## Problem Statement

Salary determination is complex and opaque. Employees and job seekers often lack access
to reliable salary data for their profile. SalaryAI addresses this by building an
intelligent analytics system that leverages machine learning to predict and analyse
salary patterns based on education, experience, job title, location, age, and gender.

---

## Objectives

- Predict expected salary using a trained ML model
- Analyse salary patterns across the dataset interactively
- Benchmark an employee's current salary against the model estimate
- Explore salary differences across demographic and professional categories
- Provide career insights based on real dataset statistics
- Compare multiple ML model performances side-by-side

---

## Features

| Module | Description |
|---|---|
| 🏠 Home | Dashboard with dataset overview and key metrics |
| 💰 Salary Prediction | ML-based salary prediction from user profile |
| 📊 Salary Analytics | Interactive charts with filters |
| ⚖️ Salary Benchmarking | Compare current vs predicted salary |
| 📈 Pay Gap Analysis | Descriptive salary differences across groups |
| 🚀 Career Insights | Dataset-driven career information |
| 🤖 Model Performance | ML model comparison with metrics |
| ℹ️ About | Project and technology information |

---

## Dataset Description

**File:** `salary_data.csv`

| Column | Type | Description |
|---|---|---|
| Education | Categorical | Highest qualification (High School / Bachelor / Master / PhD) |
| Experience | Numerical | Years of work experience |
| Location | Categorical | Work location (Urban / Suburban / Rural) |
| Job_Title | Categorical | Job role (Manager / Director / Analyst / Engineer) |
| Age | Numerical | Employee age |
| Gender | Categorical | Gender (Male / Female) |
| Salary | Numerical (Target) | Annual salary in ₹ |

---

## Machine Learning Workflow

```
salary_data.csv
      ↓
Data Loading & Validation
      ↓
Data Cleaning (whitespace, invalid salary, duplicates)
      ↓
Feature/Target Split
      ↓
Train/Test Split (80/20, random_state=42)
      ↓
ColumnTransformer:
  ├── Numerical (Experience, Age): SimpleImputer (median)
  └── Categorical (Education, Location, Job_Title, Gender):
        SimpleImputer (most_frequent) → OneHotEncoder (handle_unknown=ignore)
      ↓
Train 4 Models → Evaluate (MAE, MSE, RMSE, R²)
      ↓
Select Best Model (highest R²)
      ↓
Save Pipeline → models/salary_prediction_model.pkl
Save Results  → models/model_results.csv
```

---

## Algorithms

| Algorithm | Type |
|---|---|
| Linear Regression | Linear model (baseline) |
| Decision Tree Regressor | Non-linear tree model |
| Random Forest Regressor | Ensemble (bagging) |
| Gradient Boosting Regressor | Ensemble (boosting) |

---

## Evaluation Metrics

| Metric | Full Name | Interpretation |
|---|---|---|
| MAE | Mean Absolute Error | Avg absolute salary error |
| MSE | Mean Squared Error | Avg squared error |
| RMSE | Root Mean Squared Error | Error in salary units |
| R² | Coefficient of Determination | Variance explained (0–1) |

---

## Project Structure

```
MyBotProject/
│
├── salary_data.csv          ← Original dataset (DO NOT MODIFY)
├── app.py                   ← Main Streamlit application
├── train_model.py           ← ML training pipeline
├── data_analysis.py         ← Reusable data functions
├── requirements.txt         ← Python dependencies
├── README.md                ← This file
├── .gitignore
│
├── models/
│   ├── salary_prediction_model.pkl  ← Saved best model
│   └── model_results.csv            ← Model comparison results
│
└── assets/                  ← Static assets (if any)
```

---

## Installation & Setup (Windows)

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the virtual environment

```bash
venv\Scripts\activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Train the ML model

```bash
python train_model.py
```

This will:
- Load and validate `salary_data.csv`
- Train 4 ML models
- Select the best model by R² score
- Save `models/salary_prediction_model.pkl`
- Save `models/model_results.csv`

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

Open your browser at: http://localhost:8501

---

## How to Use Each Module

### 💰 Salary Prediction
1. Navigate to **Salary Prediction** in the sidebar
2. Enter your education level, experience, location, job title, age, and gender
3. Click **Predict Salary**
4. View the predicted salary and estimated range

### 📊 Salary Analytics
1. Navigate to **Salary Analytics**
2. Use the **Filters** expander to narrow down by education, location, job title, or gender
3. All charts update based on the selected filters

### ⚖️ Salary Benchmarking
1. Navigate to **Salary Benchmarking**
2. Enter your profile and current salary
3. Click **Benchmark My Salary**
4. View how your salary compares to the model estimate

### 📈 Pay Gap Analysis
1. Navigate to **Pay Gap Analysis**
2. Switch between tabs: Gender / Education / Location / Job Title / Experience
3. Review descriptive statistics and charts

### 🚀 Career Insights
1. Navigate to **Career Insights**
2. Select education, job title, location, and experience
3. Click **Get Career Insights**
4. View salary statistics and distribution for matching records

### 🤖 Model Performance
1. Navigate to **Model Performance**
2. View the model comparison table
3. Review bar charts for R², RMSE, and MAE
4. The best model is highlighted in green

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `salary_data.csv not found` | Ensure the file is in the `MyBotProject/` directory |
| `Model has not been trained yet` | Run `python train_model.py` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| App crashes on start | Check Python ≥ 3.8 and all packages installed |
| Empty charts after filtering | Relax the selected filters |

---

## Future Improvements

- Add more ML models (XGBoost, LightGBM, SVR)
- Add hyperparameter tuning with GridSearchCV
- Add SHAP feature importance explanations
- Add CSV upload for bulk salary prediction
- Add time-series salary trend analysis (if temporal data available)
- Add export-to-PDF report functionality
- Add user authentication for saved profiles
- Deploy to Streamlit Cloud or AWS

---

## Disclaimer

All predictions and analyses are based solely on the `salary_data.csv` dataset.
Results are for educational and demonstration purposes only.
Actual salaries depend on many factors not present in this dataset.
