"""
train_model.py
Train and compare multiple ML models for salary prediction.
Saves the best model pipeline and model_results.csv.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "salary_data.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "salary_prediction_model.pkl")
RESULTS_PATH = os.path.join(MODELS_DIR, "model_results.csv")

FEATURE_COLS     = ["Education", "Experience", "Location", "Job_Title", "Age", "Gender"]
TARGET_COL       = "Salary"
CATEGORICAL_COLS = ["Education", "Location", "Job_Title", "Gender"]
NUMERICAL_COLS   = ["Experience", "Age"]
REQUIRED_COLUMNS = FEATURE_COLS + [TARGET_COL]


# ──────────────────────────────────────────────
# Step 1: Load & Validate
# ──────────────────────────────────────────────

def load_and_validate():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"[ERROR] salary_data.csv not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[INFO] Columns: {df.columns.tolist()}")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[ERROR] Missing required columns: {missing_cols}")

    print("[INFO] All required columns present.")
    return df


# ──────────────────────────────────────────────
# Step 2: Clean Data
# ──────────────────────────────────────────────

def clean_data(df):
    df = df.copy()

    # Strip whitespace from categorical columns
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Convert salary to numeric and drop bad rows
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    before = len(df)
    df = df[df[TARGET_COL].notna() & (df[TARGET_COL] > 0)]
    after = len(df)
    if before != after:
        print(f"[INFO] Removed {before - after} rows with invalid salary values.")

    # Remove duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        print(f"[INFO] Removing {dupes} duplicate rows.")
        df = df.drop_duplicates()

    df = df.reset_index(drop=True)
    print(f"[INFO] Clean dataset: {len(df)} rows.")
    return df


# ──────────────────────────────────────────────
# Step 3: Build Preprocessing Pipeline
# ──────────────────────────────────────────────

def build_preprocessor():
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # sparse_output was added in sklearn 1.2; older versions use sparse=False
    sklearn_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if sklearn_version >= (1, 2):
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", ohe),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


# ──────────────────────────────────────────────
# Step 4: Train & Evaluate Models
# ──────────────────────────────────────────────

def evaluate_model(name, model, preprocessor, X_train, X_test, y_train, y_test,
                   run_cv: bool = True, cv_folds: int = 5):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    cv_r2_mean = cv_r2_std = None
    if run_cv:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Build a fresh pipeline for CV to avoid data leakage
                cv_pre = build_preprocessor()
                cv_pipe = Pipeline([("preprocessor", cv_pre), ("model", model)])
                cv_scores = cross_val_score(
                    cv_pipe,
                    pd.concat([X_train, X_test]),
                    pd.concat([y_train, y_test]),
                    cv=cv_folds, scoring="r2", n_jobs=-1
                )
                cv_r2_mean = round(float(cv_scores.mean()), 4)
                cv_r2_std  = round(float(cv_scores.std()),  4)
        except Exception:
            pass

    cv_label = f"{cv_r2_mean:.4f} ±{cv_r2_std:.4f}" if cv_r2_mean is not None else "N/A"
    print(f"  [{name}]  MAE={mae:,.2f}  RMSE={rmse:,.2f}  R²={r2:.4f}  CV-R²={cv_label}")
    metrics = {
        "Model":    name,
        "MAE":      round(mae, 2),
        "MSE":      round(mse, 2),
        "RMSE":     round(rmse, 2),
        "R2 Score": round(r2, 4),
    }
    if cv_r2_mean is not None:
        metrics["CV_R2_Mean"] = cv_r2_mean
        metrics["CV_R2_Std"]  = cv_r2_std
    return pipeline, metrics


# ──────────────────────────────────────────────
# Main Training Script
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SalaryAI — Model Training Pipeline")
    print("=" * 60)

    # Load & clean
    df = load_and_validate()
    df = clean_data(df)

    # Features & target
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    print(f"\n[INFO] Features: {FEATURE_COLS}")
    print(f"[INFO] Target: {TARGET_COL}")
    print(f"[INFO] Dataset size: {len(X)} samples\n")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[INFO] Train size: {len(X_train)}  |  Test size: {len(X_test)}\n")

    # Build preprocessor (fresh for each model)
    models_to_train = [
        ("Linear Regression",        LinearRegression()),
        ("Ridge Regression",         Ridge(alpha=1.0)),
        ("Decision Tree",            DecisionTreeRegressor(random_state=42)),
        ("Random Forest",            RandomForestRegressor(n_estimators=100, random_state=42)),
        ("Gradient Boosting",        GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ("Extra Trees",              ExtraTreesRegressor(n_estimators=100, random_state=42)),
    ]

    print("[INFO] Training and evaluating models...")
    results     = []
    pipelines   = {}

    for name, model in models_to_train:
        preprocessor = build_preprocessor()  # fresh for each
        pipeline, metrics = evaluate_model(
            name, model, preprocessor, X_train, X_test, y_train, y_test
        )
        results.append(metrics)
        pipelines[name] = pipeline

    # Save results
    os.makedirs(MODELS_DIR, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n[INFO] Model results saved to: {RESULTS_PATH}")

    # Select best model by R²
    best_row    = results_df.loc[results_df["R2 Score"].idxmax()]
    best_name   = best_row["Model"]
    best_r2     = best_row["R2 Score"]
    best_pipeline = pipelines[best_name]

    print(f"\n[BEST MODEL] {best_name}  (R² = {best_r2:.4f})")

    # Save best pipeline
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"[INFO] Best model pipeline saved to: {MODEL_PATH}")

    # ── Quick sanity-check prediction ──────────────────────
    print("\n[TEST] Running a sample prediction...")
    sample = pd.DataFrame([{
        "Education": "Bachelor",
        "Experience": 5,
        "Location": "Urban",
        "Job_Title": "Engineer",
        "Age": 28,
        "Gender": "Female",
    }])
    predicted = best_pipeline.predict(sample)[0]
    print(f"[TEST] Sample prediction → ₹{predicted:,.2f}")

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\n  ✓ Best Model : {best_name}")
    print(f"  ✓ R² Score   : {best_r2:.4f}")
    print(f"  ✓ Model File : {MODEL_PATH}")
    print("=" * 60)
    print("\n[DONE] Training complete. Run: streamlit run app.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] Training failed: {e}")
        sys.exit(1)
