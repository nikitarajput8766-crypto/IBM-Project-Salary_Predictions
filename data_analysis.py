"""
data_analysis.py
Reusable data loading, cleaning, statistics, filtering and analysis
functions for SalaryAI — Intelligent Salary Prediction & Analytics.
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "salary_data.csv")

REQUIRED_COLUMNS = ["Education", "Experience", "Location", "Job_Title", "Age", "Gender", "Salary"]
FEATURE_COLS     = ["Education", "Experience", "Location", "Job_Title", "Age", "Gender"]
TARGET_COL       = "Salary"

CATEGORICAL_COLS = ["Education", "Location", "Job_Title", "Gender"]
NUMERICAL_COLS   = ["Experience", "Age"]

EDUCATION_ORDER  = ["High School", "Bachelor", "Master", "PhD"]


# ──────────────────────────────────────────────
# Loading & Validation
# ──────────────────────────────────────────────

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load salary_data.csv and return a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame) -> bool:
    """Return True if all required columns are present."""
    return all(col in df.columns for col in REQUIRED_COLUMNS)


def get_missing_columns(df: pd.DataFrame) -> list:
    """Return list of required columns that are missing from df."""
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


# ──────────────────────────────────────────────
# Cleaning (in-memory only — never modifies source file)
# ──────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a clean copy of df:
    - Strip whitespace from categorical columns
    - Remove rows where Salary is null or <= 0
    - Remove duplicate rows
    Original df is never modified.
    """
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if TARGET_COL in df.columns:
        df = df[pd.to_numeric(df[TARGET_COL], errors="coerce").notna()]
        df[TARGET_COL] = df[TARGET_COL].astype(float)
        df = df[df[TARGET_COL] > 0]

    df = df.drop_duplicates()
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────
# Data Quality Report
# ──────────────────────────────────────────────

def data_quality_report(df: pd.DataFrame) -> dict:
    """
    Return a comprehensive data quality summary dict.
    Does NOT modify the dataframe.
    """
    report = {
        "total_rows":       len(df),
        "total_columns":    len(df.columns),
        "duplicate_rows":   int(df.duplicated().sum()),
        "missing_by_col":   df.isnull().sum().to_dict(),
        "total_missing":    int(df.isnull().sum().sum()),
    }

    issues = []

    # Invalid salary
    if TARGET_COL in df.columns:
        numeric_salary = pd.to_numeric(df[TARGET_COL], errors="coerce")
        bad_salary = int((numeric_salary.isna() | (numeric_salary <= 0)).sum())
        report["invalid_salary_rows"] = bad_salary
        if bad_salary:
            issues.append(f"{bad_salary} rows with invalid/missing salary")

    # Invalid age — coerce first so non-numeric values don't crash the comparison
    if "Age" in df.columns:
        age_num = pd.to_numeric(df["Age"], errors="coerce")
        bad_age = int((age_num.isna() | (age_num < 16) | (age_num > 80)).sum())
        report["invalid_age_rows"] = bad_age
        if bad_age:
            issues.append(f"{bad_age} rows with unusual age (<16 or >80)")

    # Invalid experience — coerce first so non-numeric values don't crash the comparison
    if "Experience" in df.columns:
        exp_num = pd.to_numeric(df["Experience"], errors="coerce")
        bad_exp = int((exp_num.isna() | (exp_num < 0) | (exp_num > 50)).sum())
        report["invalid_experience_rows"] = bad_exp
        if bad_exp:
            issues.append(f"{bad_exp} rows with unusual experience (<0 or >50)")

    # Categorical consistency
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            report[f"{col}_unique_values"] = sorted(df[col].dropna().unique().tolist())

    report["issues"] = issues
    return report


# ──────────────────────────────────────────────
# Summary Statistics
# ──────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame) -> dict:
    """Return a dict of key dataset statistics."""
    summary = {
        "total_records":  len(df),
        "total_columns":  len(df.columns),
        "columns":        df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if TARGET_COL in df.columns:
        salary = df[TARGET_COL]
        summary.update({
            "salary_min":    float(salary.min()),
            "salary_max":    float(salary.max()),
            "salary_mean":   float(salary.mean()),
            "salary_median": float(salary.median()),
            "salary_std":    float(salary.std()),
        })
    return summary


def salary_statistics(df: pd.DataFrame) -> dict:
    """Return salary descriptive statistics."""
    s = df[TARGET_COL]
    return {
        "count":  int(s.count()),
        "mean":   round(float(s.mean()), 2),
        "median": round(float(s.median()), 2),
        "min":    round(float(s.min()), 2),
        "max":    round(float(s.max()), 2),
        "std":    round(float(s.std()), 2),
        "25%":    round(float(s.quantile(0.25)), 2),
        "75%":    round(float(s.quantile(0.75)), 2),
    }


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising missing values per column."""
    missing_count = df.isnull().sum()
    missing_pct   = (missing_count / len(df) * 100).round(2)
    report = pd.DataFrame({
        "Column":        missing_count.index,
        "Missing Count": missing_count.values,
        "Missing %":     missing_pct.values,
    })
    return report[report["Missing Count"] > 0].reset_index(drop=True)


# ──────────────────────────────────────────────
# Unique Category Values
# ──────────────────────────────────────────────

def get_unique_values(df: pd.DataFrame, col: str) -> list:
    """Return sorted unique non-null values for a column."""
    return sorted(df[col].dropna().unique().tolist())


def get_education_levels(df: pd.DataFrame) -> list:
    """Return education levels in logical order."""
    present = df["Education"].dropna().unique().tolist()
    return [e for e in EDUCATION_ORDER if e in present]


def get_all_categories(df: pd.DataFrame) -> dict:
    """Return dict of all categorical column → sorted unique values."""
    return {col: get_unique_values(df, col) for col in CATEGORICAL_COLS if col in df.columns}


# ──────────────────────────────────────────────
# Filtering
# ──────────────────────────────────────────────

def filter_data(df: pd.DataFrame,
                education: list = None,
                location: list = None,
                job_title: list = None,
                gender: list = None,
                exp_range: tuple = None) -> pd.DataFrame:
    """
    Filter df by lists of allowed values.
    exp_range: optional (min_exp, max_exp) tuple.
    None means no filter applied for that column.
    """
    filtered = df.copy()
    if education:
        filtered = filtered[filtered["Education"].isin(education)]
    if location:
        filtered = filtered[filtered["Location"].isin(location)]
    if job_title:
        filtered = filtered[filtered["Job_Title"].isin(job_title)]
    if gender:
        filtered = filtered[filtered["Gender"].isin(gender)]
    if exp_range:
        filtered = filtered[
            (filtered["Experience"] >= exp_range[0]) &
            (filtered["Experience"] <= exp_range[1])
        ]
    return filtered.reset_index(drop=True)


# ──────────────────────────────────────────────
# Group-level Salary Analysis
# ──────────────────────────────────────────────

def salary_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Return avg, median, count, min, max salary grouped by group_col."""
    grouped = (
        df.groupby(group_col)[TARGET_COL]
        .agg(Count="count", Average="mean", Median="median", Min="min", Max="max")
        .round(2)
        .reset_index()
    )
    grouped.columns.name = None
    return grouped


def salary_by_experience_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Return average salary binned by experience ranges."""
    df2 = df.copy()
    df2["Exp_Bin"] = pd.cut(
        df2["Experience"],
        bins=[0, 5, 10, 15, 20, 25, 30, 100],
        labels=["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "30+"],
        right=True,
    )
    result = (
        df2.groupby("Exp_Bin", observed=True)[TARGET_COL]
        .agg(Count="count", Average="mean")
        .round(2)
        .reset_index()
    )
    return result


def pay_gap_analysis(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Return salary statistics grouped by group_col."""
    grouped = (
        df.groupby(group_col)[TARGET_COL]
        .agg(Count="count", Avg_Salary="mean", Median_Salary="median",
             Min_Salary="min", Max_Salary="max", Std_Salary="std")
        .round(2)
        .reset_index()
    )
    overall_mean = df[TARGET_COL].mean()
    grouped["vs_Overall_Mean_%"] = (
        (grouped["Avg_Salary"] - overall_mean) / overall_mean * 100
    ).round(2)
    return grouped


# ──────────────────────────────────────────────
# Career Insights
# ──────────────────────────────────────────────

def career_insights(df: pd.DataFrame,
                    education: str = None,
                    job_title: str = None,
                    location: str = None,
                    experience: int = None,
                    exp_tolerance: int = 5) -> dict:
    """Filter dataset and return descriptive salary stats."""
    filtered = df.copy()
    if education:
        filtered = filtered[filtered["Education"] == education]
    if job_title:
        filtered = filtered[filtered["Job_Title"] == job_title]
    if location:
        filtered = filtered[filtered["Location"] == location]

    if len(filtered) == 0:
        return {"no_data": True}

    result = {
        "total_records": len(filtered),
        "avg_salary":    round(float(filtered[TARGET_COL].mean()), 2),
        "median_salary": round(float(filtered[TARGET_COL].median()), 2),
        "min_salary":    round(float(filtered[TARGET_COL].min()), 2),
        "max_salary":    round(float(filtered[TARGET_COL].max()), 2),
    }

    if experience is not None:
        exp_filtered = filtered[
            (filtered["Experience"] >= max(0, experience - exp_tolerance)) &
            (filtered["Experience"] <= experience + exp_tolerance)
        ]
        result["similar_exp_count"] = len(exp_filtered)
        result["similar_exp_avg_salary"] = (
            round(float(exp_filtered[TARGET_COL].mean()), 2) if len(exp_filtered) > 0 else None
        )

    if job_title:
        result["locations_with_data"] = (
            df[df["Job_Title"] == job_title]["Location"].value_counts().to_dict()
        )

    if len(filtered) >= 5:
        result["exp_salary_correlation"] = round(
            float(filtered["Experience"].corr(filtered[TARGET_COL])), 3
        )
    else:
        result["exp_salary_correlation"] = None

    return result


# ──────────────────────────────────────────────
# Anomaly / Review Flag Detection
# ──────────────────────────────────────────────

def compute_residuals(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Run model predictions on all rows of df and compute residuals.
    Returns df with added columns: Predicted_Salary, Residual, Abs_Residual.
    Does NOT modify the original dataframe.
    """
    df_out = df.copy()
    predictions = model.predict(df[FEATURE_COLS])
    df_out["Predicted_Salary"] = predictions.round(2)
    df_out["Residual"]         = (df_out[TARGET_COL] - df_out["Predicted_Salary"]).round(2)
    df_out["Abs_Residual"]     = df_out["Residual"].abs().round(2)
    return df_out


def flag_anomalies(df_with_residuals: pd.DataFrame,
                   threshold_multiplier: float = 2.0) -> pd.DataFrame:
    """
    Flag rows where |Residual| > threshold_multiplier * std(Residual).
    Returns only the flagged rows with a Review_Flag column.
    A flag means the observed salary differs substantially from the model
    estimate — it is NOT proof of error or unfair compensation.
    """
    std_resid  = df_with_residuals["Abs_Residual"].std()
    mean_resid = df_with_residuals["Abs_Residual"].mean()
    threshold  = mean_resid + threshold_multiplier * std_resid

    flagged = df_with_residuals[df_with_residuals["Abs_Residual"] > threshold].copy()
    flagged["Review_Flag"] = "⚑ Large Deviation"
    return flagged.sort_values("Abs_Residual", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────
# Feature Importance
# ──────────────────────────────────────────────

def get_feature_importance(pipeline) -> pd.DataFrame | None:
    """
    Extract feature importance from a sklearn pipeline.
    Returns a DataFrame with Feature and Importance columns,
    or None if the model does not support it.
    """
    try:
        model_step  = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        # Get feature names after one-hot encoding
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                feature_names.extend(cols)
            elif name == "cat":
                ohe = trans.named_steps["encoder"]
                feature_names.extend(ohe.get_feature_names_out(cols).tolist())

        # Check for native importance
        if hasattr(model_step, "feature_importances_"):
            importances = model_step.feature_importances_
        elif hasattr(model_step, "coef_"):
            # coef_ is 1-D for single-output regressors; squeeze just in case
            importances = np.abs(np.asarray(model_step.coef_).ravel())
        else:
            return None

        imp_df = pd.DataFrame({
            "Feature":    feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

        # Aggregate by original feature name (collapse one-hot columns)
        def original_feature(name: str) -> str:
            for col in FEATURE_COLS:
                if name.startswith(col):
                    return col
            return name

        imp_df["Original_Feature"] = imp_df["Feature"].apply(original_feature)
        agg = (
            imp_df.groupby("Original_Feature")["Importance"]
            .sum()
            .reset_index()
            .rename(columns={"Original_Feature": "Feature"})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
        # Normalise to sum=1
        total = agg["Importance"].sum()
        if total > 0:
            agg["Importance"] = (agg["Importance"] / total).round(4)
        return agg

    except Exception:
        return None


# ──────────────────────────────────────────────
# Prediction Interval (simple ± residual std)
# ──────────────────────────────────────────────

def prediction_interval(df: pd.DataFrame, model,
                         confidence: float = 0.90) -> tuple:
    """
    Compute a simple empirical prediction interval by looking at the
    std deviation of residuals on the full training set.
    Returns (lower_offset, upper_offset) in salary units.
    This is an approximation — not a formal statistical interval.
    """
    try:
        preds  = model.predict(df[FEATURE_COLS])
        actual = df[TARGET_COL].values
        resids = actual - preds
        # Use symmetric percentile around 0
        margin = np.percentile(np.abs(resids), confidence * 100)
        return float(margin), float(margin)
    except Exception:
        return None, None


# ──────────────────────────────────────────────
# Report helpers
# ──────────────────────────────────────────────

def build_prediction_report(profile: dict, prediction: float,
                             current_salary: float = None,
                             model_name: str = "Unknown",
                             model_metrics: dict = None,
                             margin: float = None) -> pd.DataFrame:
    """
    Build a simple CSV-friendly report as a DataFrame.
    """
    rows = [
        ("Profile: Education",    profile.get("Education", "")),
        ("Profile: Experience",   profile.get("Experience", "")),
        ("Profile: Location",     profile.get("Location", "")),
        ("Profile: Job Title",    profile.get("Job_Title", "")),
        ("Profile: Age",          profile.get("Age", "")),
        ("Profile: Gender",       profile.get("Gender", "")),
        ("Predicted Salary",      f"₹{prediction:,.2f}"),
        ("Model Used",            model_name),
    ]
    if margin is not None:
        rows.append(("Est. Range Lower", f"₹{max(0, prediction - margin):,.2f}"))
        rows.append(("Est. Range Upper", f"₹{prediction + margin:,.2f}"))
    if current_salary is not None and current_salary > 0:
        diff     = current_salary - prediction
        pct_diff = (diff / prediction) * 100
        rows.append(("Current Salary",    f"₹{current_salary:,.2f}"))
        rows.append(("Difference",        f"₹{diff:+,.2f}"))
        rows.append(("Difference %",      f"{pct_diff:+.1f}%"))
    if model_metrics:
        for k, v in model_metrics.items():
            rows.append((f"Model {k}", str(v)))
    rows.append(("Limitation", (
        "This prediction is a model-based estimate. Actual salary depends on many factors "
        "not present in this dataset: company, industry, skills, performance, and market conditions."
    )))
    return pd.DataFrame(rows, columns=["Field", "Value"])


# ──────────────────────────────────────────────
# Percentile Ranking
# ──────────────────────────────────────────────

def salary_percentile_rank(df: pd.DataFrame, salary: float,
                            education: str = None,
                            job_title: str = None,
                            location: str = None) -> dict:
    """
    Return the percentile rank of a given salary within a filtered subset.
    Also returns p25, p50, p75, p90 benchmarks.
    """
    subset = df.copy()
    if education:
        subset = subset[subset["Education"] == education]
    if job_title:
        subset = subset[subset["Job_Title"] == job_title]
    if location:
        subset = subset[subset["Location"] == location]

    if len(subset) < 3:
        return {"no_data": True}

    s = subset[TARGET_COL]
    rank_pct = float((s < salary).sum() / len(s) * 100)
    return {
        "percentile_rank": round(rank_pct, 1),
        "sample_size":     len(subset),
        "p25":  round(float(s.quantile(0.25)), 2),
        "p50":  round(float(s.quantile(0.50)), 2),
        "p75":  round(float(s.quantile(0.75)), 2),
        "p90":  round(float(s.quantile(0.90)), 2),
        "mean": round(float(s.mean()), 2),
        "min":  round(float(s.min()), 2),
        "max":  round(float(s.max()), 2),
    }


# ──────────────────────────────────────────────
# Salary Growth Projection
# ──────────────────────────────────────────────

def salary_growth_projection(df: pd.DataFrame,
                              education: str,
                              job_title: str,
                              location: str,
                              current_exp: int,
                              years_ahead: int = 10,
                              exp_step: int = 1) -> pd.DataFrame:
    """
    Project salary growth by computing dataset averages at each experience level.
    Returns a DataFrame with columns: Experience, Avg_Salary, Count.
    """
    rows = []
    for exp in range(current_exp, current_exp + years_ahead + 1, exp_step):
        subset = df[(df["Education"] == education) &
                    (df["Job_Title"] == job_title) &
                    (df["Location"] == location)]
        # tolerance window ±2 yrs around each projected experience level
        window = subset[(subset["Experience"] >= max(0, exp - 2)) &
                        (subset["Experience"] <= exp + 2)]
        if len(window) >= 2:
            avg = float(window[TARGET_COL].mean())
            cnt = len(window)
        else:
            # fallback: use all matching records with linear extrapolation
            if len(subset) >= 3:
                corr_slope = np.polyfit(subset["Experience"], subset[TARGET_COL], 1)
                avg = float(np.polyval(corr_slope, exp))
                cnt = 0
            else:
                avg = None
                cnt = 0
        # Guard: projected salary must be positive; clip negatives to None
        safe_avg = round(avg, 2) if (avg is not None and avg > 0) else None
        rows.append({"Year": exp - current_exp, "Experience": exp,
                     "Avg_Salary": safe_avg, "Count": cnt})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Top Earner Profiles
# ──────────────────────────────────────────────

def top_earner_profiles(df: pd.DataFrame, top_n: int = 10,
                         percentile: float = 0.90) -> pd.DataFrame:
    """
    Return profiles of top earners (above the given percentile).
    Includes a salary_vs_mean column showing % above the overall mean.
    """
    threshold = df[TARGET_COL].quantile(percentile)
    top = df[df[TARGET_COL] >= threshold].copy()
    overall_mean = df[TARGET_COL].mean()
    top["Salary_vs_Mean_%"] = ((top[TARGET_COL] - overall_mean) / overall_mean * 100).round(1)
    return top.sort_values(TARGET_COL, ascending=False).head(top_n).reset_index(drop=True)


# ──────────────────────────────────────────────
# Role × Location Pivot (Heatmap Data)
# ──────────────────────────────────────────────

def role_location_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a pivot table: rows = Job_Title, columns = Location,
    values = average Salary. NaN where no data exists.
    """
    pivot = df.pivot_table(
        values=TARGET_COL, index="Job_Title", columns="Location",
        aggfunc="mean"
    ).round(0)
    return pivot


# ──────────────────────────────────────────────
# Education × Experience Salary Matrix
# ──────────────────────────────────────────────

def education_experience_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a pivot table: rows = Education, columns = Exp_Bin,
    values = average Salary.
    """
    df2 = df.copy()
    df2["Exp_Bin"] = pd.cut(
        df2["Experience"],
        bins=[0, 5, 10, 15, 20, 25, 30, 100],
        labels=["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "30+"],
        right=True,
    )
    pivot = df2.pivot_table(
        values=TARGET_COL, index="Education", columns="Exp_Bin",
        aggfunc="mean", observed=True
    ).round(0)
    # Sort rows by EDUCATION_ORDER
    ordered = [e for e in EDUCATION_ORDER if e in pivot.index]
    return pivot.loc[ordered]


# ──────────────────────────────────────────────
# Salary Buckets / Distribution Bins
# ──────────────────────────────────────────────

def salary_buckets(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Divide the salary range into n_bins equal-width buckets.
    Returns a DataFrame with: Bucket (label), Count, Pct.
    """
    salary = df[TARGET_COL]
    min_s, max_s = salary.min(), salary.max()
    edges = np.linspace(min_s, max_s, n_bins + 1)
    labels = [f"₹{edges[i]/1000:.0f}K–{edges[i+1]/1000:.0f}K" for i in range(n_bins)]
    bucketed = pd.cut(salary, bins=edges, labels=labels, include_lowest=True)
    counts = bucketed.value_counts().sort_index()
    result = pd.DataFrame({
        "Bucket": counts.index,
        "Count":  counts.values,
        "Pct":    (counts.values / len(df) * 100).round(1),
    })
    return result


# ──────────────────────────────────────────────
# What-If: Sensitivity Analysis
# ──────────────────────────────────────────────

def what_if_experience_sweep(df: pd.DataFrame, model,
                              base_profile: dict,
                              exp_range: tuple = (0, 35)) -> pd.DataFrame:
    """
    Sweep experience from exp_range[0] to exp_range[1] holding everything else
    constant, and return predicted salaries.
    """
    rows = []
    for exp in range(exp_range[0], exp_range[1] + 1):
        profile = dict(base_profile)
        profile["Experience"] = exp
        inp = pd.DataFrame([profile])
        try:
            pred = float(model.predict(inp)[0])
        except Exception:
            pred = None
        rows.append({"Experience": exp, "Predicted_Salary": round(pred, 2) if pred is not None else None})
    return pd.DataFrame(rows)


def what_if_age_sweep(df: pd.DataFrame, model,
                      base_profile: dict,
                      age_range: tuple = (18, 65)) -> pd.DataFrame:
    """
    Sweep age across a range holding everything else constant,
    and return predicted salaries.
    """
    rows = []
    for age in range(age_range[0], age_range[1] + 1, 2):
        profile = dict(base_profile)
        profile["Age"] = age
        inp = pd.DataFrame([profile])
        try:
            pred = float(model.predict(inp)[0])
        except Exception:
            pred = None
        rows.append({"Age": age, "Predicted_Salary": round(pred, 2) if pred is not None else None})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Gender Pay Gap
# ──────────────────────────────────────────────

def gender_pay_gap_by_role(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with columns: Job_Title, Male_Avg, Female_Avg, Gap, Gap_Pct.
    Rows where either gender has no data are dropped.
    """
    rows = []
    for job in sorted(df["Job_Title"].dropna().unique()):
        sub = df[df["Job_Title"] == job]
        male   = sub[sub["Gender"] == "Male"][TARGET_COL]
        female = sub[sub["Gender"] == "Female"][TARGET_COL]
        if len(male) < 2 or len(female) < 2:
            continue
        m_avg = float(male.mean())
        f_avg = float(female.mean())
        gap   = m_avg - f_avg
        gap_pct = (gap / m_avg * 100) if m_avg else 0.0
        rows.append({
            "Job_Title":  job,
            "Male_Avg":   round(m_avg, 0),
            "Female_Avg": round(f_avg, 0),
            "Gap":        round(gap, 0),
            "Gap_Pct":    round(gap_pct, 1),
        })
    return pd.DataFrame(rows).sort_values("Gap_Pct", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────
# Salary Trend by Age Band
# ──────────────────────────────────────────────

def salary_by_age_band(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return average salary per age band (20s, 30s, 40s, 50s, 60+).
    """
    df2 = df.copy()
    df2["Age_Band"] = pd.cut(
        df2["Age"],
        bins=[0, 29, 39, 49, 59, 120],
        labels=["20s", "30s", "40s", "50s", "60+"],
        right=True,
    )
    result = (
        df2.groupby("Age_Band", observed=True)[TARGET_COL]
        .agg(Count="count", Avg_Salary="mean", Median_Salary="median")
        .round(2)
        .reset_index()
    )
    return result


# ──────────────────────────────────────────────
# CSV Upload Validation
# ──────────────────────────────────────────────

def validate_uploaded_csv(uploaded_df: pd.DataFrame) -> dict:
    """
    Validate an uploaded CSV for compatibility with the salary model.
    Returns a report dict with 'valid', 'issues', and 'warnings' keys.
    """
    issues    = []
    warn_msgs = []   # renamed: avoids shadowing stdlib 'warnings' module

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in uploaded_df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    if TARGET_COL in uploaded_df.columns:
        bad_salary = int(
            (pd.to_numeric(uploaded_df[TARGET_COL], errors="coerce")
               .pipe(lambda s: s.isna() | (s <= 0))).sum()
        )
        if bad_salary:
            warn_msgs.append(f"{bad_salary} rows with invalid salary (will be dropped)")

    if "Age" in uploaded_df.columns:
        age_num = pd.to_numeric(uploaded_df["Age"], errors="coerce")
        bad_age = int((age_num.isna() | (age_num < 16) | (age_num > 80)).sum())
        if bad_age:
            warn_msgs.append(f"{bad_age} rows with unusual age (<16 or >80)")

    if "Experience" in uploaded_df.columns:
        exp_num = pd.to_numeric(uploaded_df["Experience"], errors="coerce")
        bad_exp = int((exp_num.isna() | (exp_num < 0) | (exp_num > 50)).sum())
        if bad_exp:
            warn_msgs.append(f"{bad_exp} rows with unusual experience (<0 or >50)")

    return {
        "valid":    len(issues) == 0,
        "rows":     len(uploaded_df),
        "columns":  uploaded_df.columns.tolist(),
        "issues":   issues,
        "warnings": warn_msgs,
    }


# ──────────────────────────────────────────────
# Virtual Assistant — rule-based NLP response engine
# ──────────────────────────────────────────────

import re as _re

def _fmt(v: float) -> str:
    """Format a float as an Indian rupee string."""
    return f"₹{v:,.0f}"


def va_response(user_input: str, df: pd.DataFrame, model=None) -> str:
    """
    Rule-based virtual assistant that answers natural-language questions about
    the salary dataset and the ML model.

    Parameters
    ----------
    user_input : str   Raw question from the user.
    df         : pd.DataFrame  Cleaned salary dataset.
    model      : sklearn pipeline or None.

    Returns
    -------
    str — Markdown-formatted answer.
    """
    q = user_input.strip().lower()

    # ── greeting ────────────────────────────────────────────────────────────
    if _re.search(r"\b(hi|hello|hey|howdy|good (morning|afternoon|evening))\b", q):
        return (
            "👋 **Hello! I'm SalaryAI Assistant.**\n\n"
            "I can answer questions about salaries, the dataset, and the prediction model. "
            "Try asking me things like:\n"
            "- *What is the average salary?*\n"
            "- *What is the highest paying job?*\n"
            "- *How many records are in the dataset?*\n"
            "- *What does the model use to predict salary?*\n"
            "- *What is the average salary for a PhD?*\n"
            "- *Which location pays the most?*"
        )

    # ── goodbye ──────────────────────────────────────────────────────────────
    if _re.search(r"\b(bye|goodbye|exit|quit|thanks|thank you)\b", q):
        return "👋 Goodbye! Feel free to ask me anything about salaries anytime."

    # ── help ─────────────────────────────────────────────────────────────────
    if _re.search(r"\b(help|what can you do|capabilities|commands)\b", q):
        return (
            "🤖 **I can help you with:**\n\n"
            "📊 **Dataset stats** — record count, salary range, averages\n"
            "💰 **Salary by group** — by education, job title, location, gender\n"
            "🏆 **Top paying** — highest salary jobs, locations, education levels\n"
            "📈 **Experience** — how experience affects salary\n"
            "🤖 **Model info** — which model is best, what features it uses\n"
            "🔍 **Specific queries** — e.g. *average salary for a Bachelor in Urban*\n\n"
            "Just type your question in plain English!"
        )

    # ── dataset size / record count ──────────────────────────────────────────
    if _re.search(r"\b(how many|number of|count of|total).*(record|row|person|entry|data|sample)\b", q):
        return (
            f"📂 The dataset contains **{len(df):,} records** with "
            f"**{len(df.columns)} columns**: "
            f"{', '.join(df.columns.tolist())}."
        )

    # ── overall average salary ───────────────────────────────────────────────
    if _re.search(r"\b(average|mean|avg)\b.*\bsalary\b", q) and not _re.search(
        r"\b(education|degree|bachelor|master|phd|high school|job|title|location|urban|rural|suburban|gender|male|female)\b", q
    ):
        avg = df[TARGET_COL].mean()
        med = df[TARGET_COL].median()
        return (
            f"💰 The **overall average salary** is **{_fmt(avg)}** "
            f"and the **median salary** is **{_fmt(med)}**."
        )

    # ── salary range / min / max ──────────────────────────────────────────────
    if _re.search(r"\b(min|minimum|lowest|smallest)\b.*\bsalary\b|\bsalary\b.*\b(min|minimum|lowest)\b", q):
        mn = df[TARGET_COL].min()
        row = df.loc[df[TARGET_COL].idxmin()]
        return (
            f"📉 The **minimum salary** in the dataset is **{_fmt(mn)}** "
            f"({row.get('Job_Title','?')}, {row.get('Education','?')}, {row.get('Location','?')})."
        )

    if _re.search(r"\b(max|maximum|highest|biggest|top)\b.*\bsalary\b|\bsalary\b.*\b(max|maximum|highest)\b", q):
        mx = df[TARGET_COL].max()
        row = df.loc[df[TARGET_COL].idxmax()]
        return (
            f"📈 The **maximum salary** in the dataset is **{_fmt(mx)}** "
            f"({row.get('Job_Title','?')}, {row.get('Education','?')}, {row.get('Location','?')})."
        )

    if _re.search(r"\b(salary range|range of salary|salary spread)\b", q):
        mn = df[TARGET_COL].min()
        mx = df[TARGET_COL].max()
        return f"📊 Salaries range from **{_fmt(mn)}** to **{_fmt(mx)}**."

    # ── salary by education ───────────────────────────────────────────────────
    edu_match = _re.search(
        r"\b(high school|bachelor|master|phd|doctorate|postgrad)\b", q
    )
    if edu_match or _re.search(r"\beducation\b", q):
        grp = df.groupby("Education")[TARGET_COL].mean().round(0)
        if edu_match:
            raw = edu_match.group(1)
            edu_map = {"doctorate": "PhD", "postgrad": "Master"}
            edu = edu_map.get(raw, raw.title())
            # Normalise "High School" capitalisation
            if raw == "high school":
                edu = "High School"
            if edu in grp.index:
                return (
                    f"🎓 The average salary for a **{edu}** graduate is "
                    f"**{_fmt(grp[edu])}**."
                )
            return f"⚠️ No data found for education level **{edu}**."
        # Generic education breakdown
        lines = "\n".join(
            f"- **{e}**: {_fmt(v)}" for e, v in grp.sort_values(ascending=False).items()
        )
        return f"🎓 **Average salary by education level:**\n\n{lines}"

    # ── salary by job title ───────────────────────────────────────────────────
    job_titles = df["Job_Title"].dropna().unique().tolist()
    matched_job = next(
        (j for j in job_titles if j.lower() in q), None
    )
    if matched_job:
        sub = df[df["Job_Title"] == matched_job][TARGET_COL]
        return (
            f"💼 **{matched_job}** salary stats:\n\n"
            f"- Average: **{_fmt(sub.mean())}**\n"
            f"- Median: **{_fmt(sub.median())}**\n"
            f"- Range: {_fmt(sub.min())} — {_fmt(sub.max())}\n"
            f"- Records: **{len(sub):,}**"
        )

    if _re.search(r"\b(job title|role|position|occupation)\b", q) or \
       _re.search(r"\b(highest|best|top).*(pay|paying|paid|job|role)\b", q) or \
       _re.search(r"\b(job|role).*(pay|salary|earn)\b", q):
        grp = df.groupby("Job_Title")[TARGET_COL].mean().round(0).sort_values(ascending=False)
        lines = "\n".join(f"- **{j}**: {_fmt(v)}" for j, v in grp.items())
        return f"💼 **Average salary by job title:**\n\n{lines}"

    # ── salary by location ────────────────────────────────────────────────────
    locations = df["Location"].dropna().unique().tolist()
    matched_loc = next((l for l in locations if l.lower() in q), None)
    if matched_loc:
        sub = df[df["Location"] == matched_loc][TARGET_COL]
        return (
            f"📍 **{matched_loc}** salary stats:\n\n"
            f"- Average: **{_fmt(sub.mean())}**\n"
            f"- Median: **{_fmt(sub.median())}**\n"
            f"- Range: {_fmt(sub.min())} — {_fmt(sub.max())}\n"
            f"- Records: **{len(sub):,}**"
        )

    if _re.search(r"\b(location|city|region|area|urban|rural|suburban)\b", q) or \
       _re.search(r"\b(highest|best|top).*(location|city|place|area)\b", q):
        grp = df.groupby("Location")[TARGET_COL].mean().round(0).sort_values(ascending=False)
        lines = "\n".join(f"- **{l}**: {_fmt(v)}" for l, v in grp.items())
        best = grp.index[0]
        return (
            f"📍 **Average salary by location:**\n\n{lines}\n\n"
            f"🏆 **{best}** pays the highest on average."
        )

    # ── salary by gender ──────────────────────────────────────────────────────
    if _re.search(r"\b(gender|male|female|men|women|pay gap)\b", q):
        grp = df.groupby("Gender")[TARGET_COL].mean().round(0)
        lines = "\n".join(f"- **{g}**: {_fmt(v)}" for g, v in grp.items())
        if "Male" in grp.index and "Female" in grp.index:
            gap = abs(grp.get("Male", 0) - grp.get("Female", 0))
            return (
                f"👥 **Average salary by gender:**\n\n{lines}\n\n"
                f"📊 The raw pay gap (unadjusted for experience, role, etc.) is "
                f"**{_fmt(gap)}**.\n"
                f"⚠️ This does not control for experience, role, or location."
            )
        return f"👥 **Average salary by gender:**\n\n{lines}"

    # ── experience ────────────────────────────────────────────────────────────
    if _re.search(r"\b(experience|years|yrs|seniority)\b", q):
        corr = round(float(df["Experience"].corr(df[TARGET_COL])), 3)
        avg_exp = round(df["Experience"].mean(), 1)
        direction = "positively" if corr > 0 else "negatively"
        strength = "strongly" if abs(corr) > 0.5 else "weakly"
        return (
            f"⏳ **Experience & salary insights:**\n\n"
            f"- Average experience in the dataset: **{avg_exp} years**\n"
            f"- Correlation between experience and salary: **{corr}**\n"
            f"- They are **{strength} {direction} correlated**.\n"
            f"- Experience range: {int(df['Experience'].min())}–{int(df['Experience'].max())} years."
        )

    # ── model info ────────────────────────────────────────────────────────────
    if _re.search(r"\b(model|algorithm|ml|machine learning|predict|accuracy|r2|r²|rmse|mae)\b", q):
        if model is not None:
            try:
                model_name = type(model.named_steps["model"]).__name__
            except Exception:
                model_name = "Unknown"
            return (
                f"🤖 **Model Information:**\n\n"
                f"- Active model: **{model_name}**\n"
                f"- Features used: **{', '.join(FEATURE_COLS)}**\n"
                f"- Target variable: **Salary**\n"
                f"- The model was selected as the best performer based on R² score.\n"
                f"- Use the **🤖 Model Performance** page for full metrics (MAE, RMSE, R², CV)."
            )
        return (
            "🤖 The model has not been loaded yet.\n\n"
            "Run `python train_model.py` to train, then restart the app."
        )

    if _re.search(r"\b(feature|input|variable|column)\b", q):
        return (
            f"🔬 **Model input features:**\n\n"
            + "\n".join(f"- {f}" for f in FEATURE_COLS)
            + "\n\nThe target (output) is **Salary**."
        )

    # ── dataset columns ───────────────────────────────────────────────────────
    if _re.search(r"\b(column|field|attribute|feature list)\b", q):
        return (
            f"📋 **Dataset columns ({len(df.columns)} total):**\n\n"
            + "\n".join(f"- `{c}`" for c in df.columns.tolist())
        )

    # ── unique values ─────────────────────────────────────────────────────────
    if _re.search(r"\b(unique|distinct|different).*(job|title|role)\b", q):
        jobs = sorted(df["Job_Title"].dropna().unique().tolist())
        return f"💼 **{len(jobs)} unique job titles:** {', '.join(jobs)}"

    if _re.search(r"\b(unique|distinct|different).*(location|city)\b", q):
        locs = sorted(df["Location"].dropna().unique().tolist())
        return f"📍 **{len(locs)} locations:** {', '.join(locs)}"

    if _re.search(r"\b(unique|distinct|different).*(education|degree)\b", q):
        edus = [e for e in EDUCATION_ORDER if e in df["Education"].unique()]
        return f"🎓 **{len(edus)} education levels:** {', '.join(edus)}"

    # ── percentile / top earner ───────────────────────────────────────────────
    if _re.search(r"\b(top 10|top ten|p90|90th percentile|top earner)\b", q):
        threshold = df[TARGET_COL].quantile(0.90)
        count = int((df[TARGET_COL] >= threshold).sum())
        return (
            f"🏆 The **90th percentile** salary is **{_fmt(threshold)}**.\n\n"
            f"**{count:,} records** ({count/len(df)*100:.1f}%) earn at or above this level."
        )

    # ── median ────────────────────────────────────────────────────────────────
    if _re.search(r"\b(median|50th percentile|middle)\b.*\bsalary\b", q):
        med = df[TARGET_COL].median()
        return f"📊 The **median salary** is **{_fmt(med)}**."

    # ── standard deviation / spread ───────────────────────────────────────────
    if _re.search(r"\b(std|standard deviation|spread|variance|deviation)\b", q):
        std = df[TARGET_COL].std()
        return (
            f"📐 The **standard deviation** of salaries is **{_fmt(std)}**.\n\n"
            f"This means most salaries fall roughly within "
            f"**{_fmt(df[TARGET_COL].mean() - std)}** – **{_fmt(df[TARGET_COL].mean() + std)}**."
        )

    # ── what is this app ──────────────────────────────────────────────────────
    if _re.search(r"\b(what is|what does|about|purpose|goal|this app|this tool)\b", q):
        return (
            "💼 **SalaryAI** is an intelligent salary prediction and analytics platform.\n\n"
            "It lets you:\n"
            "- 💰 Predict your expected salary from your profile\n"
            "- ⚖️ Benchmark your current salary vs the model estimate\n"
            "- 📊 Explore salary patterns across roles, locations, and education\n"
            "- 📈 Project your salary growth over time\n"
            "- 🎯 Find your salary percentile rank\n"
            "- 🗺 View role × location salary heatmaps\n"
            "- ⚡ Run what-if sensitivity analyses\n"
            "- 📂 Upload your own CSV for batch predictions\n\n"
            "Use the sidebar to navigate between modules."
        )

    # ── fallback ──────────────────────────────────────────────────────────────
    return (
        "🤔 I'm not sure I understood that. Here are some things you can ask me:\n\n"
        "- *What is the average salary?*\n"
        "- *What is the highest paying job?*\n"
        "- *Average salary for a PhD?*\n"
        "- *Which location pays the most?*\n"
        "- *How many records are in the dataset?*\n"
        "- *What features does the model use?*\n"
        "- *What is the gender pay gap?*\n\n"
        "Try rephrasing your question or type **help** to see all topics."
    )
