"""
app.py — SalaryAI
Intelligent Salary Prediction, Benchmarking & Career Analytics
Complete upgraded Streamlit application.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib

from data_analysis import (
    load_data, clean_data, dataset_summary, salary_statistics,
    get_unique_values, get_education_levels, get_all_categories,
    filter_data, salary_by_group, career_insights, data_quality_report,
    compute_residuals, flag_anomalies, get_feature_importance,
    prediction_interval, build_prediction_report, missing_value_report,
    salary_percentile_rank, salary_growth_projection, top_earner_profiles,
    role_location_heatmap, education_experience_matrix, salary_buckets,
    what_if_experience_sweep, what_if_age_sweep,
    gender_pay_gap_by_role, salary_by_age_band, validate_uploaded_csv,
    va_response,
    DATA_PATH, FEATURE_COLS, TARGET_COL,
)

# ──────────────────────────────────────────────
# Page Config  (MUST be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SalaryAI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "salary_prediction_model.pkl")
RESULTS_PATH = os.path.join(BASE_DIR, "models", "model_results.csv")

# ══════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
:root {
    --primary:#2563eb; --secondary:#7c3aed; --accent:#06b6d4;
    --success:#10b981; --warning:#f59e0b; --danger:#ef4444;
    --bg:#0f172a; --surface:#1e293b; --border:#475569;
    --text:#f1f5f9; --muted:#94a3b8;
}
html,body,[class*="css"]{ font-family:'Inter',sans-serif !important; color:var(--text); }
.stApp{
    background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 50%,#0f172a 100%);
    background-size:400% 400%; animation:gradientShift 14s ease infinite;
}
@keyframes gradientShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes slideDown{from{opacity:0;transform:translateY(-18px)}to{opacity:1;transform:translateY(0)}}
@keyframes popIn{from{opacity:0;transform:scale(0.93)}to{opacity:1;transform:scale(1)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(37,99,235,0.5)}50%{box-shadow:0 0 0 14px rgba(37,99,235,0)}}

/* Sidebar */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0f172a 0%,#1e1b4b 100%) !important;
    border-right:1px solid rgba(99,102,241,0.3) !important;
}
[data-testid="stSidebar"] *{color:#e2e8f0 !important;}
[data-testid="stSidebar"] .stRadio>label{
    background:rgba(99,102,241,0.07); border-radius:10px;
    padding:0.5rem 0.9rem; margin:3px 0;
    border:1px solid transparent; transition:all 0.22s ease; display:block;
}
[data-testid="stSidebar"] .stRadio>label:hover{
    background:rgba(99,102,241,0.2); border-color:rgba(99,102,241,0.5);
    transform:translateX(5px);
}
.sidebar-brand{text-align:center;padding:1rem 0 0.4rem;}
.sidebar-brand-title{
    font-size:1.5rem;font-weight:800;
    background:linear-gradient(90deg,#a5f3fc,#818cf8);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.sidebar-brand-sub{font-size:0.7rem;color:#64748b !important;-webkit-text-fill-color:#64748b;letter-spacing:.06em;}

/* Hero */
.hero-banner{
    background:linear-gradient(135deg,#1e1b4b 0%,#312e81 40%,#1e3a8a 100%);
    border-radius:20px; padding:2.2rem 2.5rem; margin-bottom:1.8rem;
    border:1px solid rgba(99,102,241,0.4); position:relative; overflow:hidden;
    animation:slideDown 0.55s ease-out;
}
.hero-banner::before{
    content:''; position:absolute; top:-50%; left:-50%; width:200%; height:200%;
    background:radial-gradient(circle,rgba(99,102,241,0.1) 0%,transparent 60%);
    animation:rotate 22s linear infinite;
}
@keyframes rotate{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
.hero-title{
    font-size:2.8rem; font-weight:800;
    background:linear-gradient(90deg,#a5f3fc,#818cf8,#c4b5fd);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    margin:0 0 0.35rem; position:relative; z-index:1;
}
.hero-sub{font-size:1.05rem;color:#a5b4fc;position:relative;z-index:1;margin:0;}

/* Stat Card */
.stat-card{
    background:linear-gradient(135deg,rgba(30,41,59,.95),rgba(51,65,85,.8));
    border:1px solid rgba(99,102,241,.3); border-radius:16px;
    padding:1.1rem 1.3rem; text-align:center;
    transition:all 0.3s ease; animation:popIn 0.5s ease-out both; position:relative; overflow:hidden;
}
.stat-card::after{
    content:''; position:absolute; top:0;left:0;right:0; height:3px;
    background:linear-gradient(90deg,var(--primary),var(--secondary),var(--accent));
    background-size:200% 100%; animation:shimmer 2.5s linear infinite;
}
.stat-card:hover{transform:translateY(-5px);border-color:rgba(99,102,241,.7);box-shadow:0 12px 32px rgba(99,102,241,.25);}
.stat-value{font-size:1.5rem;font-weight:700;color:#a5f3fc;}
.stat-label{font-size:0.74rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-top:.18rem;}

/* Module Card */
.module-card{
    background:linear-gradient(135deg,rgba(30,41,59,.9),rgba(30,27,75,.7));
    border:1px solid rgba(99,102,241,.22); border-radius:14px;
    padding:1.1rem 0.9rem; text-align:center;
    transition:all 0.3s cubic-bezier(.34,1.56,.64,1);
    animation:popIn 0.5s ease-out both; height:100%;
}
.module-card:hover{transform:translateY(-6px) scale(1.02);border-color:rgba(99,102,241,.6);box-shadow:0 16px 40px rgba(99,102,241,.2);}
.module-icon{font-size:1.9rem;margin-bottom:.4rem;display:block;}
.module-name{font-size:.9rem;font-weight:600;color:#c7d2fe;margin-bottom:.25rem;}
.module-desc{font-size:.75rem;color:#64748b;line-height:1.4;}

/* Section Head */
.section-head{
    font-size:1.18rem;font-weight:700;color:#a5b4fc;
    padding-bottom:.35rem;margin-bottom:.9rem;
    border-bottom:2px solid;border-image:linear-gradient(90deg,#6366f1,transparent) 1;
    animation:fadeIn .4s ease-out;
}

/* Input Card */
.input-card{
    background:rgba(30,41,59,.8); border:1px solid rgba(99,102,241,.22);
    border-radius:16px; padding:1.3rem; animation:popIn .4s ease-out; transition:border-color .3s;
}
.input-card:hover{border-color:rgba(99,102,241,.5);}
.input-card h4{color:#a5b4fc;font-size:.98rem;font-weight:600;margin-bottom:.7rem;}

/* Result Banner */
.result-banner{
    background:linear-gradient(135deg,rgba(16,185,129,.12),rgba(6,182,212,.08));
    border:1px solid rgba(16,185,129,.4); border-radius:16px;
    padding:1.8rem; text-align:center;
    animation:pulse 2s ease-in-out 3, popIn .5s ease-out; margin:1rem 0;
}
.result-salary{
    font-size:2.8rem;font-weight:800;
    background:linear-gradient(90deg,#34d399,#06b6d4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.result-label{font-size:.85rem;color:#6ee7b7;text-transform:uppercase;letter-spacing:.1em;}

/* Info/Warn boxes */
.info-box{background:rgba(37,99,235,.1);border:1px solid rgba(37,99,235,.35);border-radius:12px;padding:.9rem 1.1rem;font-size:.84rem;color:#bfdbfe;line-height:1.6;animation:fadeIn .5s ease;}
.warn-box{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.35);border-radius:12px;padding:.9rem 1.1rem;font-size:.84rem;color:#fde68a;line-height:1.6;animation:fadeIn .5s ease;}
.danger-box{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.35);border-radius:12px;padding:.9rem 1.1rem;font-size:.84rem;color:#fca5a5;line-height:1.6;}
.success-box{background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.35);border-radius:12px;padding:.9rem 1.1rem;font-size:.84rem;color:#6ee7b7;line-height:1.6;}
.status-box{border-radius:12px;padding:1.1rem 1.4rem;font-size:.93rem;font-weight:500;animation:popIn .4s ease-out;display:flex;align-items:center;gap:.8rem;}

/* Chart Card */
.chart-card{background:rgba(30,41,59,.7);border:1px solid rgba(99,102,241,.2);border-radius:14px;padding:.9rem;animation:fadeIn .5s ease-out;transition:box-shadow .3s;}
.chart-card:hover{box-shadow:0 8px 24px rgba(99,102,241,.15);}
.chart-title{font-size:.85rem;font-weight:600;color:#a5b4fc;margin-bottom:.5rem;text-transform:uppercase;letter-spacing:.05em;}

/* Streamlit Metric Override */
div[data-testid="metric-container"]{
    background:linear-gradient(135deg,rgba(30,41,59,.9),rgba(51,65,85,.7)) !important;
    border:1px solid rgba(99,102,241,.3) !important; border-radius:12px !important;
    padding:.9rem !important; transition:transform .25s,box-shadow .25s; animation:popIn .45s ease-out;
}
div[data-testid="metric-container"]:hover{transform:translateY(-3px);box-shadow:0 8px 20px rgba(99,102,241,.2);}
div[data-testid="metric-container"] label{color:#94a3b8 !important;font-size:.74rem !important;text-transform:uppercase;letter-spacing:.06em;}
div[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#a5f3fc !important;font-weight:700 !important;font-size:1.4rem !important;}

/* Button */
div.stButton>button{
    background:linear-gradient(135deg,#2563eb,#7c3aed) !important; color:white !important;
    border:none !important; border-radius:12px !important; font-weight:600 !important;
    font-size:.97rem !important; padding:.65rem 2rem !important; letter-spacing:.04em;
    transition:all .25s ease !important; box-shadow:0 4px 14px rgba(37,99,235,.35) !important;
}
div.stButton>button:hover{transform:translateY(-2px) !important;box-shadow:0 8px 24px rgba(37,99,235,.5) !important;filter:brightness(1.1);}
div.stButton>button:active{transform:translateY(0) !important;}

/* Tabs */
button[data-baseweb="tab"]{background:transparent !important;color:#94a3b8 !important;border-bottom:2px solid transparent !important;font-weight:500 !important;transition:all .2s;}
button[data-baseweb="tab"][aria-selected="true"]{color:#a5b4fc !important;border-bottom-color:#6366f1 !important;background:rgba(99,102,241,.08) !important;}

/* Dataframe */
[data-testid="stDataFrame"]{border-radius:12px !important;overflow:hidden;border:1px solid rgba(99,102,241,.2) !important;}

/* Inputs */
[data-baseweb="select"]>div,[data-baseweb="input"]>div{background:rgba(30,41,59,.9) !important;border-color:rgba(99,102,241,.35) !important;border-radius:10px !important;color:#e2e8f0 !important;}
.stSlider>div>div>div{background:rgba(99,102,241,.4) !important;}
details{background:rgba(30,41,59,.6) !important;border:1px solid rgba(99,102,241,.2) !important;border-radius:12px !important;padding:.2rem !important;}

/* Scrollbar */
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:#0f172a;}
::-webkit-scrollbar-thumb{background:#6366f1;border-radius:3px;}

/* Misc */
.fancy-divider{height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,.5),transparent);margin:1.4rem 0;border:none;}
.badge{display:inline-block;padding:.18rem .65rem;border-radius:20px;font-size:.7rem;font-weight:600;letter-spacing:.05em;}
.badge-green{background:rgba(16,185,129,.2);color:#34d399;border:1px solid rgba(16,185,129,.4);}
.badge-blue{background:rgba(37,99,235,.2);color:#93c5fd;border:1px solid rgba(37,99,235,.4);}
.badge-purple{background:rgba(124,58,237,.2);color:#c4b5fd;border:1px solid rgba(124,58,237,.4);}
.badge-orange{background:rgba(245,158,11,.2);color:#fbbf24;border:1px solid rgba(245,158,11,.4);}
.main .block-container{animation:fadeIn .3s ease-out;}

/* ── Virtual Assistant chat UI ── */
.chat-container{display:flex;flex-direction:column;gap:.75rem;max-height:520px;overflow-y:auto;padding:.4rem 0;}
.chat-bubble{display:flex;gap:.7rem;align-items:flex-start;animation:fadeIn .3s ease-out;}
.chat-bubble.user{flex-direction:row-reverse;}
.chat-avatar{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.1rem;flex-shrink:0;}
.chat-avatar.bot{background:linear-gradient(135deg,#2563eb,#7c3aed);}
.chat-avatar.user{background:linear-gradient(135deg,#0f766e,#0891b2);}
.chat-msg{max-width:80%;padding:.75rem 1rem;border-radius:14px;font-size:.875rem;line-height:1.6;}
.chat-msg.bot{background:rgba(30,41,59,.85);border:1px solid rgba(99,102,241,.25);color:#e2e8f0;border-top-left-radius:4px;}
.chat-msg.user{background:linear-gradient(135deg,rgba(37,99,235,.25),rgba(124,58,237,.2));border:1px solid rgba(99,102,241,.4);color:#e2e8f0;border-top-right-radius:4px;}
.chat-time{font-size:.68rem;color:#475569;margin-top:.2rem;}
.va-quick-btn{display:inline-block;background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.3);border-radius:20px;padding:.3rem .85rem;font-size:.78rem;color:#a5b4fc;margin:.2rem;cursor:pointer;transition:all .2s;}
.va-quick-btn:hover{background:rgba(99,102,241,.25);border-color:rgba(99,102,241,.6);}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Matplotlib dark theme
# ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1e293b", "axes.facecolor": "#1e293b",
    "axes.edgecolor":   "#475569", "axes.labelcolor": "#94a3b8",
    "xtick.color":      "#94a3b8", "ytick.color":     "#94a3b8",
    "text.color":       "#e2e8f0", "grid.color":      "#334155",
    "grid.alpha": 0.45, "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.facecolor": "#1e293b", "legend.edgecolor": "#475569",
    "legend.labelcolor": "#e2e8f0", "font.family": "sans-serif", "font.size": 9,
})
CC = ["#6366f1","#06b6d4","#10b981","#f59e0b","#ef4444","#ec4899","#8b5cf6","#14b8a6"]


# ──────────────────────────────────────────────
# Cached loaders
# ──────────────────────────────────────────────
@st.cache_data
def get_data():
    try:
        return clean_data(load_data(DATA_PATH))
    except FileNotFoundError as e:
        st.error(str(e)); st.stop()

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

@st.cache_data
def get_model_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    try:
        return pd.read_csv(RESULTS_PATH)
    except Exception:
        return None


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def ri(v): return f"₹{v:,.0f}"
def ri2(v): return f"₹{v:,.2f}"

def stat_card(label, value, delay=0):
    return (f'<div class="stat-card" style="animation-delay:{delay*0.08}s">'
            f'<div class="stat-value">{value}</div>'
            f'<div class="stat-label">{label}</div></div>')

def chart_wrap(title, fig):
    st.markdown(f'<div class="chart-card"><div class="chart-title">{title}</div>',
                unsafe_allow_html=True)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

def hero(title, sub, icon="💼"):
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title">{icon} {title}</div>
        <p class="hero-sub">{sub}</p>
    </div>""", unsafe_allow_html=True)

def divider():
    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

def get_best_model_info():
    mr = get_model_results()
    if mr is None or len(mr) == 0:
        return "Not trained", {}
    idx  = mr["R2 Score"].idxmax()
    row  = mr.loc[idx]
    return row["Model"], row.to_dict()


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
PAGES = {
    "🏠 Dashboard":           "Dashboard",
    "💰 Salary Prediction":   "Salary Prediction",
    "⚖️ Salary Benchmarking": "Salary Benchmarking",
    "🔮 Career Simulator":    "Career Simulator",
    "📊 Salary Analytics":    "Salary Analytics",
    "📈 Growth Projector":    "Growth Projector",
    "🎯 Percentile Rank":     "Percentile Rank",
    "🗺 Role Heatmap":        "Role Heatmap",
    "⚡ What-If Analyser":    "What-If Analyser",
    "🏆 Top Earners":         "Top Earners",
    "📂 Upload & Predict":    "Upload & Predict",
    "🔍 Salary Review":       "Salary Review",
    "🤖 Model Performance":   "Model Performance",
    "🗂 Dataset Explorer":    "Dataset Explorer",
    "🚀 Career Insights":     "Career Insights",
    "💬 AI Assistant":        "AI Assistant",
    "ℹ️ About":               "About",
}

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">💼 SalaryAI</div>
        <div class="sidebar-brand-sub">SALARY INTELLIGENCE PLATFORM</div>
    </div>""", unsafe_allow_html=True)
    divider()
    page_label = st.radio("Nav", list(PAGES.keys()), label_visibility="collapsed")
    divider()
    st.markdown('<small style="color:#475569;display:block;text-align:center;">Python · Scikit-learn · Streamlit</small>',
                unsafe_allow_html=True)

page = PAGES[page_label]

# Load shared data
df         = get_data()
categories = get_all_categories(df)
model      = get_model()
mr         = get_model_results()
best_name, best_row = get_best_model_info()


# ══════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════
if page == "Dashboard":
    hero("SalaryAI",
         "Intelligent Salary Prediction, Benchmarking &amp; Career Analytics &nbsp;"
         '<span class="badge badge-blue">ML Powered</span>&nbsp;'
         '<span class="badge badge-green">Live Data</span>')

    stats   = salary_statistics(df)
    summary = dataset_summary(df)

    # KPI row 1
    cols = st.columns(6)
    kpis = [
        ("Total Records",   f"{summary['total_records']:,}"),
        ("Avg Salary",      ri(stats["mean"])),
        ("Median Salary",   ri(stats["median"])),
        ("Min Salary",      ri(stats["min"])),
        ("Max Salary",      ri(stats["max"])),
        ("Avg Experience",  f"{df['Experience'].mean():.1f} yrs"),
    ]
    for i,(lbl,val) in enumerate(kpis):
        with cols[i]: st.markdown(stat_card(lbl,val,i), unsafe_allow_html=True)

    st.markdown("")
    # KPI row 2
    cols2 = st.columns(5)
    kpis2 = [
        ("Job Roles",    str(df["Job_Title"].nunique())),
        ("Locations",    str(df["Location"].nunique())),
        ("Best Model",   best_name),
        ("R² Score",     f"{best_row.get('R2 Score',0):.4f}" if isinstance(best_row,dict) and 'R2 Score' in best_row else "—"),
        ("RMSE",         ri(best_row.get("RMSE",0)) if isinstance(best_row,dict) and "RMSE" in best_row else "—"),
    ]
    for i,(lbl,val) in enumerate(kpis2):
        with cols2[i]: st.markdown(stat_card(lbl,val,i), unsafe_allow_html=True)

    divider()
    ca, cb = st.columns([3,2])
    with ca:
        st.markdown('<p class="section-head">📌 What SalaryAI Does</p>', unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
        <strong>SalaryAI</strong> is a salary intelligence and career decision-support platform that:<br><br>
        &nbsp;💰&nbsp; <strong>Predicts</strong> expected salary from a professional profile<br>
        &nbsp;⚖️&nbsp; <strong>Benchmarks</strong> your current salary vs the model estimate<br>
        &nbsp;🔮&nbsp; <strong>Simulates</strong> career scenarios side-by-side<br>
        &nbsp;📊&nbsp; <strong>Analyses</strong> salary patterns across the full dataset<br>
        &nbsp;🔍&nbsp; <strong>Flags</strong> records with unusually large salary deviations<br>
        &nbsp;🤖&nbsp; <strong>Explains</strong> which features the model relies on most<br>
        &nbsp;🚀&nbsp; <strong>Provides</strong> dataset-driven career insights<br><br>
        <em>This is a decision-support tool. It is not a substitute for HR professionals
        or compensation experts.</em>
        </div>""", unsafe_allow_html=True)

    with cb:
        st.markdown('<p class="section-head">📂 Dataset Overview</p>', unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        d1.metric("Rows",       f"{summary['total_records']:,}")
        d1.metric("Salary Min", ri(stats["min"]))
        d2.metric("Columns",    str(summary["total_columns"]))
        d2.metric("Salary Max", ri(stats["max"]))
        st.metric("Salary Median", ri(stats["median"]))

    divider()
    st.markdown('<p class="section-head">🧭 Modules</p>', unsafe_allow_html=True)
    mods = [
        ("💰","Salary Prediction","AI salary estimate from your profile."),
        ("⚖️","Benchmarking","Current vs model estimated salary."),
        ("🔮","Career Simulator","Side-by-side scenario comparison."),
        ("📊","Analytics","Interactive charts & filters."),
        ("📈","Growth Projector","Project salary over your career."),
        ("🎯","Percentile Rank","Where does your salary rank?"),
        ("🗺","Role Heatmap","Salary by role × location matrix."),
        ("⚡","What-If Analyser","Sensitivity of salary to inputs."),
        ("🏆","Top Earners","Who are the highest earners?"),
        ("📂","Upload & Predict","Batch predict from your CSV."),
        ("🔍","Salary Review","Review flags for large deviations."),
        ("🤖","Model Performance","MAE, RMSE, R² for all models."),
        ("🗂","Dataset Explorer","Preview, filter & download data."),
        ("🚀","Career Insights","Dataset stats for your career path."),
    ]
    mc = st.columns(7)
    for i,(icon,name,desc) in enumerate(mods):
        with mc[i % 7]:
            st.markdown(f"""
            <div class="module-card" style="animation-delay:{i*0.07}s">
                <span class="module-icon">{icon}</span>
                <div class="module-name">{name}</div>
                <div class="module-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    if model is None:
        st.warning("⚠️ Model not trained. Run `python train_model.py` first.")
    else:
        st.success(f"✅ Model loaded — **{best_name}** (R² = {best_row.get('R2 Score',0):.4f})")


# ══════════════════════════════════════════════
# PAGE: SALARY PREDICTION
# ══════════════════════════════════════════════
elif page == "Salary Prediction":
    hero("Salary Prediction",
         "Enter your profile to get an AI-powered salary estimate", "💰")

    if model is None:
        st.error("⚠️ Model not trained. Run `python train_model.py` first."); st.stop()

    # Compute empirical margin once
    margin, _ = prediction_interval(df, model, confidence=0.90)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="input-card"><h4>👤 Professional Profile</h4></div>',
                    unsafe_allow_html=True)
        education  = st.selectbox("🎓 Education",  get_education_levels(df),
                                   help="Highest qualification")
        experience = st.slider("💼 Experience (yrs)", 0, 35, 5)
        age        = st.slider("🎂 Age", 18, 70, 28)

    with c2:
        st.markdown('<div class="input-card"><h4>🏢 Job Details</h4></div>',
                    unsafe_allow_html=True)
        location  = st.selectbox("📍 Location",  get_unique_values(df,"Location"))
        job_title = st.selectbox("💼 Job Title", get_unique_values(df,"Job_Title"))
        gender    = st.selectbox("👤 Gender",    get_unique_values(df,"Gender"))

    divider()
    if st.button("🔮 Predict Salary", use_container_width=True, type="primary"):
        if age < experience + 16:
            st.warning("⚠️ Age seems too low for the given experience.")
        else:
            try:
                with st.spinner("Running model…"):
                    inp = pd.DataFrame([{"Education":education,"Experience":experience,
                                         "Location":location,"Job_Title":job_title,
                                         "Age":age,"Gender":gender}])
                    pred  = model.predict(inp)[0]
                    lower = max(0, pred - margin) if margin else pred*0.90
                    upper = pred + margin          if margin else pred*1.10

                st.markdown(f"""
                <div class="result-banner">
                    <div class="result-label">Estimated Annual Salary</div>
                    <div class="result-salary">{ri(pred)}</div>
                    <div style="color:#64748b;font-size:.83rem;margin-top:.4rem;">
                        90% Estimated Range &nbsp;
                        <span style="color:#6ee7b7;">{ri(lower)}</span> — <span style="color:#6ee7b7;">{ri(upper)}</span>
                        &nbsp;<span class="badge badge-orange">Estimate</span>
                    </div>
                </div>""", unsafe_allow_html=True)

                r1,r2,r3 = st.columns(3)
                r1.metric("💰 Predicted",  ri(pred))
                r2.metric("📉 Range Low",  ri(lower))
                r3.metric("📈 Range High", ri(upper))

                st.markdown(f"""<div class="info-box" style="margin-top:.9rem;">
                    <strong>📊 Model Used:</strong> {best_name} &nbsp;
                    <span class="badge badge-blue">R² {best_row.get('R2 Score',0):.4f}</span><br><br>
                    Based on a <em>{education}</em> qualification, <em>{experience}</em> years of experience,
                    as a <em>{job_title}</em> in <em>{location}</em>, the estimated salary is
                    <strong>{ri(pred)}</strong>.
                    The range {ri(lower)}–{ri(upper)} is a 90th-percentile empirical estimate
                    based on training residuals — it is not a guarantee.
                </div>""", unsafe_allow_html=True)

                st.markdown("""<div class="warn-box" style="margin-top:.7rem;">
                    ⚠️ This estimate is based on the patterns in the training dataset.
                    Actual salaries depend on company, industry, skills, performance, and
                    market conditions not captured here.
                </div>""", unsafe_allow_html=True)

                # Download report
                profile = {"Education":education,"Experience":experience,
                           "Location":location,"Job_Title":job_title,
                           "Age":age,"Gender":gender}
                metrics_dict = {k: best_row[k] for k in ["MAE","RMSE","R2 Score"] if k in best_row}
                rpt = build_prediction_report(profile, pred, model_name=best_name,
                                              model_metrics=metrics_dict, margin=margin)
                csv_bytes = rpt.to_csv(index=False).encode()
                st.download_button("📥 Download Prediction Report (CSV)",
                                   csv_bytes, "salary_prediction_report.csv",
                                   "text/csv", use_container_width=True)

                with st.expander("🔍 Similar Records from Dataset"):
                    sim = df[(df["Education"]==education)&
                             (df["Job_Title"]==job_title)&
                             (df["Location"]==location)]
                    if len(sim):
                        st.write(f"**{len(sim)}** matching records:")
                        st.dataframe(sim.sort_values("Salary",ascending=False).head(10),
                                     use_container_width=True)
                    else:
                        st.info("No exact matches for this combination.")

            except Exception as e:
                st.error(f"❌ Prediction error: {e}")


# ══════════════════════════════════════════════
# PAGE: SALARY BENCHMARKING
# ══════════════════════════════════════════════
elif page == "Salary Benchmarking":
    hero("Salary Benchmarking",
         "Compare your current salary against the model estimate", "⚖️")

    if model is None:
        st.error("⚠️ Model not trained. Run `python train_model.py` first."); st.stop()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="input-card"><h4>👤 Your Profile</h4></div>', unsafe_allow_html=True)
        bm_edu = st.selectbox("🎓 Education", get_education_levels(df), key="bm_edu")
        bm_exp = st.slider("💼 Experience", 0, 35, 5, key="bm_exp")
        bm_age = st.slider("🎂 Age", 18, 70, 30, key="bm_age")
    with c2:
        st.markdown('<div class="input-card"><h4>🏢 Job Details</h4></div>', unsafe_allow_html=True)
        bm_loc = st.selectbox("📍 Location",  get_unique_values(df,"Location"), key="bm_loc")
        bm_job = st.selectbox("💼 Job Title", get_unique_values(df,"Job_Title"), key="bm_job")
        bm_gen = st.selectbox("👤 Gender",    get_unique_values(df,"Gender"), key="bm_gen")

    bm_cur = st.number_input("💵 Current Salary (₹)", min_value=0.0,
                              value=80000.0, step=1000.0, format="%.2f")
    divider()

    if st.button("⚖️ Benchmark My Salary", use_container_width=True, type="primary"):
        if bm_cur <= 0:
            st.error("❌ Enter a current salary greater than 0.")
        else:
            try:
                with st.spinner("Calculating…"):
                    inp  = pd.DataFrame([{"Education":bm_edu,"Experience":bm_exp,
                                          "Location":bm_loc,"Job_Title":bm_job,
                                          "Age":bm_age,"Gender":bm_gen}])
                    pred     = model.predict(inp)[0]
                    diff     = bm_cur - pred
                    pct_diff = (diff / pred) * 100

                if pct_diff < -10:
                    status,sbg,sbd,stxt="📉 Below Model Estimate","rgba(239,68,68,.12)","rgba(239,68,68,.4)","#fca5a5"
                    msg="Your salary is notably below the model estimate for this profile."
                elif pct_diff > 10:
                    status,sbg,sbd,stxt="📈 Above Model Estimate","rgba(16,185,129,.12)","rgba(16,185,129,.4)","#6ee7b7"
                    msg="Your salary is notably above the model estimate for this profile."
                else:
                    status,sbg,sbd,stxt="✅ Near Model Estimate","rgba(37,99,235,.12)","rgba(37,99,235,.4)","#93c5fd"
                    msg="Your salary is close to the model estimate for this profile."

                r1,r2,r3,r4 = st.columns(4)
                r1.metric("💵 Current",  ri(bm_cur))
                r2.metric("🤖 Estimate", ri(pred))
                r3.metric("📊 Diff",     f"{'+'if diff>=0 else ''}{ri(diff)}")
                r4.metric("% Diff",      f"{'+'if pct_diff>=0 else ''}{pct_diff:.1f}%")

                st.markdown(f"""
                <div class="status-box" style="background:{sbg};border:1px solid {sbd};margin:1rem 0;">
                    <div style="font-size:1.5rem;">{status.split()[0]}</div>
                    <div><strong style="color:{stxt};">{status}</strong><br>
                    <span style="font-size:.83rem;color:#94a3b8;">{msg}</span></div>
                </div>""", unsafe_allow_html=True)

                # Chart
                fig, ax = plt.subplots(figsize=(6,3.2))
                bars = ax.bar(["Current Salary","Model Estimate"],[bm_cur,pred],
                              color=["#f59e0b","#6366f1"], edgecolor="#0f172a", width=0.45, zorder=3)
                for b,v in zip(bars,[bm_cur,pred]):
                    ax.text(b.get_x()+b.get_width()/2, b.get_height()+500, ri(v),
                            ha="center", va="bottom", fontsize=10, color="#e2e8f0")
                ax.set_ylabel("Salary (₹)")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
                plt.tight_layout()
                chart_wrap("Current vs Model Estimate", fig)

                st.markdown("""<div class="warn-box">
                    ⚠️ <strong>Important:</strong> The benchmark is a model-based estimate using the
                    variables available in this dataset. Actual compensation depends on many additional
                    factors including company, industry, skills, certifications, and performance.
                    This is not proof of underpayment or overpayment.
                </div>""", unsafe_allow_html=True)

                # Download
                profile = {"Education":bm_edu,"Experience":bm_exp,
                           "Location":bm_loc,"Job_Title":bm_job,
                           "Age":bm_age,"Gender":bm_gen}
                metrics_dict = {k: best_row[k] for k in ["MAE","RMSE","R2 Score"] if k in best_row}
                rpt = build_prediction_report(profile, pred, current_salary=bm_cur,
                                              model_name=best_name, model_metrics=metrics_dict)
                st.download_button("📥 Download Benchmark Report (CSV)",
                                   rpt.to_csv(index=False).encode(),
                                   "benchmark_report.csv","text/csv",
                                   use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error: {e}")


# ══════════════════════════════════════════════
# PAGE: CAREER SIMULATOR
# ══════════════════════════════════════════════
elif page == "Career Simulator":
    hero("Career Scenario Simulator",
         "Compare two career profiles side-by-side using model estimates", "🔮")

    if model is None:
        st.error("⚠️ Model not trained. Run `python train_model.py` first."); st.stop()

    st.markdown("""<div class="info-box">
        Use this tool to explore <strong>model-based salary estimates</strong> for two different
        career scenarios. Changing a variable shows how the model responds — it does not guarantee
        that the change will produce the same result in the real world.
    </div>""", unsafe_allow_html=True)
    divider()

    sa_col, sb_col = st.columns(2, gap="large")

    with sa_col:
        st.markdown('<p class="section-head">🅐 Scenario A</p>', unsafe_allow_html=True)
        st.markdown('<div class="input-card"><h4>Profile A</h4></div>', unsafe_allow_html=True)
        a_edu = st.selectbox("🎓 Education A",  get_education_levels(df), key="a_edu")
        a_exp = st.slider("💼 Experience A", 0, 35, 5, key="a_exp")
        a_age = st.slider("🎂 Age A", 18, 70, 28, key="a_age")
        a_loc = st.selectbox("📍 Location A",  get_unique_values(df,"Location"), key="a_loc")
        a_job = st.selectbox("💼 Job Title A", get_unique_values(df,"Job_Title"), key="a_job")
        a_gen = st.selectbox("👤 Gender A",    get_unique_values(df,"Gender"), key="a_gen")

    with sb_col:
        st.markdown('<p class="section-head">🅑 Scenario B</p>', unsafe_allow_html=True)
        st.markdown('<div class="input-card"><h4>Profile B</h4></div>', unsafe_allow_html=True)
        b_edu = st.selectbox("🎓 Education B",  get_education_levels(df),
                              index=min(2, len(get_education_levels(df))-1), key="b_edu")
        b_exp = st.slider("💼 Experience B", 0, 35, 10, key="b_exp")
        b_age = st.slider("🎂 Age B", 18, 70, 33, key="b_age")
        b_loc = st.selectbox("📍 Location B",  get_unique_values(df,"Location"), key="b_loc")
        b_job = st.selectbox("💼 Job Title B", get_unique_values(df,"Job_Title"),
                              index=min(1, len(get_unique_values(df,"Job_Title"))-1), key="b_job")
        b_gen = st.selectbox("👤 Gender B",    get_unique_values(df,"Gender"), key="b_gen")

    divider()
    if st.button("🔮 Compare Scenarios", use_container_width=True, type="primary"):
        try:
            with st.spinner("Running both scenarios…"):
                inp_a = pd.DataFrame([{"Education":a_edu,"Experience":a_exp,"Location":a_loc,
                                        "Job_Title":a_job,"Age":a_age,"Gender":a_gen}])
                inp_b = pd.DataFrame([{"Education":b_edu,"Experience":b_exp,"Location":b_loc,
                                        "Job_Title":b_job,"Age":b_age,"Gender":b_gen}])
                pred_a = model.predict(inp_a)[0]
                pred_b = model.predict(inp_b)[0]
                diff   = pred_b - pred_a
                pct    = (diff / pred_a * 100) if pred_a else 0

            r1,r2,r3,r4 = st.columns(4)
            r1.metric("🅐 Scenario A", ri(pred_a))
            r2.metric("🅑 Scenario B", ri(pred_b))
            r3.metric("Difference",    f"{'+'if diff>=0 else ''}{ri(diff)}")
            r4.metric("% Change",      f"{'+'if pct>=0 else ''}{pct:.1f}%")

            # Bar chart
            fig, ax = plt.subplots(figsize=(7,3.5))
            bars = ax.bar(["Scenario A","Scenario B"],[pred_a,pred_b],
                          color=[CC[0],CC[2]], edgecolor="#0f172a", width=0.45, zorder=3)
            for b,v in zip(bars,[pred_a,pred_b]):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+500, ri(v),
                        ha="center", va="bottom", fontsize=11, color="#e2e8f0")
            ax.set_ylabel("Estimated Salary (₹)")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
            plt.tight_layout()
            chart_wrap("Scenario A vs Scenario B — Model Estimates", fig)

            # Diff table
            divider()
            st.markdown('<p class="section-head">📋 Profile Comparison</p>', unsafe_allow_html=True)
            comp_df = pd.DataFrame({
                "Feature":    ["Education","Experience","Location","Job Title","Age","Gender","Est. Salary"],
                "Scenario A": [a_edu, a_exp, a_loc, a_job, a_age, a_gen, ri(pred_a)],
                "Scenario B": [b_edu, b_exp, b_loc, b_job, b_age, b_gen, ri(pred_b)],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            st.markdown("""<div class="warn-box" style="margin-top:.8rem;">
                ⚠️ These are model-based scenarios. They illustrate how the model responds to
                different inputs — they do not guarantee real-world salary outcomes.
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")


# ══════════════════════════════════════════════
# PAGE: SALARY ANALYTICS
# ══════════════════════════════════════════════
elif page == "Salary Analytics":
    hero("Salary Analytics", "Interactive salary patterns from the actual dataset", "📊")

    with st.expander("🔍 Filters", expanded=False):
        fc1,fc2,fc3,fc4,fc5 = st.columns(5)
        f_edu = fc1.multiselect("Education", categories["Education"],  default=categories["Education"])
        f_loc = fc2.multiselect("Location",  categories["Location"],   default=categories["Location"])
        f_job = fc3.multiselect("Job Title", categories["Job_Title"],  default=categories["Job_Title"])
        f_gen = fc4.multiselect("Gender",    categories["Gender"],     default=categories["Gender"])
        exp_min, exp_max = int(df["Experience"].min()), int(df["Experience"].max())
        f_exp = fc5.slider("Experience range", exp_min, exp_max, (exp_min, exp_max))

    fdf = filter_data(df,
                      education=f_edu or None, location=f_loc or None,
                      job_title=f_job or None, gender=f_gen or None,
                      exp_range=f_exp)
    if len(fdf)==0:
        st.warning("No records match selected filters."); st.stop()

    stats = salary_statistics(fdf)
    kpi_cols = st.columns(8)
    for i,(lbl,val) in enumerate([
        ("Records",       f"{stats['count']:,}"),
        ("Avg Salary",    ri(stats['mean'])),
        ("Median",        ri(stats['median'])),
        ("Min",           ri(stats['min'])),
        ("Max",           ri(stats['max'])),
        ("Avg Exp",       f"{fdf['Experience'].mean():.1f} yrs"),
        ("Job Roles",     str(fdf['Job_Title'].nunique())),
        ("Locations",     str(fdf['Location'].nunique())),
    ]):
        with kpi_cols[i]: st.markdown(stat_card(lbl,val,i), unsafe_allow_html=True)
    divider()

    # Row 1
    c1,c2 = st.columns(2, gap="medium")
    with c1:
        fig,ax = plt.subplots(figsize=(6,3.5))
        ax.hist(fdf["Salary"],bins=30,color=CC[0],edgecolor="#0f172a",alpha=.9)
        ax.axvline(stats["mean"],  color="#f59e0b",linestyle="--",lw=1.5,label=f"Mean {ri(stats['mean'])}")
        ax.axvline(stats["median"],color="#10b981",linestyle="--",lw=1.5,label=f"Median {ri(stats['median'])}")
        ax.set_xlabel("Salary"); ax.set_ylabel("Count"); ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
        plt.tight_layout(); chart_wrap("📊 Salary Distribution", fig)
    with c2:
        fig,ax = plt.subplots(figsize=(6,3.5))
        sc = ax.scatter(fdf["Experience"],fdf["Salary"],c=fdf["Age"],cmap="plasma",alpha=.7,s=30,edgecolors="none")
        plt.colorbar(sc,ax=ax,label="Age")
        ax.set_xlabel("Experience (yrs)"); ax.set_ylabel("Salary")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
        plt.tight_layout(); chart_wrap("📈 Salary vs Experience (colour = Age)", fig)

    # Row 2
    c3,c4 = st.columns(2, gap="medium")
    with c3:
        edu_df = salary_by_group(fdf,"Education")
        fig,ax = plt.subplots(figsize=(6,3.5))
        bars = ax.bar(edu_df["Education"],edu_df["Average"],color=CC[:len(edu_df)],edgecolor="#0f172a",zorder=3)
        for b,v in zip(bars,edu_df["Average"]):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+300,ri(v),ha="center",va="bottom",fontsize=8,color="#e2e8f0")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
        plt.tight_layout(); chart_wrap("🎓 Salary by Education", fig)
    with c4:
        loc_df = salary_by_group(fdf,"Location")
        fig,ax = plt.subplots(figsize=(6,3.5))
        bars = ax.bar(loc_df["Location"],loc_df["Average"],color=[CC[2],CC[3],CC[4]],edgecolor="#0f172a",zorder=3)
        for b,v in zip(bars,loc_df["Average"]):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+300,ri(v),ha="center",va="bottom",fontsize=8,color="#e2e8f0")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
        plt.tight_layout(); chart_wrap("📍 Salary by Location", fig)

    # Row 3
    c5,c6 = st.columns(2, gap="medium")
    with c5:
        job_df = salary_by_group(fdf,"Job_Title").sort_values("Average")
        fig,ax = plt.subplots(figsize=(6,3.5))
        ax.barh(job_df["Job_Title"],job_df["Average"],color=CC[6],edgecolor="#0f172a",zorder=3)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
        plt.tight_layout(); chart_wrap("💼 Salary by Job Title", fig)
    with c6:
        gen_df = salary_by_group(fdf,"Gender")
        colors_g = [CC[0],"#ec4899"][:len(gen_df)]
        fig,ax = plt.subplots(figsize=(6,3.5))
        bars = ax.bar(gen_df["Gender"],gen_df["Average"],color=colors_g,edgecolor="#0f172a",width=.5,zorder=3)
        for b,v in zip(bars,gen_df["Average"]):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+300,ri(v),ha="center",va="bottom",fontsize=9,color="#e2e8f0")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
        plt.tight_layout(); chart_wrap("👤 Salary by Gender", fig)

    # Row 4
    c7,c8 = st.columns(2, gap="medium")
    with c7:
        fig,ax = plt.subplots(figsize=(6,3.5))
        ax.hist(fdf["Experience"],bins=20,color=CC[3],edgecolor="#0f172a",alpha=.9)
        ax.set_xlabel("Experience (yrs)"); ax.set_ylabel("Count")
        plt.tight_layout(); chart_wrap("⏳ Experience Distribution", fig)
    with c8:
        edu_counts = fdf["Education"].value_counts()
        fig,ax = plt.subplots(figsize=(6,3.5))
        wedges,_,autotexts = ax.pie(edu_counts.values,labels=edu_counts.index,autopct="%1.1f%%",
            colors=CC[:len(edu_counts)],startangle=140,pctdistance=.82,
            wedgeprops=dict(linewidth=2,edgecolor="#0f172a"))
        for at in autotexts: at.set_color("#e2e8f0"); at.set_fontsize(9)
        ax.axis("equal"); plt.tight_layout()
        chart_wrap("🎓 Education Distribution", fig)

    with st.expander("📋 Filtered Data Table"):
        st.dataframe(fdf, use_container_width=True)
        st.download_button("📥 Download Filtered Data (CSV)",
                           fdf.to_csv(index=False).encode(),
                           "filtered_data.csv","text/csv")


# ══════════════════════════════════════════════
# PAGE: SALARY REVIEW
# ══════════════════════════════════════════════
elif page == "Salary Review":
    hero("Salary Review Flags",
         "Records where the observed salary differs substantially from the model estimate", "🔍")

    if model is None:
        st.error("⚠️ Model not trained. Run `python train_model.py` first."); st.stop()

    st.markdown("""<div class="warn-box">
        ⚠️ <strong>Important:</strong> A review flag indicates that the observed salary differs
        substantially from the model estimate. It is <strong>NOT</strong> proof of an error,
        fraud, or unfair compensation. Many legitimate factors can cause large deviations.
        Use this as a starting point for review — not as a conclusion.
    </div>""", unsafe_allow_html=True)
    divider()

    threshold = st.slider("Sensitivity (std multiplier — lower = more flags)",
                           1.0, 3.5, 2.0, 0.25,
                           help="Flags records where |residual| > mean + multiplier × std(residuals)")

    try:
        with st.spinner("Computing residuals…"):
            df_resid  = compute_residuals(df, model)
            flagged   = flag_anomalies(df_resid, threshold_multiplier=threshold)

        total_flags = len(flagged)
        pct_flagged = total_flags / len(df) * 100

        f1,f2,f3 = st.columns(3)
        f1.metric("Total Records",     f"{len(df):,}")
        f2.metric("⚑ Flagged Records", str(total_flags))
        f3.metric("Flag Rate",         f"{pct_flagged:.1f}%")
        divider()

        if total_flags == 0:
            st.success("✅ No records flagged at this sensitivity level.")
        else:
            # Summary by job title
            c1,c2 = st.columns(2, gap="medium")
            with c1:
                show_cols = ["Job_Title","Education","Experience","Location",
                             "Salary","Predicted_Salary","Residual","Abs_Residual","Review_Flag"]
                st.markdown('<p class="section-head">⚑ Flagged Records</p>', unsafe_allow_html=True)
                st.dataframe(
                    flagged[show_cols].style.format({
                        "Salary": "₹{:,.0f}", "Predicted_Salary": "₹{:,.0f}",
                        "Residual": "₹{:+,.0f}", "Abs_Residual": "₹{:,.0f}",
                    }), use_container_width=True)

            with c2:
                st.markdown('<p class="section-head">📊 Residual Distribution</p>', unsafe_allow_html=True)
                fig,ax = plt.subplots(figsize=(5,3.5))
                ax.hist(df_resid["Residual"],bins=30,color=CC[0],edgecolor="#0f172a",alpha=.9,label="All")
                ax.hist(flagged["Residual"],bins=20,color="#ef4444",edgecolor="#0f172a",alpha=.8,label="Flagged")
                ax.axvline(0,color="#f59e0b",linestyle="--",lw=1.5,label="Zero")
                ax.set_xlabel("Residual (₹)"); ax.set_ylabel("Count"); ax.legend(fontsize=8)
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
                plt.tight_layout(); chart_wrap("Residual Distribution", fig)

            # Actual vs Predicted scatter for flagged
            fig,ax = plt.subplots(figsize=(10,4))
            ax.scatter(df_resid["Salary"],df_resid["Predicted_Salary"],
                       color=CC[0],alpha=.5,s=20,label="Normal",edgecolors="none")
            ax.scatter(flagged["Salary"],flagged["Predicted_Salary"],
                       color="#ef4444",alpha=.9,s=50,label="Flagged",edgecolors="#0f172a",zorder=5)
            mn = min(df["Salary"].min(), df_resid["Predicted_Salary"].min())
            mx = max(df["Salary"].max(), df_resid["Predicted_Salary"].max())
            ax.plot([mn,mx],[mn,mx],color="#f59e0b",linestyle="--",lw=1.5,label="Perfect fit")
            ax.set_xlabel("Actual Salary"); ax.set_ylabel("Predicted Salary")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
            ax.legend(fontsize=8); plt.tight_layout()
            chart_wrap("Actual vs Predicted — Flagged Records in Red", fig)

            st.download_button("📥 Download Flagged Records (CSV)",
                               flagged[show_cols].to_csv(index=False).encode(),
                               "salary_review_flags.csv","text/csv",
                               use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error computing residuals: {e}")


# ══════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════
elif page == "Model Performance":
    hero("Model Performance", "Comparison of all trained ML models", "🤖")

    if mr is None:
        st.error("⚠️ No model results. Run `python train_model.py` first."); st.stop()

    best_idx  = mr["R2 Score"].idxmax()
    b_name    = mr.loc[best_idx,"Model"]
    b_r2      = mr.loc[best_idx,"R2 Score"]

    st.markdown(f"""
    <div class="result-banner" style="text-align:left;padding:1.4rem 2rem;">
        <div style="display:flex;align-items:center;gap:1rem;">
            <span style="font-size:2.3rem;">🏆</span>
            <div>
                <div style="font-size:.78rem;color:#6ee7b7;text-transform:uppercase;letter-spacing:.1em;">Best Model</div>
                <div style="font-size:1.7rem;font-weight:800;color:#a5f3fc;">{b_name}</div>
                <div style="color:#94a3b8;font-size:.88rem;">R² = <strong style="color:#34d399;">{b_r2:.4f}</strong></div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Per-model cards
    mcols = st.columns(len(mr))
    for i,row in mr.iterrows():
        is_best = (row["Model"]==b_name)
        badge = ' <span class="badge badge-green">BEST</span>' if is_best else ""
        with mcols[i]:
            st.markdown(f"""
            <div class="stat-card" style="{'border-color:rgba(16,185,129,.6);' if is_best else ''}animation-delay:{i*.1}s">
                <div class="stat-value" style="font-size:.95rem;">{row['Model']}{badge}</div>
                <div style="margin-top:.45rem;font-size:.78rem;color:#94a3b8;">
                    R² <strong style="color:#a5f3fc;">{row['R2 Score']:.4f}</strong><br>
                    RMSE <strong style="color:#fde68a;">₹{row['RMSE']:,.0f}</strong><br>
                    MAE <strong style="color:#c4b5fd;">₹{row['MAE']:,.0f}</strong>
                </div>
            </div>""", unsafe_allow_html=True)

    divider()
    # Styled table
    display_df = mr.copy()
    display_df["Selected"] = display_df["Model"].apply(lambda x:"✅ Best" if x==b_name else "")
    st.dataframe(
        display_df.style
            .highlight_max(subset=["R2 Score"],color="#064e3b")
            .highlight_min(subset=["MAE","RMSE"],color="#064e3b")
            .format({"MAE":"{:,.2f}","MSE":"{:,.2f}","RMSE":"{:,.2f}","R2 Score":"{:.4f}"}),
        use_container_width=True)

    # Charts row
    bar_colors = ["#10b981" if m==b_name else "#6366f1" for m in mr["Model"]]
    ch1,ch2,ch3 = st.columns(3, gap="medium")
    for col, metric, title, fmt_fn in [
        (ch1,"R2 Score","R² Score (Higher = Better)", lambda v:f"{v:.4f}"),
        (ch2,"RMSE",    "RMSE ₹ (Lower = Better)",   lambda v:f"₹{v/1000:.1f}K"),
        (ch3,"MAE",     "MAE ₹ (Lower = Better)",    lambda v:f"₹{v/1000:.1f}K"),
    ]:
        with col:
            fig,ax = plt.subplots(figsize=(4.5,3.5))
            bars = ax.bar(mr["Model"],mr[metric],color=bar_colors,edgecolor="#0f172a",zorder=3)
            for b,v in zip(bars,mr[metric]):
                ax.text(b.get_x()+b.get_width()/2,b.get_height()*1.01,
                        fmt_fn(v),ha="center",va="bottom",fontsize=8,color="#e2e8f0")
            if metric=="R2 Score": ax.set_ylim(0,1.08)
            else:
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
            plt.xticks(rotation=20,ha="right",fontsize=8)
            plt.tight_layout(); chart_wrap(title,fig)

    # Actual vs Predicted + Residuals (from saved model)
    if model is not None:
        divider()
        st.markdown('<p class="section-head">📈 Actual vs Predicted & Residuals</p>',
                    unsafe_allow_html=True)
        try:
            df_r = compute_residuals(df, model)
            c1,c2 = st.columns(2, gap="medium")
            with c1:
                fig,ax = plt.subplots(figsize=(6,4))
                ax.scatter(df_r["Salary"],df_r["Predicted_Salary"],
                           color=CC[0],alpha=.6,s=22,edgecolors="none")
                mn = df_r["Salary"].min(); mx = df_r["Salary"].max()
                ax.plot([mn,mx],[mn,mx],color="#f59e0b",linestyle="--",lw=1.5,label="Perfect fit")
                ax.set_xlabel("Actual Salary"); ax.set_ylabel("Predicted Salary")
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
                ax.legend(); plt.tight_layout()
                chart_wrap("Actual vs Predicted", fig)
            with c2:
                fig,ax = plt.subplots(figsize=(6,4))
                ax.hist(df_r["Residual"],bins=30,color=CC[1],edgecolor="#0f172a",alpha=.9)
                ax.axvline(0,color="#f59e0b",linestyle="--",lw=1.5,label="Zero residual")
                ax.set_xlabel("Residual (₹)"); ax.set_ylabel("Count"); ax.legend(fontsize=8)
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
                plt.tight_layout(); chart_wrap("Residual Distribution", fig)
        except Exception as e:
            st.warning(f"Could not compute residuals: {e}")

    # Feature importance
    if model is not None:
        divider()
        st.markdown('<p class="section-head">🔬 Model Explainability — Feature Importance</p>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
            Feature importance describes how strongly the trained model relies on each feature
            for its predictions. It does <strong>not</strong> establish causation.
            For tree/ensemble models this is native importance; for linear models it uses
            the magnitude of coefficients.
        </div>""", unsafe_allow_html=True)
        imp_df = get_feature_importance(model)
        if imp_df is not None and len(imp_df):
            fi1,fi2 = st.columns([2,3], gap="medium")
            with fi1:
                st.dataframe(imp_df.style.format({"Importance":"{:.4f}"}),
                             use_container_width=True)
            with fi2:
                fig,ax = plt.subplots(figsize=(6,3.5))
                ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1],
                        color=CC[:len(imp_df)], edgecolor="#0f172a", zorder=3)
                ax.set_xlabel("Importance (normalised)")
                plt.tight_layout(); chart_wrap("Feature Importance", fig)
        else:
            st.info("Feature importance is not available for the current best model.")

    divider()
    with st.expander("📖 Metric Definitions", expanded=True):
        m1,m2,m3,m4 = st.columns(4)
        for col,t,b in [
            (m1,"MAE","Mean Absolute Error — average absolute prediction error. Lower is better."),
            (m2,"MSE","Mean Squared Error — penalises large errors more. Lower is better."),
            (m3,"RMSE","Root Mean Squared Error — error in salary units (₹). Lower is better."),
            (m4,"R²","Coefficient of Determination — proportion of variance explained (0–1). Higher is better."),
        ]:
            with col:
                st.markdown(f"""<div class="info-box">
                    <strong style="color:#a5b4fc;">{t}</strong><br>
                    <span style="font-size:.8rem;">{b}</span>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ══════════════════════════════════════════════
elif page == "Dataset Explorer":
    hero("Dataset Explorer", "Preview, filter, inspect and download the salary dataset", "🗂")

    raw_df = load_data(DATA_PATH)   # show raw for transparency
    qr     = data_quality_report(raw_df)

    r1,r2,r3,r4,r5 = st.columns(5)
    r1.metric("Total Rows",       f"{qr['total_rows']:,}")
    r2.metric("Total Columns",    str(qr['total_columns']))
    r3.metric("Duplicate Rows",   str(qr['duplicate_rows']))
    r4.metric("Total Missing",    str(qr['total_missing']))
    r5.metric("Salary Col",       "✅ Present" if TARGET_COL in raw_df.columns else "❌ Missing")

    if qr["issues"]:
        st.markdown(f"""<div class="warn-box">⚠️ Data quality notes:<br>
            {"<br>".join("• " + i for i in qr["issues"])}</div>""",
            unsafe_allow_html=True)
    else:
        st.success("✅ No data quality issues detected in the raw dataset.")
    divider()

    tab_preview, tab_types, tab_missing, tab_stats, tab_cats = st.tabs(
        ["👁 Preview","📐 Types","❓ Missing Values","📊 Statistics","🏷 Categories"])

    with tab_preview:
        n = st.slider("Rows to preview", 5, min(100,len(raw_df)), 20)
        st.dataframe(raw_df.head(n), use_container_width=True)

    with tab_types:
        dtypes_df = pd.DataFrame({"Column": raw_df.dtypes.index,
                                   "Type":   raw_df.dtypes.astype(str).values})
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    with tab_missing:
        mv = missing_value_report(raw_df)
        if len(mv)==0:
            st.success("✅ No missing values.")
        else:
            st.dataframe(mv, use_container_width=True)

    with tab_stats:
        st.dataframe(raw_df.describe().round(2), use_container_width=True)

    with tab_cats:
        for col in ["Education","Location","Job_Title","Gender"]:
            if col in raw_df.columns:
                vc = raw_df[col].value_counts().reset_index()
                vc.columns = [col,"Count"]
                st.markdown(f"**{col}** ({raw_df[col].nunique()} unique values)")
                st.dataframe(vc, use_container_width=True, hide_index=True)

    divider()
    st.markdown('<p class="section-head">🔍 Filter & Download</p>', unsafe_allow_html=True)
    fc1,fc2,fc3,fc4 = st.columns(4)
    dl_edu = fc1.multiselect("Education", categories["Education"], default=categories["Education"], key="dl_edu")
    dl_loc = fc2.multiselect("Location",  categories["Location"],  default=categories["Location"],  key="dl_loc")
    dl_job = fc3.multiselect("Job Title", categories["Job_Title"], default=categories["Job_Title"], key="dl_job")
    dl_gen = fc4.multiselect("Gender",    categories["Gender"],    default=categories["Gender"],    key="dl_gen")

    dl_df = filter_data(df, education=dl_edu or None, location=dl_loc or None,
                         job_title=dl_job or None, gender=dl_gen or None)
    st.write(f"**{len(dl_df):,}** records match.")
    st.dataframe(dl_df, use_container_width=True)
    st.download_button("📥 Download Filtered Dataset (CSV)",
                       dl_df.to_csv(index=False).encode(),
                       "salary_data_filtered.csv","text/csv",
                       use_container_width=True)


# ══════════════════════════════════════════════
# PAGE: CAREER INSIGHTS
# ══════════════════════════════════════════════
elif page == "Career Insights":
    hero("Career Insights",
         "Dataset-driven salary statistics for your career path", "🚀")

    c1,c2 = st.columns(2, gap="large")
    with c1:
        ci_edu = st.selectbox("🎓 Education", ["Any"]+get_education_levels(df), key="ci_edu")
        ci_exp = st.slider("💼 Experience (yrs)", 0, 35, 5, key="ci_exp")
    with c2:
        ci_job = st.selectbox("💼 Job Title", ["Any"]+get_unique_values(df,"Job_Title"), key="ci_job")
        ci_loc = st.selectbox("📍 Location",  ["Any"]+get_unique_values(df,"Location"),  key="ci_loc")
    divider()

    if st.button("🔍 Get Career Insights", use_container_width=True, type="primary"):
        edu = None if ci_edu=="Any" else ci_edu
        job = None if ci_job=="Any" else ci_job
        loc = None if ci_loc=="Any" else ci_loc

        with st.spinner("Analysing dataset…"):
            ins = career_insights(df,education=edu,job_title=job,
                                   location=loc,experience=ci_exp)
        if ins.get("no_data"):
            st.warning("No records found. Try relaxing the filters.")
        else:
            st.success(f"✅ Found **{ins['total_records']}** matching records.")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("📊 Avg Salary",    ri(ins["avg_salary"]))
            c2.metric("📈 Median",        ri(ins["median_salary"]))
            c3.metric("📉 Min",           ri(ins["min_salary"]))
            c4.metric("📈 Max",           ri(ins["max_salary"]))

            ca,cb = st.columns(2, gap="large")
            with ca:
                if "similar_exp_count" in ins:
                    avg_s = ri(ins["similar_exp_avg_salary"]) if ins["similar_exp_avg_salary"] else "N/A"
                    st.markdown(f"""<div class="info-box">
                        <strong>⏳ Similar Experience (±5 yrs of {ci_exp})</strong><br>
                        Records: <strong>{ins['similar_exp_count']}</strong><br>
                        Avg Salary: <strong>{avg_s}</strong>
                    </div>""", unsafe_allow_html=True)
                if ins.get("exp_salary_correlation") is not None:
                    corr = ins["exp_salary_correlation"]
                    dir_ = "positively" if corr>0 else "negatively"
                    str_ = "strongly" if abs(corr)>0.5 else "weakly"
                    st.info(f"📊 Experience & salary are **{str_} {dir_} correlated** (r = {corr:.3f})")
            with cb:
                if "locations_with_data" in ins and ins["locations_with_data"]:
                    st.markdown("**📍 Locations with data for this role:**")
                    ld = pd.DataFrame(list(ins["locations_with_data"].items()),
                                      columns=["Location","Count"]).sort_values("Count",ascending=False)
                    st.dataframe(ld, use_container_width=True, hide_index=True)

            fs = filter_data(df, education=[edu] if edu else None,
                             job_title=[job] if job else None,
                             location=[loc] if loc else None)
            if len(fs)>=5:
                cl,cr = st.columns(2, gap="medium")
                with cl:
                    fig,ax = plt.subplots(figsize=(6,3.5))
                    ax.hist(fs["Salary"],bins=20,color=CC[0],edgecolor="#0f172a",alpha=.9)
                    ax.axvline(ins["avg_salary"],  color="#f59e0b",linestyle="--",lw=1.5,label="Avg")
                    ax.axvline(ins["median_salary"],color="#10b981",linestyle="--",lw=1.5,label="Median")
                    ax.set_xlabel("Salary"); ax.legend()
                    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
                    plt.tight_layout(); chart_wrap("Salary Distribution", fig)
                with cr:
                    fig,ax = plt.subplots(figsize=(6,3.5))
                    ax.scatter(fs["Experience"],fs["Salary"],
                               color=CC[6],alpha=.75,s=50,edgecolors="#0f172a",zorder=3)
                    ax.set_xlabel("Experience (yrs)"); ax.set_ylabel("Salary")
                    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
                    plt.tight_layout(); chart_wrap("Experience vs Salary", fig)
            else:
                st.info("Not enough records for charts.")


# ══════════════════════════════════════════════
# PAGE: GROWTH PROJECTOR
# ══════════════════════════════════════════════
elif page == "Growth Projector":
    hero("Salary Growth Projector",
         "See how your salary is projected to grow as your experience increases", "📈")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        gp_edu = st.selectbox("🎓 Education",  get_education_levels(df), key="gp_edu")
        gp_job = st.selectbox("💼 Job Title",  get_unique_values(df, "Job_Title"), key="gp_job")
    with c2:
        gp_loc = st.selectbox("📍 Location",   get_unique_values(df, "Location"), key="gp_loc")
        gp_exp = st.slider("💼 Current Experience (yrs)", 0, 25, 3, key="gp_exp")
        gp_yrs = st.slider("⏩ Years to Project", 3, 20, 10, key="gp_yrs")

    divider()
    if st.button("📈 Project Growth", use_container_width=True, type="primary"):
        with st.spinner("Projecting…"):
            proj = salary_growth_projection(df, gp_edu, gp_job, gp_loc, gp_exp, gp_yrs)

        valid = proj.dropna(subset=["Avg_Salary"])
        if len(valid) < 2:
            st.warning("Not enough data for this combination. Try relaxing the filters.")
        else:
            r1, r2, r3 = st.columns(3)
            r1.metric("Starting Salary",  ri(valid.iloc[0]["Avg_Salary"]))
            r2.metric("Projected Salary", ri(valid.iloc[-1]["Avg_Salary"]))
            growth = valid.iloc[-1]["Avg_Salary"] - valid.iloc[0]["Avg_Salary"]
            r3.metric("Total Growth",     ri(growth),
                      delta=f"+{growth/valid.iloc[0]['Avg_Salary']*100:.1f}%" if valid.iloc[0]["Avg_Salary"] else None)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(valid["Experience"], valid["Avg_Salary"],
                    color=CC[0], lw=2.5, marker="o", markersize=5, zorder=3)
            ax.fill_between(valid["Experience"], valid["Avg_Salary"],
                            alpha=0.15, color=CC[0])
            ax.axvline(gp_exp, color="#f59e0b", linestyle="--", lw=1.5,
                       label=f"Now ({gp_exp} yrs)")
            ax.set_xlabel("Experience (yrs)")
            ax.set_ylabel("Avg Salary (₹)")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
            ax.legend(fontsize=9)
            plt.tight_layout()
            chart_wrap("Projected Salary Growth Curve", fig)

            with st.expander("📋 Projection Table"):
                st.dataframe(
                    valid.style.format({"Avg_Salary": "₹{:,.0f}"}),
                    use_container_width=True, hide_index=True
                )
                st.download_button("📥 Download Projection (CSV)",
                                   valid.to_csv(index=False).encode(),
                                   "salary_growth_projection.csv", "text/csv")

        st.markdown("""<div class="warn-box" style="margin-top:.8rem;">
            ⚠️ Projections are based on observed averages in the dataset — not guaranteed future salaries.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: PERCENTILE RANK
# ══════════════════════════════════════════════
elif page == "Percentile Rank":
    hero("Salary Percentile Rank",
         "See where a salary falls in the distribution for your profile", "🎯")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        pr_sal = st.number_input("💵 Salary to Rank (₹)", min_value=0.0,
                                  value=90000.0, step=1000.0, format="%.0f")
        pr_edu = st.selectbox("🎓 Education (optional)",
                               ["Any"] + get_education_levels(df), key="pr_edu")
    with c2:
        pr_job = st.selectbox("💼 Job Title (optional)",
                               ["Any"] + get_unique_values(df, "Job_Title"), key="pr_job")
        pr_loc = st.selectbox("📍 Location (optional)",
                               ["Any"] + get_unique_values(df, "Location"), key="pr_loc")

    divider()
    if st.button("🎯 Calculate Percentile", use_container_width=True, type="primary"):
        edu = None if pr_edu == "Any" else pr_edu
        job = None if pr_job == "Any" else pr_job
        loc = None if pr_loc == "Any" else pr_loc

        result = salary_percentile_rank(df, pr_sal, education=edu,
                                         job_title=job, location=loc)
        if result.get("no_data"):
            st.warning("Not enough data for the selected filters. Try selecting 'Any'.")
        else:
            pct = result["percentile_rank"]
            color = "#10b981" if pct >= 75 else ("#f59e0b" if pct >= 40 else "#ef4444")
            tier  = "Top earner" if pct >= 90 else ("Above average" if pct >= 60
                    else ("Near average" if pct >= 40 else "Below average"))

            st.markdown(f"""
            <div class="result-banner" style="text-align:center;padding:2rem;">
                <div class="result-label">Your Percentile Rank</div>
                <div style="font-size:3.2rem;font-weight:800;color:{color};">{pct:.1f}%</div>
                <div style="font-size:.9rem;color:#94a3b8;margin-top:.4rem;">
                    {tier} — {result['sample_size']:,} records in comparison group
                </div>
            </div>""", unsafe_allow_html=True)

            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("P25", ri(result["p25"]))
            r2.metric("P50 (Median)", ri(result["p50"]))
            r3.metric("P75", ri(result["p75"]))
            r4.metric("P90", ri(result["p90"]))
            r5.metric("Mean", ri(result["mean"]))

            # Distribution with salary marker
            subset = filter_data(
                df,
                education=[edu] if edu else None,
                location=[loc] if loc else None,
                job_title=[job] if job else None,
            )
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.hist(subset["Salary"], bins=30, color=CC[0],
                    edgecolor="#0f172a", alpha=0.85, label="Distribution")
            ax.axvline(pr_sal, color="#f59e0b", lw=2.5, linestyle="--",
                       label=f"Your salary {ri(pr_sal)}")
            ax.axvline(result["p50"], color="#10b981", lw=1.5, linestyle=":",
                       label=f"Median {ri(result['p50'])}")
            ax.set_xlabel("Salary (₹)")
            ax.set_ylabel("Count")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
            ax.legend(fontsize=9)
            plt.tight_layout()
            chart_wrap("Salary Distribution — Your Position Highlighted", fig)


# ══════════════════════════════════════════════
# PAGE: ROLE HEATMAP
# ══════════════════════════════════════════════
elif page == "Role Heatmap":
    hero("Role × Location Salary Heatmap",
         "Average salary for each role across every location", "🗺")

    import matplotlib.colors as mcolors

    pivot = role_location_heatmap(df)
    edu_matrix = education_experience_matrix(df)

    tab_rl, tab_ee, tab_gender, tab_age = st.tabs(
        ["🗺 Role × Location", "🎓 Education × Experience",
         "👥 Gender Pay Gap", "🎂 Age Band Salary"])

    with tab_rl:
        st.markdown("""<div class="info-box">
            Average salary (₹) for each Job Title × Location combination.
            Darker = higher salary. White = no data.
        </div>""", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.55)))
        cmap = plt.cm.YlOrRd
        masked = np.ma.masked_invalid(pivot.values.astype(float))
        im = ax.imshow(masked, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        plt.colorbar(im, ax=ax, label="Avg Salary (₹)")
        for r in range(len(pivot.index)):
            for c in range(len(pivot.columns)):
                val = pivot.values[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"₹{val/1000:.0f}K",
                            ha="center", va="center", fontsize=7.5,
                            color="white" if val > pivot.values[~np.isnan(pivot.values)].mean() else "#1e293b")
        plt.tight_layout()
        chart_wrap("Role × Location Average Salary Heatmap", fig)
        st.dataframe(pivot.style.format("₹{:,.0f}", na_rep="—")
                     .background_gradient(cmap="YlOrRd", axis=None),
                     use_container_width=True)

    with tab_ee:
        st.markdown("""<div class="info-box">
            Average salary by Education level and Experience band.
        </div>""", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        masked_ee = np.ma.masked_invalid(edu_matrix.values.astype(float))
        im2 = ax.imshow(masked_ee, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(edu_matrix.columns)))
        ax.set_xticklabels(edu_matrix.columns, fontsize=9)
        ax.set_yticks(range(len(edu_matrix.index)))
        ax.set_yticklabels(edu_matrix.index, fontsize=9)
        plt.colorbar(im2, ax=ax, label="Avg Salary (₹)")
        for r in range(len(edu_matrix.index)):
            for c in range(len(edu_matrix.columns)):
                val = edu_matrix.values[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"₹{val/1000:.0f}K",
                            ha="center", va="center", fontsize=7.5, color="#e2e8f0")
        plt.tight_layout()
        chart_wrap("Education × Experience Salary Matrix", fig)

    with tab_gender:
        gap_df = gender_pay_gap_by_role(df)
        if len(gap_df) == 0:
            st.info("Not enough data to compute gender pay gap by role.")
        else:
            c1, c2 = st.columns([2, 3], gap="medium")
            with c1:
                st.dataframe(
                    gap_df.style.format({
                        "Male_Avg": "₹{:,.0f}", "Female_Avg": "₹{:,.0f}",
                        "Gap": "₹{:+,.0f}", "Gap_Pct": "{:+.1f}%",
                    }),
                    use_container_width=True, hide_index=True,
                )
            with c2:
                fig, ax = plt.subplots(figsize=(6, max(3, len(gap_df) * 0.45)))
                y = range(len(gap_df))
                ax.barh(y, gap_df["Male_Avg"],   color=CC[0], label="Male",   alpha=0.85)
                ax.barh(y, gap_df["Female_Avg"], color="#ec4899", label="Female", alpha=0.85)
                ax.set_yticks(list(y))
                ax.set_yticklabels(gap_df["Job_Title"], fontsize=8)
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
                ax.legend(fontsize=9)
                plt.tight_layout()
                chart_wrap("Male vs Female Average Salary by Role", fig)

            st.markdown("""<div class="warn-box" style="margin-top:.8rem;">
                ⚠️ These are simple averages — they do not control for experience, age,
                location, or other confounding variables. Do not interpret as proof of discrimination.
            </div>""", unsafe_allow_html=True)

    with tab_age:
        age_df = salary_by_age_band(df)
        c1, c2 = st.columns([2, 3], gap="medium")
        with c1:
            st.dataframe(
                age_df.style.format({"Avg_Salary": "₹{:,.0f}", "Median_Salary": "₹{:,.0f}"}),
                use_container_width=True, hide_index=True
            )
        with c2:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            bars = ax.bar(age_df["Age_Band"].astype(str), age_df["Avg_Salary"],
                          color=CC[:len(age_df)], edgecolor="#0f172a", zorder=3)
            for b, v in zip(bars, age_df["Avg_Salary"]):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 300,
                        ri(v), ha="center", va="bottom", fontsize=9, color="#e2e8f0")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
            ax.set_xlabel("Age Band"); ax.set_ylabel("Avg Salary (₹)")
            plt.tight_layout()
            chart_wrap("Average Salary by Age Band", fig)


# ══════════════════════════════════════════════
# PAGE: WHAT-IF ANALYSER
# ══════════════════════════════════════════════
elif page == "What-If Analyser":
    hero("What-If Salary Analyser",
         "See how changing a single factor shifts your predicted salary", "⚡")

    if model is None:
        st.error("⚠️ Model not trained. Run `python train_model.py` first."); st.stop()

    st.markdown("""<div class="info-box">
        Fix all profile inputs, then choose a variable to sweep.
        The chart shows how the model's prediction changes as that one variable moves.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        wi_edu = st.selectbox("🎓 Education",  get_education_levels(df), key="wi_edu")
        wi_job = st.selectbox("💼 Job Title",  get_unique_values(df, "Job_Title"), key="wi_job")
        wi_loc = st.selectbox("📍 Location",   get_unique_values(df, "Location"), key="wi_loc")
    with c2:
        wi_gen = st.selectbox("👤 Gender",     get_unique_values(df, "Gender"), key="wi_gen")
        wi_exp = st.slider("💼 Base Experience (yrs)", 0, 35, 5, key="wi_exp")
        wi_age = st.slider("🎂 Base Age", 18, 70, 28, key="wi_age")

    sweep_var = st.radio("📊 Variable to Sweep", ["Experience", "Age"], horizontal=True)
    divider()

    if st.button("⚡ Run What-If Analysis", use_container_width=True, type="primary"):
        base_profile = {
            "Education": wi_edu, "Experience": wi_exp,
            "Location": wi_loc, "Job_Title": wi_job,
            "Age": wi_age, "Gender": wi_gen,
        }
        with st.spinner("Running sweep…"):
            if sweep_var == "Experience":
                sweep_df = what_if_experience_sweep(df, model, base_profile, (0, 35))
                x_col, x_label = "Experience", "Experience (yrs)"
            else:
                sweep_df = what_if_age_sweep(df, model, base_profile, (18, 65))
                x_col, x_label = "Age", "Age"

        sweep_df = sweep_df.dropna(subset=["Predicted_Salary"])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sweep_df[x_col], sweep_df["Predicted_Salary"],
                color=CC[0], lw=2.5, zorder=3)
        ax.fill_between(sweep_df[x_col], sweep_df["Predicted_Salary"],
                        alpha=0.15, color=CC[0])
        base_val = wi_exp if sweep_var == "Experience" else wi_age
        base_pred = sweep_df[sweep_df[x_col] == base_val]["Predicted_Salary"]
        if len(base_pred):
            ax.axvline(base_val, color="#f59e0b", lw=1.8, linestyle="--",
                       label=f"Base {sweep_var}={base_val}")
            ax.scatter([base_val], [base_pred.values[0]],
                       color="#f59e0b", s=80, zorder=5)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Predicted Salary (₹)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
        ax.legend(fontsize=9)
        plt.tight_layout()
        chart_wrap(f"Predicted Salary vs {sweep_var} (all else constant)", fig)

        min_s = sweep_df["Predicted_Salary"].min()
        max_s = sweep_df["Predicted_Salary"].max()
        r1, r2, r3 = st.columns(3)
        r1.metric("Min Predicted", ri(min_s))
        r2.metric("Max Predicted", ri(max_s))
        r3.metric("Range", ri(max_s - min_s),
                  delta=f"+{(max_s-min_s)/min_s*100:.1f}%" if min_s else None)

        with st.expander("📋 Sweep Data Table"):
            st.dataframe(sweep_df.style.format({"Predicted_Salary": "₹{:,.0f}"}),
                         use_container_width=True, hide_index=True)
            st.download_button("📥 Download (CSV)",
                               sweep_df.to_csv(index=False).encode(),
                               f"whatif_{sweep_var.lower()}_sweep.csv", "text/csv")


# ══════════════════════════════════════════════
# PAGE: TOP EARNERS
# ══════════════════════════════════════════════
elif page == "Top Earners":
    hero("Top Earner Profiles",
         "Who are the highest earners in the dataset, and what do they have in common?", "🏆")

    pct_slider = st.slider("Show top earners above percentile (%)", 70, 99, 90, 1,
                            help="E.g. 90 = top 10% earners")
    top_n = st.slider("Maximum records to display", 10, 100, 20)

    top = top_earner_profiles(df, top_n=top_n, percentile=pct_slider / 100)
    threshold = df["Salary"].quantile(pct_slider / 100)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Salary Threshold", ri(threshold))
    r2.metric("Records Shown",    str(len(top)))
    r3.metric("Top Avg Salary",   ri(top["Salary"].mean()))
    r4.metric("Overall Mean",     ri(df["Salary"].mean()))
    divider()

    tab_table, tab_edu, tab_job, tab_loc = st.tabs(
        ["📋 Records", "🎓 Education Mix", "💼 Job Mix", "📍 Location Mix"])

    with tab_table:
        display_cols = ["Job_Title", "Education", "Experience",
                        "Location", "Gender", "Age", "Salary", "Salary_vs_Mean_%"]
        st.dataframe(
            top[display_cols].style.format({
                "Salary": "₹{:,.0f}",
                "Salary_vs_Mean_%": "{:+.1f}%",
            }).background_gradient(subset=["Salary"], cmap="Greens"),
            use_container_width=True, hide_index=True
        )
        st.download_button("📥 Download Top Earners (CSV)",
                           top[display_cols].to_csv(index=False).encode(),
                           "top_earners.csv", "text/csv", use_container_width=True)

    with tab_edu:
        edu_counts = top["Education"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        wedges, _, autotexts = ax.pie(
            edu_counts.values, labels=edu_counts.index,
            autopct="%1.0f%%", colors=CC[:len(edu_counts)],
            startangle=140, wedgeprops=dict(linewidth=2, edgecolor="#0f172a"))
        for at in autotexts:
            at.set_color("#e2e8f0"); at.set_fontsize(9)
        ax.axis("equal"); plt.tight_layout()
        chart_wrap("Education Distribution Among Top Earners", fig)

    with tab_job:
        job_counts = top["Job_Title"].value_counts()
        fig, ax = plt.subplots(figsize=(6, max(3, len(job_counts) * 0.45)))
        ax.barh(job_counts.index[::-1], job_counts.values[::-1],
                color=CC[6], edgecolor="#0f172a", zorder=3)
        ax.set_xlabel("Count")
        plt.tight_layout()
        chart_wrap("Top Earner Job Title Distribution", fig)

    with tab_loc:
        loc_counts = top["Location"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(loc_counts.index, loc_counts.values,
               color=[CC[2], CC[3], CC[4]][:len(loc_counts)], edgecolor="#0f172a", zorder=3)
        ax.set_xlabel("Location"); ax.set_ylabel("Count")
        plt.tight_layout()
        chart_wrap("Top Earner Location Distribution", fig)


# ══════════════════════════════════════════════
# PAGE: UPLOAD & PREDICT
# ══════════════════════════════════════════════
elif page == "Upload & Predict":
    hero("Upload & Batch Predict",
         "Upload your own CSV to get AI salary predictions for every row", "📂")

    if model is None:
        st.error("⚠️ Model not trained. Run `python train_model.py` first."); st.stop()

    st.markdown(f"""<div class="info-box">
        Upload a CSV with the following columns:<br>
        <code>Education, Experience, Location, Job_Title, Age, Gender</code><br>
        A <code>Salary</code> column is optional — if present, a residual analysis is included.
        <br><br>Accepted education values: <strong>{", ".join(get_education_levels(df))}</strong><br>
        Accepted locations: <strong>{", ".join(get_unique_values(df, "Location"))}</strong>
    </div>""", unsafe_allow_html=True)
    divider()

    uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            validation  = validate_uploaded_csv(uploaded_df)

            r1, r2, r3 = st.columns(3)
            r1.metric("Uploaded Rows",    str(validation["rows"]))
            r2.metric("Columns Found",    str(len(validation["columns"])))
            r3.metric("Validation",       "✅ Valid" if validation["valid"] else "❌ Issues")

            if validation["issues"]:
                for issue in validation["issues"]:
                    st.error(f"❌ {issue}")
                st.stop()

            if validation["warnings"]:
                for w in validation["warnings"]:
                    st.warning(f"⚠️ {w}")

            st.success("✅ File validated. Running predictions…")
            with st.spinner("Predicting all rows…"):
                predict_df  = clean_data(uploaded_df) if "Salary" in uploaded_df.columns else uploaded_df.copy()
                feat_cols_present = [c for c in FEATURE_COLS if c in predict_df.columns]
                if len(feat_cols_present) < len(FEATURE_COLS):
                    missing = [c for c in FEATURE_COLS if c not in predict_df.columns]
                    st.error(f"❌ Missing feature columns: {missing}")
                    st.stop()

                predictions = model.predict(predict_df[FEATURE_COLS])
                predict_df["Predicted_Salary"] = predictions.round(0)

                margin, _ = prediction_interval(df, model, confidence=0.90)
                if margin:
                    predict_df["Range_Low"]  = (predict_df["Predicted_Salary"] - margin).clip(lower=0).round(0)
                    predict_df["Range_High"] = (predict_df["Predicted_Salary"] + margin).round(0)

                if TARGET_COL in predict_df.columns:
                    predict_df["Residual"] = (
                        predict_df[TARGET_COL] - predict_df["Predicted_Salary"]
                    ).round(0)

            st.markdown(f"**{len(predict_df):,}** predictions generated.")
            st.dataframe(
                predict_df.style.format({
                    "Predicted_Salary": "₹{:,.0f}",
                    "Range_Low":        "₹{:,.0f}",
                    "Range_High":       "₹{:,.0f}",
                    **({"Salary": "₹{:,.0f}", "Residual": "₹{:+,.0f}"}
                       if TARGET_COL in predict_df.columns else {}),
                }),
                use_container_width=True
            )

            # Summary stats for batch predictions
            divider()
            st.markdown('<p class="section-head">📊 Prediction Summary</p>', unsafe_allow_html=True)
            kc = st.columns(4)
            kc[0].metric("Avg Predicted",    ri(predict_df["Predicted_Salary"].mean()))
            kc[1].metric("Median Predicted", ri(predict_df["Predicted_Salary"].median()))
            kc[2].metric("Min Predicted",    ri(predict_df["Predicted_Salary"].min()))
            kc[3].metric("Max Predicted",    ri(predict_df["Predicted_Salary"].max()))

            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.hist(predict_df["Predicted_Salary"], bins=30,
                    color=CC[2], edgecolor="#0f172a", alpha=0.9)
            ax.axvline(predict_df["Predicted_Salary"].mean(), color="#f59e0b",
                       linestyle="--", lw=1.5, label="Mean")
            ax.set_xlabel("Predicted Salary (₹)")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
            ax.legend(fontsize=9)
            plt.tight_layout()
            chart_wrap("Distribution of Batch Predictions", fig)

            if TARGET_COL in predict_df.columns and "Residual" in predict_df.columns:
                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.hist(predict_df["Residual"], bins=30,
                        color=CC[4], edgecolor="#0f172a", alpha=0.9)
                ax.axvline(0, color="#f59e0b", linestyle="--", lw=1.5, label="Zero")
                ax.set_xlabel("Residual (₹ — Actual minus Predicted)")
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
                ax.legend(fontsize=9)
                plt.tight_layout()
                chart_wrap("Residual Distribution (Actual vs Predicted)", fig)

            st.download_button("📥 Download Predictions (CSV)",
                               predict_df.to_csv(index=False).encode(),
                               "batch_predictions.csv", "text/csv",
                               use_container_width=True)

            st.markdown("""<div class="warn-box" style="margin-top:.8rem;">
                ⚠️ Predictions are made using the model trained on the built-in dataset.
                Accuracy may vary if your data has different distributions or categories.
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    else:
        st.markdown("""<div class="info-box" style="text-align:center;padding:2rem;">
            📁 Upload a CSV above to begin batch predictions.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: AI ASSISTANT
# ══════════════════════════════════════════════
elif page == "AI Assistant":
    hero("AI Salary Assistant",
         "Ask anything about salaries, the dataset, or the prediction model", "💬")

    # ── session state for chat history ──────────────────────────────────────
    if "va_history" not in st.session_state:
        st.session_state.va_history = [
            {
                "role": "bot",
                "text": (
                    "👋 **Hello! I'm your SalaryAI Assistant.**\n\n"
                    "I can answer questions about salaries, the dataset, and the ML model — "
                    "powered entirely by the live data in your project.\n\n"
                    "Type a question below, or tap a suggestion to get started!"
                ),
            }
        ]

    # ── quick-suggest chips ──────────────────────────────────────────────────
    suggestions = [
        "What is the average salary?",
        "Which job title pays the most?",
        "Average salary for a PhD?",
        "Which location pays the most?",
        "What is the gender pay gap?",
        "How many records are in the dataset?",
        "What features does the model use?",
        "What is the salary range?",
    ]

    st.markdown('<p class="section-head">💡 Quick Questions</p>', unsafe_allow_html=True)
    chip_cols = st.columns(4)
    for ci, sug in enumerate(suggestions):
        if chip_cols[ci % 4].button(sug, key=f"va_chip_{ci}", use_container_width=True):
            st.session_state.va_history.append({"role": "user", "text": sug})
            reply = va_response(sug, df, model)
            st.session_state.va_history.append({"role": "bot", "text": reply})

    divider()

    # ── render chat history ──────────────────────────────────────────────────
    st.markdown('<p class="section-head">💬 Conversation</p>', unsafe_allow_html=True)

    # Render each message using st.markdown for proper markdown support
    for msg in st.session_state.va_history:
        role = msg["role"]
        avatar = "🤖" if role == "bot" else "👤"
        bg = "rgba(30,41,59,.85);border:1px solid rgba(99,102,241,.25)" if role == "bot" \
             else "linear-gradient(135deg,rgba(37,99,235,.2),rgba(124,58,237,.15));border:1px solid rgba(99,102,241,.35)"
        radius = "14px 14px 14px 4px" if role == "bot" else "14px 14px 4px 14px"

        col_a, col_b, col_c = st.columns([1, 10, 1])
        if role == "bot":
            with col_a:
                st.markdown(
                    f'<div style="width:34px;height:34px;border-radius:50%;'
                    f'background:linear-gradient(135deg,#2563eb,#7c3aed);'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'font-size:1.1rem;margin-top:.3rem;">{avatar}</div>',
                    unsafe_allow_html=True,
                )
            with col_b:
                st.markdown(
                    f'<div style="background:{bg};border-radius:{radius};'
                    f'padding:.75rem 1rem;font-size:.875rem;line-height:1.6;color:#e2e8f0;">'
                    + msg["text"].replace("\n", "<br>")
                    + "</div>",
                    unsafe_allow_html=True,
                )
        else:
            with col_b:
                st.markdown(
                    f'<div style="background:{bg};border-radius:{radius};'
                    f'padding:.75rem 1rem;font-size:.875rem;line-height:1.6;'
                    f'color:#e2e8f0;text-align:right;">'
                    + msg["text"]
                    + "</div>",
                    unsafe_allow_html=True,
                )
            with col_c:
                st.markdown(
                    f'<div style="width:34px;height:34px;border-radius:50%;'
                    f'background:linear-gradient(135deg,#0f766e,#0891b2);'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'font-size:1.1rem;margin-top:.3rem;">{avatar}</div>',
                    unsafe_allow_html=True,
                )

    divider()

    # ── input box ────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    /* VA input bar — white text, glowing border on focus */
    div[data-testid="stForm"] [data-baseweb="input"] > div {
        background: rgba(15, 23, 42, 0.85) !important;
        border: 2px solid rgba(99,102,241,0.4) !important;
        border-radius: 14px !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    }
    div[data-testid="stForm"] [data-baseweb="input"] > div:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.25), 0 0 18px rgba(99,102,241,0.2) !important;
    }
    div[data-testid="stForm"] [data-baseweb="input"] input {
        color: #ffffff !important;
        caret-color: #a5b4fc !important;
        font-size: 0.97rem !important;
        padding: 0.6rem 1rem !important;
    }
    div[data-testid="stForm"] [data-baseweb="input"] input::placeholder {
        color: #475569 !important;
        font-style: italic;
    }
    div[data-testid="stForm"] [data-baseweb="input"] input:focus::placeholder {
        color: #6366f1 !important;
        opacity: 0.6;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.form("va_form", clear_on_submit=True):
        user_q = st.text_input(
            "Your question",
            placeholder="💬 Ask me anything about salaries…",
            label_visibility="collapsed",
        )
        sent = st.form_submit_button("Send ➤", use_container_width=True, type="primary")

    if sent and user_q.strip():
        st.session_state.va_history.append({"role": "user", "text": user_q.strip()})
        with st.spinner("Thinking…"):
            reply = va_response(user_q.strip(), df, model)
        st.session_state.va_history.append({"role": "bot", "text": reply})
        st.rerun()

    # ── clear chat ────────────────────────────────────────────────────────────
    col_clr, _ = st.columns([1, 5])
    if col_clr.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.va_history = [
            {
                "role": "bot",
                "text": "🗑 Chat cleared. Ask me anything!",
            }
        ]
        st.rerun()

    st.markdown("""<div class="info-box" style="margin-top:.9rem;">
        ℹ️ The assistant answers from the <strong>live dataset</strong> — no external API needed.
        For ML predictions, go to <strong>💰 Salary Prediction</strong>.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════
elif page == "About":
    hero("About SalaryAI",
         "Real-world problems addressed, technology stack, and limitations", "ℹ️")

    ca,cb = st.columns(2, gap="large")
    with ca:
        st.markdown('<p class="section-head">🌍 Real-World Problems</p>', unsafe_allow_html=True)
        for i,(title,desc) in enumerate([
            ("Salary Estimation",    "Help employees and job seekers estimate expected salary from a profile."),
            ("Salary Benchmarking",  "Compare current compensation against a model-based estimate."),
            ("Career Exploration",   "Explore how different career choices appear in the dataset."),
            ("Salary Trend Analysis","Understand salary patterns across education, location, and role."),
            ("Compensation Review",  "Identify records with large deviations for HR review — not as proof of error."),
            ("Data-Driven Decisions","Provide a decision-support layer before HR or compensation discussions."),
        ], 1):
            st.markdown(f"""<div class="stat-card" style="text-align:left;padding:.8rem 1rem;margin-bottom:.45rem;">
                <strong style="color:#a5b4fc;">{i}. {title}</strong><br>
                <span style="color:#64748b;font-size:.8rem;">{desc}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="warn-box" style="margin-top:.9rem;">
            ⚠️ SalaryAI is a <strong>decision-support tool</strong>.
            It is not a substitute for HR professionals, compensation consultants,
            or formal salary surveys.
        </div>""", unsafe_allow_html=True)

    with cb:
        st.markdown('<p class="section-head">🛠️ Technology Stack</p>', unsafe_allow_html=True)
        tech = {"Component":["Data Processing","Machine Learning","Web App",
                              "Visualisation","Model Persistence"],
                "Library":  ["Pandas, NumPy","Scikit-learn (Pipeline + ColumnTransformer)",
                              "Streamlit","Matplotlib","Joblib"]}
        st.table(pd.DataFrame(tech))

        st.markdown('<p class="section-head" style="margin-top:1.2rem;">🤖 ML Models</p>', unsafe_allow_html=True)
        for name,desc in [("Linear Regression","Baseline; highest R² on this dataset"),
                           ("Decision Tree","Non-linear tree model"),
                           ("Random Forest","Ensemble bagging model"),
                           ("Gradient Boosting","Ensemble boosting model")]:
            st.markdown(f"""<div class="stat-card" style="text-align:left;padding:.7rem .9rem;margin-bottom:.4rem;">
                <strong style="color:#a5b4fc;">{name}</strong>
                <span style="color:#64748b;font-size:.8rem;"> — {desc}</span>
            </div>""", unsafe_allow_html=True)

    divider()
    st.markdown('<p class="section-head">⚠️ Limitations</p>', unsafe_allow_html=True)
    st.markdown("""<div class="danger-box">
        The model and this application have important limitations:<br><br>
        • Salary depends on <strong>company size, industry, specific skills, certifications,
          performance reviews, and benefits</strong> — none of which are in this dataset.<br>
        • Predictions reflect patterns in <em>this dataset only</em> — they may not represent
          current market conditions.<br>
        • The salary range shown is an <strong>empirical approximation</strong>, not a formal
          statistical confidence interval.<br>
        • A "review flag" means the model residual is large — it does <strong>NOT</strong>
          mean the salary is wrong, unfair, or fraudulent.<br>
        • Feature importance shows model reliance on variables —
          it does <strong>NOT</strong> establish causation.<br>
        • This tool is for <strong>educational and prototype purposes</strong>.
          Do not use it as the sole basis for compensation decisions.
    </div>""", unsafe_allow_html=True)
    divider()
    st.markdown('<p class="section-head">⚡ Quick Start</p>', unsafe_allow_html=True)
    st.code("""cd "C:\\Users\\Nikita\\Desktop\\IBM Project\\MyBotProject"
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python train_model.py      # train once
streamlit run app.py       # launch app""", language="bash")
