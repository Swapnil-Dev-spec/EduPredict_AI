import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore") 
# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────

conn = sqlite3.connect(
    "student_predictions.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_name TEXT,
    student_id TEXT,
    study_hours REAL,
    attendance REAL,
    sleep_hours REAL,
    previous_marks REAL,
    predicted_marks REAL,
    grade TEXT
)
""")

conn.commit()

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EduPredict | AI Performance System",
    page_icon="assets/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS — DARK PROFESSIONAL THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e2e8f0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #13161e;
    border-right: 1px solid #1e2230;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Cards ── */
.card {
    background: #13161e;
    border: 1px solid #1e2230;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
}
.card-accent {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border: 1px solid #2a4a5e;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
}

/* ── Metric tiles ── */
.metric-tile {
    background: #13161e;
    border: 1px solid #1e2230;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-top: 0.3rem;
}

/* ── Section headings ── */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2230;
}

/* ── Grade badge ── */
.grade-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 3.5rem;
    font-weight: 700;
    color: #ffffff;
    padding: 0.3rem 1rem;
    border-radius: 8px;
}
.grade-a  { background: linear-gradient(135deg, #064e3b, #059669); border: 1px solid #10b981; }
.grade-b  { background: linear-gradient(135deg, #1e3a5f, #1d4ed8); border: 1px solid #3b82f6; }
.grade-c  { background: linear-gradient(135deg, #3b2a06, #b45309); border: 1px solid #f59e0b; }
.grade-f  { background: linear-gradient(135deg, #450a0a, #b91c1c); border: 1px solid #ef4444; }

/* ── Progress bar override ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #0ea5e9, #38bdf8);
    border-radius: 99px;
}
.stProgress > div > div > div {
    background: #1e2230;
    border-radius: 99px;
}

/* ── Inputs & sliders ── */
.stSlider [data-baseweb="slider"] { padding: 0.3rem 0; }
.stTextInput input, .stSelectbox select {
    background: #1e2230 !important;
    border: 1px solid #2d3548 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── Buttons ── */
.stButton button {
    background: linear-gradient(135deg, #0369a1, #0ea5e9) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.85 !important; }

/* ── Divider ── */
hr { border-color: #1e2230 !important; }

/* ── Recommendation pills ── */
.rec-pill {
    display: inline-block;
    background: #1e2230;
    border: 1px solid #2d3548;
    border-radius: 99px;
    padding: 0.35rem 0.9rem;
    font-size: 0.82rem;
    margin: 0.25rem 0.2rem;
    color: #94a3b8;
}
.rec-pill.warn {
    border-color: #b45309;
    color: #fcd34d;
    background: #1c1505;
}
.rec-pill.good {
    border-color: #059669;
    color: #6ee7b7;
    background: #021a0f;
}

/* ── Login wrapper ── */
.login-wrapper {
    max-width: 400px;
    margin: 6rem auto;
    background: #13161e;
    border: 1px solid #1e2230;
    border-radius: 16px;
    padding: 2.5rem;
}
.login-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #38bdf8;
    text-align: center;
    margin-bottom: 0.3rem;
}
.login-sub {
    text-align: center;
    font-size: 0.82rem;
    color: #475569;
    margin-bottom: 1.8rem;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    background: #13161e;
    border-radius: 10px;
    padding: 0.3rem;
    gap: 0.2rem;
    border: 1px solid #1e2230;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748b;
    font-weight: 500;
    padding: 0.45rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: #1e2230 !important;
    color: #38bdf8 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────
#  SAMPLE DATA GENERATOR  (no CSV needed)
# ─────────────────────────────────────────────
@st.cache_data
def generate_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    study   = rng.uniform(1, 12, n)
    attend  = rng.uniform(40, 100, n)
    sleep   = rng.uniform(4, 10, n)
    prev    = rng.uniform(30, 100, n)
    noise   = rng.normal(0, 4, n)
    marks   = (
        study * 2.5
        + attend * 0.25
        + sleep * 1.2
        + prev * 0.35
        + noise
    ).clip(0, 100)
    return pd.DataFrame({
        "study_hours":    np.round(study, 1),
        "attendance":     np.round(attend, 1),
        "sleep_hours":    np.round(sleep, 1),
        "previous_marks": np.round(prev, 1),
        "final_marks":    np.round(marks, 1),
    })

# ─────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_resource
def train_models(data):
    X = data[["study_hours", "attendance", "sleep_hours", "previous_marks"]]
    y = data["final_marks"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression().fit(X_tr, y_tr)
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)

    metrics = {}
    for name, m in [("Linear Regression", lr), ("Random Forest", rf)]:
        pred = m.predict(X_te)
        metrics[name] = {
            "MAE":  round(mean_absolute_error(y_te, pred), 2),
            "R2":   round(r2_score(y_te, pred), 3),
        }
    return lr, rf, metrics

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def save_prediction(
    student_name,
    student_id,
    study_hours,
    attendance,
    sleep_hours,
    previous_marks,
    predicted_marks,
    grade
):

    cursor.execute("""
    INSERT INTO predictions (
        student_name,
        student_id,
        study_hours,
        attendance,
        sleep_hours,
        previous_marks,
        predicted_marks,
        grade
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        student_name,
        student_id,
        study_hours,
        attendance,
        sleep_hours,
        previous_marks,
        predicted_marks,
        grade
    ))

    conn.commit()
def grade_info(marks):
    if marks >= 85: return "A", "Exceptional", "grade-a"
    if marks >= 70: return "B", "Proficient",  "grade-b"
    if marks >= 50: return "C", "Developing",  "grade-c"
    return "F", "At Risk", "grade-f"

def set_mpl_dark():
    plt.rcParams.update({
        "figure.facecolor":  "#13161e",
        "axes.facecolor":    "#13161e",
        "axes.edgecolor":    "#1e2230",
        "axes.labelcolor":   "#94a3b8",
        "xtick.color":       "#64748b",
        "ytick.color":       "#64748b",
        "grid.color":        "#1e2230",
        "text.color":        "#e2e8f0",
        "axes.titlecolor":   "#e2e8f0",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

# ─────────────────────────────────────────────
#  LOGIN PAGE
# ─────────────────────────────────────────────
if not st.session_state.logged_in:
    st.markdown("""
        <div class="login-wrapper">
            <div class="login-title">EduPredict</div>
            <div class="login-sub">AI-Powered Student Performance System</div>
        </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        if st.button("Sign In", use_container_width=True):
            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials. Try admin / 1234")
    st.stop()

# ─────────────────────────────────────────────
#  DATA & MODELS
# ─────────────────────────────────────────────
data = generate_data()
lr_model, rf_model, model_metrics = train_models(data)
set_mpl_dark()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">EduPredict</div>', unsafe_allow_html=True)
    st.caption("AI Student Performance System v2.0")
    st.divider()

    st.markdown('<div class="section-title">Model Selection</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Algorithm",
        ["Linear Regression", "Random Forest"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    m = model_metrics[model_choice]
    st.markdown(f"""
        <div class="metric-tile" style="margin-bottom:0.6rem">
            <div class="metric-value">{m['R2']}</div>
            <div class="metric-label">R² Score</div>
        </div>
        <div class="metric-tile">
            <div class="metric-value">{m['MAE']}</div>
            <div class="metric-label">Mean Abs Error</div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-title">Session</div>', unsafe_allow_html=True)
    st.caption(f"Predictions made: {len(st.session_state.history)}")
    if st.button("Sign Out", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# ─────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
    <div style="margin-bottom: 2rem;">
        <div style="font-family:'Space Mono',monospace; font-size:0.65rem;
                    letter-spacing:0.2em; text-transform:uppercase;
                    color:#38bdf8; margin-bottom:0.4rem;">
            AI Performance Intelligence
        </div>
        <h1 style="font-size:2.2rem; font-weight:700; margin:0; color:#f1f5f9;">
            Student Performance Predictor
        </h1>
        <p style="color:#475569; margin-top:0.4rem; font-size:0.9rem;">
            Enter student parameters below to generate an AI-powered academic forecast.
        </p>
    </div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Prediction", "Analytics", "Dataset"])

# ═══════════════════════════════════════════
#  TAB 1 — PREDICTION
# ═══════════════════════════════════════════
with tab1:

    st.markdown(
        '<div class="section-title">Student Parameters</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        student_name = st.text_input(
            "Student Name",
            placeholder="Full name"
        )

        student_id = st.text_input(
            "Student ID",
            placeholder="e.g. STU-2024-001"
        )

    with col2:
        study_hours = st.slider(
            "Study Hours / Day",
            0, 12, 6
        )

        attendance = st.slider(
            "Attendance (%)",
            0, 100, 80
        )

    with col3:
        sleep_hours = st.slider(
            "Sleep Hours / Day",
            0, 12, 7
        )

        previous_marks = st.slider(
            "Previous Exam Marks",
            0, 100, 65
        )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_btn = st.button(
        "Generate Forecast",
        use_container_width=False
    )

    if predict_btn:

        with st.spinner("Running predictive analysis..."):

            model = (
                lr_model
                if model_choice == "Linear Regression"
                else rf_model
            )

            raw = model.predict([[
                study_hours,
                attendance,
                sleep_hours,
                previous_marks
            ]])[0]

            predicted = round(
                float(np.clip(raw, 0, 100)),
                1
            )

            grade, level, badge_cls = grade_info(predicted)
        # ── Store history ──
        st.session_state.history.append({
            "Name": student_name or "Unknown",
            "ID":   student_id   or "—",
            "Study Hrs": study_hours,
            "Attendance": attendance,
            "Sleep Hrs":  sleep_hours,
            "Prev Marks": previous_marks,
            "Predicted":  predicted,
            "Grade":      grade,
        })

        save_prediction(
    student_name,
    student_id,
    study_hours,
    attendance,
    sleep_hours,
    previous_marks,
    predicted,
    grade
)

        st.divider()

        # ── Result header ──
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-value">{predicted}</div>
                    <div class="metric-label">Predicted Score</div>
                </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-value" style="font-size:2.4rem;">{grade}</div>
                    <div class="metric-label">Letter Grade</div>
                </div>
            """, unsafe_allow_html=True)
        with r3:
            status = "PASS" if predicted >= 40 else "FAIL"
            s_color = "#10b981" if status == "PASS" else "#ef4444"
            st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-value" style="color:{s_color}; font-size:1.6rem;">{status}</div>
                    <div class="metric-label">Result</div>
                </div>
            """, unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-value" style="font-size:1.3rem; color:#a78bfa;">{level}</div>
                    <div class="metric-label">Performance Tier</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Score bar ──
        st.markdown('<div class="section-title">Score Breakdown</div>', unsafe_allow_html=True)
        st.progress(int(predicted))
        st.caption(f"{predicted} / 100")

       
     # ── Feature contribution chart ──
st.markdown('<div class="section-title">Feature Influence</div>', unsafe_allow_html=True)

if model_choice == "Linear Regression":
            coefs = lr_model.coef_
            labels = ["Study Hours", "Attendance", "Sleep Hours", "Prev Marks"]
            values = [round(c * v, 2) for c, v in zip(coefs, [study_hours, attendance, sleep_hours, previous_marks])]
            colors = ["#38bdf8" if v >= 0 else "#f87171" for v in values]

            fig_bar, ax_bar = plt.subplots(figsize=(6, 2.5))
            bars = ax_bar.barh(labels, values, color=colors, height=0.5)
            ax_bar.axvline(0, color="#2d3548", linewidth=1)
            ax_bar.set_xlabel("Score Contribution")
            ax_bar.grid(axis="x", linestyle="--", alpha=0.3)
            fig_bar.tight_layout()
            st.pyplot(fig_bar)
else:
            importances = rf_model.feature_importances_
            labels = ["Study Hours", "Attendance", "Sleep Hours", "Prev Marks"]
            fig_imp, ax_imp = plt.subplots(figsize=(6, 2.5))
            ax_imp.barh(labels, importances, color="#38bdf8", height=0.5)
            ax_imp.set_xlabel("Importance Score")
            ax_imp.grid(axis="x", linestyle="--", alpha=0.3)
            fig_imp.tight_layout()
            st.pyplot(fig_imp)

        
        
        # ── Recommendations ──
st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)
recs = []
if study_hours   < 4:  recs.append(("warn", "Increase daily study hours to at least 4–6 hrs"))
if attendance    < 75: recs.append(("warn", "Attendance is below 75% — attend more classes"))
if sleep_hours   < 6:  recs.append(("warn", "Sleep deprivation impacts retention — aim for 7–8 hrs"))
if previous_marks< 50: recs.append(("warn", "Revisit previous topics and practice past papers"))

if not recs:
            st.markdown('<span class="rec-pill good">Strong academic habits — maintain consistency</span>', unsafe_allow_html=True)
else:
            html_recs = "".join(f'<span class="rec-pill warn">{r}</span>' for _, r in recs)
            st.markdown(html_recs, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
    
   

   # ── Database History ──

st.divider()

st.markdown(
    '<div class="section-title">Database Prediction History</div>',
    unsafe_allow_html=True
)

db_data = pd.read_sql_query(
    "SELECT * FROM predictions",
    conn
)

st.dataframe(
    db_data,
    use_container_width=True,
    hide_index=True
)

csv = db_data.to_csv(index=False).encode("utf-8")

st.download_button(
    "Export Database CSV",
    data=csv,
    file_name="database_predictions.csv",
    mime="text/csv"
)

# ═══════════════════════════════════════════
#  TAB 2 — ANALYTICS
# ═══════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

    o1, o2, o3, o4 = st.columns(4)
    stats = [
        ("Total Students",   len(data),                         ""),
        ("Avg Final Score",  round(data["final_marks"].mean(),1),"/ 100"),
        ("Avg Study Hours",  round(data["study_hours"].mean(),1),"hrs/day"),
        ("Avg Attendance",   round(data["attendance"].mean(),1), "%"),
    ]
    for col, (label, val, unit) in zip([o1, o2, o3, o4], stats):
        col.markdown(f"""
            <div class="metric-tile">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label} {unit}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Study Hours vs Final Marks</div>', unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        sc = ax1.scatter(data["study_hours"], data["final_marks"],
                         c=data["final_marks"], cmap="cool", alpha=0.6, s=20)
        m_lr, b_lr = np.polyfit(data["study_hours"], data["final_marks"], 1)
        x_line = np.linspace(data["study_hours"].min(), data["study_hours"].max(), 100)
        ax1.plot(x_line, m_lr * x_line + b_lr, color="#38bdf8", linewidth=1.5, label="Trend")
        ax1.set_xlabel("Study Hours")
        ax1.set_ylabel("Final Marks")
        ax1.legend(fontsize=8)
        ax1.grid(True, linestyle="--", alpha=0.2)
        fig1.tight_layout()
        st.pyplot(fig1)

    with c2:
        st.markdown('<div class="section-title">Attendance vs Final Marks</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.scatter(data["attendance"], data["final_marks"],
                    c=data["attendance"], cmap="plasma", alpha=0.6, s=20)
        m2, b2 = np.polyfit(data["attendance"], data["final_marks"], 1)
        x2 = np.linspace(data["attendance"].min(), data["attendance"].max(), 100)
        ax2.plot(x2, m2 * x2 + b2, color="#a78bfa", linewidth=1.5, label="Trend")
        ax2.set_xlabel("Attendance (%)")
        ax2.set_ylabel("Final Marks")
        ax2.legend(fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.2)
        fig2.tight_layout()
        st.pyplot(fig2)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-title">Sleep Hours Distribution</div>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.hist(data["sleep_hours"], bins=15, color="#0ea5e9", edgecolor="#0d0f14", alpha=0.85)
        ax3.set_xlabel("Sleep Hours")
        ax3.set_ylabel("Count")
        ax3.grid(True, linestyle="--", alpha=0.2)
        fig3.tight_layout()
        st.pyplot(fig3)

    with c4:
        st.markdown('<div class="section-title">Grade Distribution</div>', unsafe_allow_html=True)
        bins   = [0, 40, 50, 70, 85, 101]
        labels = ["F (<40)", "C (40-50)", "C+ (50-70)", "B (70-85)", "A (85+)"]
        colors = ["#ef4444", "#f97316", "#facc15", "#3b82f6", "#10b981"]
        counts = pd.cut(data["final_marks"], bins=bins, labels=labels, right=False).value_counts()
        counts = counts.reindex(labels)

        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        bars = ax4.bar(labels, counts.values, color=colors, edgecolor="#0d0f14", width=0.6)
        ax4.set_ylabel("Number of Students")
        ax4.set_xticklabels(labels, fontsize=7.5)
        ax4.grid(axis="y", linestyle="--", alpha=0.2)
        for bar, val in zip(bars, counts.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     str(val), ha="center", va="bottom", fontsize=8, color="#94a3b8")
        fig4.tight_layout()
        st.pyplot(fig4)

    # ── Correlation heatmap ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Feature Correlation Matrix</div>', unsafe_allow_html=True)
    corr = data.corr()
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    im = ax5.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax5.set_xticks(range(len(corr.columns)))
    ax5.set_yticks(range(len(corr.columns)))
    ax5.set_xticklabels(corr.columns, rotation=35, ha="right", fontsize=8)
    ax5.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax5.text(j, i, f"{corr.values[i,j]:.2f}",
                     ha="center", va="center", fontsize=7.5,
                     color="white" if abs(corr.values[i,j]) > 0.5 else "#94a3b8")
    plt.colorbar(im, ax=ax5, fraction=0.03)
    fig5.tight_layout()
    st.pyplot(fig5)

    # ── Model comparison ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
    comp_df = pd.DataFrame(model_metrics).T.reset_index()
    comp_df.columns = ["Model", "MAE (lower is better)", "R² Score (higher is better)"]
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════
#  TAB 3 — DATASET
# ═══════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Raw Dataset</div>', unsafe_allow_html=True)
    st.caption(f"{len(data)} records — auto-generated synthetic student data")

    search = st.text_input("Filter rows (min study hours)", placeholder="e.g. 6")
    filtered = data.copy()
    if search.strip():
        try:
            filtered = data[data["study_hours"] >= float(search)]
        except ValueError:
            pass

    st.dataframe(filtered, use_container_width=True, hide_index=True)

    csv_raw = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Export Dataset as CSV", data=csv_raw,
                       file_name="student_data.csv", mime="text/csv")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
    <div style="text-align:center; padding:2rem 0 1rem; color:#334155;
                font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase;">
        EduPredict &nbsp;|&nbsp; Python &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Scikit-learn
    </div>
""", unsafe_allow_html=True)