"""
🏎️ Motor Insurance Pricing Dashboard
=====================================
A comprehensive pricing actuary toolkit demonstrating:
- Portfolio analysis & segmentation
- GLM frequency modeling with relativities
- Pure Premium calculation (Frequency × Severity)
- GLM vs XGBoost benchmarking with SHAP interpretability
- Model monitoring & drift detection
- Commercial pricing, elasticity & competitive positioning

Built by François — Actuarial Data Scientist
Dataset: freMTPL2freq + freMTPL2sev (French Motor Third-Party Liability)
"""

import streamlit as st
import warnings

from src.config import CSS_STYLES

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Motor Insurance Pricing Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CSS_STYLES, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL CACHING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading freMTPL2freq + freMTPL2sev datasets...")
def load_cached_data():
    from src.data_loader import load_data
    return load_data()


@st.cache_resource(show_spinner="Preparing modeling features...")
def get_cached_modeling_data(_df):
    from src.data_loader import get_modeling_data
    return get_modeling_data(_df)


@st.cache_resource(show_spinner="Training GLM (Frequency + Severity) & XGBoost models...")
def run_cached_models(_X, _y, _w, _cc, _df_model):
    from src.models import run_models
    return run_models(_X, _y, _w, _cc, _df_model)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏎️ Pricing Dashboard")
    st.markdown("---")

    page = st.radio(
        "**Navigation**",
        [
            "📊 Portfolio Overview",
            "🎯 GLM Pricing Model",
            "💰 Pure Premium",
            "🤖 GLM vs XGBoost",
            "📈 Model Monitoring",
            "💼 Commercial Pricing & Elasticity",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("""
    **Dataset:** freMTPL2freq + sev
    *French Motor TPL Insurance*
    **Period:** 2003 – 2005

    **Models:**
    - Poisson GLM (frequency)
    - Gamma GLM (severity)
    - XGBoost (Poisson objective)

    **Pure Premium:**
    Frequency × Severity

    **Commercial Pricing:**
    Elasticity, Waterfall, Benchmarking

    ---
    **Built by François**
    *Actuarial Data Scientist*
    *Pricing & Machine Learning*
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

df = load_cached_data()

# ── About section ─────────────────────────────────────────────────────────────

with st.expander("ℹ️ About This Application", expanded=False):
    _n_policies = f"{len(df):,.0f}".replace(",", " ")
    _n_claims = f"{int(df['ClaimNb'].sum()):,.0f}".replace(",", " ")
    _n_with_sev = f"{int((df['TotalClaimAmount'] > 0).sum()):,.0f}".replace(",", " ")
    st.markdown(f"""
    ### 📋 Dataset
    This dashboard uses the **freMTPL2freq** and **freMTPL2sev** datasets — standard actuarial
    benchmarks for motor insurance pricing.

    | | |
    |---|---|
    | **Source** | OpenML (datasets 41214 + 41215) |
    | **Geography** | France |
    | **Period** | 2003 – 2005 |
    | **Policies** | {_n_policies} |
    | **Total claims** | {_n_claims} |
    | **Policies with severity data** | {_n_with_sev} |

    ---
    ### 🎯 Objective
    An **end-to-end pricing actuary toolkit** covering the complete motor insurance pricing workflow:
    1. **📊 Portfolio Analysis** — Risk distribution, segmentation, exposure concentration, and claims trend
    2. **🎯 GLM Modeling** — Poisson GLM (frequency) + Gamma GLM (severity) with relativities
    3. **💰 Pure Premium** — Frequency × Severity decomposition and severity distribution analysis
    4. **🤖 ML Benchmarking** — GLM vs XGBoost comparison with SHAP interpretability
    5. **📈 Model Monitoring** — A/E ratios, PSI drift detection, residual analysis
    6. **💼 Commercial Pricing** — Pricing waterfall, price elasticity, competitive positioning, data quality

    ---
    ### ⚙️ Methodology Note
    **Portfolio analysis** (tab 1) uses the **full dataset** ({_n_policies} policies).
    **Modeling** (tabs 2–5) uses a **stratified sub-sample of 100 000 policies** to respect
    cloud memory constraints. All claims are retained; only non-claim policies are sampled.
    This preserves the statistical validity of GLM coefficients and relativities.
    """)

# ── Dismissible navigation hint ──────────────────────────────────────────────

if "show_nav_hint" not in st.session_state:
    st.session_state.show_nav_hint = True

if st.session_state.show_nav_hint:
    hint_col, close_col = st.columns([11, 1])
    with hint_col:
        st.markdown(
            '<div class="sidebar-hint">'
            '⬅️ <strong>Navigation menu:</strong> click the <code>&gt;</code> arrow at the top left '
            'to open the side panel and navigate between tabs.'
            '</div>',
            unsafe_allow_html=True,
        )
    with close_col:
        if st.button("✕", key="dismiss_nav_hint", help="Dismiss this message"):
            st.session_state.show_nav_hint = False
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📊 Portfolio Overview":
    from src.pages.portfolio import render
    render(df)

elif page == "🎯 GLM Pricing Model":
    X, y, w, claim_count, df_model = get_cached_modeling_data(df)
    results = run_cached_models(X, y, w, claim_count, df_model)
    from src.pages.glm_model import render
    render(df, X, y, w, claim_count, df_model, results)

elif page == "💰 Pure Premium":
    X, y, w, claim_count, df_model = get_cached_modeling_data(df)
    results = run_cached_models(X, y, w, claim_count, df_model)
    from src.pages.pure_premium import render
    render(df, X, y, w, claim_count, df_model, results)

elif page == "🤖 GLM vs XGBoost":
    X, y, w, claim_count, df_model = get_cached_modeling_data(df)
    results = run_cached_models(X, y, w, claim_count, df_model)
    from src.pages.comparison import render
    render(df, X, y, w, claim_count, df_model, results)

elif page == "📈 Model Monitoring":
    X, y, w, claim_count, df_model = get_cached_modeling_data(df)
    results = run_cached_models(X, y, w, claim_count, df_model)
    from src.pages.monitoring import render
    render(df, X, y, w, claim_count, df_model, results)

elif page == "💼 Commercial Pricing & Elasticity":
    X, y, w, claim_count, df_model = get_cached_modeling_data(df)
    results = run_cached_models(X, y, w, claim_count, df_model)
    from src.pages.commercial import render
    render(df, X, y, w, claim_count, df_model, results)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
    <strong>Motor Insurance Pricing Dashboard</strong> — Built by François | Actuarial Data Scientist<br>
    Dataset: freMTPL2freq + freMTPL2sev (French Motor TPL) | Models: Poisson GLM (freq) + Gamma GLM (sev) + XGBoost<br>
    <em>End-to-end pricing actuary capabilities: EDA → Freq/Sev Modeling → Pure Premium → Elasticity → Commercial Pricing → Monitoring</em>
</div>
""", unsafe_allow_html=True)
