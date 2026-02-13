"""
🏎️ Motor Insurance Pricing Dashboard
=====================================
A comprehensive pricing actuary toolkit demonstrating:
- Portfolio analysis & segmentation
- GLM frequency modeling with relativities
- Pure Premium calculation (Frequency × Severity)
- GLM vs XGBoost benchmarking with SHAP interpretability
- Model monitoring & drift detection

Built by François — Actuarial Data Scientist
Dataset: freMTPL2freq + freMTPL2sev (French Motor Third-Party Liability)
"""

import streamlit as st
import warnings

from src.config import PAGES
from src.styles import inject_css
from src.pages import portfolio, glm_model, pure_premium, model_comparison, monitoring

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLES
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Motor Insurance Pricing Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

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

    page = st.radio("**Navigation**", PAGES, index=0)

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

    ---
    **Built by François**
    *Actuarial Data Scientist*
    *Pricing & Machine Learning*
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

df = load_cached_data()

# ═══════════════════════════════════════════════════════════════════════════════
# ABOUT SECTION (COLLAPSIBLE)
# ═══════════════════════════════════════════════════════════════════════════════

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

    **Key variables:** Driver age, Bonus-Malus coefficient, Vehicle power/age/brand/fuel type,
    Area (population density), Region, Exposure (policy duration in years).

    ---

    ### 🎯 Objective

    An **end-to-end pricing actuary toolkit** covering the complete motor insurance pricing workflow:

    1. **📊 Portfolio Analysis** — Risk distribution, segmentation, and exposure concentration
    2. **🎯 GLM Modeling** — Poisson GLM (frequency) + Gamma GLM (severity) with relativities
    3. **💰 Pure Premium** — Frequency × Severity decomposition and severity distribution analysis
    4. **🤖 ML Benchmarking** — GLM vs XGBoost comparison with SHAP interpretability
    5. **📈 Model Monitoring** — A/E ratios, PSI drift detection, residual analysis
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# DISMISSIBLE NAVIGATION HINT
# ═══════════════════════════════════════════════════════════════════════════════

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

if page == PAGES[0]:
    portfolio.render(df)
elif page == PAGES[1]:
    glm_model.render(df, get_cached_modeling_data, run_cached_models)
elif page == PAGES[2]:
    pure_premium.render(df, get_cached_modeling_data, run_cached_models)
elif page == PAGES[3]:
    model_comparison.render(df, get_cached_modeling_data, run_cached_models)
elif page == PAGES[4]:
    monitoring.render(df, get_cached_modeling_data, run_cached_models)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
    <strong>Motor Insurance Pricing Dashboard</strong> — Built by François | Actuarial Data Scientist<br>
    Dataset: freMTPL2freq + freMTPL2sev (French Motor TPL) | Models: Poisson GLM (freq) + Gamma GLM (sev) + XGBoost<br>
    <em>End-to-end pricing actuary capabilities: EDA → Freq/Sev Modeling → Pure Premium → Monitoring</em>
</div>
""", unsafe_allow_html=True)
