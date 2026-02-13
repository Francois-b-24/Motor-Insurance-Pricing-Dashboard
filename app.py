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
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Motor Insurance Pricing Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .kpi-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5986 100%);
        border-radius: 12px; padding: 20px 24px; text-align: center;
        color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 10px;
    }
    .kpi-card h3 { margin:0; font-size:14px; font-weight:400; opacity:0.85;
                    text-transform:uppercase; letter-spacing:0.5px; }
    .kpi-card h1 { margin:8px 0 0 0; font-size:32px; font-weight:700; }
    .kpi-card-green {
        background: linear-gradient(135deg, #1a7a4c 0%, #2ecc71 100%);
        border-radius: 12px; padding: 20px 24px; text-align: center;
        color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 10px;
    }
    .kpi-card-green h3 { margin:0; font-size:14px; font-weight:400; opacity:0.85;
                          text-transform:uppercase; letter-spacing:0.5px; }
    .kpi-card-green h1 { margin:8px 0 0 0; font-size:32px; font-weight:700; }
    .section-header {
        background: linear-gradient(90deg, #1e3a5f, #3d7bc7);
        color: white; padding: 10px 20px; border-radius: 8px;
        margin: 20px 0 15px 0; font-size: 18px; font-weight: 600;
    }
    .sidebar-hint {
        background: #f0f2f6; border-radius: 8px; padding: 10px 16px;
        margin-bottom: 12px; font-size: 14px; color: #555;
        border-left: 4px solid #3d7bc7;
    }
    .footer {
        text-align:center; padding:20px; color:#888; font-size:13px;
        border-top:1px solid #eee; margin-top:30px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


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
        ["📊 Portfolio Overview",
         "🎯 GLM Pricing Model",
         "💰 Pure Premium",
         "🤖 GLM vs XGBoost",
         "📈 Model Monitoring"],
        index=0
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
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_number(value, decimals=0, prefix="", suffix=""):
    """Format a number with spaces as thousand separators."""
    if decimals == 0:
        formatted = f"{value:,.0f}".replace(",", " ")
    else:
        formatted = f"{value:,.{decimals}f}".replace(",", " ")
    return f"{prefix}{formatted}{suffix}"


def kpi_card(title, value, green=False):
    css_class = "kpi-card-green" if green else "kpi-card"
    st.markdown(f"""
    <div class="{css_class}">
        <h3>{title}</h3>
        <h1>{value}</h1>
    </div>
    """, unsafe_allow_html=True)


def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


COLORS = {
    "primary": "#1e3a5f",
    "secondary": "#3d7bc7",
    "accent": "#e74c3c",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "palette": ["#1e3a5f", "#3d7bc7", "#5ba3e6", "#8ec3f5", "#c4dff6",
                 "#e74c3c", "#f39c12", "#2ecc71"]
}

SEGMENT_LABELS = {
    "DrivAge_bin": "Driver Age",
    "VehAge_bin": "Vehicle Age",
    "VehPower_bin": "Vehicle Power",
    "BonusMalus_bin": "Bonus-Malus",
    "Area": "Area (Density)",
    "VehGas": "Fuel Type",
}


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
    1. **📊 Portfolio Analysis** — Risk distribution, segmentation, and exposure concentration
    2. **🎯 GLM Modeling** — Poisson GLM (frequency) + Gamma GLM (severity) with relativities
    3. **💰 Pure Premium** — Frequency × Severity decomposition and severity distribution analysis
    4. **🤖 ML Benchmarking** — GLM vs XGBoost comparison with SHAP interpretability
    5. **📈 Model Monitoring** — A/E ratios, PSI drift detection, residual analysis

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
# PAGE 1 — PORTFOLIO OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📊 Portfolio Overview":

    st.title("📊 Portfolio Overview")
    st.markdown("*Explore the structure, risk distribution, and segmentation of the motor insurance portfolio.*")

    # --- KPI Row ---
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        kpi_card("Policies", fmt_number(len(df)))
    with col2:
        kpi_card("Total Exposure", fmt_number(df['Exposure'].sum(), suffix="PY"))
    with col3:
        kpi_card("Total Claims", fmt_number(df['ClaimNb'].sum()))
    with col4:
        kpi_card("Avg Frequency", f"{df['ClaimNb'].sum() / df['Exposure'].sum():.4f}")
    with col5:
        avg_pp = df["TotalClaimAmount"].sum() / df["Exposure"].sum()
        kpi_card("Avg Pure Premium", fmt_number(avg_pp, prefix="€"))
    with col6:
        # Average severity (Loss Ratio requires model predictions, shown in Pure Premium page)
        avg_sev = df.loc[df["ClaimNb"] > 0, "TotalClaimAmount"].sum() / df.loc[df["ClaimNb"] > 0, "ClaimNb"].sum() if (df["ClaimNb"] > 0).any() else 0
        kpi_card("Avg Severity", fmt_number(avg_sev, prefix="€"))

    st.markdown("")

    # --- Exposure & Frequency by Segment ---
    section_header("📋 Risk Segmentation Analysis")

    segment_var = st.selectbox(
        "Select segmentation variable:",
        ["DrivAge_bin", "VehAge_bin", "VehPower_bin", "BonusMalus_bin", "Area", "VehGas"],
        format_func=lambda x: SEGMENT_LABELS.get(x, x)
    )

    seg_data = df.groupby(segment_var, observed=True).agg(
        Policies=("ClaimNb", "count"),
        Exposure=("Exposure", "sum"),
        Claims=("ClaimNb", "sum"),
        TotalAmount=("TotalClaimAmount", "sum"),
    ).reset_index()
    seg_data["Frequency"] = seg_data["Claims"] / seg_data["Exposure"]
    seg_data["PurePremium"] = seg_data["TotalAmount"] / seg_data["Exposure"]
    seg_data["Severity"] = np.where(seg_data["Claims"] > 0,
                                     seg_data["TotalAmount"] / seg_data["Claims"], 0)
    seg_data["Exposure_pct"] = seg_data["Exposure"] / seg_data["Exposure"].sum() * 100

    col_left, col_right = st.columns(2)

    with col_left:
        fig_exp = go.Figure()
        fig_exp.add_trace(go.Bar(
            x=seg_data[segment_var].astype(str), y=seg_data["Exposure_pct"],
            marker_color=COLORS["secondary"], name="Exposure %",
            text=seg_data["Exposure_pct"].round(1).astype(str) + "%", textposition="outside"
        ))
        fig_exp.update_layout(
            title="Exposure Distribution (%)",
            xaxis_title=SEGMENT_LABELS.get(segment_var, segment_var),
            yaxis_title="% of Total Exposure",
            template="plotly_white", height=400, showlegend=False,
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig_exp, width='stretch')

    with col_right:
        avg_freq = df["ClaimNb"].sum() / df["Exposure"].sum()
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Bar(
            x=seg_data[segment_var].astype(str), y=seg_data["Frequency"],
            marker_color=[COLORS["accent"] if f > avg_freq else COLORS["success"]
                          for f in seg_data["Frequency"]],
            text=seg_data["Frequency"].round(4).astype(str), textposition="outside"
        ))
        fig_freq.add_hline(y=avg_freq, line_dash="dash", line_color=COLORS["primary"],
                           annotation_text=f"Portfolio avg: {avg_freq:.4f}")
        fig_freq.update_layout(
            title="Claim Frequency by Segment",
            xaxis_title=SEGMENT_LABELS.get(segment_var, segment_var),
            yaxis_title="Frequency (claims/exposure)",
            template="plotly_white", height=400, showlegend=False,
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig_freq, width='stretch')

    # --- Heatmap ---
    section_header("🔥 Risk Heatmap — Driver Age × Bonus-Malus")

    heat_data = df.groupby(["DrivAge_bin", "BonusMalus_bin"], observed=True).agg(
        Exposure=("Exposure", "sum"), Claims=("ClaimNb", "sum")
    ).reset_index()
    heat_data["Frequency"] = heat_data["Claims"] / heat_data["Exposure"]

    heat_pivot = heat_data.pivot_table(
        index="DrivAge_bin", columns="BonusMalus_bin", values="Frequency", aggfunc="mean"
    )

    fig_heat = px.imshow(
        heat_pivot, color_continuous_scale="RdYlGn_r",
        labels=dict(x="Bonus-Malus", y="Driver Age", color="Frequency"),
        text_auto=".4f", aspect="auto"
    )
    fig_heat.update_layout(
        title="Claim Frequency: Driver Age × Bonus-Malus",
        template="plotly_white", height=420,
        xaxis=dict(tickangle=-45, dtick=1),
        yaxis=dict(dtick=1),
    )
    st.plotly_chart(fig_heat, width='stretch')

    # --- BM Distribution ---
    section_header("📉 Bonus-Malus Distribution")

    col_bm1, col_bm2 = st.columns(2)
    with col_bm1:
        fig_bm = px.histogram(df, x="BonusMalus", nbins=50,
                               color_discrete_sequence=[COLORS["secondary"]],
                               title="Bonus-Malus Coefficient Distribution")
        fig_bm.update_layout(
            template="plotly_white", height=350,
            xaxis_title="Bonus-Malus", yaxis_title="Number of Policies",
        )
        st.plotly_chart(fig_bm, width='stretch')

    with col_bm2:
        bm_freq = df.groupby("BonusMalus_bin", observed=True).agg(
            Exposure=("Exposure", "sum"), Claims=("ClaimNb", "sum")
        ).reset_index()
        bm_freq["Frequency"] = bm_freq["Claims"] / bm_freq["Exposure"]
        fig_bm_freq = px.bar(bm_freq, x="BonusMalus_bin", y="Frequency",
                              color="Frequency", color_continuous_scale="Reds",
                              title="Frequency by Bonus-Malus Band")
        fig_bm_freq.update_layout(
            template="plotly_white", height=350,
            xaxis_title="Bonus-Malus", yaxis_title="Frequency",
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig_bm_freq, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — GLM PRICING MODEL
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🎯 GLM Pricing Model":

    st.title("🎯 GLM Pricing Model")
    st.markdown("*Poisson GLM for frequency modeling and Gamma GLM for severity — the industry standards for motor insurance pricing.*")

    X, y, w, claim_count, df_model = get_cached_modeling_data(df)
    results = run_cached_models(X, y, w, claim_count, df_model)
    glm = results["glm_results"]

    st.caption(f"📐 *Model trained on a stratified sample of {len(X):,.0f} policies (all claims retained) from {len(df):,.0f} total policies.*")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Model Type", "Poisson GLM")
    with col2:
        kpi_card("Features", f"{len(X.columns)}")
    with col3:
        kpi_card("Test Gini", f"{results['glm_metrics_test']['Gini']:.3f}")
    with col4:
        kpi_card("Test RMSE", f"{results['glm_metrics_test']['RMSE']:.4f}")

    # --- Frequency Relativities ---
    section_header("📐 GLM Frequency Relativities (exp(β))")

    st.markdown("""
    Relativities show the **multiplicative effect** of each variable on the predicted frequency.
    A relativity > 1 means **higher risk**; < 1 means **lower risk** than the reference level.
    """)

    coef_df = pd.DataFrame({
        "Variable": glm.params.index,
        "Coefficient": glm.params.values,
        "Std Error": glm.bse.values,
        "P-value": glm.pvalues.values,
        "Relativity": np.exp(glm.params.values)
    })
    coef_df["Significant"] = coef_df["P-value"] < 0.05

    key_vars = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity"]
    key_coef = coef_df[coef_df["Variable"].isin(key_vars)].copy()

    if not key_coef.empty:
        fig_rel = go.Figure()
        colors = [COLORS["accent"] if r > 1 else COLORS["success"]
                  for r in key_coef["Relativity"]]
        fig_rel.add_trace(go.Bar(
            x=key_coef["Variable"], y=key_coef["Relativity"],
            marker_color=colors, text=key_coef["Relativity"].round(4), textposition="outside"
        ))
        fig_rel.add_hline(y=1, line_dash="dash", line_color="gray",
                          annotation_text="Reference = 1.0")
        fig_rel.update_layout(title="Relativities — Key Continuous Variables",
                               yaxis_title="Relativity (exp(β))",
                               template="plotly_white", height=400)
        st.plotly_chart(fig_rel, width='stretch')

    # Area relativities
    area_coefs = coef_df[coef_df["Variable"].str.startswith("Area_")].copy()
    if not area_coefs.empty:
        area_coefs["Area"] = area_coefs["Variable"].str.replace("Area_", "")
        ref_row = pd.DataFrame({
            "Variable": ["Area_A (ref)"], "Area": ["A (ref)"],
            "Relativity": [1.0], "Coefficient": [0.0],
            "Std Error": [0.0], "P-value": [0.0], "Significant": [True]
        })
        area_coefs = pd.concat([ref_row, area_coefs], ignore_index=True)

        fig_area = px.bar(area_coefs, x="Area", y="Relativity",
                           color="Relativity", color_continuous_scale="RdYlGn_r",
                           title="Area Relativities (A = reference)")
        fig_area.add_hline(y=1, line_dash="dash", line_color="gray")
        fig_area.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig_area, width='stretch')

    with st.expander("📄 View Full GLM Frequency Coefficient Table"):
        display_df = coef_df[["Variable", "Coefficient", "Std Error", "P-value",
                               "Relativity", "Significant"]].copy()
        display_df["Coefficient"] = display_df["Coefficient"].round(5)
        display_df["Std Error"] = display_df["Std Error"].round(5)
        display_df["P-value"] = display_df["P-value"].apply(
            lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
        display_df["Relativity"] = display_df["Relativity"].round(4)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --- Severity Model (moved from Pure Premium page) ---
    section_header("📐 Severity Model — Gamma GLM")

    glm_sev = results.get("glm_sev_results")
    if glm_sev is not None:
        st.markdown("""
        The severity model is a **Gamma GLM with log link**, fitted only on policies with at least one claim.
        This is the standard actuarial approach: model severity separately from frequency, then multiply.
        """)

        sev_coef_df = pd.DataFrame({
            "Variable": glm_sev.params.index,
            "Coefficient": glm_sev.params.values,
            "Relativity": np.exp(glm_sev.params.values),
            "P-value": glm_sev.pvalues.values,
        })

        key_sev = sev_coef_df[sev_coef_df["Variable"].isin(
            ["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity"]
        )].copy()

        if not key_sev.empty:
            fig_sev_rel = go.Figure()
            colors_sev = [COLORS["accent"] if r > 1 else COLORS["success"]
                          for r in key_sev["Relativity"]]
            fig_sev_rel.add_trace(go.Bar(
                x=key_sev["Variable"], y=key_sev["Relativity"],
                marker_color=colors_sev,
                text=key_sev["Relativity"].round(4), textposition="outside"
            ))
            fig_sev_rel.add_hline(y=1, line_dash="dash", line_color="gray",
                                   annotation_text="Reference = 1.0")
            fig_sev_rel.update_layout(
                title="Severity Relativities — Key Variables",
                yaxis_title="Relativity (exp(β))",
                template="plotly_white", height=400
            )
            st.plotly_chart(fig_sev_rel, width='stretch')

        with st.expander("📄 View Full Severity GLM Coefficient Table"):
            sev_display = sev_coef_df.copy()
            sev_display["Coefficient"] = sev_display["Coefficient"].round(5)
            sev_display["Relativity"] = sev_display["Relativity"].round(4)
            sev_display["P-value"] = sev_display["P-value"].apply(
                lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
            st.dataframe(sev_display, use_container_width=True, hide_index=True)
    else:
        st.info("Severity model could not be fitted (insufficient claims data). Using portfolio average severity.")

    # --- Premium Simulator ---
    section_header("🧮 Premium Simulator")
    st.markdown("*Simulate the predicted frequency and pure premium for a given driver/vehicle profile.*")

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        sim_vehpower = st.slider("Vehicle Power", 4, 15, 7)
        sim_vehage = st.slider("Vehicle Age", 0, 30, 5)
    with sim_col2:
        sim_drivage = st.slider("Driver Age", 18, 90, 40)
        sim_bonusmalus = st.slider("Bonus-Malus", 50, 230, 80)
    with sim_col3:
        sim_area = st.selectbox("Area", ["A", "B", "C", "D", "E", "F"], index=2)
        sim_vehgas = st.selectbox("Fuel Type", ["Regular", "Diesel"], index=0)

    sim_input = pd.DataFrame({col: [0.0] for col in X.columns})
    sim_input["VehPower"] = sim_vehpower
    sim_input["VehAge"] = sim_vehage
    sim_input["DrivAge"] = sim_drivage
    sim_input["BonusMalus"] = sim_bonusmalus
    sim_input["LogDensity"] = np.log1p(1500)

    for a in ["B", "C", "D", "E", "F"]:
        col_name = f"Area_{a}"
        if col_name in sim_input.columns:
            sim_input[col_name] = 1.0 if sim_area == a else 0.0

    if "VehGas_Regular" in sim_input.columns:
        sim_input["VehGas_Regular"] = 1.0 if sim_vehgas == "Regular" else 0.0
    elif "VehGas_Diesel" in sim_input.columns:
        sim_input["VehGas_Diesel"] = 1.0 if sim_vehgas == "Diesel" else 0.0

    sim_input_c = sm.add_constant(sim_input, has_constant="add")

    try:
        sim_freq = glm.predict(sim_input_c)[0]
    except Exception:
        sim_freq = df["Frequency"].mean()

    # Predict severity too
    glm_sev = results.get("glm_sev_results")
    if glm_sev is not None:
        try:
            sim_sev = glm_sev.predict(sim_input_c)[0]
        except Exception:
            sim_sev = results["avg_severity"]
    else:
        sim_sev = results["avg_severity"]

    sim_pp = sim_freq * sim_sev
    avg_freq = df["ClaimNb"].sum() / df["Exposure"].sum()
    avg_pp = df["TotalClaimAmount"].sum() / df["Exposure"].sum()

    sim_r1, sim_r2, sim_r3, sim_r4 = st.columns(4)
    with sim_r1:
        kpi_card("Predicted Frequency", f"{sim_freq:.4f}")
    with sim_r2:
        kpi_card("Predicted Severity", fmt_number(sim_sev, prefix="€"))
    with sim_r3:
        kpi_card("Pure Premium", fmt_number(sim_pp, prefix="€"), green=True)
    with sim_r4:
        kpi_card("vs Portfolio Avg", f"{sim_pp / avg_pp:.2f}x")

    relativity_to_avg = sim_freq / avg_freq
    if relativity_to_avg > 1.2:
        st.warning(f"⚠️ This profile is **{(relativity_to_avg-1)*100:.0f}% above** portfolio average — high-risk segment.")
    elif relativity_to_avg < 0.8:
        st.success(f"✅ This profile is **{(1-relativity_to_avg)*100:.0f}% below** portfolio average — low-risk segment.")
    else:
        st.info(f"ℹ️ This profile is within **±20%** of portfolio average — standard risk.")

    # --- GLM Statistical Diagnostics ---
    section_header("🔬 GLM Statistical Diagnostics")
    
    diagnostics = results.get("glm_diagnostics", {})
    if diagnostics and len(diagnostics) > 0:
        st.markdown("""
        **Statistical tests** validate the quality and assumptions of the GLM model.
        These diagnostics help identify potential issues like overdispersion, poor fit, or missing variables.
        """)
        
        diag_col1, diag_col2, diag_col3, diag_col4 = st.columns(4)
        with diag_col1:
            st.metric("Pseudo R²", f"{diagnostics.get('pseudo_r2', 0):.4f}")
            st.caption("Model explanatory power")
        with diag_col2:
            st.metric("AIC", f"{diagnostics.get('aic', 0):.2f}")
            st.caption("Lower is better")
        with diag_col3:
            st.metric("BIC", f"{diagnostics.get('bic', 0):.2f}")
            st.caption("Lower is better")
        with diag_col4:
            dispersion = diagnostics.get('dispersion_ratio', 1.0)
            st.metric("Dispersion Ratio", f"{dispersion:.3f}")
            if dispersion > 1.5:
                st.caption("⚠️ Overdispersed (consider Negative Binomial)")
            else:
                st.caption("✅ Poisson appropriate")
        
        # Test results
        st.markdown("#### Test Results")
        test_results = pd.DataFrame({
            "Test": ["Likelihood Ratio Test", "Pearson Chi-square", "Dean's Overdispersion Test"],
            "Statistic": [
                f"{diagnostics.get('lr_statistic', 0):.2f}",
                f"{diagnostics.get('pearson_chi2', 0):.2f}",
                f"{diagnostics.get('dean_statistic', 0):.4f}"
            ],
            "P-value": [
                f"{diagnostics.get('lr_pvalue', 1):.2e}" if diagnostics.get('lr_pvalue', 1) < 0.001 else f"{diagnostics.get('lr_pvalue', 1):.4f}",
                f"{diagnostics.get('pearson_chi2_pvalue', 1):.2e}" if diagnostics.get('pearson_chi2_pvalue', 1) < 0.001 else f"{diagnostics.get('pearson_chi2_pvalue', 1):.4f}",
                f"{diagnostics.get('dean_pvalue', 1):.2e}" if diagnostics.get('dean_pvalue', 1) < 0.001 else f"{diagnostics.get('dean_pvalue', 1):.4f}"
            ],
            "Interpretation": [
                "✅ Model significantly better than null" if diagnostics.get('lr_pvalue', 1) < 0.05 else "❌ Model not significantly better",
                "✅ Good fit" if 0.05 < diagnostics.get('pearson_chi2_pvalue', 1) < 0.95 else "⚠️ Check model fit",
                "✅ No overdispersion" if diagnostics.get('dean_pvalue', 1) > 0.05 else "⚠️ Overdispersion detected"
            ]
        })
        st.dataframe(test_results, use_container_width=True, hide_index=True)
        
        # Residual plots
        st.markdown("#### Residual Analysis")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            pearson_res = diagnostics.get('pearson_residuals', np.array([]))
            if len(pearson_res) > 0:
                # Sample if too large to avoid performance issues
                sample_size = min(10000, len(pearson_res))
                if sample_size < len(pearson_res):
                    sample_idx = np.random.choice(len(pearson_res), size=sample_size, replace=False)
                    pearson_res_sample = pearson_res[sample_idx]
                else:
                    pearson_res_sample = pearson_res
                fig_pearson = px.histogram(x=pearson_res_sample, nbins=50,
                                           title="Pearson Residuals Distribution",
                                           color_discrete_sequence=[COLORS["secondary"]])
                fig_pearson.add_vline(x=0, line_dash="dash", line_color="red")
                fig_pearson.update_layout(template="plotly_white", height=350,
                                         xaxis_title="Pearson Residual", yaxis_title="Frequency")
                st.plotly_chart(fig_pearson, width='stretch')
        
        with res_col2:
            deviance_res = diagnostics.get('deviance_residuals', np.array([]))
            if len(deviance_res) > 0:
                # Sample if too large
                sample_size = min(10000, len(deviance_res))
                if sample_size < len(deviance_res):
                    sample_idx = np.random.choice(len(deviance_res), size=sample_size, replace=False)
                    deviance_res_sample = deviance_res[sample_idx]
                else:
                    deviance_res_sample = deviance_res
                fig_dev = px.histogram(x=deviance_res_sample, nbins=50,
                                      title="Deviance Residuals Distribution",
                                      color_discrete_sequence=[COLORS["accent"]])
                fig_dev.add_vline(x=0, line_dash="dash", line_color="red")
                fig_dev.update_layout(template="plotly_white", height=350,
                                    xaxis_title="Deviance Residual", yaxis_title="Frequency")
                st.plotly_chart(fig_dev, width='stretch')

    # --- Pricing Table Export ---
    section_header("📋 Pricing Table Generator")
    
    st.markdown("""
    Generate a **pricing table** with pure premium and commercial premium (with loadings) for different risk profiles.
    This table can be exported to CSV/Excel for use in production systems.
    """)
    
    # Loadings configuration
    load_col1, load_col2, load_col3 = st.columns(3)
    with load_col1:
        expense_loading = st.slider("Expense Loading (%)", 0.0, 50.0, 15.0) / 100
    with load_col2:
        profit_margin = st.slider("Profit Margin (%)", 0.0, 20.0, 5.0) / 100
    with load_col3:
        reinsurance_loading = st.slider("Reinsurance Loading (%)", 0.0, 15.0, 3.0) / 100
    
    total_loading = 1 + expense_loading + profit_margin + reinsurance_loading
    
    # Generate pricing table by segment
    pricing_segment = st.selectbox(
        "Select segmentation for pricing table:",
        ["Area", "DrivAge_bin", "VehAge_bin", "VehPower_bin", "BonusMalus_bin"],
        format_func=lambda x: SEGMENT_LABELS.get(x, x),
        key="pricing_seg"
    )
    
    # Calculate average pure premium by segment
    seg_pricing = df.groupby(pricing_segment, observed=True).agg(
        Exposure=("Exposure", "sum"),
        TotalAmount=("TotalClaimAmount", "sum"),
        Policies=("ClaimNb", "count")
    ).reset_index()
    seg_pricing["PurePremium"] = seg_pricing["TotalAmount"] / seg_pricing["Exposure"]
    seg_pricing["CommercialPremium"] = seg_pricing["PurePremium"] * total_loading
    seg_pricing["Relativity"] = seg_pricing["PurePremium"] / seg_pricing["PurePremium"].mean()
    
    # Display table
    pricing_display = seg_pricing[[pricing_segment, "Policies", "Exposure", "PurePremium", 
                                   "CommercialPremium", "Relativity"]].copy()
    pricing_display.columns = [
        SEGMENT_LABELS.get(pricing_segment, pricing_segment),
        "Policies", "Exposure (PY)", "Pure Premium (€)", "Commercial Premium (€)", "Relativity"
    ]
    pricing_display["Pure Premium (€)"] = pricing_display["Pure Premium (€)"].apply(lambda x: fmt_number(x, prefix="€", decimals=2))
    pricing_display["Commercial Premium (€)"] = pricing_display["Commercial Premium (€)"].apply(lambda x: fmt_number(x, prefix="€", decimals=2))
    pricing_display["Relativity"] = pricing_display["Relativity"].round(3)
    
    st.dataframe(pricing_display, use_container_width=True, hide_index=True)
    
    # Export button
    csv_data = seg_pricing.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Pricing Table (CSV)",
        data=csv_data,
        file_name=f"pricing_table_{pricing_segment}.csv",
        mime="text/csv"
    )
    
    st.info(f"💡 **Total Loading Factor:** {total_loading:.2%} (Pure Premium × {total_loading:.2f} = Commercial Premium)")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PURE PREMIUM
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "💰 Pure Premium":

    st.title("💰 Pure Premium Analysis")
    st.markdown("""
    *The **Pure Premium** is the expected cost per unit of exposure:*
    ### Pure Premium = Frequency × Severity
    *This is the foundation of any insurance pricing structure, before loading for expenses, profit margin, and reinsurance.*
    """)

    X, y, w, claim_count, df_model = get_cached_modeling_data(df)
    results = run_cached_models(X, y, w, claim_count, df_model)

    # --- KPI Row ---
    total_exposure = df["Exposure"].sum()
    total_claims = df["ClaimNb"].sum()
    total_amount = df["TotalClaimAmount"].sum()
    avg_freq = total_claims / total_exposure
    claims_mask = df["ClaimNb"] > 0
    avg_sev = df.loc[claims_mask, "TotalClaimAmount"].sum() / df.loc[claims_mask, "ClaimNb"].sum() if claims_mask.any() else 0
    avg_pp = total_amount / total_exposure

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Avg Frequency", f"{avg_freq:.4f}")
    with col2:
        kpi_card("Avg Severity", fmt_number(avg_sev, prefix="€"))
    with col3:
        kpi_card("Avg Pure Premium", fmt_number(avg_pp, prefix="€"), green=True)
    with col4:
        kpi_card("Total Incurred", fmt_number(total_amount, prefix="€"))

    # --- Decomposition by Segment ---
    section_header("📊 Pure Premium Decomposition by Segment")

    pp_segment = st.selectbox(
        "Select segmentation variable:",
        ["DrivAge_bin", "VehAge_bin", "VehPower_bin", "BonusMalus_bin", "Area", "VehGas"],
        format_func=lambda x: SEGMENT_LABELS.get(x, x),
        key="pp_segment"
    )

    pp_data = df.groupby(pp_segment, observed=True).agg(
        Exposure=("Exposure", "sum"),
        Claims=("ClaimNb", "sum"),
        TotalAmount=("TotalClaimAmount", "sum"),
    ).reset_index()
    pp_data["Frequency"] = pp_data["Claims"] / pp_data["Exposure"]
    pp_data["Severity"] = np.where(pp_data["Claims"] > 0,
                                    pp_data["TotalAmount"] / pp_data["Claims"], 0)
    pp_data["PurePremium"] = pp_data["TotalAmount"] / pp_data["Exposure"]

    # Stacked view: Frequency × Severity = Pure Premium
    fig_decomp = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Frequency", "Avg Severity (€)", "Pure Premium (€)"),
        shared_yaxes=False
    )

    fig_decomp.add_trace(go.Bar(
        x=pp_data[pp_segment].astype(str), y=pp_data["Frequency"],
        marker_color=COLORS["secondary"], name="Frequency",
        text=pp_data["Frequency"].round(4), textposition="outside"
    ), row=1, col=1)

    fig_decomp.add_trace(go.Bar(
        x=pp_data[pp_segment].astype(str), y=pp_data["Severity"],
        marker_color=COLORS["warning"], name="Severity",
        text=pp_data["Severity"].round(0).astype(int).astype(str) + "€", textposition="outside"
    ), row=1, col=2)

    fig_decomp.add_trace(go.Bar(
        x=pp_data[pp_segment].astype(str), y=pp_data["PurePremium"],
        marker_color=COLORS["success"], name="Pure Premium",
        text=pp_data["PurePremium"].round(0).astype(int).astype(str) + "€", textposition="outside"
    ), row=1, col=3)

    fig_decomp.update_layout(
        title=f"Pure Premium Decomposition by {SEGMENT_LABELS.get(pp_segment, pp_segment)}",
        template="plotly_white", height=450, showlegend=False,
    )
    # Ensure all x-axis labels are visible
    fig_decomp.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_decomp, width='stretch')

    # --- Pure Premium Heatmap ---
    section_header("🔥 Pure Premium Heatmap — Driver Age × Bonus-Malus")

    pp_heat = df.groupby(["DrivAge_bin", "BonusMalus_bin"], observed=True).agg(
        Exposure=("Exposure", "sum"),
        TotalAmount=("TotalClaimAmount", "sum"),
    ).reset_index()
    pp_heat["PurePremium"] = pp_heat["TotalAmount"] / pp_heat["Exposure"]

    pp_pivot = pp_heat.pivot_table(
        index="DrivAge_bin", columns="BonusMalus_bin",
        values="PurePremium", aggfunc="mean"
    )

    fig_pp_heat = px.imshow(
        pp_pivot, color_continuous_scale="RdYlGn_r",
        labels=dict(x="Bonus-Malus", y="Driver Age", color="Pure Premium (€)"),
        text_auto=",.0f", aspect="auto"
    )
    fig_pp_heat.update_layout(
        title="Pure Premium (€): Driver Age × Bonus-Malus",
        template="plotly_white", height=420,
        xaxis=dict(tickangle=-45, dtick=1),
        yaxis=dict(dtick=1),
    )
    st.plotly_chart(fig_pp_heat, width='stretch')

    # --- Severity Distribution Analysis ---
    section_header("📊 Severity Distribution Analysis")
    
    st.markdown("""
    **Deep dive into claim severity distribution** — understanding the tail risk, extreme values, and distribution characteristics.
    Critical for pricing, reserving, and reinsurance decisions.
    """)
    
    severity_analysis = results.get("severity_analysis", {})
    sev_train = results.get("sev_train", np.array([]))
    cc_train = results.get("cc_train", np.array([]))
    
    if severity_analysis and len(sev_train) > 0:
        # Filter non-zero severities
        sev_nonzero = sev_train[sev_train > 0]
        
        # Summary statistics
        sev_col1, sev_col2, sev_col3, sev_col4, sev_col5 = st.columns(5)
        with sev_col1:
            kpi_card("Mean Severity", fmt_number(severity_analysis.get('mean', 0), prefix="€"))
        with sev_col2:
            kpi_card("Median Severity", fmt_number(severity_analysis.get('median', 0), prefix="€"))
        with sev_col3:
            kpi_card("Coefficient of Variation", f"{severity_analysis.get('cv', 0):.3f}")
        with sev_col4:
            kpi_card("Skewness", f"{severity_analysis.get('skewness', 0):.2f}")
        with sev_col5:
            kpi_card("Kurtosis", f"{severity_analysis.get('kurtosis', 0):.2f}")
        
        # Distribution visualization
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            # Histogram with log scale
            fig_sev_hist = go.Figure()
            fig_sev_hist.add_trace(go.Histogram(
                x=sev_nonzero, nbinsx=100,
                marker_color=COLORS["secondary"],
                name="Severity Distribution"
            ))
            fig_sev_hist.add_vline(x=severity_analysis.get('mean', 0), line_dash="dash",
                                   line_color="red", annotation_text="Mean")
            fig_sev_hist.add_vline(x=severity_analysis.get('median', 0), line_dash="dash",
                                   line_color="blue", annotation_text="Median")
            fig_sev_hist.update_layout(
                title="Severity Distribution (Linear Scale)",
                xaxis_title="Claim Amount (€)",
                yaxis_title="Frequency",
                template="plotly_white", height=400
            )
            st.plotly_chart(fig_sev_hist, width='stretch')
        
        with dist_col2:
            # Log scale histogram
            log_sev = np.log(sev_nonzero[sev_nonzero > 0])
            fig_sev_log = go.Figure()
            fig_sev_log.add_trace(go.Histogram(
                x=log_sev, nbinsx=50,
                marker_color=COLORS["accent"],
                name="Log(Severity) Distribution"
            ))
            fig_sev_log.add_vline(x=severity_analysis.get('log_mean', 0), line_dash="dash",
                                  line_color="red", annotation_text="Log Mean")
            fig_sev_log.update_layout(
                title="Log(Severity) Distribution",
                xaxis_title="Log(Claim Amount)",
                yaxis_title="Frequency",
                template="plotly_white", height=400
            )
            st.plotly_chart(fig_sev_log, width='stretch')
        
        # Risk measures
        st.markdown("#### Risk Measures (VaR & TVaR)")
        risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
        with risk_col1:
            kpi_card("VaR 95%", fmt_number(severity_analysis.get('var_95', 0), prefix="€"))
        with risk_col2:
            kpi_card("VaR 99%", fmt_number(severity_analysis.get('var_99', 0), prefix="€"))
        with risk_col3:
            kpi_card("TVaR 95%", fmt_number(severity_analysis.get('tvar_95', 0), prefix="€"))
        with risk_col4:
            kpi_card("TVaR 99%", fmt_number(severity_analysis.get('tvar_99', 0), prefix="€"))
        
        st.caption("**VaR (Value at Risk):** Maximum loss at given confidence level | **TVaR (Tail Value at Risk):** Expected loss beyond VaR threshold")
        
        # Percentiles table
        st.markdown("#### Severity Percentiles")
        percentiles_data = {
            "Percentile": ["50th (Median)", "75th", "90th", "95th", "99th", "99.5th", "99.9th"],
            "Value (€)": [
                fmt_number(severity_analysis.get('p50', 0), prefix="€"),
                fmt_number(severity_analysis.get('p75', 0), prefix="€"),
                fmt_number(severity_analysis.get('p90', 0), prefix="€"),
                fmt_number(severity_analysis.get('p95', 0), prefix="€"),
                fmt_number(severity_analysis.get('p99', 0), prefix="€"),
                fmt_number(severity_analysis.get('p99.5', 0), prefix="€"),
                fmt_number(severity_analysis.get('p99.9', 0), prefix="€"),
            ]
        }
        st.dataframe(pd.DataFrame(percentiles_data), use_container_width=True, hide_index=True)
        
        # Extreme values analysis
        st.markdown("#### Extreme Values Analysis")
        extreme_threshold = severity_analysis.get('p95', 0)
        extreme_claims = sev_nonzero[sev_nonzero >= extreme_threshold]
        extreme_pct = len(extreme_claims) / len(sev_nonzero) * 100 if len(sev_nonzero) > 0 else 0
        extreme_total = extreme_claims.sum() if len(extreme_claims) > 0 else 0
        extreme_pct_of_total = extreme_total / sev_nonzero.sum() * 100 if len(sev_nonzero) > 0 else 0
        
        ext_col1, ext_col2, ext_col3 = st.columns(3)
        with ext_col1:
            st.metric("Claims ≥ 95th percentile", f"{len(extreme_claims)} ({extreme_pct:.2f}%)")
        with ext_col2:
            st.metric("Total Amount (Extreme)", fmt_number(extreme_total, prefix="€"))
        with ext_col3:
            st.metric("% of Total Severity", f"{extreme_pct_of_total:.1f}%")
        
        st.info(f"💡 **Insight:** {extreme_pct:.1f}% of claims (≥95th percentile) represent {extreme_pct_of_total:.1f}% of total claim costs. This highlights the importance of tail risk management.")
    else:
        st.info("Severity analysis requires claims data. Please ensure the dataset contains claim severity information.")

    # --- Model vs Actual Pure Premium ---
    section_header("🎯 Modeled vs Actual Pure Premium")

    pp_metrics = results.get("pp_metrics_glm", {})
    xgb_pp_metrics = results.get("pp_metrics_xgb", {})

    if pp_metrics:
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("#### GLM Pure Premium (Freq × Sev)")
            st.metric("Mean Predicted", fmt_number(pp_metrics.get('Mean Predicted', 0), prefix="€"))
            st.metric("Mean Actual", fmt_number(pp_metrics.get('Mean Actual', 0), prefix="€"))
            st.metric("Gini", f"{pp_metrics.get('Gini', 0):.4f}")
        with mc2:
            st.markdown("#### XGBoost Freq × GLM Sev")
            st.metric("Mean Predicted", fmt_number(xgb_pp_metrics.get('Mean Predicted', 0), prefix="€"))
            st.metric("Mean Actual", fmt_number(xgb_pp_metrics.get('Mean Actual', 0), prefix="€"))
            st.metric("Gini", f"{xgb_pp_metrics.get('Gini', 0):.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — GLM vs XGBOOST
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 GLM vs XGBoost":

    st.title("🤖 GLM vs XGBoost — Model Comparison")
    st.markdown("""
    *Benchmarking the traditional actuarial approach (Poisson GLM) against a modern ML model (XGBoost).
    Both approaches have merits — the key is understanding when to use each.*
    """)

    X, y, w, claim_count, df_model = get_cached_modeling_data(df)
    results = run_cached_models(X, y, w, claim_count, df_model)

    # --- Metrics ---
    section_header("📊 Performance Comparison (Test Set)")

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("GLM Gini", f"{results['glm_metrics_test']['Gini']:.4f}")
        st.metric("GLM RMSE", f"{results['glm_metrics_test']['RMSE']:.5f}")
    with m_col2:
        st.metric("XGBoost Gini", f"{results['xgb_metrics_test']['Gini']:.4f}",
                   delta=f"{results['xgb_metrics_test']['Gini'] - results['glm_metrics_test']['Gini']:.4f}")
        st.metric("XGBoost RMSE", f"{results['xgb_metrics_test']['RMSE']:.5f}",
                   delta=f"{results['xgb_metrics_test']['RMSE'] - results['glm_metrics_test']['RMSE']:.5f}",
                   delta_color="inverse")
    with m_col3:
        st.markdown("#### 🏆 Summary")
        glm_gini = results['glm_metrics_test']['Gini']
        xgb_gini = results['xgb_metrics_test']['Gini']
        if xgb_gini > glm_gini:
            st.markdown(f"""
            XGBoost achieves a **{((xgb_gini - glm_gini)/glm_gini)*100:.1f}% higher Gini**,
            capturing more risk differentiation. However, GLM offers **full transparency**
            and regulatory compliance.
            """)
        else:
            st.markdown("""
            GLM performs competitively with XGBoost, while offering **full interpretability**
            and regulatory compliance.
            """)

    # --- Lift Curves ---
    section_header("📈 Lift Curves — Actual vs Predicted by Decile")

    glm_lift = results["glm_lift"]
    xgb_lift = results["xgb_lift"]

    fig_lift = make_subplots(rows=1, cols=2,
                              subplot_titles=("GLM Lift Curve", "XGBoost Lift Curve"))

    fig_lift.add_trace(go.Bar(x=glm_lift["decile_label"], y=glm_lift["avg_actual"],
                               name="Actual", marker_color=COLORS["primary"]), row=1, col=1)
    fig_lift.add_trace(go.Scatter(x=glm_lift["decile_label"], y=glm_lift["avg_predicted"],
                                   name="GLM Predicted", mode="lines+markers",
                                   line=dict(color=COLORS["accent"], width=3)), row=1, col=1)

    fig_lift.add_trace(go.Bar(x=xgb_lift["decile_label"], y=xgb_lift["avg_actual"],
                               name="Actual", marker_color=COLORS["primary"],
                               showlegend=False), row=1, col=2)
    fig_lift.add_trace(go.Scatter(x=xgb_lift["decile_label"], y=xgb_lift["avg_predicted"],
                                   name="XGBoost Predicted", mode="lines+markers",
                                   line=dict(color=COLORS["warning"], width=3)), row=1, col=2)

    fig_lift.update_layout(template="plotly_white", height=420,
                            title_text="Ordered Lift: Do models correctly rank risk?")
    fig_lift.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_lift, width='stretch')

    # --- Double Lift ---
    section_header("⚖️ Double Lift — Where Do Models Disagree?")

    y_test = results["y_test"]
    glm_pred = results["glm_pred_test"]
    xgb_pred = results["xgb_pred_test"]
    w_test = results["w_test"]

    ratio = xgb_pred / np.clip(glm_pred, 1e-6, None)
    dl_df = pd.DataFrame({
        "ratio": ratio, "y_true": y_test,
        "glm_pred": glm_pred, "xgb_pred": xgb_pred, "w": w_test
    })
    dl_df["decile"] = pd.qcut(dl_df["ratio"], q=10, labels=False, duplicates="drop")

    dl_agg = dl_df.groupby("decile")[["y_true", "glm_pred", "xgb_pred", "w"]].apply(
        lambda g: pd.Series({
            "avg_actual": np.average(g["y_true"], weights=g["w"]),
            "avg_glm": np.average(g["glm_pred"], weights=g["w"]),
            "avg_xgb": np.average(g["xgb_pred"], weights=g["w"]),
        })
    ).reset_index()
    dl_agg["label"] = [f"D{i+1}" for i in range(len(dl_agg))]

    fig_dl = go.Figure()
    fig_dl.add_trace(go.Bar(x=dl_agg["label"], y=dl_agg["avg_actual"],
                             name="Actual", marker_color=COLORS["primary"]))
    fig_dl.add_trace(go.Scatter(x=dl_agg["label"], y=dl_agg["avg_glm"],
                                 name="GLM", mode="lines+markers",
                                 line=dict(color=COLORS["accent"], width=2)))
    fig_dl.add_trace(go.Scatter(x=dl_agg["label"], y=dl_agg["avg_xgb"],
                                 name="XGBoost", mode="lines+markers",
                                 line=dict(color=COLORS["warning"], width=2)))
    fig_dl.update_layout(
        title="Double Lift: Deciles sorted by XGBoost/GLM ratio",
        xaxis_title="Decile (sorted by XGB/GLM ratio)",
        yaxis_title="Average Frequency",
        template="plotly_white", height=400
    )
    st.plotly_chart(fig_dl, width='stretch')

    # --- SHAP ---
    section_header("🔍 SHAP Interpretability — XGBoost")

    st.markdown("""
    **SHAP values** explain each feature's contribution to individual predictions —
    bridging ML accuracy with actuarial interpretability.
    """)

    try:
        import shap

        xgb_model = results["xgb_model"]
        X_test = results["X_test"]
        sample_size = min(1000, len(X_test))
        X_sample = X_test.iloc[:sample_size]

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_sample)

        mean_shap = pd.DataFrame({
            "Feature": X_sample.columns,
            "Mean |SHAP|": np.abs(shap_values).mean(axis=0)
        }).sort_values("Mean |SHAP|", ascending=True).tail(15)

        fig_shap = px.bar(mean_shap, x="Mean |SHAP|", y="Feature", orientation="h",
                           color="Mean |SHAP|", color_continuous_scale="Reds",
                           title="Top 15 Features by Mean |SHAP| Value")
        fig_shap.update_layout(template="plotly_white", height=500, showlegend=False)
        st.plotly_chart(fig_shap, width='stretch')

        if "BonusMalus" in X_sample.columns:
            st.markdown("#### SHAP Dependence — Bonus-Malus")
            bm_idx = list(X_sample.columns).index("BonusMalus")
            fig_dep = px.scatter(
                x=X_sample["BonusMalus"], y=shap_values[:, bm_idx],
                color=X_sample["DrivAge"] if "DrivAge" in X_sample.columns else None,
                labels={"x": "Bonus-Malus", "y": "SHAP Value", "color": "Driver Age"},
                title="SHAP Dependence: Bonus-Malus (colored by Driver Age)",
                color_continuous_scale="Viridis", opacity=0.5
            )
            fig_dep.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_dep, width='stretch')

    except Exception as e:
        st.warning(f"SHAP analysis requires the `shap` library. Error: {e}")

    # --- Pros & Cons ---
    section_header("⚖️ GLM vs XGBoost — When to Use What")

    pros_cons = pd.DataFrame({
        "Criterion": ["Interpretability", "Regulatory Compliance", "Non-linear Patterns",
                       "Training Speed", "Feature Interactions", "Stability Over Time",
                       "Production Deployment", "Actuarial Sign-off"],
        "GLM": ["✅ Full transparency", "✅ Industry standard", "❌ Manual interactions",
                 "✅ Very fast", "❌ Must specify manually", "✅ Very stable",
                 "✅ Simple & proven", "✅ Easy to validate"],
        "XGBoost": ["⚠️ Requires SHAP", "⚠️ Extra documentation", "✅ Automatic",
                     "⚠️ Slower", "✅ Auto-captured", "⚠️ Can overfit",
                     "⚠️ More complex", "⚠️ Needs explainability layer"]
    })
    st.dataframe(pros_cons, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Model Monitoring":

    st.title("📈 Model Monitoring & Drift Detection")
    st.markdown("""
    *Continuous monitoring ensures pricing models remain accurate.
    These are the key KPIs a pricing actuary tracks regularly.*
    """)

    X, y, w, claim_count, df_model = get_cached_modeling_data(df)
    results = run_cached_models(X, y, w, claim_count, df_model)

    # Simulate monthly periods
    np.random.seed(42)
    n_test = len(results["y_test"])
    months = pd.date_range("2024-01-01", periods=12, freq="MS")
    month_assign = np.random.choice(range(12), size=n_test)

    mon_df = pd.DataFrame({
        "month": [months[i] for i in month_assign],
        "y_true": results["y_test"],
        "glm_pred": results["glm_pred_test"],
        "xgb_pred": results["xgb_pred_test"],
        "w": results["w_test"]
    })

    # --- A/E ---
    section_header("📋 Actual vs Expected (A/E Ratio)")

    ae_monthly = mon_df.groupby("month")[["y_true", "glm_pred", "xgb_pred", "w"]].apply(
        lambda g: pd.Series({
            "actual": np.average(g["y_true"], weights=g["w"]),
            "glm_expected": np.average(g["glm_pred"], weights=g["w"]),
            "xgb_expected": np.average(g["xgb_pred"], weights=g["w"]),
        })
    ).reset_index()
    ae_monthly["AE_GLM"] = ae_monthly["actual"] / ae_monthly["glm_expected"]
    ae_monthly["AE_XGB"] = ae_monthly["actual"] / ae_monthly["xgb_expected"]

    fig_ae = go.Figure()
    fig_ae.add_trace(go.Scatter(x=ae_monthly["month"], y=ae_monthly["AE_GLM"],
                                 mode="lines+markers", name="GLM A/E",
                                 line=dict(color=COLORS["accent"], width=2.5)))
    fig_ae.add_trace(go.Scatter(x=ae_monthly["month"], y=ae_monthly["AE_XGB"],
                                 mode="lines+markers", name="XGBoost A/E",
                                 line=dict(color=COLORS["warning"], width=2.5)))
    fig_ae.add_hrect(y0=0.95, y1=1.05, fillcolor="green", opacity=0.1,
                      annotation_text="Acceptable range (±5%)")
    fig_ae.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig_ae.update_layout(title="Monthly A/E Ratio", xaxis_title="Month",
                          yaxis_title="Actual / Expected",
                          template="plotly_white", height=420)
    st.plotly_chart(fig_ae, width='stretch')

    # --- PSI ---
    section_header("📊 Population Stability Index (PSI)")

    def compute_psi(expected, actual, bins=10):
        breakpoints = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
        if len(breakpoints) < 3:
            return 0.0
        exp_counts = np.histogram(expected, bins=breakpoints)[0]
        act_counts = np.histogram(actual, bins=breakpoints)[0]
        exp_pct = np.clip(exp_counts / exp_counts.sum(), 0.001, None)
        act_pct = np.clip(act_counts / act_counts.sum(), 0.001, None)
        return np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))

    first_half = mon_df[mon_df["month"] < "2024-07-01"]
    second_half = mon_df[mon_df["month"] >= "2024-07-01"]

    psi_glm = compute_psi(first_half["glm_pred"].values, second_half["glm_pred"].values)
    psi_xgb = compute_psi(first_half["xgb_pred"].values, second_half["xgb_pred"].values)

    psi_col1, psi_col2 = st.columns(2)
    with psi_col1:
        kpi_card("GLM Prediction PSI", f"{psi_glm:.4f}")
        if psi_glm < 0.1:
            st.success("✅ Stable — no drift detected")
        elif psi_glm < 0.25:
            st.warning("⚠️ Moderate drift — investigate")
        else:
            st.error("🚨 Significant drift — recalibrate")

    with psi_col2:
        kpi_card("XGBoost Prediction PSI", f"{psi_xgb:.4f}")
        if psi_xgb < 0.1:
            st.success("✅ Stable — no drift detected")
        elif psi_xgb < 0.25:
            st.warning("⚠️ Moderate drift — investigate")
        else:
            st.error("🚨 Significant drift — recalibrate")

    # --- Residuals ---
    section_header("🔬 Residual Analysis")

    resid_model = st.selectbox("Select model:", ["GLM", "XGBoost"])
    pred = np.array(results["glm_pred_test"] if resid_model == "GLM" else results["xgb_pred_test"])
    residuals = np.array(results["y_test"]) - pred

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        fig_rh = px.histogram(x=residuals, nbins=80,
                               title=f"{resid_model} — Residual Distribution",
                               color_discrete_sequence=[COLORS["secondary"]])
        fig_rh.add_vline(x=0, line_dash="dash", line_color="red")
        fig_rh.update_layout(
            template="plotly_white", height=380,
            xaxis_title="Residual", yaxis_title="Count",
        )
        st.plotly_chart(fig_rh, width='stretch')

    with col_res2:
        sample_idx = np.random.choice(len(pred), size=min(3000, len(pred)), replace=False)
        fig_rs = px.scatter(x=pred[sample_idx], y=residuals[sample_idx], opacity=0.3,
                             title=f"{resid_model} — Residuals vs Predicted",
                             color_discrete_sequence=[COLORS["primary"]])
        fig_rs.add_hline(y=0, line_dash="dash", line_color="red")
        fig_rs.update_layout(
            template="plotly_white", height=380,
            xaxis_title="Predicted Value", yaxis_title="Residual",
        )
        st.plotly_chart(fig_rs, width='stretch')

    # --- Segment A/E ---
    section_header("🎯 Segment-Level A/E Monitoring")

    seg_var_mon = st.selectbox(
        "Select segment:", ["DrivAge_bin", "VehAge_bin", "VehPower_bin", "BonusMalus_bin", "Area"],
        format_func=lambda x: SEGMENT_LABELS.get(x, x), key="seg_mon"
    )

    X_test = results["X_test"]
    test_idx = X_test.index
    if seg_var_mon in df_model.columns:
        seg_values = df_model.loc[test_idx, seg_var_mon].values
    else:
        seg_values = np.random.choice(df[seg_var_mon].dropna().unique(), size=len(results["y_test"]))

    seg_ae_df = pd.DataFrame({
        "segment": seg_values, "y_true": results["y_test"],
        "glm_pred": results["glm_pred_test"], "w": results["w_test"]
    })

    seg_ae = seg_ae_df.groupby("segment", observed=True)[["y_true", "glm_pred", "w"]].apply(
        lambda g: pd.Series({
            "actual_freq": np.average(g["y_true"], weights=g["w"]),
            "predicted_freq": np.average(g["glm_pred"], weights=g["w"]),
        })
    ).reset_index()
    seg_ae["AE_ratio"] = seg_ae["actual_freq"] / seg_ae["predicted_freq"]

    fig_sae = go.Figure()
    fig_sae.add_trace(go.Bar(
        x=seg_ae["segment"].astype(str), y=seg_ae["AE_ratio"],
        marker_color=[COLORS["accent"] if abs(ae - 1) > 0.05 else COLORS["success"]
                      for ae in seg_ae["AE_ratio"]],
        text=seg_ae["AE_ratio"].round(3), textposition="outside"
    ))
    fig_sae.add_hrect(y0=0.95, y1=1.05, fillcolor="green", opacity=0.1)
    fig_sae.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig_sae.update_layout(
        title=f"A/E Ratio by {SEGMENT_LABELS.get(seg_var_mon, seg_var_mon)}",
        template="plotly_white", height=400,
        xaxis=dict(tickangle=-45),
        xaxis_title=SEGMENT_LABELS.get(seg_var_mon, seg_var_mon),
        yaxis_title="A/E Ratio",
    )
    st.plotly_chart(fig_sae, width='stretch')

    # --- Summary ---
    section_header("📝 Monitoring Summary & Recommendations")

    overall_ae = (np.average(results["y_test"], weights=results["w_test"]) /
                  np.average(results["glm_pred_test"], weights=results["w_test"]))

    st.markdown(f"""
    | Metric | Value | Status |
    |--------|-------|--------|
    | Overall A/E (GLM) | {overall_ae:.3f} | {"✅ OK" if abs(overall_ae - 1) < 0.05 else "⚠️ Investigate"} |
    | GLM Prediction PSI | {psi_glm:.4f} | {"✅ Stable" if psi_glm < 0.1 else "⚠️ Drift"} |
    | XGBoost Prediction PSI | {psi_xgb:.4f} | {"✅ Stable" if psi_xgb < 0.1 else "⚠️ Drift"} |

    **Recommended Actions:**
    - Monitor A/E ratio monthly at portfolio and segment level
    - Trigger model review if A/E deviates beyond ±5% for 3 consecutive months
    - Recalibrate if PSI exceeds 0.25 on key variables
    - Compare GLM vs XGBoost predictions quarterly to detect non-linear patterns
    - Document all monitoring findings for regulatory reporting
    """)


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
