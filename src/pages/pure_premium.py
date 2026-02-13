"""
Page 3 — Pure Premium Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import COLORS, SEGMENT_LABELS
from src.utils import fmt_number, kpi_card, section_header


def render(df, get_modeling_data, run_models):
    """Render the Pure Premium Analysis page."""

    st.title("💰 Pure Premium Analysis")
    st.markdown("""
    *The **Pure Premium** is the expected cost per unit of exposure:*
    ### Pure Premium = Frequency × Severity
    *This is the foundation of any insurance pricing structure, before loading for expenses, profit margin, and reinsurance.*
    """)

    X, y, w, claim_count, df_model = get_modeling_data(df)
    results = run_models(X, y, w, claim_count, df_model)

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
        key="pp_segment",
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

    fig_decomp = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Frequency", "Avg Severity (€)", "Pure Premium (€)"),
        shared_yaxes=False,
    )

    fig_decomp.add_trace(go.Bar(
        x=pp_data[pp_segment].astype(str), y=pp_data["Frequency"],
        marker_color=COLORS["secondary"], name="Frequency",
        text=pp_data["Frequency"].round(4), textposition="outside",
    ), row=1, col=1)

    fig_decomp.add_trace(go.Bar(
        x=pp_data[pp_segment].astype(str), y=pp_data["Severity"],
        marker_color=COLORS["warning"], name="Severity",
        text=pp_data["Severity"].round(0).astype(int).astype(str) + "€", textposition="outside",
    ), row=1, col=2)

    fig_decomp.add_trace(go.Bar(
        x=pp_data[pp_segment].astype(str), y=pp_data["PurePremium"],
        marker_color=COLORS["success"], name="Pure Premium",
        text=pp_data["PurePremium"].round(0).astype(int).astype(str) + "€", textposition="outside",
    ), row=1, col=3)

    fig_decomp.update_layout(
        title=f"Pure Premium Decomposition by {SEGMENT_LABELS.get(pp_segment, pp_segment)}",
        template="plotly_white", height=450, showlegend=False,
    )
    fig_decomp.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_decomp, use_container_width=True)

    # --- Pure Premium Heatmap ---
    section_header("🔥 Pure Premium Heatmap — Driver Age × Bonus-Malus")

    pp_heat = df.groupby(["DrivAge_bin", "BonusMalus_bin"], observed=True).agg(
        Exposure=("Exposure", "sum"),
        TotalAmount=("TotalClaimAmount", "sum"),
    ).reset_index()
    pp_heat["PurePremium"] = pp_heat["TotalAmount"] / pp_heat["Exposure"]

    pp_pivot = pp_heat.pivot_table(
        index="DrivAge_bin", columns="BonusMalus_bin",
        values="PurePremium", aggfunc="mean",
    )

    fig_pp_heat = px.imshow(
        pp_pivot, color_continuous_scale="RdYlGn_r",
        labels=dict(x="Bonus-Malus", y="Driver Age", color="Pure Premium (€)"),
        text_auto=",.0f", aspect="auto",
    )
    fig_pp_heat.update_layout(
        title="Pure Premium (€): Driver Age × Bonus-Malus",
        template="plotly_white", height=420,
        xaxis=dict(tickangle=-45, dtick=1),
        yaxis=dict(dtick=1),
    )
    st.plotly_chart(fig_pp_heat, use_container_width=True)

    # --- Severity Distribution Analysis ---
    section_header("📊 Severity Distribution Analysis")

    st.markdown("""
    **Deep dive into claim severity distribution** — understanding the tail risk, extreme values, and distribution characteristics.
    Critical for pricing, reserving, and reinsurance decisions.
    """)

    severity_analysis = results.get("severity_analysis", {})
    sev_train = results.get("sev_train", np.array([]))

    if severity_analysis and len(sev_train) > 0:
        sev_nonzero = sev_train[sev_train > 0]

        sev_col1, sev_col2, sev_col3, sev_col4, sev_col5 = st.columns(5)
        with sev_col1:
            kpi_card("Mean Severity", fmt_number(severity_analysis.get("mean", 0), prefix="€"))
        with sev_col2:
            kpi_card("Median Severity", fmt_number(severity_analysis.get("median", 0), prefix="€"))
        with sev_col3:
            kpi_card("Coefficient of Variation", f"{severity_analysis.get('cv', 0):.3f}")
        with sev_col4:
            kpi_card("Skewness", f"{severity_analysis.get('skewness', 0):.2f}")
        with sev_col5:
            kpi_card("Kurtosis", f"{severity_analysis.get('kurtosis', 0):.2f}")

        dist_col1, dist_col2 = st.columns(2)

        with dist_col1:
            fig_sev_hist = go.Figure()
            fig_sev_hist.add_trace(go.Histogram(
                x=sev_nonzero, nbinsx=100,
                marker_color=COLORS["secondary"],
                name="Severity Distribution",
            ))
            fig_sev_hist.add_vline(x=severity_analysis.get("mean", 0), line_dash="dash",
                                   line_color="red", annotation_text="Mean")
            fig_sev_hist.add_vline(x=severity_analysis.get("median", 0), line_dash="dash",
                                   line_color="blue", annotation_text="Median")
            fig_sev_hist.update_layout(
                title="Severity Distribution (Linear Scale)",
                xaxis_title="Claim Amount (€)",
                yaxis_title="Frequency",
                template="plotly_white", height=400,
            )
            st.plotly_chart(fig_sev_hist, use_container_width=True)

        with dist_col2:
            log_sev = np.log(sev_nonzero[sev_nonzero > 0])
            fig_sev_log = go.Figure()
            fig_sev_log.add_trace(go.Histogram(
                x=log_sev, nbinsx=50,
                marker_color=COLORS["accent"],
                name="Log(Severity) Distribution",
            ))
            fig_sev_log.add_vline(x=severity_analysis.get("log_mean", 0), line_dash="dash",
                                  line_color="red", annotation_text="Log Mean")
            fig_sev_log.update_layout(
                title="Log(Severity) Distribution",
                xaxis_title="Log(Claim Amount)",
                yaxis_title="Frequency",
                template="plotly_white", height=400,
            )
            st.plotly_chart(fig_sev_log, use_container_width=True)

        st.markdown("#### Risk Measures (VaR & TVaR)")
        risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
        with risk_col1:
            kpi_card("VaR 95%", fmt_number(severity_analysis.get("var_95", 0), prefix="€"))
        with risk_col2:
            kpi_card("VaR 99%", fmt_number(severity_analysis.get("var_99", 0), prefix="€"))
        with risk_col3:
            kpi_card("TVaR 95%", fmt_number(severity_analysis.get("tvar_95", 0), prefix="€"))
        with risk_col4:
            kpi_card("TVaR 99%", fmt_number(severity_analysis.get("tvar_99", 0), prefix="€"))

        st.caption("**VaR (Value at Risk):** Maximum loss at given confidence level | **TVaR (Tail Value at Risk):** Expected loss beyond VaR threshold")

        st.markdown("#### Severity Percentiles")
        percentiles_data = {
            "Percentile": ["50th (Median)", "75th", "90th", "95th", "99th", "99.5th", "99.9th"],
            "Value (€)": [
                fmt_number(severity_analysis.get("p50", 0), prefix="€"),
                fmt_number(severity_analysis.get("p75", 0), prefix="€"),
                fmt_number(severity_analysis.get("p90", 0), prefix="€"),
                fmt_number(severity_analysis.get("p95", 0), prefix="€"),
                fmt_number(severity_analysis.get("p99", 0), prefix="€"),
                fmt_number(severity_analysis.get("p99.5", 0), prefix="€"),
                fmt_number(severity_analysis.get("p99.9", 0), prefix="€"),
            ],
        }
        st.dataframe(pd.DataFrame(percentiles_data), use_container_width=True, hide_index=True)

        st.markdown("#### Extreme Values Analysis")
        extreme_threshold = severity_analysis.get("p95", 0)
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
            st.metric("Mean Predicted", fmt_number(pp_metrics.get("Mean Predicted", 0), prefix="€"))
            st.metric("Mean Actual", fmt_number(pp_metrics.get("Mean Actual", 0), prefix="€"))
            st.metric("Gini", f"{pp_metrics.get('Gini', 0):.4f}")
        with mc2:
            st.markdown("#### XGBoost Freq × GLM Sev")
            st.metric("Mean Predicted", fmt_number(xgb_pp_metrics.get("Mean Predicted", 0), prefix="€"))
            st.metric("Mean Actual", fmt_number(xgb_pp_metrics.get("Mean Actual", 0), prefix="€"))
            st.metric("Gini", f"{xgb_pp_metrics.get('Gini', 0):.4f}")
