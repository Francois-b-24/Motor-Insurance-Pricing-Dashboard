"""
Page 2 — GLM Pricing Model
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

from src.config import COLORS, SEGMENT_LABELS
from src.utils import fmt_number, kpi_card, section_header


def render(df, get_modeling_data, run_models):
    """Render the GLM Pricing Model page."""

    st.title("🎯 GLM Pricing Model")
    st.markdown("*Poisson GLM for frequency modeling and Gamma GLM for severity — the industry standards for motor insurance pricing.*")

    X, y, w, claim_count, df_model = get_modeling_data(df)
    results = run_models(X, y, w, claim_count, df_model)
    glm = results["glm_results"]

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
        "Relativity": np.exp(glm.params.values),
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
            marker_color=colors, text=key_coef["Relativity"].round(4), textposition="outside",
        ))
        fig_rel.add_hline(y=1, line_dash="dash", line_color="gray",
                          annotation_text="Reference = 1.0")
        fig_rel.update_layout(title="Relativities — Key Continuous Variables",
                               yaxis_title="Relativity (exp(β))",
                               template="plotly_white", height=400)
        st.plotly_chart(fig_rel, use_container_width=True)

    # Area relativities
    area_coefs = coef_df[coef_df["Variable"].str.startswith("Area_")].copy()
    if not area_coefs.empty:
        area_coefs["Area"] = area_coefs["Variable"].str.replace("Area_", "")
        ref_row = pd.DataFrame({
            "Variable": ["Area_A (ref)"], "Area": ["A (ref)"],
            "Relativity": [1.0], "Coefficient": [0.0],
            "Std Error": [0.0], "P-value": [0.0], "Significant": [True],
        })
        area_coefs = pd.concat([ref_row, area_coefs], ignore_index=True)

        fig_area = px.bar(area_coefs, x="Area", y="Relativity",
                           color="Relativity", color_continuous_scale="RdYlGn_r",
                           title="Area Relativities (A = reference)")
        fig_area.add_hline(y=1, line_dash="dash", line_color="gray")
        fig_area.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig_area, use_container_width=True)

    with st.expander("📄 View Full GLM Frequency Coefficient Table"):
        display_df = coef_df[["Variable", "Coefficient", "Std Error", "P-value",
                               "Relativity", "Significant"]].copy()
        display_df["Coefficient"] = display_df["Coefficient"].round(5)
        display_df["Std Error"] = display_df["Std Error"].round(5)
        display_df["P-value"] = display_df["P-value"].apply(
            lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
        display_df["Relativity"] = display_df["Relativity"].round(4)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --- Severity Model ---
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
                text=key_sev["Relativity"].round(4), textposition="outside",
            ))
            fig_sev_rel.add_hline(y=1, line_dash="dash", line_color="gray",
                                   annotation_text="Reference = 1.0")
            fig_sev_rel.update_layout(
                title="Severity Relativities — Key Variables",
                yaxis_title="Relativity (exp(β))",
                template="plotly_white", height=400,
            )
            st.plotly_chart(fig_sev_rel, use_container_width=True)

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
            dispersion = diagnostics.get("dispersion_ratio", 1.0)
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
                f"{diagnostics.get('dean_statistic', 0):.4f}",
            ],
            "P-value": [
                f"{diagnostics.get('lr_pvalue', 1):.2e}" if diagnostics.get("lr_pvalue", 1) < 0.001 else f"{diagnostics.get('lr_pvalue', 1):.4f}",
                f"{diagnostics.get('pearson_chi2_pvalue', 1):.2e}" if diagnostics.get("pearson_chi2_pvalue", 1) < 0.001 else f"{diagnostics.get('pearson_chi2_pvalue', 1):.4f}",
                f"{diagnostics.get('dean_pvalue', 1):.2e}" if diagnostics.get("dean_pvalue", 1) < 0.001 else f"{diagnostics.get('dean_pvalue', 1):.4f}",
            ],
            "Interpretation": [
                "✅ Model significantly better than null" if diagnostics.get("lr_pvalue", 1) < 0.05 else "❌ Model not significantly better",
                "✅ Good fit" if 0.05 < diagnostics.get("pearson_chi2_pvalue", 1) < 0.95 else "⚠️ Check model fit",
                "✅ No overdispersion" if diagnostics.get("dean_pvalue", 1) > 0.05 else "⚠️ Overdispersion detected",
            ],
        })
        st.dataframe(test_results, use_container_width=True, hide_index=True)

        # Residual plots
        st.markdown("#### Residual Analysis")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            pearson_res = diagnostics.get("pearson_residuals", np.array([]))
            if len(pearson_res) > 0:
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
                st.plotly_chart(fig_pearson, use_container_width=True)

        with res_col2:
            deviance_res = diagnostics.get("deviance_residuals", np.array([]))
            if len(deviance_res) > 0:
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
                st.plotly_chart(fig_dev, use_container_width=True)

    # --- Pricing Table Export ---
    section_header("📋 Pricing Table Generator")

    st.markdown("""
    Generate a **pricing table** with pure premium and commercial premium (with loadings) for different risk profiles.
    This table can be exported to CSV/Excel for use in production systems.
    """)

    load_col1, load_col2, load_col3 = st.columns(3)
    with load_col1:
        expense_loading = st.slider("Expense Loading (%)", 0.0, 50.0, 15.0) / 100
    with load_col2:
        profit_margin = st.slider("Profit Margin (%)", 0.0, 20.0, 5.0) / 100
    with load_col3:
        reinsurance_loading = st.slider("Reinsurance Loading (%)", 0.0, 15.0, 3.0) / 100

    total_loading = 1 + expense_loading + profit_margin + reinsurance_loading

    pricing_segment = st.selectbox(
        "Select segmentation for pricing table:",
        ["Area", "DrivAge_bin", "VehAge_bin", "VehPower_bin", "BonusMalus_bin"],
        format_func=lambda x: SEGMENT_LABELS.get(x, x),
        key="pricing_seg",
    )

    seg_pricing = df.groupby(pricing_segment, observed=True).agg(
        Exposure=("Exposure", "sum"),
        TotalAmount=("TotalClaimAmount", "sum"),
        Policies=("ClaimNb", "count"),
    ).reset_index()
    seg_pricing["PurePremium"] = seg_pricing["TotalAmount"] / seg_pricing["Exposure"]
    seg_pricing["CommercialPremium"] = seg_pricing["PurePremium"] * total_loading
    seg_pricing["Relativity"] = seg_pricing["PurePremium"] / seg_pricing["PurePremium"].mean()

    pricing_display = seg_pricing[[pricing_segment, "Policies", "Exposure", "PurePremium",
                                   "CommercialPremium", "Relativity"]].copy()
    pricing_display.columns = [
        SEGMENT_LABELS.get(pricing_segment, pricing_segment),
        "Policies", "Exposure (PY)", "Pure Premium (€)", "Commercial Premium (€)", "Relativity",
    ]
    pricing_display["Pure Premium (€)"] = pricing_display["Pure Premium (€)"].apply(
        lambda x: fmt_number(x, prefix="€", decimals=2))
    pricing_display["Commercial Premium (€)"] = pricing_display["Commercial Premium (€)"].apply(
        lambda x: fmt_number(x, prefix="€", decimals=2))
    pricing_display["Relativity"] = pricing_display["Relativity"].round(3)

    st.dataframe(pricing_display, use_container_width=True, hide_index=True)

    csv_data = seg_pricing.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Pricing Table (CSV)",
        data=csv_data,
        file_name=f"pricing_table_{pricing_segment}.csv",
        mime="text/csv",
    )

    st.info(f"💡 **Total Loading Factor:** {total_loading:.2%} (Pure Premium × {total_loading:.2f} = Commercial Premium)")
