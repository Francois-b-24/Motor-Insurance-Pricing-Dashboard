"""
Page 5 — Model Monitoring & Drift Detection
A/E ratios, PSI drift detection, residual analysis, segment-level monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import COLORS, SEGMENT_LABELS
from src.components import kpi_card, section_header


def _compute_psi(expected, actual, bins=10):
    """Population Stability Index between two prediction distributions."""
    breakpoints = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    if len(breakpoints) < 3:
        return 0.0
    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual, bins=breakpoints)[0]
    exp_pct = np.clip(exp_counts / exp_counts.sum(), 0.001, None)
    act_pct = np.clip(act_counts / act_counts.sum(), 0.001, None)
    return np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))


def render(df, X, y, w, claim_count, df_model, results):
    """Render the Model Monitoring page."""

    st.title("📈 Model Monitoring & Drift Detection")
    st.markdown("""
    *Continuous monitoring ensures pricing models remain accurate.
    These are the key KPIs a pricing actuary tracks regularly.*
    """)

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
        "w": results["w_test"],
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
    fig_ae.add_trace(go.Scatter(
        x=ae_monthly["month"], y=ae_monthly["AE_GLM"],
        mode="lines+markers", name="GLM A/E",
        line=dict(color=COLORS["accent"], width=2.5),
    ))
    fig_ae.add_trace(go.Scatter(
        x=ae_monthly["month"], y=ae_monthly["AE_XGB"],
        mode="lines+markers", name="XGBoost A/E",
        line=dict(color=COLORS["warning"], width=2.5),
    ))
    fig_ae.add_hrect(y0=0.95, y1=1.05, fillcolor="green", opacity=0.1,
                      annotation_text="Acceptable range (\u00b15%)")
    fig_ae.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig_ae.update_layout(
        title="Monthly A/E Ratio", xaxis_title="Month",
        yaxis_title="Actual / Expected",
        template="plotly_white", height=420,
    )
    st.plotly_chart(fig_ae, width='stretch')

    # --- PSI ---
    section_header("📊 Population Stability Index (PSI)")

    first_half = mon_df[mon_df["month"] < "2024-07-01"]
    second_half = mon_df[mon_df["month"] >= "2024-07-01"]

    psi_glm = _compute_psi(first_half["glm_pred"].values, second_half["glm_pred"].values)
    psi_xgb = _compute_psi(first_half["xgb_pred"].values, second_half["xgb_pred"].values)

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
        fig_rh = px.histogram(
            x=residuals, nbins=80,
            title=f"{resid_model} — Residual Distribution",
            color_discrete_sequence=[COLORS["secondary"]],
        )
        fig_rh.add_vline(x=0, line_dash="dash", line_color="red")
        fig_rh.update_layout(
            template="plotly_white", height=380,
            xaxis_title="Residual", yaxis_title="Count",
        )
        st.plotly_chart(fig_rh, width='stretch')

    with col_res2:
        sample_idx = np.random.choice(len(pred), size=min(3000, len(pred)), replace=False)
        fig_rs = px.scatter(
            x=pred[sample_idx], y=residuals[sample_idx], opacity=0.3,
            title=f"{resid_model} — Residuals vs Predicted",
            color_discrete_sequence=[COLORS["primary"]],
        )
        fig_rs.add_hline(y=0, line_dash="dash", line_color="red")
        fig_rs.update_layout(
            template="plotly_white", height=380,
            xaxis_title="Predicted Value", yaxis_title="Residual",
        )
        st.plotly_chart(fig_rs, width='stretch')

    # --- Segment A/E ---
    section_header("🎯 Segment-Level A/E Monitoring")

    seg_var_mon = st.selectbox(
        "Select segment:",
        ["DrivAge_bin", "VehAge_bin", "VehPower_bin", "BonusMalus_bin", "Area"],
        format_func=lambda x: SEGMENT_LABELS.get(x, x),
        key="seg_mon",
    )

    X_test = results["X_test"]
    test_idx = X_test.index
    if seg_var_mon in df_model.columns:
        seg_values = df_model.loc[test_idx, seg_var_mon].values
    else:
        seg_values = np.random.choice(df[seg_var_mon].dropna().unique(), size=len(results["y_test"]))

    seg_ae_df = pd.DataFrame({
        "segment": seg_values, "y_true": results["y_test"],
        "glm_pred": results["glm_pred_test"], "w": results["w_test"],
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
        marker_color=[
            COLORS["accent"] if abs(ae - 1) > 0.05 else COLORS["success"]
            for ae in seg_ae["AE_ratio"]
        ],
        text=seg_ae["AE_ratio"].round(3), textposition="outside",
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

    overall_ae = (
        np.average(results["y_test"], weights=results["w_test"])
        / np.average(results["glm_pred_test"], weights=results["w_test"])
    )

    st.markdown(f"""
    | Metric | Value | Status |
    |--------|-------|--------|
    | Overall A/E (GLM) | {overall_ae:.3f} | {"✅ OK" if abs(overall_ae - 1) < 0.05 else "⚠️ Investigate"} |
    | GLM Prediction PSI | {psi_glm:.4f} | {"✅ Stable" if psi_glm < 0.1 else "⚠️ Drift"} |
    | XGBoost Prediction PSI | {psi_xgb:.4f} | {"✅ Stable" if psi_xgb < 0.1 else "⚠️ Drift"} |

    **Recommended Actions:**
    - Monitor A/E ratio monthly at portfolio and segment level
    - Trigger model review if A/E deviates beyond \u00b15% for 3 consecutive months
    - Recalibrate if PSI exceeds 0.25 on key variables
    - Compare GLM vs XGBoost predictions quarterly to detect non-linear patterns
    - Document all monitoring findings for regulatory reporting
    """)
