"""
Page 4 — GLM vs XGBoost Model Comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import COLORS
from src.utils import section_header


def render(df, get_modeling_data, run_models):
    """Render the GLM vs XGBoost comparison page."""

    st.title("🤖 GLM vs XGBoost — Model Comparison")
    st.markdown("""
    *Benchmarking the traditional actuarial approach (Poisson GLM) against a modern ML model (XGBoost).
    Both approaches have merits — the key is understanding when to use each.*
    """)

    X, y, w, claim_count, df_model = get_modeling_data(df)
    results = run_models(X, y, w, claim_count, df_model)

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
        glm_gini = results["glm_metrics_test"]["Gini"]
        xgb_gini = results["xgb_metrics_test"]["Gini"]
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
    st.plotly_chart(fig_lift, width="stretch")

    # --- Double Lift ---
    section_header("⚖️ Double Lift — Where Do Models Disagree?")

    y_test = results["y_test"]
    glm_pred = results["glm_pred_test"]
    xgb_pred = results["xgb_pred_test"]
    w_test = results["w_test"]

    ratio = xgb_pred / np.clip(glm_pred, 1e-6, None)
    dl_df = pd.DataFrame({
        "ratio": ratio, "y_true": y_test,
        "glm_pred": glm_pred, "xgb_pred": xgb_pred, "w": w_test,
    })
    dl_df["decile"] = pd.qcut(dl_df["ratio"], q=10, labels=False, duplicates="drop")

    dl_agg = dl_df.groupby("decile").apply(
        lambda g: pd.Series({
            "avg_actual": np.average(g["y_true"], weights=g["w"]),
            "avg_glm": np.average(g["glm_pred"], weights=g["w"]),
            "avg_xgb": np.average(g["xgb_pred"], weights=g["w"]),
        }),
        include_groups=False,
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
        template="plotly_white", height=400,
    )
    st.plotly_chart(fig_dl, width="stretch")

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
            "Mean |SHAP|": np.abs(shap_values).mean(axis=0),
        }).sort_values("Mean |SHAP|", ascending=True).tail(15)

        fig_shap = px.bar(mean_shap, x="Mean |SHAP|", y="Feature", orientation="h",
                           color="Mean |SHAP|", color_continuous_scale="Reds",
                           title="Top 15 Features by Mean |SHAP| Value")
        fig_shap.update_layout(template="plotly_white", height=500, showlegend=False)
        st.plotly_chart(fig_shap, width="stretch")

        if "BonusMalus" in X_sample.columns:
            st.markdown("#### SHAP Dependence — Bonus-Malus")
            bm_idx = list(X_sample.columns).index("BonusMalus")
            fig_dep = px.scatter(
                x=X_sample["BonusMalus"], y=shap_values[:, bm_idx],
                color=X_sample["DrivAge"] if "DrivAge" in X_sample.columns else None,
                labels={"x": "Bonus-Malus", "y": "SHAP Value", "color": "Driver Age"},
                title="SHAP Dependence: Bonus-Malus (colored by Driver Age)",
                color_continuous_scale="Viridis", opacity=0.5,
            )
            fig_dep.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_dep, width="stretch")

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
                     "⚠️ More complex", "⚠️ Needs explainability layer"],
    })
    st.dataframe(pros_cons, width="stretch", hide_index=True)
