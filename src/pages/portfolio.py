"""
Page 1 — Portfolio Overview
"""

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import COLORS, SEGMENT_LABELS
from src.utils import fmt_number, kpi_card, section_header


def render(df):
    """Render the Portfolio Overview page."""

    st.title("📊 Portfolio Overview")
    st.markdown("*Explore the structure, risk distribution, and segmentation of the motor insurance portfolio.*")
    st.caption("**Analysis period:** 2003 – 2005 | Source: freMTPL2freq + freMTPL2sev (French Motor TPL)")

    # --- KPI Row ---
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        kpi_card("Policies", fmt_number(len(df)))
    with col2:
        kpi_card("Total Exposure", fmt_number(df["Exposure"].sum(), suffix="PY"))
    with col3:
        kpi_card("Total Claims", fmt_number(df["ClaimNb"].sum()))
    with col4:
        kpi_card("Avg Frequency", f"{df['ClaimNb'].sum() / df['Exposure'].sum():.4f}")
    with col5:
        avg_pp = df["TotalClaimAmount"].sum() / df["Exposure"].sum()
        kpi_card("Avg Pure Premium", fmt_number(avg_pp, prefix="€"))
    with col6:
        avg_sev = (
            df.loc[df["ClaimNb"] > 0, "TotalClaimAmount"].sum()
            / df.loc[df["ClaimNb"] > 0, "ClaimNb"].sum()
            if (df["ClaimNb"] > 0).any()
            else 0
        )
        kpi_card("Avg Severity", fmt_number(avg_sev, prefix="€"))

    st.markdown("")

    # --- Exposure & Frequency by Segment ---
    section_header("📋 Risk Segmentation Analysis")

    segment_var = st.selectbox(
        "Select segmentation variable:",
        ["DrivAge_bin", "VehAge_bin", "VehPower_bin", "BonusMalus_bin", "Area", "VehGas"],
        format_func=lambda x: SEGMENT_LABELS.get(x, x),
    )

    seg_data = df.groupby(segment_var, observed=True).agg(
        Policies=("ClaimNb", "count"),
        Exposure=("Exposure", "sum"),
        Claims=("ClaimNb", "sum"),
        TotalAmount=("TotalClaimAmount", "sum"),
    ).reset_index()
    seg_data["Frequency"] = seg_data["Claims"] / seg_data["Exposure"]
    seg_data["PurePremium"] = seg_data["TotalAmount"] / seg_data["Exposure"]
    seg_data["Severity"] = np.where(
        seg_data["Claims"] > 0, seg_data["TotalAmount"] / seg_data["Claims"], 0
    )
    seg_data["Exposure_pct"] = seg_data["Exposure"] / seg_data["Exposure"].sum() * 100

    col_left, col_right = st.columns(2)

    with col_left:
        fig_exp = go.Figure()
        fig_exp.add_trace(go.Bar(
            x=seg_data[segment_var].astype(str), y=seg_data["Exposure_pct"],
            marker_color=COLORS["secondary"], name="Exposure %",
            text=seg_data["Exposure_pct"].round(1).astype(str) + "%", textposition="outside",
        ))
        fig_exp.update_layout(
            title="Exposure Distribution (%)",
            xaxis_title=SEGMENT_LABELS.get(segment_var, segment_var),
            yaxis_title="% of Total Exposure",
            template="plotly_white", height=400, showlegend=False,
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig_exp, width="stretch")

    with col_right:
        avg_freq = df["ClaimNb"].sum() / df["Exposure"].sum()
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Bar(
            x=seg_data[segment_var].astype(str), y=seg_data["Frequency"],
            marker_color=[
                COLORS["accent"] if f > avg_freq else COLORS["success"]
                for f in seg_data["Frequency"]
            ],
            text=seg_data["Frequency"].round(4).astype(str), textposition="outside",
        ))
        fig_freq.add_hline(
            y=avg_freq, line_dash="dash", line_color=COLORS["primary"],
            annotation_text=f"Portfolio avg: {avg_freq:.4f}",
        )
        fig_freq.update_layout(
            title="Claim Frequency by Segment",
            xaxis_title=SEGMENT_LABELS.get(segment_var, segment_var),
            yaxis_title="Frequency (claims/exposure)",
            template="plotly_white", height=400, showlegend=False,
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig_freq, width="stretch")

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
        text_auto=".4f", aspect="auto",
    )
    fig_heat.update_layout(
        title="Claim Frequency: Driver Age × Bonus-Malus",
        template="plotly_white", height=420,
        xaxis=dict(tickangle=-45, dtick=1),
        yaxis=dict(dtick=1),
    )
    st.plotly_chart(fig_heat, width="stretch")

    # --- BM Distribution ---
    section_header("📉 Bonus-Malus Distribution")

    col_bm1, col_bm2 = st.columns(2)
    with col_bm1:
        fig_bm = px.histogram(
            df, x="BonusMalus", nbins=50,
            color_discrete_sequence=[COLORS["secondary"]],
            title="Bonus-Malus Coefficient Distribution",
        )
        fig_bm.update_layout(
            template="plotly_white", height=350,
            xaxis_title="Bonus-Malus", yaxis_title="Number of Policies",
        )
        st.plotly_chart(fig_bm, width="stretch")

    with col_bm2:
        bm_freq = df.groupby("BonusMalus_bin", observed=True).agg(
            Exposure=("Exposure", "sum"), Claims=("ClaimNb", "sum")
        ).reset_index()
        bm_freq["Frequency"] = bm_freq["Claims"] / bm_freq["Exposure"]
        fig_bm_freq = px.bar(
            bm_freq, x="BonusMalus_bin", y="Frequency",
            color="Frequency", color_continuous_scale="Reds",
            title="Frequency by Bonus-Malus Band",
        )
        fig_bm_freq.update_layout(
            template="plotly_white", height=350,
            xaxis_title="Bonus-Malus", yaxis_title="Frequency",
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig_bm_freq, width="stretch")
