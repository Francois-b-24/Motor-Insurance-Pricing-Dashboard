"""
Page 6 — Commercial Pricing & Elasticity
Executive summary, pricing waterfall, price elasticity simulation,
segment-level optimization, competitive positioning, data quality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import COLORS, SEGMENT_LABELS
from src.components import fmt_number, kpi_card, section_header


def render(df, X, y, w, claim_count, df_model, results):
    """Render the Commercial Pricing & Elasticity page."""

    st.title("💼 Commercial Pricing & Elasticity")
    st.markdown("""
    *From technical risk premium to commercial pricing: incorporating **expenses**, **profit targets**,
    **competitive positioning**, and **price elasticity** to build a market-ready tariff.*

    This page demonstrates the end-to-end pricing workflow that bridges actuarial modeling
    with business strategy — a core competency for pricing actuaries in the automotive insurance space.
    """)

    # ── Executive Summary ────────────────────────────────────────────
    section_header("📋 Executive Summary")

    total_exposure = df["Exposure"].sum()
    total_claims_n = df["ClaimNb"].sum()
    total_amount = df["TotalClaimAmount"].sum()
    avg_freq = total_claims_n / total_exposure
    claims_mask = df["ClaimNb"] > 0
    avg_sev = (
        df.loc[claims_mask, "TotalClaimAmount"].sum()
        / df.loc[claims_mask, "ClaimNb"].sum()
        if claims_mask.any() else 0
    )
    avg_pp = total_amount / total_exposure

    exec_col1, exec_col2, exec_col3 = st.columns(3)
    with exec_col1:
        kpi_card("Portfolio Pure Premium", fmt_number(avg_pp, prefix="\u20ac", decimals=2))
    with exec_col2:
        target_lr = 0.65
        implied_commercial = avg_pp / target_lr
        kpi_card("Implied Commercial Premium", fmt_number(implied_commercial, prefix="\u20ac", decimals=2), green=True)
    with exec_col3:
        kpi_card("Target Loss Ratio", f"{target_lr:.0%}")

    st.markdown(f"""
    **Key Portfolio Indicators:**
    | Metric | Value | Benchmark |
    |--------|-------|-----------|
    | Average Frequency | {avg_freq:.4f} | ~6-8% for motor TPL |
    | Average Severity | {fmt_number(avg_sev, prefix='\u20ac')} | Varies by market |
    | Pure Premium | {fmt_number(avg_pp, prefix='\u20ac', decimals=2)} | Technical price floor |
    | Implied Commercial Premium (at {target_lr:.0%} LR) | {fmt_number(implied_commercial, prefix='\u20ac', decimals=2)} | Market-dependent |

    *These figures establish the **technical price floor**. The commercial premium must cover risk, expenses,
    profit margin, and reinsurance costs while remaining competitive.*
    """)

    # ── Commercial Pricing Waterfall ─────────────────────────────────
    section_header("💧 Commercial Pricing Waterfall")

    st.markdown("""
    The **pricing waterfall** shows how the technical risk premium is loaded to arrive at the
    final commercial premium. Each layer represents a strategic decision aligned with business objectives.
    """)

    wf_col1, wf_col2 = st.columns([1, 2])

    with wf_col1:
        st.markdown("#### Loading Parameters")
        wf_expense = st.slider("Acquisition & Admin Expenses (%)", 5.0, 35.0, 18.0, key="wf_exp") / 100
        wf_commission = st.slider("Distribution Commission (%)", 0.0, 20.0, 8.0, key="wf_comm") / 100
        wf_profit = st.slider("Target Profit Margin (%)", 0.0, 15.0, 5.0, key="wf_prof") / 100
        wf_reins = st.slider("Reinsurance Cost (%)", 0.0, 10.0, 3.0, key="wf_reins") / 100
        wf_tax = st.slider("Insurance Tax (%)", 0.0, 20.0, 9.0, key="wf_tax") / 100

    with wf_col2:
        risk_premium = avg_pp
        expense_amount = risk_premium * wf_expense
        commission_amount = risk_premium * wf_commission
        profit_amount = risk_premium * wf_profit
        reins_amount = risk_premium * wf_reins
        subtotal_before_tax = risk_premium + expense_amount + commission_amount + profit_amount + reins_amount
        tax_amount = subtotal_before_tax * wf_tax
        commercial_premium = subtotal_before_tax + tax_amount

        fig_waterfall = go.Figure(go.Waterfall(
            name="Pricing Waterfall",
            orientation="v",
            x=["Risk Premium", "Expenses", "Commission", "Profit Margin",
               "Reinsurance", "Subtotal", "Tax", "Commercial Premium"],
            y=[risk_premium, expense_amount, commission_amount, profit_amount,
               reins_amount, 0, tax_amount, 0],
            measure=["absolute", "relative", "relative", "relative",
                      "relative", "total", "relative", "total"],
            textposition="outside",
            text=[f"\u20ac{risk_premium:.2f}", f"+\u20ac{expense_amount:.2f}",
                  f"+\u20ac{commission_amount:.2f}", f"+\u20ac{profit_amount:.2f}",
                  f"+\u20ac{reins_amount:.2f}", f"\u20ac{subtotal_before_tax:.2f}",
                  f"+\u20ac{tax_amount:.2f}", f"\u20ac{commercial_premium:.2f}"],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": COLORS["secondary"]}},
            decreasing={"marker": {"color": COLORS["accent"]}},
            totals={"marker": {"color": COLORS["primary"]}},
        ))
        fig_waterfall.update_layout(
            title="From Risk Premium to Commercial Premium",
            yaxis_title="Amount (\u20ac)", template="plotly_white", height=450,
            showlegend=False,
        )
        st.plotly_chart(fig_waterfall, width='stretch')

    expected_lr = risk_premium / commercial_premium * (1 + wf_tax)
    combined_ratio = (risk_premium + expense_amount + commission_amount) / (commercial_premium / (1 + wf_tax))

    lr_col1, lr_col2, lr_col3, lr_col4 = st.columns(4)
    with lr_col1:
        kpi_card("Expected Loss Ratio", f"{expected_lr:.1%}")
    with lr_col2:
        kpi_card("Combined Ratio", f"{combined_ratio:.1%}")
    with lr_col3:
        kpi_card("Commercial Premium", fmt_number(commercial_premium, prefix="\u20ac", decimals=2), green=True)
    with lr_col4:
        kpi_card("Loading Factor", f"{commercial_premium / risk_premium:.2f}x")

    # ── Price Elasticity Simulation ──────────────────────────────────
    section_header("📉 Price Elasticity Simulation")

    st.markdown("""
    **Price elasticity** measures how demand (policy count) responds to premium changes.
    This is essential for optimizing revenue: setting the price where
    marginal revenue equals marginal cost.

    The simulation below uses a **logit-based demand model** — standard in insurance pricing —
    where the probability of purchasing a policy decreases as the premium increases.
    """)

    el_col1, el_col2 = st.columns([1, 2])

    with el_col1:
        st.markdown("#### Elasticity Parameters")
        base_elasticity = st.slider(
            "Base Price Elasticity (\u03b5)", -3.0, -0.1, -0.8, step=0.1,
            help="\u03b5 < -1: elastic demand | \u03b5 > -1: inelastic demand",
        )
        current_policies = len(df)
        current_premium = commercial_premium
        premium_range_pct = st.slider("Premium Variation Range (%)", 10, 50, 30)

        st.markdown(f"""
        **Current state:**
        - Policies: {current_policies:,}
        - Premium: \u20ac{current_premium:.2f}
        - Elasticity: {base_elasticity:.1f}

        *Elasticity of {base_elasticity:.1f} means a 10% price increase
        leads to a ~{abs(base_elasticity) * 10:.0f}% drop in demand.*
        """)

    with el_col2:
        price_changes = np.linspace(-premium_range_pct / 100, premium_range_pct / 100, 50)
        prices = current_premium * (1 + price_changes)

        demand = current_policies * np.exp(base_elasticity * price_changes)
        revenue = prices * demand
        profit = (prices - risk_premium) * demand
        loss_ratio = risk_premium / prices

        optimal_idx = np.argmax(profit)
        optimal_price = prices[optimal_idx]
        optimal_demand = demand[optimal_idx]
        optimal_profit = profit[optimal_idx]

        fig_elast = make_subplots(specs=[[{"secondary_y": True}]])

        fig_elast.add_trace(
            go.Scatter(
                x=prices, y=demand, name="Demand (policies)",
                line=dict(color=COLORS["secondary"], width=2.5),
                fill="tozeroy", fillcolor="rgba(61,123,199,0.1)",
            ),
            secondary_y=False,
        )
        fig_elast.add_trace(
            go.Scatter(
                x=prices, y=profit, name="Profit (\u20ac)",
                line=dict(color=COLORS["success"], width=2.5),
            ),
            secondary_y=True,
        )

        fig_elast.add_trace(
            go.Scatter(
                x=[optimal_price], y=[optimal_demand],
                name=f"Optimal: \u20ac{optimal_price:.2f}",
                mode="markers",
                marker=dict(size=14, color=COLORS["accent"], symbol="star"),
            ),
            secondary_y=False,
        )

        fig_elast.add_vline(x=current_premium, line_dash="dash", line_color="gray",
                             annotation_text="Current price")

        fig_elast.update_layout(
            title="Price Elasticity: Demand & Profit vs Premium Level",
            template="plotly_white", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig_elast.update_yaxes(title_text="Demand (policies)", secondary_y=False)
        fig_elast.update_yaxes(title_text="Total Profit (\u20ac)", secondary_y=True)
        st.plotly_chart(fig_elast, width='stretch')

    opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
    with opt_col1:
        kpi_card("Optimal Premium", fmt_number(optimal_price, prefix="\u20ac", decimals=2), green=True)
    with opt_col2:
        kpi_card("Expected Demand", fmt_number(optimal_demand))
    with opt_col3:
        kpi_card("Max Profit", fmt_number(optimal_profit, prefix="\u20ac"))
    with opt_col4:
        price_delta = (optimal_price - current_premium) / current_premium * 100
        kpi_card("vs Current Price", f"{price_delta:+.1f}%")

    # ── Segment-Level Elasticity ─────────────────────────────────────
    section_header("🎯 Segment-Level Pricing Optimization")

    st.markdown("""
    Different customer segments have different price sensitivities. Young drivers and high-risk
    segments typically show **lower elasticity** (captive demand), while standard-risk segments
    are more price-sensitive (competitive market).
    """)

    seg_elast_var = st.selectbox(
        "Segmentation variable:",
        ["DrivAge_bin", "BonusMalus_bin", "Area", "VehPower_bin", "VehGas"],
        format_func=lambda x: SEGMENT_LABELS.get(x, x),
        key="seg_elast",
    )

    seg_pp = df.groupby(seg_elast_var, observed=True).agg(
        Policies=("ClaimNb", "count"),
        Exposure=("Exposure", "sum"),
        Claims=("ClaimNb", "sum"),
        TotalAmount=("TotalClaimAmount", "sum"),
    ).reset_index()
    seg_pp["Frequency"] = seg_pp["Claims"] / seg_pp["Exposure"]
    seg_pp["PurePremium"] = seg_pp["TotalAmount"] / seg_pp["Exposure"]
    seg_pp["CommercialPremium"] = seg_pp["PurePremium"] * (1 + wf_expense + wf_commission + wf_profit + wf_reins) * (1 + wf_tax)
    seg_pp["MarketShare"] = seg_pp["Exposure"] / seg_pp["Exposure"].sum() * 100

    np.random.seed(42)
    n_segs = len(seg_pp)
    seg_pp["Elasticity"] = np.round(np.random.uniform(-1.2, -0.3, n_segs), 2)
    seg_pp["OptimalAdjustment"] = seg_pp["Elasticity"].apply(
        lambda e: f"{-1/(2*abs(e)) * 10:+.1f}%" if abs(e) > 0 else "+0.0%"
    )
    seg_pp["RiskCategory"] = seg_pp["Frequency"].apply(
        lambda f: "High Risk" if f > avg_freq * 1.2 else ("Low Risk" if f < avg_freq * 0.8 else "Standard")
    )

    display_cols = seg_pp[[seg_elast_var, "Policies", "MarketShare", "PurePremium",
                            "CommercialPremium", "Frequency", "Elasticity", "RiskCategory"]].copy()
    display_cols.columns = [
        SEGMENT_LABELS.get(seg_elast_var, seg_elast_var),
        "Policies", "Market Share (%)", "Pure Premium (\u20ac)",
        "Commercial Premium (\u20ac)", "Frequency", "Elasticity (\u03b5)", "Risk Category",
    ]
    display_cols["Pure Premium (\u20ac)"] = display_cols["Pure Premium (\u20ac)"].round(2)
    display_cols["Commercial Premium (\u20ac)"] = display_cols["Commercial Premium (\u20ac)"].round(2)
    display_cols["Market Share (%)"] = display_cols["Market Share (%)"].round(1)
    display_cols["Frequency"] = display_cols["Frequency"].round(4)

    st.dataframe(display_cols, use_container_width=True, hide_index=True)

    fig_seg_opt = make_subplots(specs=[[{"secondary_y": True}]])
    fig_seg_opt.add_trace(
        go.Bar(
            x=seg_pp[seg_elast_var].astype(str), y=seg_pp["CommercialPremium"],
            name="Commercial Premium (\u20ac)", marker_color=COLORS["secondary"],
        ),
        secondary_y=False,
    )
    fig_seg_opt.add_trace(
        go.Scatter(
            x=seg_pp[seg_elast_var].astype(str), y=seg_pp["Elasticity"],
            name="Elasticity (\u03b5)", mode="lines+markers",
            line=dict(color=COLORS["accent"], width=2.5),
        ),
        secondary_y=True,
    )
    fig_seg_opt.update_layout(
        title=f"Commercial Premium & Price Elasticity by {SEGMENT_LABELS.get(seg_elast_var, seg_elast_var)}",
        template="plotly_white", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(tickangle=-45),
    )
    fig_seg_opt.update_yaxes(title_text="Commercial Premium (\u20ac)", secondary_y=False)
    fig_seg_opt.update_yaxes(title_text="Elasticity (\u03b5)", secondary_y=True)
    st.plotly_chart(fig_seg_opt, width='stretch')

    # ── Competitive Positioning ──────────────────────────────────────
    section_header("🏁 Competitive Positioning Analysis")

    st.markdown("""
    *Simulated competitive landscape analysis. In production, this would be fed by market data,
    competitor rate filings, and web-scraped quotes. For automotive insurers like Stellantis Insurance,
    the competitive set includes both traditional insurers and OEM-backed programs.*
    """)

    np.random.seed(42)
    competitors = [
        "Our Premium", "Competitor A\n(Traditional)", "Competitor B\n(Digital)",
        "Competitor C\n(OEM Program)", "Competitor D\n(Bancassurance)",
    ]
    comp_premiums = [
        commercial_premium,
        commercial_premium * np.random.uniform(0.9, 1.15),
        commercial_premium * np.random.uniform(0.85, 1.0),
        commercial_premium * np.random.uniform(0.95, 1.1),
        commercial_premium * np.random.uniform(0.88, 1.05),
    ]

    comp_colors = [COLORS["primary"], "#95a5a6", "#95a5a6", "#95a5a6", "#95a5a6"]
    comp_borders = [COLORS["accent"], "rgba(0,0,0,0)", "rgba(0,0,0,0)", "rgba(0,0,0,0)", "rgba(0,0,0,0)"]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=competitors, y=comp_premiums,
        marker_color=comp_colors,
        marker_line_color=comp_borders,
        marker_line_width=3,
        text=[f"\u20ac{p:.2f}" for p in comp_premiums],
        textposition="outside",
    ))
    fig_comp.add_hline(y=risk_premium, line_dash="dot", line_color=COLORS["accent"],
                        annotation_text=f"Risk Premium: \u20ac{risk_premium:.2f}")
    fig_comp.update_layout(
        title="Competitive Premium Benchmark (Simulated)",
        yaxis_title="Annual Premium (\u20ac)", template="plotly_white", height=420,
        showlegend=False,
    )
    st.plotly_chart(fig_comp, width='stretch')

    our_rank = sorted(range(len(comp_premiums)), key=lambda i: comp_premiums[i]).index(0) + 1
    avg_competitor = np.mean(comp_premiums[1:])
    position_vs_market = (commercial_premium - avg_competitor) / avg_competitor * 100

    pos_col1, pos_col2, pos_col3 = st.columns(3)
    with pos_col1:
        kpi_card("Market Rank", f"{our_rank}/{len(competitors)}")
    with pos_col2:
        kpi_card("vs Market Average", f"{position_vs_market:+.1f}%")
    with pos_col3:
        kpi_card("Margin vs Risk Premium", f"{((commercial_premium / risk_premium) - 1) * 100:.1f}%", green=True)

    # ── Data Quality Monitoring ──────────────────────────────────────
    section_header("🔍 Data Quality Assessment")

    st.markdown("""
    *Reliable pricing requires reliable data. This section monitors key data quality indicators —
    a critical responsibility highlighted in the role.*
    """)

    dq_col1, dq_col2, dq_col3 = st.columns(3)

    with dq_col1:
        st.markdown("#### Completeness")
        completeness = {}
        key_fields = [
            "Exposure", "ClaimNb", "DrivAge", "VehAge", "VehPower",
            "BonusMalus", "Area", "VehGas", "TotalClaimAmount",
        ]
        for field in key_fields:
            if field in df.columns:
                pct = (1 - df[field].isna().mean()) * 100
                completeness[field] = pct

        dq_df = pd.DataFrame({
            "Field": completeness.keys(),
            "Completeness (%)": [f"{v:.1f}%" for v in completeness.values()],
            "Status": [
                "✅" if v >= 99 else ("⚠️" if v >= 95 else "❌")
                for v in completeness.values()
            ],
        })
        st.dataframe(dq_df, use_container_width=True, hide_index=True)

    with dq_col2:
        st.markdown("#### Outlier Detection")
        outliers = {}
        outlier_fields = {
            "BonusMalus": (50, 230), "Exposure": (0.01, 1.0),
            "DrivAge": (18, 100), "VehAge": (0, 50),
        }
        for field, (lo, hi) in outlier_fields.items():
            if field in df.columns:
                n_out = ((df[field] < lo) | (df[field] > hi)).sum()
                outliers[field] = f"{n_out} ({n_out/len(df)*100:.2f}%)"

        out_df = pd.DataFrame({
            "Field": outliers.keys(),
            "Out-of-Range": outliers.values(),
        })
        st.dataframe(out_df, use_container_width=True, hide_index=True)

    with dq_col3:
        st.markdown("#### Distribution Checks")
        st.metric("Total Policies", f"{len(df):,}")
        st.metric("Policies with Claims", f"{(df['ClaimNb'] > 0).sum():,} ({(df['ClaimNb'] > 0).mean()*100:.1f}%)")
        st.metric("Avg Exposure", f"{df['Exposure'].mean():.3f} PY")
        zero_exposure = (df["Exposure"] <= 0.01).sum()
        st.metric("Near-Zero Exposure", f"{zero_exposure} ({zero_exposure/len(df)*100:.2f}%)")

    # ── Pricing Recommendations ──────────────────────────────────────
    section_header("📝 Pricing Recommendations & Next Steps")

    seg_analysis = df.groupby("DrivAge_bin", observed=True).agg(
        Exposure=("Exposure", "sum"), Claims=("ClaimNb", "sum"),
        Amount=("TotalClaimAmount", "sum"),
    ).reset_index()
    seg_analysis["Frequency"] = seg_analysis["Claims"] / seg_analysis["Exposure"]
    seg_analysis["PP"] = seg_analysis["Amount"] / seg_analysis["Exposure"]
    highest_risk = seg_analysis.loc[seg_analysis["Frequency"].idxmax(), "DrivAge_bin"]
    lowest_risk = seg_analysis.loc[seg_analysis["Frequency"].idxmin(), "DrivAge_bin"]

    st.markdown(f"""
    #### Rate Adequacy Assessment
    | Finding | Detail | Action |
    |---------|--------|--------|
    | **Highest risk segment** | Driver age: {highest_risk} | Consider surcharge or targeted UBI program |
    | **Lowest risk segment** | Driver age: {lowest_risk} | Potential for competitive pricing to gain market share |
    | **Bonus-Malus effectiveness** | Strong predictor (GLM relativity visible in modeling tab) | Maintain as primary rating factor |
    | **Area effect** | Urban areas show higher frequency | Align with density-based rating, consider telematics |

    #### Strategic Recommendations
    1. **Rate Refresh:** Implement annual rate review cycle — current GLM coefficients should be
       re-estimated with the most recent 24 months of experience data
    2. **UBI Score Integration:** For Stellantis vehicles equipped with connected car technology,
       develop a Usage-Based Insurance score as a supplementary rating factor
    3. **Elasticity-Based Optimization:** Apply segment-level elasticity to optimize the rate
       structure — reduce premiums where demand is elastic, increase where captive
    4. **Multi-Market Harmonization:** Adapt this pricing framework for deployment across
       European markets, adjusting for local regulation, competition, and loss patterns
    5. **Product Bundling:** Leverage Stellantis ecosystem to offer bundled pricing
       (Car Insurance + Breakdown + GAP) with cross-sell elasticity modeling

    #### Model Governance
    - Quarterly model performance review (A/E, PSI, Gini stability)
    - Annual full model re-estimation with expanded feature set
    - Document all pricing decisions for regulatory compliance (Solvency II)
    - Maintain audit trail for rate changes and their justification
    """)
