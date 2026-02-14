"""
Reusable Streamlit UI components shared across all pages.
"""

import streamlit as st


def fmt_number(value, decimals=0, prefix="", suffix=""):
    """Format a number with spaces as thousand separators."""
    if decimals == 0:
        formatted = f"{value:,.0f}".replace(",", " ")
    else:
        formatted = f"{value:,.{decimals}f}".replace(",", " ")
    return f"{prefix}{formatted}{suffix}"


def kpi_card(title, value, green=False):
    """Render a styled KPI card via unsafe HTML."""
    css_class = "kpi-card-green" if green else "kpi-card"
    st.markdown(f"""
    <div class="{css_class}">
        <h3>{title}</h3>
        <h1>{value}</h1>
    </div>
    """, unsafe_allow_html=True)


def section_header(text):
    """Render a styled section header via unsafe HTML."""
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)
