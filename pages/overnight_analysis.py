"""
Overnight vs Regular Session Analysis Page — Session Decomposition & Movement Attribution
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.calculations import compute_cumulative_performance
from src.statistics import compute_distribution_summary
from src.charts import (
    plot_cumulative_performance,
    apply_chart_theme,
)


def render_overnight_page(df_daily: pd.DataFrame):
    st.markdown("## 🌙 Overnight vs Regular Session Dynamics")
    st.caption("Investigating movement attribution: Where does NASDAQ-100 historical drift and volatility occur?")

    if df_daily.empty:
        st.warning("No data available.")
        return

    df_clean = df_daily.dropna(subset=["overnight_return_pct", "regular_return_pct", "total_return_pct"]).copy()
    n = len(df_clean)

    # Statistical Comparison Metrics
    over_mean = df_clean["overnight_return_pct"].mean()
    reg_mean = df_clean["regular_return_pct"].mean()
    tot_mean = df_clean["total_return_pct"].mean()

    over_vol = df_clean["overnight_return"].std() * np.sqrt(252) * 100
    reg_vol = df_clean["regular_return"].std() * np.sqrt(252) * 100
    tot_vol = df_clean["total_return"].std() * np.sqrt(252) * 100

    over_pos = (df_clean["overnight_return_pct"] > 0).sum() / n * 100
    reg_pos = (df_clean["regular_return_pct"] > 0).sum() / n * 100

    # Variance Attribution Share
    var_over = df_clean["overnight_return"].var()
    var_reg = df_clean["regular_return"].var()
    over_var_share = (var_over / (var_over + var_reg)) * 100 if (var_over + var_reg) > 0 else 50.0

    # KPI Top Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overnight Mean Return", f"{over_mean:.3f}%", f"Vol: {over_vol:.1f}%")
    c2.metric("Regular Session Mean", f"{reg_mean:.3f}%", f"Vol: {reg_vol:.1f}%")
    c3.metric("Overnight Variance Share", f"{over_var_share:.1f}%", f"Regular: {100 - over_var_share:.1f}%")
    c4.metric("Positive Overnight Drift", f"{over_pos:.1f}%", f"Regular: {reg_pos:.1f}%")

    st.markdown("---")

    # Cumulative Returns Decomposition
    st.markdown("### 📈 Cumulative Return Trajectories: Overnight vs Regular Session")
    st.caption("Simulates compound growth of pure overnight holding (Close_{t-1} → Open_t) vs pure regular session holding (Open_t → Close_t).")
    df_cum = compute_cumulative_performance(df_clean)
    fig_cum = plot_cumulative_performance(df_cum)
    st.plotly_chart(fig_cum, use_container_width=True)

    # Side-by-side Comparative Distributions
    col_dist1, col_dist2 = st.columns(2)

    with col_dist1:
        st.markdown("### 🌙 Overnight Return Distribution")
        fig_over = go.Figure()
        fig_over.add_trace(go.Histogram(
            x=df_clean["overnight_return_pct"],
            nbinsx=45,
            marker=dict(color="#f59e0b", line=dict(color="white", width=0.5)),
            name="Overnight",
            hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
        ))
        fig_over.add_vline(x=over_mean, line=dict(color="#10b981", width=2, dash="dash"), annotation_text=f"Mean: {over_mean:.3f}%")
        fig_over.update_xaxes(title="Overnight Return (%)", ticksuffix="%")
        fig_over.update_yaxes(title="Count")
        st.plotly_chart(apply_chart_theme(fig_over, "Overnight Return Histogram"), use_container_width=True)

    with col_dist2:
        st.markdown("### ☀️ Regular Session Return Distribution")
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Histogram(
            x=df_clean["regular_return_pct"],
            nbinsx=45,
            marker=dict(color="#a855f7", line=dict(color="white", width=0.5)),
            name="Regular Session",
            hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
        ))
        fig_reg.add_vline(x=reg_mean, line=dict(color="#10b981", width=2, dash="dash"), annotation_text=f"Mean: {reg_mean:.3f}%")
        fig_reg.update_xaxes(title="Regular Session Return (%)", ticksuffix="%")
        fig_reg.update_yaxes(title="Count")
        st.plotly_chart(apply_chart_theme(fig_reg, "Regular Session Return Histogram"), use_container_width=True)

    # Detailed Statistical Attribution Table
    st.markdown("### 📋 Statistical Attribution Comparison Table")
    dist_over = compute_distribution_summary(df_clean["overnight_return_pct"], label="Overnight")
    dist_reg = compute_distribution_summary(df_clean["regular_return_pct"], label="Regular Session")
    dist_tot = compute_distribution_summary(df_clean["total_return_pct"], label="Total Close-to-Close")

    comp_table = pd.DataFrame([
        ("Sample Size (Sessions)", f"{n:,}", f"{n:,}", f"{n:,}"),
        ("Mean Return", f"{dist_over['mean']:.3f}%", f"{dist_reg['mean']:.3f}%", f"{dist_tot['mean']:.3f}%"),
        ("Median Return", f"{dist_over['median']:.3f}%", f"{dist_reg['median']:.3f}%", f"{dist_tot['median']:.3f}%"),
        ("Standard Deviation", f"{dist_over['std']:.3f}%", f"{dist_reg['std']:.3f}%", f"{dist_tot['std']:.3f}%"),
        ("Annualized Volatility", f"{over_vol:.2f}%", f"{reg_vol:.2f}%", f"{tot_vol:.2f}%"),
        ("Average Absolute Move", f"{df_clean['abs_overnight_return_pct'].mean():.3f}%", f"{df_clean['abs_regular_return_pct'].mean():.3f}%", f"{df_clean['abs_total_return_pct'].mean():.3f}%"),
        ("Positive Sessions %", f"{dist_over['positive_pct']:.1f}%", f"{dist_reg['positive_pct']:.1f}%", f"{dist_tot['positive_pct']:.1f}%"),
        ("Negative Sessions %", f"{dist_over['negative_pct']:.1f}%", f"{dist_reg['negative_pct']:.1f}%", f"{dist_tot['negative_pct']:.1f}%"),
        ("10th Percentile", f"{dist_over['p10']:.3f}%", f"{dist_reg['p10']:.3f}%", f"{dist_tot['p10']:.3f}%"),
        ("90th Percentile", f"{dist_over['p90']:.3f}%", f"{dist_reg['p90']:.3f}%", f"{dist_tot['p90']:.3f}%"),
        ("Skewness", f"{dist_over['skewness']:.3f}", f"{dist_reg['skewness']:.3f}", f"{dist_tot['skewness']:.3f}"),
    ], columns=["Statistic", "Overnight (Close_{t-1} → Open_t)", "Regular Session (Open_t → Close_t)", "Total (Close_{t-1} → Close_t)"])

    st.dataframe(comp_table, hide_index=True, use_container_width=True)
