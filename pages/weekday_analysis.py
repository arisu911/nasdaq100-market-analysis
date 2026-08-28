"""
Weekday & Seasonality Analysis Page — Monday through Friday, Monthly & Yearly Aggregations
"""
import streamlit as st
import pandas as pd
import numpy as np

from src.statistics import (
    compute_weekday_statistics,
    compute_monthly_statistics,
    compute_yearly_statistics,
)
from src.charts import (
    plot_weekday_comparison,
    apply_chart_theme,
)
import plotly.graph_objects as go


def render_weekday_page(df_daily: pd.DataFrame):
    st.markdown("## 📅 Day-of-Week & Calendar Analysis")
    st.caption("Descriptive performance, volatility, and range comparisons across Monday–Friday, calendar months, and years.")

    if df_daily.empty:
        st.warning("No data available.")
        return

    tab_week, tab_month, tab_year = st.tabs(["📆 Weekday Breakdown", "🗓️ Monthly Seasonality", "📊 Yearly Performance"])

    with tab_week:
        st.markdown("### 📊 Monday to Friday Statistical Comparison")
        weekday_df = compute_weekday_statistics(df_daily)

        fig_week = plot_weekday_comparison(weekday_df)
        st.plotly_chart(fig_week, use_container_width=True)

        # Comparative Metrics Cards
        st.markdown("### 📋 Weekday Metrics Summary Table")
        cols_display = [
            "Day of Week", "Sample Size (n)", "Mean Return", "Median Return", "Std Dev",
            "Annualized Vol", "Positive %", "Negative %", "Avg Range %", "90th Pct Abs Move", "Avg Volume"
        ]
        st.dataframe(weekday_df[cols_display], hide_index=True, use_container_width=True)

        # Range and Volatility comparison chart
        st.markdown("### ⚡ Daily Range & Volatility by Weekday")
        fig_range = go.Figure()
        fig_range.add_trace(go.Bar(
            x=weekday_df["Day of Week"],
            y=weekday_df["_raw_avg_range"],
            name="Average Daily Range (%)",
            marker=dict(color="#38bdf8"),
            hovertemplate="Avg Range: %{y:.2f}%<extra></extra>",
        ))
        fig_range.add_trace(go.Scatter(
            x=weekday_df["Day of Week"],
            y=weekday_df["_raw_pos_pct"],
            name="Positive Day %",
            yaxis="y2",
            mode="lines+markers",
            marker=dict(size=8, color="#10b981"),
            line=dict(color="#10b981", width=2),
            hovertemplate="Positive: %{y:.1f}%<extra></extra>",
        ))
        fig_range.update_layout(
            yaxis=dict(title="Average Range (%)", ticksuffix="%"),
            yaxis2=dict(title="Positive %", ticksuffix="%", overlaying="y", side="right", showgrid=False),
        )
        st.plotly_chart(apply_chart_theme(fig_range, "Weekday Range vs Positive Probability"), use_container_width=True)

    with tab_month:
        st.markdown("### 🗓️ Month-by-Month Historical Behavior (Jan–Dec)")
        month_df = compute_monthly_statistics(df_daily)

        fig_month = go.Figure()
        fig_month.add_trace(go.Bar(
            x=month_df["Month"],
            y=month_df["_raw_mean_ret"],
            name="Mean Daily Return (%)",
            marker=dict(
                color=np.where(month_df["_raw_mean_ret"] >= 0, "#10b981", "#ef4444"),
            ),
            hovertemplate="Month: %{x}<br>Mean Return: %{y:.3f}%<extra></extra>",
        ))
        fig_month.update_yaxes(title="Mean Return (%)", ticksuffix="%")
        st.plotly_chart(apply_chart_theme(fig_month, "Monthly Average Daily Return"), use_container_width=True)

        st.dataframe(
            month_df.drop(columns=[c for c in month_df.columns if c.startswith("_")]),
            hide_index=True,
            use_container_width=True,
        )

    with tab_year:
        st.markdown("### 📊 Year-by-Year Historical Breakdown")
        year_df = compute_yearly_statistics(df_daily)

        fig_year = go.Figure()
        fig_year.add_trace(go.Bar(
            x=year_df["Year"],
            y=year_df["_raw_cum_ret"],
            name="Annual Return (%)",
            marker=dict(
                color=np.where(year_df["_raw_cum_ret"] >= 0, "#38bdf8", "#ef4444"),
            ),
            hovertemplate="Year: %{x}<br>Annual Return: %{y:.2f}%<extra></extra>",
        ))
        fig_year.update_yaxes(title="Annual Return (%)", ticksuffix="%")
        st.plotly_chart(apply_chart_theme(fig_year, "Annual Return History"), use_container_width=True)

        st.dataframe(
            year_df.drop(columns=[c for c in year_df.columns if c.startswith("_")]),
            hide_index=True,
            use_container_width=True,
        )
