"""
Volatility Analysis Page — Intraday Realized Volatility, Rolling Horizons, and Regime Shifts
"""
import streamlit as st
import pandas as pd
import numpy as np

from src.calculations import compute_rolling_metrics
from src.regimes import (
    classify_volatility_regimes,
    compute_regime_comparison_statistics,
)
from src.statistics import compute_intraday_time_bucket_statistics
from src.charts import (
    plot_rolling_volatility,
    plot_intraday_profile,
    apply_chart_theme,
)
import plotly.graph_objects as go


def render_volatility_page(df_daily: pd.DataFrame, df_intraday_5m: pd.DataFrame):
    st.markdown("## ⚡ Volatility Dynamics & Regime Analysis")
    st.caption("Investigating how NASDAQ-100 behavior changes across rolling volatility horizons and market regimes.")

    if df_daily.empty:
        st.warning("No data available.")
        return

    tab_regimes, tab_rolling, tab_profile = st.tabs(["🌪️ Volatility Regimes", "📈 Rolling Volatility Horizons", "⏱️ Session Volatility Profile"])

    with tab_regimes:
        st.markdown("### 🎯 Percentile-Based Volatility Regimes")
        st.caption("Classifies historical days into Low, Normal, and High volatility environments.")

        # Interactive Threshold Controls
        col_ctrl1, col_ctrl2, col_metric = st.columns([1, 1, 1.2])
        with col_ctrl1:
            low_pct = st.slider("Low Volatility Threshold (%ile):", min_value=10.0, max_value=40.0, value=25.0, step=5.0)
        with col_ctrl2:
            high_pct = st.slider("High Volatility Threshold (%ile):", min_value=60.0, max_value=90.0, value=75.0, step=5.0)
        with col_metric:
            regime_metric = st.selectbox(
                "Classification Metric:",
                options=[
                    ("daily_range_pct", "Daily High-Low Range (%)"),
                    ("rolling_vol_20d", "Trailing 20-Day Annualized Volatility (%)"),
                ],
                format_func=lambda x: x[1],
                index=0,
            )[0]

        df_daily_calc = compute_rolling_metrics(df_daily)
        classified_df, thresh = classify_volatility_regimes(
            df_daily_calc,
            metric_col=regime_metric,
            low_pct=low_pct,
            high_pct=high_pct,
        )

        st.info(
            f"**Empirical Cutoffs**: Low Regime: `< {thresh['low_threshold']:.2f}%` "
            f"| Normal Regime: `{thresh['low_threshold']:.2f}% – {thresh['high_threshold']:.2f}%` "
            f"| High Regime: `> {thresh['high_threshold']:.2f}%`"
        )

        regime_stats = compute_regime_comparison_statistics(classified_df)

        # Bar comparison of Range and Returns across regimes
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Bar(
            x=regime_stats["Volatility Regime"],
            y=regime_stats["_raw_range"],
            name="Average Daily Range (%)",
            marker=dict(color=["#38bdf8", "#3b82f6", "#ef4444"]),
            hovertemplate="Regime: %{x}<br>Avg Range: %{y:.2f}%<extra></extra>",
        ))
        fig_reg.add_trace(go.Scatter(
            x=regime_stats["Volatility Regime"],
            y=regime_stats["_raw_vol"],
            name="Annualized Volatility (%)",
            yaxis="y2",
            mode="lines+markers",
            marker=dict(size=10, color="#f59e0b"),
            line=dict(color="#f59e0b", width=2),
            hovertemplate="Vol: %{y:.2f}%<extra></extra>",
        ))
        fig_reg.update_layout(
            yaxis=dict(title="Average Range (%)", ticksuffix="%"),
            yaxis2=dict(title="Annualized Vol (%)", ticksuffix="%", overlaying="y", side="right", showgrid=False),
        )
        st.plotly_chart(apply_chart_theme(fig_reg, "Daily Range and Volatility by Market Regime"), use_container_width=True)

        st.markdown("### 📋 Regime Behavioral Comparison Table")
        cols_display = [
            "Volatility Regime", "Sessions (n)", "Sample Share", "Mean Daily Return", "Daily Std Dev",
            "Annualized Vol", "Positive Days %", "Avg Daily Range %", "Avg Overnight Return", "Avg Regular Return",
            "Avg Abs Gap %", "Gap Fill Rate"
        ]
        st.dataframe(regime_stats[cols_display], hide_index=True, use_container_width=True)

    with tab_rolling:
        st.markdown("### 📈 Rolling Realized Volatility Horizons (20d, 60d, 120d)")
        df_rolling = compute_rolling_metrics(df_daily)
        fig_roll = plot_rolling_volatility(df_rolling)
        st.plotly_chart(fig_roll, use_container_width=True)

    with tab_profile:
        st.markdown("### ⏱️ Session Volatility Progression in MYT")
        if not df_intraday_5m.empty:
            stats_df = compute_intraday_time_bucket_statistics(df_intraday_5m)
            fig_prof = plot_intraday_profile(stats_df, metric="realized_vol_pct", title="Intraday 5m Realized Volatility Curve (MYT)")
            st.plotly_chart(fig_prof, use_container_width=True)
        else:
            st.info("Intraday volatility profiling requires intraday data.")
