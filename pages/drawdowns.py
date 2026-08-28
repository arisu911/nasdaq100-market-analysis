"""
Drawdowns Page — Historical Drawdown Episodes, Underwater Curves & Recovery Dynamics
"""
import streamlit as st
import pandas as pd
import numpy as np

from src.calculations import (
    compute_drawdown_series,
    compute_drawdown_episodes,
)
from src.charts import (
    plot_drawdown_curve,
    apply_chart_theme,
)
import plotly.graph_objects as go


def render_drawdowns_page(df_daily: pd.DataFrame):
    st.markdown("## 📉 Drawdown Dynamics & Stress Analysis")
    st.caption("Historical peak-to-trough drawdowns, underwater duration, and recovery trajectories.")

    if df_daily.empty:
        st.warning("No data available.")
        return

    returns = df_daily["total_return"].dropna()
    wealth_index, dd_series, max_dd = compute_drawdown_series(returns)
    curr_dd = float(dd_series.iloc[-1]) if not dd_series.empty else 0.0
    avg_dd = float(dd_series[dd_series < 0].mean()) if (dd_series < 0).sum() > 0 else 0.0

    # KPI Cards
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Maximum Historical Drawdown", f"{max_dd:.2f}%")
    d2.metric("Current Drawdown from Peak", f"{curr_dd:.2f}%")
    d3.metric("Average In-Drawdown Depth", f"{avg_dd:.2f}%")
    d4.metric("Total Historical Sessions", f"{len(returns):,}")

    st.markdown("---")

    # Underwater Plot
    st.markdown("### 🌊 Historical Underwater Drawdown Curve")
    fig_dd = plot_drawdown_curve(dd_series)
    st.plotly_chart(fig_dd, use_container_width=True)

    # Drawdown Episodes Table
    st.markdown("### 📋 Major Historical Drawdown Episodes")
    st.caption("Top peak-to-trough decline episodes, durations, and recovery paths.")
    episodes_df = compute_drawdown_episodes(dd_series, top_n=8)

    if not episodes_df.empty:
        cols_display = [
            "Max Drawdown", "Peak Date (MYT)", "Trough Date (MYT)", "Recovery Date (MYT)",
            "Decline (Sessions)", "Recovery (Sessions)", "Total Duration"
        ]
        st.dataframe(episodes_df[cols_display], hide_index=True, use_container_width=True)
    else:
        st.info("No significant drawdown episodes recorded.")
