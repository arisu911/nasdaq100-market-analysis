"""
Intraday Analysis Page — Session Time-of-Day Profiles in MYT
"""
import streamlit as st
import pandas as pd
import numpy as np

from src.market_sessions import resample_intraday
from src.statistics import compute_intraday_time_bucket_statistics
from src.charts import (
    plot_intraday_profile,
    plot_volume_vs_volatility_scatter,
)


def render_intraday_page(df_intraday_5m: pd.DataFrame, coverage_info: dict):
    st.markdown("## ⏱️ Intraday Time-of-Day Behavior")
    st.caption("Statistical profiles across the regular trading session. All time axes and tables are in Asia/Kuala_Lumpur (MYT).")

    if df_intraday_5m.empty:
        st.warning("No intraday data available for the selected parameters.")
        return

    # Transparency Notice
    st.info(
        f"💡 **Data Coverage Note**: {coverage_info.get('limitation_text')} "
        f"Current intraday dataset covers **{coverage_info.get('total_sessions')} sessions** "
        f"from `{coverage_info.get('start_myt')}` to `{coverage_info.get('end_myt')}`."
    )

    # Resolution Selector
    col_res, col_metric = st.columns([1, 2])
    with col_res:
        resolution = st.selectbox(
            "Select Intraday Timeframe:",
            options=["5m", "10m", "15m", "30m", "60m"],
            index=0,
        )

    with col_metric:
        profile_metric = st.selectbox(
            "Primary Profile Metric:",
            options=[
                ("mean_return_pct", "Average Bar Return (%)"),
                ("median_return_pct", "Median Bar Return (%)"),
                ("mean_abs_return_pct", "Average Absolute Movement (%)"),
                ("realized_vol_pct", "Realized Volatility (Annualized %)"),
                ("mean_range_pct", "Average High-Low Range (%)"),
                ("relative_volume", "Relative Volume (Multiple of Session Mean)"),
            ],
            format_func=lambda x: x[1],
            index=2,
        )[0]

    # Resample intraday data
    df_resampled = resample_intraday(df_intraday_5m, interval_str=resolution)
    stats_df = compute_intraday_time_bucket_statistics(df_resampled)

    if stats_df.empty:
        st.warning("Could not calculate statistics for the chosen timeframe.")
        return

    # Profile Chart
    st.markdown("---")
    st.markdown(f"### 📈 Intraday {resolution} Profile across Session (MYT)")
    fig_prof = plot_intraday_profile(stats_df, metric=profile_metric, title=f"NASDAQ-100 {resolution} Time-of-Day Profile (MYT)")
    st.plotly_chart(fig_prof, use_container_width=True)

    # Dual Column: Volatility Profile + Volume vs Volatility
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ⚡ Realized Volatility Across Session")
        fig_vol = plot_intraday_profile(stats_df, metric="realized_vol_pct", title="Session Volatility Curve (MYT)")
        st.plotly_chart(fig_vol, use_container_width=True)

    with c2:
        st.markdown("### 📊 Volume vs Volatility / Movement")
        fig_scatter = plot_volume_vs_volatility_scatter(stats_df)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Detailed Time Bucket Table in MYT
    st.markdown("### 📋 Intraday Time Bucket Statistical Table (MYT)")
    display_table = stats_df.copy()
    display_table["Time (MYT)"] = display_table["display_label_myt"]
    display_table["Time (ET Ref)"] = display_table["time_ny"] + " ET"
    display_table["Avg Return"] = display_table["mean_return_pct"].apply(lambda x: f"{x:.3f}%")
    display_table["Median Return"] = display_table["median_return_pct"].apply(lambda x: f"{x:.3f}%")
    display_table["Avg Abs Movement"] = display_table["mean_abs_return_pct"].apply(lambda x: f"{x:.3f}%")
    display_table["Avg Range"] = display_table["mean_range_pct"].apply(lambda x: f"{x:.3f}%")
    display_table["Annualized Vol"] = display_table["realized_vol_pct"].apply(lambda x: f"{x:.2f}%")
    display_table["Positive %"] = display_table["positive_pct"].apply(lambda x: f"{x:.1f}%")
    display_table["Relative Volume"] = display_table["relative_volume"].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "N/A")
    display_table["Sample Size (n)"] = display_table["sample_size"]

    cols_to_show = [
        "Time (MYT)", "Time (ET Ref)", "Sample Size (n)", "Avg Return", "Median Return",
        "Avg Abs Movement", "Avg Range", "Annualized Vol", "Positive %", "Relative Volume"
    ]
    st.dataframe(display_table[cols_to_show], hide_index=True, use_container_width=True)
