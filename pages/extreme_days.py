"""
Extreme Days Page — Tail Event Analysis, Large Movement Sessions, and MFE/MAE Paths
"""
import streamlit as st
import pandas as pd
import numpy as np

from src.timezone_utils import format_myt_timestamp
from src.charts import (
    plot_mfe_mae_scatter,
    apply_chart_theme,
)


def render_extreme_days_page(df_daily: pd.DataFrame):
    st.markdown("## 🌋 Extreme Historical Days & Tail Events")
    st.caption("Investigating the largest historical outliers, gap days, and path dynamics (MFE vs MAE).")

    if df_daily.empty:
        st.warning("No data available.")
        return

    df_clean = df_daily.dropna(subset=["total_return_pct", "daily_range_pct"]).copy()

    # Percentile Threshold Selector
    col_p, col_info = st.columns([1, 2])
    with col_p:
        pct_threshold = st.slider("Extreme Tail Percentile Threshold:", min_value=90.0, max_value=99.0, value=95.0, step=1.0)

    cutoff_abs_ret = np.percentile(df_clean["abs_total_return_pct"], pct_threshold)
    extreme_subset = df_clean[df_clean["abs_total_return_pct"] >= cutoff_abs_ret]

    with col_info:
        st.info(
            f"**{pct_threshold}th Percentile Cutoff**: `|Daily Return| ≥ {cutoff_abs_ret:.2f}%` "
            f"({len(extreme_subset)} sessions identified out of {len(df_clean)} total)."
        )

    tab_tables, tab_path = st.tabs(["📋 Top Historical Tail Events", "🧭 Path Dynamics: MFE vs MAE"])

    with tab_tables:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🟢 Top 10 Largest Positive Days")
            top_pos = df_clean.sort_values(by="total_return_pct", ascending=False).head(10).copy()
            top_pos["Date (MYT)"] = top_pos.index.map(lambda d: format_myt_timestamp(d, include_date=True))
            top_pos["Weekday"] = top_pos["weekday"] if "weekday" in top_pos.columns else top_pos.index.strftime("%A")
            top_pos["Total Return"] = top_pos["total_return_pct"].apply(lambda x: f"+{x:.2f}%")
            top_pos["Overnight"] = top_pos["overnight_return_pct"].apply(lambda x: f"{x:+.2f}%")
            top_pos["Regular Session"] = top_pos["regular_return_pct"].apply(lambda x: f"{x:+.2f}%")
            top_pos["Daily Range"] = top_pos["daily_range_pct"].apply(lambda x: f"{x:.2f}%")
            st.dataframe(top_pos[["Date (MYT)", "Weekday", "Total Return", "Overnight", "Regular Session", "Daily Range"]], hide_index=True, use_container_width=True)

        with c2:
            st.markdown("### 🔴 Top 10 Largest Negative Days")
            top_neg = df_clean.sort_values(by="total_return_pct", ascending=True).head(10).copy()
            top_neg["Date (MYT)"] = top_neg.index.map(lambda d: format_myt_timestamp(d, include_date=True))
            top_neg["Weekday"] = top_neg["weekday"] if "weekday" in top_neg.columns else top_neg.index.strftime("%A")
            top_neg["Total Return"] = top_neg["total_return_pct"].apply(lambda x: f"{x:.2f}%")
            top_neg["Overnight"] = top_neg["overnight_return_pct"].apply(lambda x: f"{x:+.2f}%")
            top_neg["Regular Session"] = top_neg["regular_return_pct"].apply(lambda x: f"{x:+.2f}%")
            top_neg["Daily Range"] = top_neg["daily_range_pct"].apply(lambda x: f"{x:.2f}%")
            st.dataframe(top_neg[["Date (MYT)", "Weekday", "Total Return", "Overnight", "Regular Session", "Daily Range"]], hide_index=True, use_container_width=True)

        st.markdown("---")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("### ⚡ Top 10 Highest Volatility / Range Days")
            top_rng = df_clean.sort_values(by="daily_range_pct", ascending=False).head(10).copy()
            top_rng["Date (MYT)"] = top_rng.index.map(lambda d: format_myt_timestamp(d, include_date=True))
            top_rng["Daily Range"] = top_rng["daily_range_pct"].apply(lambda x: f"{x:.2f}%")
            top_rng["Total Return"] = top_rng["total_return_pct"].apply(lambda x: f"{x:+.2f}%")
            top_rng["MFE (Upward)"] = top_rng["mfe_pct"].apply(lambda x: f"+{x:.2f}%")
            top_rng["MAE (Downward)"] = top_rng["mae_pct"].apply(lambda x: f"{x:.2f}%")
            st.dataframe(top_rng[["Date (MYT)", "Daily Range", "Total Return", "MFE (Upward)", "MAE (Downward)"]], hide_index=True, use_container_width=True)

        with c4:
            st.markdown("### 🚀 Top 10 Largest Opening Gaps")
            top_gap = df_clean.reindex(df_clean["gap_pct"].abs().sort_values(ascending=False).index).head(10).copy()
            top_gap["Date (MYT)"] = top_gap.index.map(lambda d: format_myt_timestamp(d, include_date=True))
            top_gap["Opening Gap"] = top_gap["gap_pct"].apply(lambda x: f"{x:+.2f}%")
            top_gap["Gap Filled?"] = top_gap["gap_filled"].apply(lambda x: "Yes ✅" if x else "No ❌")
            top_gap["Total Return"] = top_gap["total_return_pct"].apply(lambda x: f"{x:+.2f}%")
            st.dataframe(top_gap[["Date (MYT)", "Opening Gap", "Gap Filled?", "Total Return"]], hide_index=True, use_container_width=True)

    with tab_path:
        st.markdown("### 🧭 Path Excursion Dynamics: MFE vs MAE")
        st.caption(
            "**MFE (Maximum Favorable Excursion)**: Peak upward move from regular session open. "
            "**MAE (Maximum Adverse Excursion)**: Deepest downward move from regular session open."
        )
        fig_mfe = plot_mfe_mae_scatter(df_clean)
        st.plotly_chart(fig_mfe, use_container_width=True)
