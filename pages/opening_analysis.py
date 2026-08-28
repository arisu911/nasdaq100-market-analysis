"""
Opening Analysis Page — First 5m/15m/30m/60m Windows, Opening Gaps, and Gap Fill Dynamics
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.calculations import (
    compute_opening_window_metrics,
    compute_gap_fill_intraday,
)
from src.statistics import compute_distribution_summary
from src.charts import (
    plot_distribution_histogram,
    apply_chart_theme,
)


def render_opening_page(df_daily: pd.DataFrame, df_intraday_5m: pd.DataFrame):
    st.markdown("## 🔔 Opening Session Dynamics & Gap Analysis")
    st.caption("Descriptive evaluation of regular session opening windows (09:30 ET open) and overnight opening gap behavior in MYT.")

    if df_daily.empty:
        st.warning("No data available.")
        return

    tab_gap, tab_windows = st.tabs(["🚀 Opening Gap & Gap-Fill Analysis", "⏱️ Opening Windows (5m / 15m / 30m / 60m)"])

    with tab_gap:
        st.markdown("### 📊 Opening Gap Distribution (Close_{t-1} → Open_t)")
        gap_series = df_daily["gap_pct"].dropna()
        n_gaps = len(gap_series)

        # Gap KPI Metrics
        up_gaps = (gap_series > 0).sum()
        down_gaps = (gap_series < 0).sum()
        flat_gaps = (gap_series == 0).sum()
        gap_filled_pct = (df_daily["gap_filled"].sum() / n_gaps * 100.0) if "gap_filled" in df_daily.columns else 0.0

        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Mean Opening Gap", f"{gap_series.mean():.2f}%")
        g2.metric("Median Opening Gap", f"{gap_series.median():.2f}%")
        g3.metric("Positive Gaps (Up)", f"{up_gaps / n_gaps * 100:.1f}%", f"{up_gaps} days")
        g4.metric("Negative Gaps (Down)", f"{down_gaps / n_gaps * 100:.1f}%", f"{down_gaps} days")
        g5.metric("Historical Gap Fill Rate", f"{gap_filled_pct:.1f}%")

        st.markdown("---")
        # Histogram of Gaps
        fig_gap = plot_distribution_histogram(gap_series, "Historical Opening Gap % Distribution", "Opening Gap (%)")
        st.plotly_chart(fig_gap, use_container_width=True)

        # Gap Fill Analysis (Daily vs Intraday Time to Fill)
        c_gap_stats, c_fill_time = st.columns([1.2, 1])
        with c_gap_stats:
            st.markdown("#### 📋 Opening Gap Descriptive Statistics")
            dist = compute_distribution_summary(gap_series, label="Opening Gap %")
            gap_table = pd.DataFrame([
                ("Sample Sessions (n)", f"{dist['count']:,}"),
                ("Mean Gap", f"{dist['mean']:.3f}%"),
                ("Median Gap", f"{dist['median']:.3f}%"),
                ("Standard Deviation", f"{dist['std']:.3f}%"),
                ("10th Percentile", f"{dist['p10']:.3f}%"),
                ("25th Percentile", f"{dist['p25']:.3f}%"),
                ("75th Percentile", f"{dist['p75']:.3f}%"),
                ("90th Percentile", f"{dist['p90']:.3f}%"),
                ("95th Percentile", f"{dist['p95']:.3f}%"),
                ("Gap Fill Frequency (Up Gaps)", f"{(df_daily[df_daily['gap_pct'] > 0]['gap_filled'].sum() / max(1, up_gaps) * 100):.1f}%"),
                ("Gap Fill Frequency (Down Gaps)", f"{(df_daily[df_daily['gap_pct'] < 0]['gap_filled'].sum() / max(1, down_gaps) * 100):.1f}%"),
            ], columns=["Gap Metric", "Value"])
            st.dataframe(gap_table, hide_index=True, use_container_width=True)

        with c_fill_time:
            st.markdown("#### ⏱️ Intraday Time-to-Fill Distribution")
            if not df_intraday_5m.empty:
                gap_fill_intra = compute_gap_fill_intraday(df_intraday_5m, df_daily)
                if not gap_fill_intra.empty and gap_fill_intra["gap_filled"].sum() > 0:
                    filled_subset = gap_fill_intra[gap_fill_intra["gap_filled"]]
                    med_time = filled_subset["minutes_to_fill"].median()
                    avg_time = filled_subset["minutes_to_fill"].mean()
                    st.success(f"**Median Time to Fill**: `{med_time:.0f} minutes` from regular open ({med_time / 60:.1f}h)")
                    st.info(f"**Average Time to Fill**: `{avg_time:.0f} minutes` across {len(filled_subset)} filled days.")

                    fig_fill = go.Figure()
                    fig_fill.add_trace(go.Histogram(
                        x=filled_subset["minutes_to_fill"],
                        nbinsx=15,
                        marker=dict(color="#10b981", line=dict(color="white", width=0.5)),
                        hovertemplate="Minutes from Open: %{x}<br>Count: %{y}<extra></extra>",
                    ))
                    fig_fill.update_xaxes(title="Minutes from Market Open (09:30 ET)", ticksuffix="m")
                    fig_fill.update_yaxes(title="Filled Sessions Count")
                    st.plotly_chart(apply_chart_theme(fig_fill, "Minutes to Gap Fill Distribution"), use_container_width=True)
                else:
                    st.info("No intraday gap fills detected in current 60-day window.")
            else:
                st.info("Intraday gap-fill timing requires intraday data.")

    with tab_windows:
        st.markdown("### ⏱️ Early Session Performance (First 5m / 15m / 30m / 60m)")
        if df_intraday_5m.empty:
            st.warning("Opening window analysis requires intraday 5m data.")
            return

        open_metrics_df = compute_opening_window_metrics(df_intraday_5m, df_daily)
        if open_metrics_df.empty:
            st.info("No opening window metrics calculated.")
            return

        # Opening Range vs Full Day Range
        st.markdown("#### 📏 Opening Range as % of Full Regular Session Range")
        st.caption("Measures how much of the day's total high-low amplitude is established early in the session.")

        ratio_data = [
            ("First 5 Minutes (09:30–09:35 ET / 21:30–21:35 MYT)", open_metrics_df["range_ratio_5m_pct"].mean(), open_metrics_df["ret_5m_pct"].mean()),
            ("First 15 Minutes (09:30–09:45 ET / 21:30–21:45 MYT)", open_metrics_df["range_ratio_15m_pct"].mean(), open_metrics_df["ret_15m_pct"].mean()),
            ("First 30 Minutes (09:30–10:00 ET / 21:30–22:00 MYT)", open_metrics_df["range_ratio_30m_pct"].mean(), open_metrics_df["ret_30m_pct"].mean()),
            ("First 60 Minutes (09:30–10:30 ET / 21:30–22:30 MYT)", open_metrics_df["range_ratio_60m_pct"].mean(), open_metrics_df["ret_60m_pct"].mean()),
        ]
        ratio_df = pd.DataFrame(ratio_data, columns=["Opening Window", "Avg % of Full Day Range", "Avg Window Return %"])

        fig_ratios = go.Figure()
        fig_ratios.add_trace(go.Bar(
            x=["5m Window", "15m Window", "30m Window", "60m Window"],
            y=ratio_df["Avg % of Full Day Range"],
            marker=dict(color=["#38bdf8", "#06b6d4", "#3b82f6", "#a855f7"]),
            text=ratio_df["Avg % of Full Day Range"].apply(lambda x: f"{x:.1f}%"),
            textposition="auto",
            hovertemplate="%{x}<br>Range Ratio: %{y:.1f}%<extra></extra>",
        ))
        fig_ratios.update_yaxes(title="Share of Total Day Range (%)", ticksuffix="%")
        st.plotly_chart(apply_chart_theme(fig_ratios, "Opening Range Share vs Full Session High-Low Range"), use_container_width=True)

        st.dataframe(
            ratio_df.style.format({
                "Avg % of Full Day Range": "{:.2f}%",
                "Avg Window Return %": "{:.3f}%",
            }),
            hide_index=True,
            use_container_width=True,
        )
