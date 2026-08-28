"""
Overview Page — Macro Performance, Data Coverage & MYT Market Clock
"""
import streamlit as st
import pandas as pd
import numpy as np

from src.timezone_utils import (
    get_current_market_clock_info,
    get_myt_session_times,
    format_myt_timestamp,
)
from src.calculations import (
    compute_cumulative_performance,
    compute_drawdown_series,
)
from src.statistics import (
    compute_distribution_summary,
    compute_historical_frequencies,
)
from src.data_cleaner import format_audit_report_table
from src.charts import (
    plot_cumulative_performance,
    plot_distribution_histogram,
    plot_drawdown_curve,
)


def render_overview_page(df_daily: pd.DataFrame, coverage_info: dict, audit_report: dict):
    st.markdown("## 📊 NASDAQ-100 Market Behavior Overview")
    st.caption("Descriptive statistical summary and session structure formatted in Asia/Kuala_Lumpur (MYT).")

    # 1. Market Session Clock in MYT Banner
    clock = get_current_market_clock_info()
    today_session = clock["today_session"]

    col_clock, col_info = st.columns([1.5, 2.5])
    with col_clock:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #111827, #1f2937); border-radius: 10px; padding: 16px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 15px;">
                <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; letter-spacing: 1px;">Current Real-Time Clock (MYT)</div>
                <div style="font-size: 22px; font-weight: 700; color: #f8fafc; margin-top: 4px;">{clock['now_myt'].strftime('%H:%M:%S')} <span style="font-size: 13px; color: #38bdf8;">MYT (UTC+8)</span></div>
                <div style="margin-top: 10px; display: flex; align-items: center; gap: 8px;">
                    <span style="height: 10px; width: 10px; background-color: {clock['status_color']}; border-radius: 50%; display: inline-block;"></span>
                    <span style="font-size: 12px; font-weight: 600; color: {clock['status_color']};">{clock['status_text']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_info:
        st.markdown(
            f"""
            <div style="background: #111827; border-radius: 10px; padding: 16px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 15px;">
                <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; letter-spacing: 1px;">NASDAQ Regular Session (09:30–16:00 ET)</div>
                <div style="font-size: 16px; font-weight: 600; color: #38bdf8; margin-top: 4px;">
                    MYT Dynamic Hours: <span style="color: #f8fafc;">{today_session['myt_session_label']}</span>
                </div>
                <div style="font-size: 12px; color: #94a3b8; margin-top: 6px;">
                    US Status: <b>{today_session['tz_abbr']}</b> ({'Daylight Saving Time UTC-4' if today_session['is_dst'] else 'Standard Time UTC-5'}) 
                    &bull; Open: <b>{today_session['myt_open_str']}</b> &bull; Close: <b>{today_session['myt_close_str']}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Key Statistical Metrics Cards
    if df_daily.empty:
        st.warning("No data available for the selected filters.")
        return

    n_sessions = len(df_daily)
    mean_ret = df_daily["total_return_pct"].mean()
    ann_vol = df_daily["total_return"].std() * np.sqrt(252) * 100
    mean_range = df_daily["daily_range_pct"].mean()
    pos_pct = (df_daily["total_return_pct"] > 0).sum() / n_sessions * 100
    _, _, max_dd = compute_drawdown_series(df_daily["total_return"])

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Trading Sessions (n)", f"{n_sessions:,}")
    m2.metric("Mean Daily Return", f"{mean_ret:.3f}%", delta=f"{mean_ret:.3f}%", delta_color="normal")
    m3.metric("Annualized Volatility", f"{ann_vol:.2f}%")
    m4.metric("Average Daily Range", f"{mean_range:.2f}%")
    m5.metric("Positive Sessions", f"{pos_pct:.1f}%")
    m6.metric("Max Drawdown", f"{max_dd:.2f}%")

    st.markdown("---")

    # 3. Cumulative Growth & Contribution Decomposition
    st.markdown("### 📈 Cumulative Return Decomposition")
    st.caption("Comparing Total Close-to-Close performance against pure Overnight vs pure Regular-Session holding.")
    df_cum = compute_cumulative_performance(df_daily)
    fig_cum = plot_cumulative_performance(df_cum)
    st.plotly_chart(fig_cum, use_container_width=True)

    # 4. Summary & Historical Frequency Breakdown
    c_left, c_right = st.columns([1.2, 1])

    with c_left:
        st.markdown("### 📋 Historical Event Frequencies")
        freq_df = compute_historical_frequencies(df_daily)
        st.dataframe(
            freq_df,
            hide_index=True,
            use_container_width=True,
        )

    with c_right:
        st.markdown("### 📊 Return Distribution Summary")
        dist = compute_distribution_summary(df_daily["total_return_pct"], label="Daily Return %")
        dist_table = pd.DataFrame([
            ("Mean", f"{dist['mean']:.3f}%"),
            ("Median", f"{dist['median']:.3f}%"),
            ("Std Dev", f"{dist['std']:.3f}%"),
            ("10th Percentile", f"{dist['p10']:.3f}%"),
            ("25th Percentile", f"{dist['p25']:.3f}%"),
            ("75th Percentile", f"{dist['p75']:.3f}%"),
            ("90th Percentile", f"{dist['p90']:.3f}%"),
            ("Skewness", f"{dist['skewness']:.3f}"),
            ("Kurtosis", f"{dist['kurtosis']:.3f}"),
        ], columns=["Statistic", "Value"])
        st.dataframe(dist_table, hide_index=True, use_container_width=True)

    # 5. Data Coverage & Quality Audit Accordion
    st.markdown("---")
    with st.expander("🔍 Data Quality Audit & Coverage Transparency", expanded=False):
        c_cov, c_aud = st.columns(2)
        with c_cov:
            st.markdown("#### Coverage & Limitations")
            st.info(
                f"**Instrument**: {coverage_info.get('instrument_note')}\n\n"
                f"**Date Range (MYT)**: `{coverage_info.get('start_myt')}` to `{coverage_info.get('end_myt')}`\n\n"
                f"**Total Observations**: `{coverage_info.get('total_bars', 0):,}` bars ({coverage_info.get('total_sessions', 0):,} trading sessions)\n\n"
                f"**Coverage Note**: {coverage_info.get('limitation_text')}"
            )
        with c_aud:
            st.markdown("#### Quality Checks")
            audit_df = format_audit_report_table(audit_report)
            st.dataframe(audit_df, hide_index=True, use_container_width=True)
