"""
Correlations Page — NASDAQ-100 vs S&P 500 Cross-Market Dynamics & Regime-Dependent Correlation
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data_loader import load_benchmark_data
from src.calculations import compute_daily_returns_and_sessions
from src.regimes import classify_volatility_regimes
from src.charts import (
    plot_correlation_rolling,
    apply_chart_theme,
)


def render_correlations_page(df_daily: pd.DataFrame):
    st.markdown("## 🔗 Cross-Market Correlation Dynamics (NASDAQ-100 vs S&P 500)")
    st.caption("Descriptive statistical co-movement, rolling correlation, and regime-dependent behavior.")

    if df_daily.empty:
        st.warning("No NASDAQ-100 data available.")
        return

    with st.spinner("Fetching S&P 500 benchmark data..."):
        try:
            spx_raw = load_benchmark_data(ticker="^GSPC", period="10y")
            spx_daily = compute_daily_returns_and_sessions(spx_raw)
        except Exception as e:
            st.error(f"Error loading S&P 500 benchmark data: {e}")
            return

    # Align dates
    combined = pd.DataFrame({
        "ndx_return": df_daily["total_return_pct"],
        "spx_return": spx_daily["total_return_pct"],
        "daily_range_pct": df_daily["daily_range_pct"],
    }).dropna()

    if len(combined) < 20:
        st.warning("Insufficient overlapping sessions to compute correlations.")
        return

    # Overall Correlation & Regression Metrics
    corr_overall = combined["ndx_return"].corr(combined["spx_return"])
    cov_matrix = np.cov(combined["spx_return"], combined["ndx_return"])
    beta = cov_matrix[0, 1] / cov_matrix[0, 0] if cov_matrix[0, 0] > 0 else 1.0
    r_squared = corr_overall ** 2

    # KPI Top Cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Overall Return Correlation (r)", f"{corr_overall:.3f}")
    k2.metric("NASDAQ-100 Beta vs SPX (β)", f"{beta:.2f}x")
    k3.metric("R-Squared (R²)", f"{r_squared:.3f}")
    k4.metric("Co-Sample Sessions (n)", f"{len(combined):,}")

    st.markdown("---")

    # Rolling Correlation
    combined["rolling_corr_60d"] = combined["ndx_return"].rolling(60).corr(combined["spx_return"])
    fig_roll_corr = plot_correlation_rolling(combined)
    st.plotly_chart(fig_roll_corr, use_container_width=True)

    # Scatter Plot + Regime Analysis
    c_scat, c_reg_corr = st.columns(2)

    with c_scat:
        st.markdown("### 📊 Daily Return Co-Movement Scatter")
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=combined["spx_return"],
            y=combined["ndx_return"],
            mode="markers",
            marker=dict(
                size=5,
                color="rgba(56, 189, 248, 0.6)",
                line=dict(width=0.5, color="white"),
            ),
            name="Daily Returns",
            hovertemplate="S&P 500: %{x:.2f}%<br>NASDAQ-100: %{y:.2f}%<extra></extra>",
        ))

        # Regression fit line
        z = np.polyfit(combined["spx_return"], combined["ndx_return"], 1)
        p = np.poly1d(z)
        x_grid = np.linspace(combined["spx_return"].min(), combined["spx_return"].max(), 50)
        fig_scatter.add_trace(go.Scatter(
            x=x_grid,
            y=p(x_grid),
            mode="lines",
            name=f"Fit (β={beta:.2f})",
            line=dict(color="#10b981", width=2, dash="dash"),
        ))
        fig_scatter.update_xaxes(title="S&P 500 Daily Return (%)", ticksuffix="%")
        fig_scatter.update_yaxes(title="NASDAQ-100 Daily Return (%)", ticksuffix="%")
        st.plotly_chart(apply_chart_theme(fig_scatter, "Return Scatter & Linear Co-Movement"), use_container_width=True)

    with c_reg_corr:
        st.markdown("### 🌪️ Correlation Across Volatility Regimes")
        classified_df, _ = classify_volatility_regimes(combined, metric_col="daily_range_pct")

        reg_corrs = []
        for regime in ["Low Volatility", "Normal Volatility", "High Volatility"]:
            sub = classified_df[classified_df["vol_regime"] == regime]
            if len(sub) > 5:
                r_val = sub["ndx_return"].corr(sub["spx_return"])
                reg_corrs.append({
                    "Volatility Regime": regime,
                    "Sessions (n)": len(sub),
                    "Return Correlation (r)": f"{r_val:.3f}",
                    "_raw_corr": r_val,
                })

        reg_corr_df = pd.DataFrame(reg_corrs)
        st.dataframe(
            reg_corr_df[["Volatility Regime", "Sessions (n)", "Return Correlation (r)"]],
            hide_index=True,
            use_container_width=True,
        )

        fig_reg_bar = go.Figure()
        fig_reg_bar.add_trace(go.Bar(
            x=reg_corr_df["Volatility Regime"],
            y=reg_corr_df["_raw_corr"],
            marker=dict(color=["#38bdf8", "#3b82f6", "#ef4444"]),
            hovertemplate="Regime: %{x}<br>Correlation: %{y:.3f}<extra></extra>",
        ))
        fig_reg_bar.update_yaxes(title="Correlation (r)", range=[0.5, 1.0])
        st.plotly_chart(apply_chart_theme(fig_reg_bar, "Regime-Specific Return Correlation"), use_container_width=True)
