"""
Interactive Plotly chart generation module.
Follows quantitative institutional research visual standards:
- Dark slate theme with high contrast typography
- Vibrant neon/cyber color palettes (Cyan, Emerald, Amber, Crimson, Purple)
- All time-series axes and hover labels explicitly formatted in MYT (Asia/Kuala_Lumpur)
- Clean responsive layouts with crosshairs and hover templates
"""
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Theme Colors
BG_DARK = "#0b0f19"
BG_CARD = "#111827"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"
TEXT_PRIMARY = "#f8fafc"
TEXT_MUTED = "#94a3b8"

COLOR_CYAN = "#00f0ff"
COLOR_GREEN = "#10b981"
COLOR_AMBER = "#f59e0b"
COLOR_RED = "#ef4444"
COLOR_PURPLE = "#a855f7"
COLOR_BLUE = "#3b82f6"


def apply_chart_theme(fig: go.Figure, title: str = "", height: int = 440) -> go.Figure:
    """Apply unified dark aesthetic styling to Plotly figures."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=15, color=TEXT_PRIMARY, family="Inter, -apple-system, sans-serif"),
            x=0.01,
            y=0.96,
        ),
        height=height,
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_DARK,
        margin=dict(l=45, r=25, t=55, b=45),
        font=dict(color=TEXT_MUTED, family="Inter, -apple-system, sans-serif", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=TEXT_PRIMARY),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
            showline=True,
            linecolor=GRID_COLOR,
            tickfont=dict(color=TEXT_MUTED, size=11),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=True,
            zerolinecolor="rgba(255, 255, 255, 0.15)",
            showline=True,
            linecolor=GRID_COLOR,
            tickfont=dict(color=TEXT_MUTED, size=11),
        ),
        hovermode="x unified",
    )
    return fig


def plot_cumulative_performance(df_cum: pd.DataFrame) -> go.Figure:
    """
    Plot cumulative return comparison: Full Close-to-Close vs Overnight vs Regular Session.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_cum.index,
        y=df_cum["cum_total"] * 100,
        name="Total (Close-to-Close)",
        line=dict(color=COLOR_CYAN, width=2.2),
        hovertemplate="Total: %{y:.2f}%<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df_cum.index,
        y=df_cum["cum_overnight"] * 100,
        name="Overnight Return Only",
        line=dict(color=COLOR_AMBER, width=1.8, dash="solid"),
        hovertemplate="Overnight: %{y:.2f}%<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df_cum.index,
        y=df_cum["cum_regular"] * 100,
        name="Regular Session Return Only",
        line=dict(color=COLOR_PURPLE, width=1.8, dash="dot"),
        hovertemplate="Regular Session: %{y:.2f}%<extra></extra>",
    ))

    fig.update_yaxes(title="Cumulative Return (%)")
    fig.update_xaxes(title="Historical Date")
    return apply_chart_theme(fig, "Cumulative Return: Total vs Overnight vs Regular Session")


def plot_drawdown_curve(drawdown_series: pd.Series) -> go.Figure:
    """Plot underwater drawdown curve with max drawdown indicator."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown_series.index,
        y=drawdown_series,
        name="Drawdown (%)",
        fill="tozeroy",
        fillcolor="rgba(239, 68, 68, 0.22)",
        line=dict(color=COLOR_RED, width=1.6),
        hovertemplate="Drawdown: %{y:.2f}%<extra></extra>",
    ))

    min_val = drawdown_series.min()
    min_idx = drawdown_series.idxmin()

    fig.add_annotation(
        x=min_idx,
        y=min_val,
        text=f"Max DD: {min_val:.2f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLOR_RED,
        arrowsize=1,
        arrowwidth=1.5,
        ax=0,
        ay=-35,
        font=dict(color=TEXT_PRIMARY, size=11),
        bgcolor=BG_DARK,
        bordercolor=COLOR_RED,
        borderwidth=1,
    )

    fig.update_yaxes(title="Drawdown (%)", ticksuffix="%")
    fig.update_xaxes(title="Date")
    return apply_chart_theme(fig, "Historical Drawdown (Underwater Curve)")


def plot_intraday_profile(
    df_profile: pd.DataFrame,
    metric: str = "mean_return_pct",
    title: str = "Intraday Profile by Time of Day (MYT)"
) -> go.Figure:
    """
    Plot intraday session progression by MYT time bucket.
    """
    fig = go.Figure()

    metric_labels = {
        "mean_return_pct": ("Mean Bar Return (%)", COLOR_CYAN, "%{y:.3f}%"),
        "median_return_pct": ("Median Bar Return (%)", COLOR_BLUE, "%{y:.3f}%"),
        "mean_abs_return_pct": ("Mean Absolute Movement (%)", COLOR_AMBER, "%{y:.3f}%"),
        "realized_vol_pct": ("Realized Volatility (Annualized %)", COLOR_PURPLE, "%{y:.2f}%"),
        "mean_range_pct": ("Average High-Low Range (%)", COLOR_GREEN, "%{y:.3f}%"),
        "relative_volume": ("Relative Volume (vs Session Mean)", COLOR_CYAN, "%{y:.2f}x"),
    }

    label, color, h_fmt = metric_labels.get(metric, (metric, COLOR_CYAN, "%{y:.2f}"))

    fig.add_trace(go.Scatter(
        x=df_profile["display_label_myt"],
        y=df_profile[metric],
        mode="lines+markers",
        name=label,
        line=dict(color=color, width=2.2),
        marker=dict(size=5, color=color),
        hovertemplate=f"Time (MYT): %{{x}}<br>{label}: {h_fmt}<extra></extra>",
    ))

    # Add reference zero line if return
    if "return" in metric:
        fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"))

    fig.update_xaxes(title="Session Time (Asia/Kuala_Lumpur MYT)", tickangle=-45)
    fig.update_yaxes(title=label)
    return apply_chart_theme(fig, title)


def plot_weekday_comparison(df_weekday: pd.DataFrame) -> go.Figure:
    """Plot weekday comparison for Mean Return vs Average Range."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_weekday["Day of Week"],
        y=df_weekday["_raw_mean_ret"],
        name="Mean Return (%)",
        marker=dict(
            color=np.where(df_weekday["_raw_mean_ret"] >= 0, COLOR_GREEN, COLOR_RED),
            line=dict(width=1, color="rgba(255,255,255,0.2)"),
        ),
        hovertemplate="Day: %{x}<br>Mean Return: %{y:.3f}%<extra></extra>",
    ))

    fig.update_layout(bargap=0.35)
    fig.update_yaxes(title="Mean Return (%)", ticksuffix="%")
    fig.update_xaxes(title="Day of Week")
    return apply_chart_theme(fig, "NASDAQ-100 Day-of-Week Mean Return Comparison")


def plot_distribution_histogram(
    series: pd.Series,
    title: str,
    xlabel: str,
    is_percentage: bool = True
) -> go.Figure:
    """Plot distribution histogram with mean, median, and 10th/90th percentile markers."""
    clean_s = series.dropna()
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=clean_s,
        nbinsx=40,
        marker=dict(
            color="rgba(56, 189, 248, 0.7)",
            line=dict(color=COLOR_CYAN, width=1),
        ),
        name="Frequency Count",
        hovertemplate="Bin: %{x}<br>Count: %{y}<extra></extra>",
    ))

    mean_val = clean_s.mean()
    median_val = clean_s.median()
    p10, p90 = np.percentile(clean_s, [10, 90])

    fig.add_vline(x=mean_val, line=dict(color=COLOR_AMBER, width=2, dash="dash"), annotation_text=f"Mean: {mean_val:.2f}{'%' if is_percentage else ''}", annotation_position="top right")
    fig.add_vline(x=median_val, line=dict(color=COLOR_GREEN, width=2, dash="dot"), annotation_text=f"Median: {median_val:.2f}{'%' if is_percentage else ''}", annotation_position="top left")

    suffix = "%" if is_percentage else ""
    fig.update_xaxes(title=xlabel, ticksuffix=suffix)
    fig.update_yaxes(title="Frequency Count")
    return apply_chart_theme(fig, title)


def plot_volume_vs_volatility_scatter(df_profile: pd.DataFrame) -> go.Figure:
    """Plot relative volume vs realized volatility / absolute movement with trendline."""
    clean_df = df_profile.dropna(subset=["relative_volume", "mean_abs_return_pct"]).copy()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=clean_df["relative_volume"],
        y=clean_df["mean_abs_return_pct"],
        mode="markers+text",
        text=clean_df["time_myt"],
        textposition="top center",
        textfont=dict(size=9, color=TEXT_MUTED),
        marker=dict(
            size=10,
            color=clean_df["mean_abs_return_pct"],
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Abs Ret %", len=0.7),
            line=dict(width=1, color="white"),
        ),
        name="Time Buckets (MYT)",
        hovertemplate="Time (MYT): %{text}<br>Rel Volume: %{x:.2f}x<br>Abs Return: %{y:.3f}%<extra></extra>",
    ))

    # Add OLS regression line if enough data
    if len(clean_df) > 3:
        z = np.polyfit(clean_df["relative_volume"], clean_df["mean_abs_return_pct"], 1)
        p = np.poly1d(z)
        x_vals = np.linspace(clean_df["relative_volume"].min(), clean_df["relative_volume"].max(), 50)
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=p(x_vals),
            mode="lines",
            name="Linear Trend",
            line=dict(color=COLOR_CYAN, width=1.5, dash="dash"),
            hovertemplate="Trend Fit<extra></extra>",
        ))

    fig.update_xaxes(title="Relative Volume (Multiple of Session Mean)", ticksuffix="x")
    fig.update_yaxes(title="Average Absolute Movement (%)", ticksuffix="%")
    return apply_chart_theme(fig, "Volume vs Volatility / Absolute Movement (MYT Time Buckets)")


def plot_rolling_volatility(df_rolling: pd.DataFrame) -> go.Figure:
    """Plot rolling annualized volatility curves across 20d, 60d, and 120d windows."""
    fig = go.Figure()

    if "rolling_vol_20d" in df_rolling.columns:
        fig.add_trace(go.Scatter(
            x=df_rolling.index,
            y=df_rolling["rolling_vol_20d"],
            name="20-Session Volatility",
            line=dict(color=COLOR_CYAN, width=1.8),
            hovertemplate="20d Vol: %{y:.2f}%<extra></extra>",
        ))

    if "rolling_vol_60d" in df_rolling.columns:
        fig.add_trace(go.Scatter(
            x=df_rolling.index,
            y=df_rolling["rolling_vol_60d"],
            name="60-Session Volatility",
            line=dict(color=COLOR_AMBER, width=1.8),
            hovertemplate="60d Vol: %{y:.2f}%<extra></extra>",
        ))

    if "rolling_vol_120d" in df_rolling.columns:
        fig.add_trace(go.Scatter(
            x=df_rolling.index,
            y=df_rolling["rolling_vol_120d"],
            name="120-Session Volatility",
            line=dict(color=COLOR_PURPLE, width=1.8),
            hovertemplate="120d Vol: %{y:.2f}%<extra></extra>",
        ))

    fig.update_yaxes(title="Annualized Volatility (%)", ticksuffix="%")
    fig.update_xaxes(title="Date")
    return apply_chart_theme(fig, "Rolling Realized Volatility Horizons (20d, 60d, 120d)")


def plot_mfe_mae_scatter(df_daily: pd.DataFrame) -> go.Figure:
    """Plot MFE vs MAE path statistics scatter plot."""
    df_clean = df_daily.dropna(subset=["mfe_pct", "mae_pct", "regular_return_pct"]).copy()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_clean["mae_pct"],
        y=df_clean["mfe_pct"],
        mode="markers",
        marker=dict(
            size=6,
            color=df_clean["regular_return_pct"],
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="Final Return %", len=0.7),
            line=dict(width=0.5, color="rgba(0,0,0,0.5)"),
        ),
        name="Sessions",
        hovertemplate="MAE (Adverse): %{x:.2f}%<br>MFE (Favorable): %{y:.2f}%<br>Final Return: %{marker.color:.2f}%<extra></extra>",
    ))

    fig.update_xaxes(title="Maximum Adverse Excursion (MAE % from Open)", ticksuffix="%")
    fig.update_yaxes(title="Maximum Favorable Excursion (MFE % from Open)", ticksuffix="%")
    return apply_chart_theme(fig, "Path Dynamics: Maximum Favorable (MFE) vs Adverse (MAE) Excursions")


def plot_correlation_rolling(df_corr: pd.DataFrame) -> go.Figure:
    """Plot rolling correlation between NASDAQ-100 and S&P 500."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_corr.index,
        y=df_corr["rolling_corr_60d"],
        name="60-Session Rolling Correlation",
        line=dict(color=COLOR_CYAN, width=2.0),
        hovertemplate="Correlation (60d): %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"))
    fig.add_hline(y=1.0, line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"))

    fig.update_yaxes(title="Correlation Coefficient (r)", range=[-0.2, 1.05])
    fig.update_xaxes(title="Date")
    return apply_chart_theme(fig, "Rolling 60-Session Return Correlation: NASDAQ-100 vs S&P 500")
