"""
FastAPI Quantitative Market Analytics Backend for NASDAQ-100 & QQQ.
Decoupled web architecture replacing Streamlit with high-performance JSON REST APIs.
Serves static frontend assets and serializes Plotly charts in dark financial terminal theme.
"""
from typing import Dict, Any, List, Optional
import sys
import os
import json
import functools
import datetime
from pathlib import Path

# Add project root to sys.path so we can import src and config directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config.settings import (
    INDEX_TICKER,
    ETF_TICKER,
    BENCHMARK_INDEX,
    BENCHMARK_ETF,
    SOURCE_TZ,
    TARGET_TZ,
)
from src.timezone_utils import (
    get_current_market_clock_info,
    convert_to_new_york,
    convert_to_myt,
    format_myt_timestamp,
)
from src.data_loader import (
    load_daily_data,
    load_intraday_data,
    load_benchmark_data,
    download_and_cache_data,
    get_data_coverage_info,
)
from src.data_cleaner import clean_market_data, format_audit_report_table
from src.calculations import (
    compute_daily_returns_and_sessions,
    compute_cumulative_performance,
    compute_drawdown_series,
    compute_drawdown_episodes,
    compute_rolling_metrics,
    compute_opening_window_metrics,
    compute_gap_fill_intraday,
)
from src.statistics import (
    compute_distribution_summary,
    compute_historical_frequencies,
    compute_weekday_statistics,
    compute_monthly_statistics,
    compute_yearly_statistics,
    compute_intraday_time_bucket_statistics,
)
from src.regimes import (
    classify_volatility_regimes,
    compute_regime_comparison_statistics,
)
from src.market_sessions import (
    resample_intraday,
    filter_regular_session,
    tag_session_phase,
)
from src.charts import (
    plot_cumulative_performance,
    plot_distribution_histogram,
    plot_drawdown_curve,
    plot_weekday_comparison,
    plot_intraday_profile,
    plot_volume_vs_volatility_scatter,
    plot_rolling_volatility,
    plot_mfe_mae_scatter,
    plot_correlation_rolling,
    apply_chart_theme,
)

# -----------------------------------------------------------------------------
# FastAPI Application Initialization
# -----------------------------------------------------------------------------
app = FastAPI(
    title="NASDAQ-100 Market Behavior Analytics API",
    description="Quantitative Institutional Research & Intraday Microstructure Engine",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data In-Memory Cache
_PROCESSED_CACHE: Dict[str, Any] = {}


# -----------------------------------------------------------------------------
# Chart & Table Formatting Helpers
# -----------------------------------------------------------------------------

def sanitize_for_json(o: Any) -> Any:
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        if np.isnan(o) or np.isinf(o):
            return None
        return float(o)
    if isinstance(o, np.ndarray):
        return [sanitize_for_json(x) for x in o.tolist()]
    if isinstance(o, dict):
        return {str(k): sanitize_for_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize_for_json(x) for x in o]
    if isinstance(o, (pd.Timestamp, datetime.datetime, datetime.date)):
        return str(o)
    if pd.isna(o):
        return None
    return o


def safe_api(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        res = await func(*args, **kwargs)
        if isinstance(res, (dict, list)):
            return JSONResponse(content=sanitize_for_json(res))
        return res
    return wrapper

def serialize_figure(fig: go.Figure, title: str = "", height: int = 420) -> dict:
    """
    Apply unified dark terminal palette (#0e1117 / #161b22) and serialize to Plotly JSON.
    """
    fig.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#0e1117",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color="#c9d1d9",
            size=12,
        ),
        margin=dict(l=55, r=30, t=55, b=45),
        height=height,
    )
    if title:
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=14, color="#f0f6fc"),
                x=0.01,
                y=0.97,
            )
        )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#30363d",
        zeroline=False,
        showline=True,
        linecolor="#30363d",
        tickfont=dict(color="#8b949e", size=11),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#30363d",
        zeroline=True,
        zerolinecolor="rgba(255,255,255,0.12)",
        showline=True,
        linecolor="#30363d",
        tickfont=dict(color="#8b949e", size=11),
    )
    return json.loads(fig.to_json())


def df_to_safe_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert pandas summary DataFrame to JSON records with guaranteed index preservation.
    Strictly follows constraint:
    Call .reset_index() first and explicitly name the index column 'label'
    (df.reset_index().rename(columns={'index': 'label'}))
    """
    if df is None or df.empty:
        return []

    d = df.copy()

    # Drop internal hidden columns starting with '_'
    hidden_cols = [c for c in d.columns if str(c).startswith("_")]
    if hidden_cols:
        d = d.drop(columns=hidden_cols)

    # Reset index and explicitly rename index column to 'label'
    d = d.reset_index()
    if "index" in d.columns:
        d = d.rename(columns={"index": "label"})
    elif "level_0" in d.columns:
        d = d.rename(columns={"level_0": "label"})

    first_col = d.columns[0]
    records = []
    for row in d.to_dict(orient="records"):
        clean_row = {}
        for k, v in row.items():
            if pd.isna(v):
                clean_row[k] = "—"
            elif isinstance(v, (np.floating, float)):
                if np.isnan(v) or np.isinf(v):
                    clean_row[k] = "—"
                else:
                    clean_row[k] = round(v, 4)
            elif isinstance(v, (np.integer, int)):
                clean_row[k] = int(v)
            elif isinstance(v, (pd.Timestamp, datetime.datetime, datetime.date)):
                clean_row[k] = str(v)
            else:
                clean_row[k] = str(v)

        # Fallback guarantee for table renderers
        if "label" not in clean_row:
            clean_row["label"] = clean_row.get(first_col, "—")
        records.append(clean_row)

    return records


# -----------------------------------------------------------------------------
# Data Ingestion & Caching
# -----------------------------------------------------------------------------
def get_processed_data(ticker: str = INDEX_TICKER, force_refresh: bool = False) -> Dict[str, Any]:
    """Load, clean, precalculate, and cache daily and intraday series."""
    cache_key = f"{ticker}_{force_refresh}"
    if not force_refresh and cache_key in _PROCESSED_CACHE:
        return _PROCESSED_CACHE[cache_key]

    raw_daily = load_daily_data(ticker=ticker, period="10y", force_refresh=force_refresh)
    clean_daily, audit_daily = clean_market_data(raw_daily, interval="1d")
    calc_daily = compute_daily_returns_and_sessions(clean_daily)
    classified_daily, _ = classify_volatility_regimes(calc_daily)
    cov_daily = get_data_coverage_info(clean_daily, interval="1d", ticker=ticker)

    raw_intra = load_intraday_data(ticker=ticker, interval="5m", force_refresh=force_refresh)
    clean_intra, audit_intra = clean_market_data(raw_intra, interval="5m")
    cov_intra = get_data_coverage_info(clean_intra, interval="5m", ticker=ticker)

    data = {
        "daily": classified_daily,
        "intra": clean_intra,
        "cov_daily": cov_daily,
        "cov_intra": cov_intra,
        "audit_daily": audit_daily,
        "audit_intra": audit_intra,
    }
    _PROCESSED_CACHE[cache_key] = data
    return data


def filter_daily_data(
    df_daily: pd.DataFrame,
    period: str = "10Y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    weekday: str = "All Days",
    regime: str = "All Regimes",
) -> pd.DataFrame:
    """Filter daily dataset by historical lookback period, day of week, and volatility regime."""
    if df_daily.empty:
        return df_daily

    filtered = df_daily.copy()
    now_ny = convert_to_new_york(datetime.datetime.now())

    # Period filter
    if period == "1Y":
        start_dt = now_ny - datetime.timedelta(days=365)
        filtered = filtered[filtered.index >= start_dt]
    elif period == "3Y":
        start_dt = now_ny - datetime.timedelta(days=365 * 3)
        filtered = filtered[filtered.index >= start_dt]
    elif period == "5Y":
        start_dt = now_ny - datetime.timedelta(days=365 * 5)
        filtered = filtered[filtered.index >= start_dt]
    elif period == "10Y":
        start_dt = now_ny - datetime.timedelta(days=365 * 10)
        filtered = filtered[filtered.index >= start_dt]
    elif period == "Custom" and start_date and end_date:
        try:
            s_dt = pd.to_datetime(start_date).date()
            e_dt = pd.to_datetime(end_date).date()
            filtered = filtered[(filtered.index.date >= s_dt) & (filtered.index.date <= e_dt)]
        except Exception:
            pass

    # Weekday filter
    if weekday != "All Days" and "weekday" in filtered.columns:
        filtered = filtered[filtered["weekday"] == weekday]

    # Regime filter
    if regime != "All Regimes" and "vol_regime" in filtered.columns:
        filtered = filtered[filtered["vol_regime"] == regime]

    return filtered


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/config")
@safe_api
async def get_dashboard_config():
    """Returns metadata configuration for available instruments, periods, and filters."""
    return {
        "instruments": [
            {"ticker": INDEX_TICKER, "name": "NASDAQ-100 Index (^NDX)"},
            {"ticker": ETF_TICKER, "name": "Invesco QQQ Trust (QQQ ETF)"},
        ],
        "periods": ["1Y", "3Y", "5Y", "10Y", "Max", "Custom"],
        "weekdays": ["All Days", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "regimes": ["All Regimes", "Low Volatility", "Normal Volatility", "High Volatility"],
        "resolutions": ["5m", "10m", "15m", "30m", "60m"],
        "benchmarks": [
            {"ticker": "^GSPC", "name": "S&P 500 Index (^GSPC)"},
            {"ticker": "SPY", "name": "SPDR S&P 500 ETF (SPY)"},
            {"ticker": "QQQ", "name": "Invesco QQQ Trust (QQQ)"},
            {"ticker": "TQQQ", "name": "ProShares UltraPro QQQ (TQQQ)"},
        ],
    }


@app.get("/api/market-clock")
@safe_api
async def get_market_clock():
    """Returns current real-time clock and NASDAQ session timings formatted in MYT."""
    clock = get_current_market_clock_info()
    today_session = clock.get("today_session", {})
    return {
        "now_myt": clock["now_myt"].strftime("%H:%M:%S"),
        "now_myt_full": clock["now_myt"].strftime("%Y-%m-%d %H:%M:%S MYT"),
        "status_text": clock["status_text"],
        "status_color": clock["status_color"],
        "is_active": clock.get("is_regular_open", False),
        "session_label": today_session.get("myt_session_label", "21:30 – 04:00+1 MYT"),
        "tz_abbr": today_session.get("tz_abbr", "EDT"),
        "is_dst": today_session.get("is_dst", True),
        "myt_open_str": today_session.get("myt_open_str", "21:30 MYT"),
        "myt_close_str": today_session.get("myt_close_str", "04:00 MYT"),
        "dst_description": "Daylight Saving Time (UTC-4)" if today_session.get("is_dst", True) else "Standard Time (UTC-5)",
    }


# 1. Overview & Seasonality
@app.get("/api/overview")
@safe_api
async def get_overview_data(
    instrument: str = Query(INDEX_TICKER),
    period: str = Query("10Y"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    weekday: str = Query("All Days"),
    regime: str = Query("All Regimes"),
    force_refresh: bool = Query(False),
):
    """
    Overview & Macro Seasonality:
    Cumulative return decomposition, yearly performance, monthly seasonality, and summary tables.
    """
    data = get_processed_data(ticker=instrument, force_refresh=force_refresh)
    df_filtered = filter_daily_data(data["daily"], period, start_date, end_date, weekday, regime)

    if df_filtered.empty:
        raise HTTPException(status_code=404, detail="No data available for the selected filters.")

    n_sessions = len(df_filtered)
    mean_ret = float(df_filtered["total_return_pct"].mean())
    ann_vol = float(df_filtered["total_return"].std() * np.sqrt(252) * 100.0)
    mean_range = float(df_filtered["daily_range_pct"].mean())
    pos_pct = float((df_filtered["total_return_pct"] > 0).sum() / n_sessions * 100.0)
    _, _, max_dd = compute_drawdown_series(df_filtered["total_return"])

    # 1. Cumulative returns chart
    df_cum = compute_cumulative_performance(df_filtered)
    fig_cum = plot_cumulative_performance(df_cum)
    chart_cumulative = serialize_figure(fig_cum, "NASDAQ-100 Cumulative Return Trajectory (Total vs Overnight vs Regular)", height=430)

    # 2. Return distribution histogram
    fig_dist = plot_distribution_histogram(df_filtered["total_return_pct"], "Daily Return % Distribution", "Total Return (%)")
    chart_distribution = serialize_figure(fig_dist, "Historical Return Distribution", height=380)

    # 3. Monthly seasonality chart
    month_df = compute_monthly_statistics(df_filtered)
    fig_month = go.Figure()
    if not month_df.empty and "_raw_mean_ret" in month_df.columns:
        fig_month.add_trace(go.Bar(
            x=month_df["Month"],
            y=month_df["_raw_mean_ret"],
            marker=dict(color=np.where(month_df["_raw_mean_ret"] >= 0, "#2ea043", "#da3633")),
            hovertemplate="Month: %{x}<br>Mean Return: %{y:.3f}%<extra></extra>",
        ))
        fig_month.update_yaxes(ticksuffix="%")
    chart_monthly = serialize_figure(fig_month, "Monthly Average Daily Return (Jan–Dec)", height=380)

    # 4. Yearly performance chart
    year_df = compute_yearly_statistics(df_filtered)
    fig_year = go.Figure()
    if not year_df.empty and "_raw_cum_ret" in year_df.columns:
        fig_year.add_trace(go.Bar(
            x=year_df["Year"],
            y=year_df["_raw_cum_ret"],
            marker=dict(color=np.where(year_df["_raw_cum_ret"] >= 0, "#58a6ff", "#da3633")),
            hovertemplate="Year: %{x}<br>Annual Return: %{y:.2f}%<extra></extra>",
        ))
        fig_year.update_yaxes(ticksuffix="%")
    chart_yearly = serialize_figure(fig_year, "Year-by-Year Cumulative Performance History", height=380)

    # Summary Tables
    freq_df = compute_historical_frequencies(df_filtered)
    dist = compute_distribution_summary(df_filtered["total_return_pct"], label="Daily Return %")
    dist_table = pd.DataFrame([
        ("Mean Return", f"{dist['mean']:.3f}%"),
        ("Median Return", f"{dist['median']:.3f}%"),
        ("Std Dev", f"{dist['std']:.3f}%"),
        ("10th Percentile", f"{dist['p10']:.3f}%"),
        ("25th Percentile", f"{dist['p25']:.3f}%"),
        ("75th Percentile", f"{dist['p75']:.3f}%"),
        ("90th Percentile", f"{dist['p90']:.3f}%"),
        ("Skewness", f"{dist['skewness']:.3f}"),
        ("Kurtosis", f"{dist['kurtosis']:.3f}"),
    ], columns=["Statistic", "Value"])

    audit_df = format_audit_report_table(data["audit_daily"])
    cov_info = get_data_coverage_info(df_filtered, interval="1d", ticker=instrument)

    return {
        "kpis": {
            "trading_sessions": f"{n_sessions:,}",
            "mean_daily_return": f"{mean_ret:+.3f}%",
            "annualized_volatility": f"{ann_vol:.2f}%",
            "avg_daily_range": f"{mean_range:.2f}%",
            "positive_sessions_pct": f"{pos_pct:.1f}%",
            "max_drawdown": f"{max_dd:.2f}%",
        },
        "charts": {
            "cumulative": chart_cumulative,
            "distribution": chart_distribution,
            "monthly": chart_monthly,
            "yearly": chart_yearly,
        },
        "tables": {
            "historical_frequencies": df_to_safe_records(freq_df),
            "distribution_summary": df_to_safe_records(dist_table),
            "monthly_summary": df_to_safe_records(month_df),
            "yearly_summary": df_to_safe_records(year_df),
            "audit_report": df_to_safe_records(audit_df),
        },
        "coverage": cov_info,
    }


# 2. Weekday Seasonality
@app.get("/api/weekday")
@safe_api
async def get_weekday_data(
    instrument: str = Query(INDEX_TICKER),
    period: str = Query("10Y"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    regime: str = Query("All Regimes"),
    force_refresh: bool = Query(False),
):
    """
    Weekday Seasonality:
    Day-of-week win rates, range distributions, and monthly/yearly seasonality tables.
    """
    data = get_processed_data(ticker=instrument, force_refresh=force_refresh)
    df_filtered = filter_daily_data(data["daily"], period, start_date, end_date, "All Days", regime)

    if df_filtered.empty:
        raise HTTPException(status_code=404, detail="No data available.")

    weekday_df = compute_weekday_statistics(df_filtered)
    month_df = compute_monthly_statistics(df_filtered)
    year_df = compute_yearly_statistics(df_filtered)

    # 1. Weekday comparison chart
    fig_week = plot_weekday_comparison(weekday_df)
    chart_weekday = serialize_figure(fig_week, "Monday–Friday Return & Win Rate Profile", height=400)

    # 2. Range & Volatility by weekday
    fig_range = go.Figure()
    if "_raw_avg_range" in weekday_df.columns:
        fig_range.add_trace(go.Bar(
            x=weekday_df["Day of Week"],
            y=weekday_df["_raw_avg_range"],
            name="Avg Daily Range (%)",
            marker=dict(color="#58a6ff"),
            hovertemplate="Avg Range: %{y:.2f}%<extra></extra>",
        ))
    if "_raw_pos_pct" in weekday_df.columns:
        fig_range.add_trace(go.Scatter(
            x=weekday_df["Day of Week"],
            y=weekday_df["_raw_pos_pct"],
            name="Positive Day %",
            yaxis="y2",
            mode="lines+markers",
            marker=dict(size=8, color="#2ea043"),
            line=dict(color="#2ea043", width=2),
            hovertemplate="Positive: %{y:.1f}%<extra></extra>",
        ))
    fig_range.update_layout(
        yaxis=dict(title="Average Range (%)", ticksuffix="%"),
        yaxis2=dict(title="Positive %", ticksuffix="%", overlaying="y", side="right", showgrid=False),
    )
    chart_range_vol = serialize_figure(fig_range, "Weekday Range Amplitude vs Positive Probability", height=400)

    # 3. Monthly seasonality chart
    fig_month = go.Figure()
    if "_raw_mean_ret" in month_df.columns:
        fig_month.add_trace(go.Bar(
            x=month_df["Month"],
            y=month_df["_raw_mean_ret"],
            marker=dict(color=np.where(month_df["_raw_mean_ret"] >= 0, "#2ea043", "#da3633")),
            hovertemplate="Month: %{x}<br>Mean Return: %{y:.3f}%<extra></extra>",
        ))
        fig_month.update_yaxes(ticksuffix="%")
    chart_month = serialize_figure(fig_month, "Monthly Return Performance (Jan–Dec)", height=380)

    # 4. Weekday box plots
    fig_box = px.box(
        df_filtered,
        x="weekday",
        y="total_return_pct",
        category_orders={"weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]},
        labels={"weekday": "Day of Week", "total_return_pct": "Daily Return (%)"},
        color="weekday",
        color_discrete_sequence=["#58a6ff", "#3fb950", "#d29922", "#bc8cff", "#f85149"],
    )
    chart_box = serialize_figure(fig_box, "Day-of-Week Return Dispersion (Box Plot)", height=380)

    return {
        "charts": {
            "weekday_comparison": chart_weekday,
            "weekday_range_vol": chart_range_vol,
            "monthly_seasonality": chart_month,
            "weekday_box": chart_box,
        },
        "tables": {
            "weekday_summary": df_to_safe_records(weekday_df),
            "monthly_summary": df_to_safe_records(month_df),
            "yearly_summary": df_to_safe_records(year_df),
        },
    }


# 3. Intraday Profiling
@app.get("/api/intraday")
@safe_api
async def get_intraday_data(
    instrument: str = Query(INDEX_TICKER),
    resolution: str = Query("5m"),
    metric: str = Query("mean_abs_return_pct"),
    force_refresh: bool = Query(False),
):
    """
    Intraday Profiling:
    RTH regular trading hours 09:30–16:00 ET, time-window return bars, volume curves, and MYT time buckets.
    """
    data = get_processed_data(ticker=instrument, force_refresh=force_refresh)
    df_intra = data["intra"]

    if df_intra.empty:
        raise HTTPException(status_code=404, detail="No intraday data available.")

    df_resampled = resample_intraday(df_intra, interval_str=resolution)
    stats_df = compute_intraday_time_bucket_statistics(df_resampled)

    if stats_df.empty:
        raise HTTPException(status_code=404, detail="Could not calculate intraday statistics.")

    # 1. Main profile chart
    fig_prof = plot_intraday_profile(stats_df, metric=metric, title=f"NASDAQ-100 {resolution} Time-of-Day Profile (MYT)")
    chart_profile = serialize_figure(fig_prof, f"Intraday {resolution} Profile across Session (MYT)", height=420)

    # 2. Volatility profile
    fig_vol = plot_intraday_profile(stats_df, metric="realized_vol_pct", title="Session Volatility Curve (MYT)")
    chart_vol = serialize_figure(fig_vol, "Intraday Realized Volatility Curve (MYT)", height=380)

    # 3. Volume vs volatility scatter
    fig_scatter = plot_volume_vs_volatility_scatter(stats_df)
    chart_scatter = serialize_figure(fig_scatter, "Relative Volume vs Volatility / Movement", height=380)

    # Formatted display table
    display_table = stats_df.copy()
    display_table["Time (MYT)"] = display_table["display_label_myt"]
    display_table["Time (ET Ref)"] = display_table["time_ny"] + " ET"
    display_table["Avg Return"] = display_table["mean_return_pct"].apply(lambda x: f"{x:+.3f}%")
    display_table["Median Return"] = display_table["median_return_pct"].apply(lambda x: f"{x:+.3f}%")
    display_table["Avg Abs Move"] = display_table["mean_abs_return_pct"].apply(lambda x: f"{x:.3f}%")
    display_table["Avg Range"] = display_table["mean_range_pct"].apply(lambda x: f"{x:.3f}%")
    display_table["Annualized Vol"] = display_table["realized_vol_pct"].apply(lambda x: f"{x:.2f}%")
    display_table["Positive %"] = display_table["positive_pct"].apply(lambda x: f"{x:.1f}%")
    display_table["Relative Volume"] = display_table["relative_volume"].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "—")
    display_table["Sample Size (n)"] = display_table["sample_size"]

    cols_order = [
        "Time (MYT)", "Time (ET Ref)", "Sample Size (n)", "Avg Return", "Median Return",
        "Avg Abs Move", "Avg Range", "Annualized Vol", "Positive %", "Relative Volume"
    ]
    table_records = df_to_safe_records(display_table[cols_order])

    return {
        "charts": {
            "profile": chart_profile,
            "volatility": chart_vol,
            "volume_scatter": chart_scatter,
        },
        "tables": {
            "time_buckets": table_records,
        },
        "coverage": data["cov_intra"],
    }


# 4. Opening Behavior & Gaps
@app.get("/api/opening")
@safe_api
async def get_opening_data(
    instrument: str = Query(INDEX_TICKER),
    period: str = Query("10Y"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    weekday: str = Query("All Days"),
    regime: str = Query("All Regimes"),
    force_refresh: bool = Query(False),
):
    """
    Opening Behavior & Gaps:
    Cash open gap distributions, gap fill probabilities, and opening range expansion 15m/30m/60m.
    """
    data = get_processed_data(ticker=instrument, force_refresh=force_refresh)
    df_filtered = filter_daily_data(data["daily"], period, start_date, end_date, weekday, regime)
    df_intra = data["intra"]

    if df_filtered.empty:
        raise HTTPException(status_code=404, detail="No data available.")

    gap_series = df_filtered["gap_pct"].dropna()
    n_gaps = len(gap_series)
    up_gaps = int((gap_series > 0).sum())
    down_gaps = int((gap_series < 0).sum())
    gap_filled_pct = float(df_filtered["gap_filled"].sum() / max(1, n_gaps) * 100.0) if "gap_filled" in df_filtered.columns else 0.0

    # 1. Opening gap distribution histogram
    fig_gap = plot_distribution_histogram(gap_series, "Historical Opening Gap % Distribution", "Opening Gap (%)")
    chart_gap_dist = serialize_figure(fig_gap, "Opening Gap % Distribution (Close_{t-1} → Open_t)", height=400)

    # 2. Time-to-fill distribution
    chart_fill_time = None
    filled_sessions_count = 0
    median_time_to_fill = None
    if not df_intra.empty:
        gap_fill_intra = compute_gap_fill_intraday(df_intra, df_filtered)
        if not gap_fill_intra.empty and gap_fill_intra["gap_filled"].sum() > 0:
            filled_subset = gap_fill_intra[gap_fill_intra["gap_filled"]]
            filled_sessions_count = len(filled_subset)
            median_time_to_fill = float(filled_subset["minutes_to_fill"].median())
            fig_fill = go.Figure()
            fig_fill.add_trace(go.Histogram(
                x=filled_subset["minutes_to_fill"],
                nbinsx=15,
                marker=dict(color="#2ea043", line=dict(color="#30363d", width=0.5)),
                hovertemplate="Minutes from Open: %{x}<br>Count: %{y}<extra></extra>",
            ))
            fig_fill.update_xaxes(title="Minutes from Market Open (09:30 ET)", ticksuffix="m")
            fig_fill.update_yaxes(title="Filled Sessions Count")
            chart_fill_time = serialize_figure(fig_fill, "Intraday Minutes to Gap Fill Distribution", height=400)

    # 3. Opening range expansion (5m, 15m, 30m, 60m)
    chart_opening_ratios = None
    ratio_records = []
    if not df_intra.empty:
        open_metrics_df = compute_opening_window_metrics(df_intra, df_filtered)
        if not open_metrics_df.empty:
            ratio_data = [
                ("First 5m Window (09:30–09:35 ET / 21:30–21:35 MYT)", float(open_metrics_df["range_ratio_5m_pct"].mean()), float(open_metrics_df["ret_5m_pct"].mean())),
                ("First 15m Window (09:30–09:45 ET / 21:30–21:45 MYT)", float(open_metrics_df["range_ratio_15m_pct"].mean()), float(open_metrics_df["ret_15m_pct"].mean())),
                ("First 30m Window (09:30–10:00 ET / 21:30–22:00 MYT)", float(open_metrics_df["range_ratio_30m_pct"].mean()), float(open_metrics_df["ret_30m_pct"].mean())),
                ("First 60m Window (09:30–10:30 ET / 21:30–22:30 MYT)", float(open_metrics_df["range_ratio_60m_pct"].mean()), float(open_metrics_df["ret_60m_pct"].mean())),
            ]
            ratio_df = pd.DataFrame(ratio_data, columns=["Opening Window", "Avg % of Full Day Range", "Avg Window Return %"])
            fig_ratios = go.Figure()
            fig_ratios.add_trace(go.Bar(
                x=["5m Window", "15m Window", "30m Window", "60m Window"],
                y=ratio_df["Avg % of Full Day Range"],
                marker=dict(color=["#58a6ff", "#39c5cf", "#3b82f6", "#bc8cff"]),
                text=ratio_df["Avg % of Full Day Range"].apply(lambda x: f"{x:.1f}%"),
                textposition="auto",
                hovertemplate="%{x}<br>Range Share: %{y:.1f}%<extra></extra>",
            ))
            fig_ratios.update_yaxes(title="Share of Total Day Range (%)", ticksuffix="%")
            chart_opening_ratios = serialize_figure(fig_ratios, "Opening Range Share vs Full Session High-Low Range", height=400)
            ratio_records = df_to_safe_records(ratio_df)

    # Descriptive gap stats table
    dist = compute_distribution_summary(gap_series, label="Opening Gap %")
    gap_table = pd.DataFrame([
        ("Sample Sessions (n)", f"{dist['count']:,}"),
        ("Mean Opening Gap", f"{dist['mean']:+.3f}%"),
        ("Median Opening Gap", f"{dist['median']:+.3f}%"),
        ("Standard Deviation", f"{dist['std']:.3f}%"),
        ("10th Percentile", f"{dist['p10']:+.3f}%"),
        ("25th Percentile", f"{dist['p25']:+.3f}%"),
        ("75th Percentile", f"{dist['p75']:+.3f}%"),
        ("90th Percentile", f"{dist['p90']:+.3f}%"),
        ("Up Gap Frequency", f"{up_gaps / max(1, n_gaps) * 100:.1f}% ({up_gaps} days)"),
        ("Down Gap Frequency", f"{down_gaps / max(1, n_gaps) * 100:.1f}% ({down_gaps} days)"),
        ("Up Gap Fill Rate", f"{(df_filtered[df_filtered['gap_pct'] > 0]['gap_filled'].sum() / max(1, up_gaps) * 100):.1f}%"),
        ("Down Gap Fill Rate", f"{(df_filtered[df_filtered['gap_pct'] < 0]['gap_filled'].sum() / max(1, down_gaps) * 100):.1f}%"),
    ], columns=["Gap Metric", "Value"])

    return {
        "kpis": {
            "mean_gap": f"{gap_series.mean():+.2f}%",
            "median_gap": f"{gap_series.median():+.2f}%",
            "up_gaps_pct": f"{up_gaps / max(1, n_gaps) * 100:.1f}%",
            "down_gaps_pct": f"{down_gaps / max(1, n_gaps) * 100:.1f}%",
            "gap_fill_rate": f"{gap_filled_pct:.1f}%",
            "filled_sessions_count": filled_sessions_count,
            "median_time_to_fill": f"{median_time_to_fill:.0f}m" if median_time_to_fill else "N/A",
        },
        "charts": {
            "gap_distribution": chart_gap_dist,
            "time_to_fill": chart_fill_time,
            "opening_range_share": chart_opening_ratios,
        },
        "tables": {
            "gap_statistics": df_to_safe_records(gap_table),
            "opening_windows": ratio_records,
        },
    }


# 5. Overnight & Globex Dynamics
@app.get("/api/overnight")
@safe_api
async def get_overnight_data(
    instrument: str = Query(INDEX_TICKER),
    period: str = Query("10Y"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    weekday: str = Query("All Days"),
    regime: str = Query("All Regimes"),
    force_refresh: bool = Query(False),
):
    """
    Overnight vs Regular Session:
    Electronic trading hours vs RTH cash session comparison and overnight drift vs day session return.
    """
    data = get_processed_data(ticker=instrument, force_refresh=force_refresh)
    df_filtered = filter_daily_data(data["daily"], period, start_date, end_date, weekday, regime)
    df_clean = df_filtered.dropna(subset=["overnight_return_pct", "regular_return_pct", "total_return_pct"]).copy()

    if df_clean.empty:
        raise HTTPException(status_code=404, detail="No data available.")

    n = len(df_clean)
    over_mean = float(df_clean["overnight_return_pct"].mean())
    reg_mean = float(df_clean["regular_return_pct"].mean())
    tot_mean = float(df_clean["total_return_pct"].mean())

    over_vol = float(df_clean["overnight_return"].std() * np.sqrt(252) * 100.0)
    reg_vol = float(df_clean["regular_return"].std() * np.sqrt(252) * 100.0)
    tot_vol = float(df_clean["total_return"].std() * np.sqrt(252) * 100.0)

    over_pos = float((df_clean["overnight_return_pct"] > 0).sum() / n * 100.0)
    reg_pos = float((df_clean["regular_return_pct"] > 0).sum() / n * 100.0)

    var_over = float(df_clean["overnight_return"].var())
    var_reg = float(df_clean["regular_return"].var())
    over_var_share = (var_over / (var_over + var_reg) * 100.0) if (var_over + var_reg) > 0 else 50.0

    # 1. Cumulative performance
    df_cum = compute_cumulative_performance(df_clean)
    fig_cum = plot_cumulative_performance(df_cum)
    chart_cum = serialize_figure(fig_cum, "Cumulative Return Trajectories: Overnight vs Regular Session", height=430)

    # 2. Overnight distribution
    fig_over = go.Figure()
    fig_over.add_trace(go.Histogram(
        x=df_clean["overnight_return_pct"],
        nbinsx=45,
        marker=dict(color="#d29922", line=dict(color="#30363d", width=0.5)),
        name="Overnight",
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
    ))
    fig_over.add_vline(x=over_mean, line=dict(color="#2ea043", width=2, dash="dash"), annotation_text=f"Mean: {over_mean:.3f}%")
    fig_over.update_xaxes(title="Overnight Return (%)", ticksuffix="%")
    chart_over_dist = serialize_figure(fig_over, "Overnight Return Distribution (Close_{t-1} → Open_t)", height=380)

    # 3. Regular session distribution
    fig_reg = go.Figure()
    fig_reg.add_trace(go.Histogram(
        x=df_clean["regular_return_pct"],
        nbinsx=45,
        marker=dict(color="#bc8cff", line=dict(color="#30363d", width=0.5)),
        name="Regular Session",
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
    ))
    fig_reg.add_vline(x=reg_mean, line=dict(color="#2ea043", width=2, dash="dash"), annotation_text=f"Mean: {reg_mean:.3f}%")
    fig_reg.update_xaxes(title="Regular Session Return (%)", ticksuffix="%")
    chart_reg_dist = serialize_figure(fig_reg, "Regular Session Return Distribution (Open_t → Close_t)", height=380)

    # Statistical attribution table
    dist_over = compute_distribution_summary(df_clean["overnight_return_pct"], label="Overnight")
    dist_reg = compute_distribution_summary(df_clean["regular_return_pct"], label="Regular Session")
    dist_tot = compute_distribution_summary(df_clean["total_return_pct"], label="Total Close-to-Close")

    comp_table = pd.DataFrame([
        ("Sample Size (Sessions)", f"{n:,}", f"{n:,}", f"{n:,}"),
        ("Mean Return", f"{dist_over['mean']:+.3f}%", f"{dist_reg['mean']:+.3f}%", f"{dist_tot['mean']:+.3f}%"),
        ("Median Return", f"{dist_over['median']:+.3f}%", f"{dist_reg['median']:+.3f}%", f"{dist_tot['median']:+.3f}%"),
        ("Standard Deviation", f"{dist_over['std']:.3f}%", f"{dist_reg['std']:.3f}%", f"{dist_tot['std']:.3f}%"),
        ("Annualized Volatility", f"{over_vol:.2f}%", f"{reg_vol:.2f}%", f"{tot_vol:.2f}%"),
        ("Avg Absolute Move", f"{df_clean['abs_overnight_return_pct'].mean():.3f}%", f"{df_clean['abs_regular_return_pct'].mean():.3f}%", f"{df_clean['abs_total_return_pct'].mean():.3f}%"),
        ("Positive Sessions %", f"{dist_over['positive_pct']:.1f}%", f"{dist_reg['positive_pct']:.1f}%", f"{dist_tot['positive_pct']:.1f}%"),
        ("Negative Sessions %", f"{dist_over['negative_pct']:.1f}%", f"{dist_reg['negative_pct']:.1f}%", f"{dist_tot['negative_pct']:.1f}%"),
        ("10th Percentile", f"{dist_over['p10']:+.3f}%", f"{dist_reg['p10']:+.3f}%", f"{dist_tot['p10']:+.3f}%"),
        ("90th Percentile", f"{dist_over['p90']:+.3f}%", f"{dist_reg['p90']:+.3f}%", f"{dist_tot['p90']:+.3f}%"),
        ("Skewness", f"{dist_over['skewness']:.3f}", f"{dist_reg['skewness']:.3f}", f"{dist_tot['skewness']:.3f}"),
    ], columns=["Statistic", "Overnight (Close_{t-1} → Open_t)", "Regular Session (Open_t → Close_t)", "Total (Close_{t-1} → Close_t)"])

    return {
        "kpis": {
            "overnight_mean": f"{over_mean:+.3f}%",
            "regular_mean": f"{reg_mean:+.3f}%",
            "overnight_vol": f"{over_vol:.2f}%",
            "regular_vol": f"{reg_vol:.2f}%",
            "overnight_var_share": f"{over_var_share:.1f}%",
            "overnight_pos_pct": f"{over_pos:.1f}%",
            "regular_pos_pct": f"{reg_pos:.1f}%",
        },
        "charts": {
            "cumulative": chart_cum,
            "overnight_dist": chart_over_dist,
            "regular_dist": chart_reg_dist,
        },
        "tables": {
            "attribution": df_to_safe_records(comp_table),
        },
    }


# 6. Volatility & Excursions
@app.get("/api/volatility")
@safe_api
async def get_volatility_data(
    instrument: str = Query(INDEX_TICKER),
    period: str = Query("10Y"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    low_pct: float = Query(25.0),
    high_pct: float = Query(75.0),
    metric: str = Query("daily_range_pct"),
    force_refresh: bool = Query(False),
):
    """
    Volatility & Excursions:
    Rolling realized volatility 20d/60d/120d, percentile-based volatility regimes, and session volatility profiles.
    """
    data = get_processed_data(ticker=instrument, force_refresh=force_refresh)
    df_filtered = filter_daily_data(data["daily"], period, start_date, end_date, "All Days", "All Regimes")
    df_intra = data["intra"]

    if df_filtered.empty:
        raise HTTPException(status_code=404, detail="No data available.")

    # Rolling metrics
    df_rolling = compute_rolling_metrics(df_filtered)
    fig_roll = plot_rolling_volatility(df_rolling)
    chart_rolling = serialize_figure(fig_roll, "Rolling Annualized Realized Volatility Horizons (20d, 60d, 120d)", height=420)

    # Volatility Regimes classification
    classified_df, thresh = classify_volatility_regimes(
        df_rolling,
        metric_col=metric,
        low_pct=low_pct,
        high_pct=high_pct,
    )
    regime_stats = compute_regime_comparison_statistics(classified_df)

    # Regime bar & scatter chart
    fig_reg = go.Figure()
    if "_raw_range" in regime_stats.columns:
        fig_reg.add_trace(go.Bar(
            x=regime_stats["Volatility Regime"],
            y=regime_stats["_raw_range"],
            name="Avg Daily Range (%)",
            marker=dict(color=["#58a6ff", "#3b82f6", "#da3633"]),
            hovertemplate="Regime: %{x}<br>Avg Range: %{y:.2f}%<extra></extra>",
        ))
    if "_raw_vol" in regime_stats.columns:
        fig_reg.add_trace(go.Scatter(
            x=regime_stats["Volatility Regime"],
            y=regime_stats["_raw_vol"],
            name="Annualized Volatility (%)",
            yaxis="y2",
            mode="lines+markers",
            marker=dict(size=10, color="#d29922"),
            line=dict(color="#d29922", width=2),
            hovertemplate="Vol: %{y:.2f}%<extra></extra>",
        ))
    fig_reg.update_layout(
        yaxis=dict(title="Average Range (%)", ticksuffix="%"),
        yaxis2=dict(title="Annualized Vol (%)", ticksuffix="%", overlaying="y", side="right", showgrid=False),
    )
    chart_regimes = serialize_figure(fig_reg, "Daily Range and Realized Volatility by Market Regime", height=400)

    # Intraday session volatility profile
    chart_profile = None
    if not df_intra.empty:
        stats_df = compute_intraday_time_bucket_statistics(df_intra)
        fig_prof = plot_intraday_profile(stats_df, metric="realized_vol_pct", title="Session Volatility Curve (MYT)")
        chart_profile = serialize_figure(fig_prof, "Intraday Session Realized Volatility Curve (MYT)", height=400)

    cols_regime = [
        "Volatility Regime", "Sessions (n)", "Sample Share", "Mean Daily Return", "Daily Std Dev",
        "Annualized Vol", "Positive Days %", "Avg Daily Range %", "Avg Overnight Return", "Avg Regular Return",
        "Avg Abs Gap %", "Gap Fill Rate"
    ]
    display_regime = regime_stats[[c for c in cols_regime if c in regime_stats.columns]]

    return {
        "thresholds": {
            "low_threshold": f"{thresh['low_threshold']:.2f}%",
            "high_threshold": f"{thresh['high_threshold']:.2f}%",
            "metric_col": metric,
        },
        "charts": {
            "rolling_vol": chart_rolling,
            "regime_comparison": chart_regimes,
            "session_volatility": chart_profile,
        },
        "tables": {
            "regime_summary": df_to_safe_records(display_regime),
        },
    }


# 7. Drawdown & Regime Analysis
@app.get("/api/drawdowns")
@safe_api
async def get_drawdown_data(
    instrument: str = Query(INDEX_TICKER),
    period: str = Query("10Y"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    weekday: str = Query("All Days"),
    regime: str = Query("All Regimes"),
    force_refresh: bool = Query(False),
):
    """
    Drawdown & Stress Analysis:
    Underwater equity curves, duration distributions, and major historical drawdown episodes.
    """
    data = get_processed_data(ticker=instrument, force_refresh=force_refresh)
    df_filtered = filter_daily_data(data["daily"], period, start_date, end_date, weekday, regime)

    if df_filtered.empty:
        raise HTTPException(status_code=404, detail="No data available.")

    returns = df_filtered["total_return"].dropna()
    wealth_index, dd_series, max_dd = compute_drawdown_series(returns)
    curr_dd = float(dd_series.iloc[-1]) if not dd_series.empty else 0.0
    avg_dd = float(dd_series[dd_series < 0].mean()) if (dd_series < 0).sum() > 0 else 0.0

    # 1. Underwater drawdown plot
    fig_dd = plot_drawdown_curve(dd_series)
    chart_dd = serialize_figure(fig_dd, "Historical Underwater Drawdown Curve", height=420)

    # 2. Drawdown Episodes
    episodes_df = compute_drawdown_episodes(dd_series, top_n=10)

    # 3. Drawdown duration bar chart
    fig_dur = go.Figure()
    if not episodes_df.empty and "max_drawdown_pct" in episodes_df.columns:
        fig_dur.add_trace(go.Bar(
            x=episodes_df["Peak Date (MYT)"],
            y=episodes_df["max_drawdown_pct"],
            marker=dict(color="#da3633"),
            hovertemplate="Peak: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
        ))
        fig_dur.update_yaxes(title="Peak-to-Trough Drawdown (%)", ticksuffix="%")
    chart_episodes_bar = serialize_figure(fig_dur, "Major Historical Drawdown Depths (%)", height=380)

    cols_ep = [
        "Max Drawdown", "Peak Date (MYT)", "Trough Date (MYT)", "Recovery Date (MYT)",
        "Decline (Sessions)", "Recovery (Sessions)", "Total Duration"
    ]
    display_ep = episodes_df[[c for c in cols_ep if c in episodes_df.columns]] if not episodes_df.empty else pd.DataFrame()

    return {
        "kpis": {
            "max_drawdown": f"{max_dd:.2f}%",
            "current_drawdown": f"{curr_dd:.2f}%",
            "avg_in_drawdown": f"{avg_dd:.2f}%",
            "total_sessions": f"{len(returns):,}",
        },
        "charts": {
            "underwater": chart_dd,
            "episodes_bar": chart_episodes_bar,
        },
        "tables": {
            "episodes": df_to_safe_records(display_ep),
        },
    }


# 8. Extreme Day Profiles
@app.get("/api/extreme-days")
@safe_api
async def get_extreme_days_data(
    instrument: str = Query(INDEX_TICKER),
    period: str = Query("10Y"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    percentile: float = Query(95.0),
    force_refresh: bool = Query(False),
):
    """
    Extreme Day Profiles:
    Tail event filtering, MFE vs MAE scatter paths, and top historical gainers, losers, and range days.
    """
    data = get_processed_data(ticker=instrument, force_refresh=force_refresh)
    df_filtered = filter_daily_data(data["daily"], period, start_date, end_date, "All Days", "All Regimes")
    df_clean = df_filtered.dropna(subset=["total_return_pct", "daily_range_pct"]).copy()

    if df_clean.empty:
        raise HTTPException(status_code=404, detail="No data available.")

    cutoff_abs_ret = float(np.percentile(df_clean["abs_total_return_pct"], percentile))
    extreme_subset = df_clean[df_clean["abs_total_return_pct"] >= cutoff_abs_ret]

    # 1. MFE vs MAE Scatter
    fig_mfe = plot_mfe_mae_scatter(df_clean)
    chart_mfe_mae = serialize_figure(fig_mfe, "Path Excursion Dynamics: MFE (Peak Upward) vs MAE (Deepest Downward)", height=450)

    # 2. Extreme Returns Bar Distribution
    fig_ext = go.Figure()
    fig_ext.add_trace(go.Histogram(
        x=extreme_subset["total_return_pct"],
        nbinsx=30,
        marker=dict(color="#f0883e", line=dict(color="#30363d", width=0.5)),
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
    ))
    fig_ext.update_xaxes(title="Daily Return (%)", ticksuffix="%")
    chart_ext_dist = serialize_figure(fig_ext, f"Extreme Day Return Distribution (Top {100 - percentile:.0f}% Tail Events)", height=380)

    # Top Gainers
    top_pos = df_clean.sort_values(by="total_return_pct", ascending=False).head(10).copy()
    top_pos["Date (MYT)"] = top_pos.index.map(lambda d: format_myt_timestamp(d, include_date=True))
    top_pos["Weekday"] = top_pos["weekday"]
    top_pos["Total Return"] = top_pos["total_return_pct"].apply(lambda x: f"+{x:.2f}%")
    top_pos["Overnight"] = top_pos["overnight_return_pct"].apply(lambda x: f"{x:+.2f}%")
    top_pos["Regular Session"] = top_pos["regular_return_pct"].apply(lambda x: f"{x:+.2f}%")
    top_pos["Daily Range"] = top_pos["daily_range_pct"].apply(lambda x: f"{x:.2f}%")

    # Top Losers
    top_neg = df_clean.sort_values(by="total_return_pct", ascending=True).head(10).copy()
    top_neg["Date (MYT)"] = top_neg.index.map(lambda d: format_myt_timestamp(d, include_date=True))
    top_neg["Weekday"] = top_neg["weekday"]
    top_neg["Total Return"] = top_neg["total_return_pct"].apply(lambda x: f"{x:.2f}%")
    top_neg["Overnight"] = top_neg["overnight_return_pct"].apply(lambda x: f"{x:+.2f}%")
    top_neg["Regular Session"] = top_neg["regular_return_pct"].apply(lambda x: f"{x:+.2f}%")
    top_neg["Daily Range"] = top_neg["daily_range_pct"].apply(lambda x: f"{x:.2f}%")

    # Top Range Days
    top_rng = df_clean.sort_values(by="daily_range_pct", ascending=False).head(10).copy()
    top_rng["Date (MYT)"] = top_rng.index.map(lambda d: format_myt_timestamp(d, include_date=True))
    top_rng["Daily Range"] = top_rng["daily_range_pct"].apply(lambda x: f"{x:.2f}%")
    top_rng["Total Return"] = top_rng["total_return_pct"].apply(lambda x: f"{x:+.2f}%")
    top_rng["MFE (Upward)"] = top_rng["mfe_pct"].apply(lambda x: f"+{x:.2f}%")
    top_rng["MAE (Downward)"] = top_rng["mae_pct"].apply(lambda x: f"{x:.2f}%")

    # Top Gaps
    top_gap = df_clean.reindex(df_clean["gap_pct"].abs().sort_values(ascending=False).index).head(10).copy()
    top_gap["Date (MYT)"] = top_gap.index.map(lambda d: format_myt_timestamp(d, include_date=True))
    top_gap["Opening Gap"] = top_gap["gap_pct"].apply(lambda x: f"{x:+.2f}%")
    top_gap["Gap Filled?"] = top_gap["gap_filled"].apply(lambda x: "Yes ✅" if x else "No ❌")
    top_gap["Total Return"] = top_gap["total_return_pct"].apply(lambda x: f"{x:+.2f}%")

    return {
        "cutoff_info": {
            "percentile": percentile,
            "cutoff_abs_return": f"{cutoff_abs_ret:.2f}%",
            "sessions_count": len(extreme_subset),
            "total_sessions": len(df_clean),
        },
        "charts": {
            "mfe_mae": chart_mfe_mae,
            "tail_distribution": chart_ext_dist,
        },
        "tables": {
            "top_gainers": df_to_safe_records(top_pos[["Date (MYT)", "Weekday", "Total Return", "Overnight", "Regular Session", "Daily Range"]]),
            "top_losers": df_to_safe_records(top_neg[["Date (MYT)", "Weekday", "Total Return", "Overnight", "Regular Session", "Daily Range"]]),
            "top_range": df_to_safe_records(top_rng[["Date (MYT)", "Daily Range", "Total Return", "MFE (Upward)", "MAE (Downward)"]]),
            "top_gaps": df_to_safe_records(top_gap[["Date (MYT)", "Opening Gap", "Gap Filled?", "Total Return"]]),
        },
    }


# 9. Cross-Market Correlations
@app.get("/api/correlations")
@safe_api
async def get_correlations_data(
    instrument: str = Query(INDEX_TICKER),
    benchmark: str = Query("^GSPC"),
    period: str = Query("10Y"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    force_refresh: bool = Query(False),
):
    """
    Cross-Market Correlations:
    Rolling correlations, beta regression co-movement, and regime-dependent correlation against benchmarks (QQQ, TQQQ, NDX, SPY, ^GSPC).
    """
    data = get_processed_data(ticker=instrument, force_refresh=force_refresh)
    df_filtered = filter_daily_data(data["daily"], period, start_date, end_date, "All Days", "All Regimes")

    if df_filtered.empty:
        raise HTTPException(status_code=404, detail="No primary instrument data available.")

    # Load benchmark data
    try:
        bm_raw = download_and_cache_data(ticker=benchmark, period="10y", interval="1d", force_refresh=force_refresh)
        bm_clean, _ = clean_market_data(bm_raw, interval="1d")
        bm_daily = compute_daily_returns_and_sessions(bm_clean)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load benchmark {benchmark}: {str(e)}")

    combined = pd.DataFrame({
        "inst_return": df_filtered["total_return_pct"],
        "bm_return": bm_daily["total_return_pct"],
        "daily_range_pct": df_filtered["daily_range_pct"],
    }).dropna()

    if len(combined) < 20:
        raise HTTPException(status_code=400, detail="Insufficient overlapping sessions to compute correlations.")

    corr_overall = float(combined["inst_return"].corr(combined["bm_return"]))
    cov_matrix = np.cov(combined["bm_return"], combined["inst_return"])
    beta = float(cov_matrix[0, 1] / cov_matrix[0, 0]) if cov_matrix[0, 0] > 0 else 1.0
    r_squared = float(corr_overall ** 2)

    # 1. Rolling 60d correlation chart
    combined["rolling_corr_60d"] = combined["inst_return"].rolling(60).corr(combined["bm_return"])
    combined["ndx_return"] = combined["inst_return"]
    combined["spx_return"] = combined["bm_return"]
    fig_roll = plot_correlation_rolling(combined)
    chart_rolling_corr = serialize_figure(fig_roll, f"Rolling 60-Session Return Correlation ({instrument} vs {benchmark})", height=420)

    # 2. Scatter & linear regression
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=combined["bm_return"],
        y=combined["inst_return"],
        mode="markers",
        marker=dict(size=5, color="rgba(88, 166, 255, 0.6)", line=dict(width=0.5, color="#30363d")),
        name="Daily Returns",
        hovertemplate=f"{benchmark}: %{{x:.2f}}%<br>{instrument}: %{{y:.2f}}%<extra></extra>",
    ))
    z = np.polyfit(combined["bm_return"], combined["inst_return"], 1)
    p = np.poly1d(z)
    x_grid = np.linspace(combined["bm_return"].min(), combined["bm_return"].max(), 50)
    fig_scatter.add_trace(go.Scatter(
        x=x_grid,
        y=p(x_grid),
        mode="lines",
        name=f"Fit (β={beta:.2f})",
        line=dict(color="#2ea043", width=2, dash="dash"),
    ))
    fig_scatter.update_xaxes(title=f"{benchmark} Daily Return (%)", ticksuffix="%")
    fig_scatter.update_yaxes(title=f"{instrument} Daily Return (%)", ticksuffix="%")
    chart_scatter = serialize_figure(fig_scatter, "Co-Movement Scatter & Empirical OLS Beta Line", height=400)

    # 3. Correlation across volatility regimes
    classified_df, _ = classify_volatility_regimes(combined, metric_col="daily_range_pct")
    reg_corrs = []
    for regime in ["Low Volatility", "Normal Volatility", "High Volatility"]:
        sub = classified_df[classified_df["vol_regime"] == regime]
        if len(sub) > 5:
            r_val = float(sub["inst_return"].corr(sub["bm_return"]))
            reg_corrs.append({
                "Volatility Regime": regime,
                "Sessions (n)": len(sub),
                "Return Correlation (r)": f"{r_val:.3f}",
                "_raw_corr": r_val,
            })
    reg_corr_df = pd.DataFrame(reg_corrs)

    fig_reg_bar = go.Figure()
    if not reg_corr_df.empty and "_raw_corr" in reg_corr_df.columns:
        fig_reg_bar.add_trace(go.Bar(
            x=reg_corr_df["Volatility Regime"],
            y=reg_corr_df["_raw_corr"],
            marker=dict(color=["#58a6ff", "#3b82f6", "#da3633"]),
            hovertemplate="Regime: %{x}<br>Correlation: %{y:.3f}<extra></extra>",
        ))
        fig_reg_bar.update_yaxes(title="Correlation (r)", range=[0.4, 1.0])
    chart_regime_bar = serialize_figure(fig_reg_bar, "Regime-Specific Return Correlation", height=400)

    return {
        "kpis": {
            "correlation": f"{corr_overall:.3f}",
            "beta": f"{beta:.2f}x",
            "r_squared": f"{r_squared:.3f}",
            "sessions_count": f"{len(combined):,}",
        },
        "charts": {
            "rolling_correlation": chart_rolling_corr,
            "scatter": chart_scatter,
            "regime_correlation": chart_regime_bar,
        },
        "tables": {
            "regime_summary": df_to_safe_records(reg_corr_df),
        },
    }


# -----------------------------------------------------------------------------
# Static Files & SPA Mounting
# -----------------------------------------------------------------------------
STATIC_DIR = Path(__file__).resolve().parent / "static"
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    """Serve the single-page frontend application."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_file)
