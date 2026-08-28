"""
NASDAQ-100 Historical Market Behavior Analytics Dashboard.
Primary timezone: Asia/Kuala_Lumpur (MYT / UTC+8).
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

from config.settings import (
    INDEX_TICKER,
    ETF_TICKER,
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
    get_data_coverage_info,
)
from src.data_cleaner import clean_market_data
from src.calculations import compute_daily_returns_and_sessions
from src.regimes import classify_volatility_regimes

# Import Modular Page Renderers
from pages.overview import render_overview_page
from pages.intraday_analysis import render_intraday_page
from pages.weekday_analysis import render_weekday_page
from pages.opening_analysis import render_opening_page
from pages.overnight_analysis import render_overnight_page
from pages.volatility_analysis import render_volatility_page
from pages.extreme_days import render_extreme_days_page
from pages.drawdowns import render_drawdowns_page
from pages.correlations import render_correlations_page

# ---------------------------------------------------------
# Page Setup & Aesthetic Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="NASDAQ-100 Market Behavior Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"], 
        [data-testid="stSidebarNavItems"], 
        [data-testid="stSidebarNavSeparator"] {
            display: none !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /* Dark Quantitative Terminal Styling */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
    }
    .metric-card {
        background-color: #111827;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetricValue"] {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        color: #94a3b8 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        border-radius: 6px;
        color: #94a3b8;
        padding: 8px 16px;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border-color: #38bdf8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Cached Data Pipeline
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_processed_data(ticker: str, force_refresh: bool = False):
    """Download, cache, clean, and precalculate daily and intraday series."""
    # 1. Daily data (10Y)
    raw_daily = load_daily_data(ticker=ticker, period="10y", force_refresh=force_refresh)
    clean_daily, audit_daily = clean_market_data(raw_daily, interval="1d")
    calc_daily = compute_daily_returns_and_sessions(clean_daily)
    classified_daily, _ = classify_volatility_regimes(calc_daily)
    cov_daily = get_data_coverage_info(clean_daily, interval="1d", ticker=ticker)

    # 2. Intraday data (5m, max 60d)
    raw_intra = load_intraday_data(ticker=ticker, interval="5m", force_refresh=force_refresh)
    clean_intra, audit_intra = clean_market_data(raw_intra, interval="5m")
    cov_intra = get_data_coverage_info(clean_intra, interval="5m", ticker=ticker)

    return classified_daily, clean_intra, cov_daily, cov_intra, audit_daily, audit_intra


# ---------------------------------------------------------
# Sidebar Navigation & Filters
# ---------------------------------------------------------
st.sidebar.markdown("## 📊 NASDAQ-100 Analytics")
st.sidebar.caption("Institutional Market Behavior Research")

# Navigation Menu
page_selection = st.sidebar.radio(
    "Research Navigation:",
    [
        "1. Overview & Market Clock",
        "2. Intraday Analysis",
        "3. Weekday & Seasonality",
        "4. Opening & Gap Dynamics",
        "5. Overnight vs Regular Session",
        "6. Volatility & Regimes",
        "7. Extreme Days & Tail Events",
        "8. Drawdowns & Stress History",
        "9. S&P 500 Correlations",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Analytical Filters")

# Instrument Selector
instrument_choice = st.sidebar.selectbox(
    "Market Instrument:",
    options=[
        (INDEX_TICKER, "NASDAQ-100 Index (^NDX)"),
        (ETF_TICKER, "Invesco QQQ Trust (QQQ ETF)"),
    ],
    format_func=lambda x: x[1],
    index=0,
)[0]

# Analysis Period Filter
period_choice = st.sidebar.selectbox(
    "Historical Period:",
    options=["1Y", "3Y", "5Y", "10Y", "Custom"],
    index=3,
)

# Weekday Filter
weekday_filter = st.sidebar.selectbox(
    "Day of Week Filter:",
    options=["All Days", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    index=0,
)

# Volatility Regime Filter
regime_filter = st.sidebar.selectbox(
    "Volatility Regime Filter:",
    options=["All Regimes", "Low Volatility", "Normal Volatility", "High Volatility"],
    index=0,
)

force_refresh = st.sidebar.button("🔄 Force Refresh Data Cache")

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
with st.spinner("Loading and processing NASDAQ-100 historical datasets..."):
    df_daily_all, df_intra_all, cov_daily, cov_intra, audit_daily, audit_intra = get_processed_data(
        ticker=instrument_choice,
        force_refresh=force_refresh,
    )

# Filter by Period
now_ny = convert_to_new_york(datetime.now())
if period_choice == "1Y":
    start_dt = now_ny - timedelta(days=365)
    filtered_daily = df_daily_all[df_daily_all.index >= start_dt].copy()
elif period_choice == "3Y":
    start_dt = now_ny - timedelta(days=365 * 3)
    filtered_daily = df_daily_all[df_daily_all.index >= start_dt].copy()
elif period_choice == "5Y":
    start_dt = now_ny - timedelta(days=365 * 5)
    filtered_daily = df_daily_all[df_daily_all.index >= start_dt].copy()
elif period_choice == "10Y":
    start_dt = now_ny - timedelta(days=365 * 10)
    filtered_daily = df_daily_all[df_daily_all.index >= start_dt].copy()
else:  # Custom
    min_date = df_daily_all.index[0].date()
    max_date = df_daily_all.index[-1].date()
    c_start, c_end = st.sidebar.date_input("Select Custom Dates (MYT/ET):", value=(min_date, max_date))
    filtered_daily = df_daily_all[
        (df_daily_all.index.date >= c_start) & (df_daily_all.index.date <= c_end)
    ].copy()

# Filter by Weekday
if weekday_filter != "All Days":
    filtered_daily = filtered_daily[filtered_daily["weekday"] == weekday_filter].copy()

# Filter by Volatility Regime
if regime_filter != "All Regimes":
    filtered_daily = filtered_daily[filtered_daily["vol_regime"] == regime_filter].copy()

# Update dynamic coverage for filtered subset
cov_filtered = get_data_coverage_info(filtered_daily, interval="1d", ticker=instrument_choice)

# ---------------------------------------------------------
# Route Page Content
# ---------------------------------------------------------
if page_selection.startswith("1."):
    render_overview_page(filtered_daily, cov_filtered, audit_daily)
elif page_selection.startswith("2."):
    render_intraday_page(df_intra_all, cov_intra)
elif page_selection.startswith("3."):
    render_weekday_page(filtered_daily)
elif page_selection.startswith("4."):
    render_opening_page(filtered_daily, df_intra_all)
elif page_selection.startswith("5."):
    render_overnight_page(filtered_daily)
elif page_selection.startswith("6."):
    render_volatility_page(filtered_daily, df_intra_all)
elif page_selection.startswith("7."):
    render_extreme_days_page(filtered_daily)
elif page_selection.startswith("8."):
    render_drawdowns_page(filtered_daily)
elif page_selection.startswith("9."):
    render_correlations_page(filtered_daily)
