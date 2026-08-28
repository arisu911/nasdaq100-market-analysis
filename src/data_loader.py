"""
Data loading and caching module.
Fetches historical daily and intraday data for NASDAQ-100 (^NDX), QQQ, and S&P 500 (^GSPC)
via yfinance with robust local Parquet caching and metadata inspection.
"""
from typing import Dict, Any, Optional, Tuple
import datetime
from pathlib import Path
import pandas as pd
import yfinance as yf

from config.settings import (
    RAW_DATA_DIR,
    CACHE_DATA_DIR,
    PROCESSED_DATA_DIR,
    INDEX_TICKER,
    ETF_TICKER,
    BENCHMARK_INDEX,
    SOURCE_TZ,
    TARGET_TZ,
)
from src.timezone_utils import (
    convert_to_new_york,
    convert_to_myt,
    ensure_timezone_aware,
    format_myt_timestamp,
)


def _get_cache_path(ticker: str, interval: str, prefix: str = "raw") -> Path:
    """Generate safe filename for parquet caching."""
    clean_ticker = ticker.replace("^", "INDEX_")
    base_dir = RAW_DATA_DIR if prefix == "raw" else CACHE_DATA_DIR
    return base_dir / f"{clean_ticker}_{interval}.parquet"


def download_and_cache_data(
    ticker: str,
    period: str = "10y",
    interval: str = "1d",
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Download historical market data from yfinance and save to Parquet cache.
    Preserves timezone awareness (America/New_York) and cleans standard columns.
    """
    cache_path = _get_cache_path(ticker, interval, prefix="raw")

    # If cache exists and not forcing refresh, load from cache
    if cache_path.exists() and not force_refresh:
        try:
            df = pd.read_parquet(cache_path)
            if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                # Ensure index has proper timezone
                if df.index.tz is None:
                    df.index = df.index.tz_localize(SOURCE_TZ)
                return df
        except Exception:
            pass  # Fall back to redownloading on corrupt cache

    # yfinance download
    # For intraday 5m, yfinance supports max 60d; 1m supports 7d; 1h supports 730d (2y)
    yf_period = period
    if interval in ["1m", "2m", "5m", "15m", "30m"]:
        if interval == "1m":
            yf_period = "7d"
        elif interval in ["2m", "5m", "15m", "30m"]:
            yf_period = "60d"
    elif interval in ["60m", "1h"]:
        yf_period = "730d"

    yf_ticker = yf.Ticker(ticker)
    df = yf_ticker.history(period=yf_period, interval=interval, auto_adjust=False)

    if df.empty:
        # Retry with standard download
        df = yf.download(ticker, period=yf_period, interval=interval, progress=False, auto_adjust=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker} with interval {interval} and period {yf_period}")

    # Standardize columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep standard OHLCV columns
    cols_to_keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[cols_to_keep].copy()

    # Ensure index is timezone-aware in America/New_York
    if df.index.tz is None:
        df.index = df.index.tz_localize(SOURCE_TZ)
    else:
        df.index = df.index.tz_convert(SOURCE_TZ)

    # Save to raw cache
    df.to_parquet(cache_path)
    return df


def load_daily_data(
    ticker: str = INDEX_TICKER,
    period: str = "10y",
    force_refresh: bool = False
) -> pd.DataFrame:
    """Load daily historical data (up to 10+ years)."""
    return download_and_cache_data(ticker=ticker, period=period, interval="1d", force_refresh=force_refresh)


def load_intraday_data(
    ticker: str = INDEX_TICKER,
    interval: str = "5m",
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Load intraday historical data (e.g. 5m resolution, up to 60 calendar days).
    """
    return download_and_cache_data(ticker=ticker, period="60d", interval=interval, force_refresh=force_refresh)


def load_benchmark_data(
    ticker: str = BENCHMARK_INDEX,
    period: str = "10y",
    force_refresh: bool = False
) -> pd.DataFrame:
    """Load benchmark comparison data (e.g. S&P 500 ^GSPC)."""
    return download_and_cache_data(ticker=ticker, period=period, interval="1d", force_refresh=force_refresh)


def get_data_coverage_info(
    df: pd.DataFrame,
    interval: str = "1d",
    ticker: str = INDEX_TICKER
) -> Dict[str, Any]:
    """
    Inspect dataset coverage, resolution, date boundaries, and data limitations.
    All displayed dates and boundaries are formatted in MYT.
    """
    if df.empty:
        return {
            "ticker": ticker,
            "interval": interval,
            "total_bars": 0,
            "total_sessions": 0,
            "start_myt": "N/A",
            "end_myt": "N/A",
            "volume_available": False,
            "coverage_description": "No data available",
        }

    idx_myt = convert_to_myt(df.index)
    start_dt = idx_myt[0]
    end_dt = idx_myt[-1]

    # Calculate number of unique trading sessions (based on NY date)
    idx_ny = convert_to_new_york(df.index)
    unique_sessions = len(pd.Series(idx_ny.date).unique())

    has_volume = "Volume" in df.columns and (df["Volume"] > 0).sum() > 0
    zero_vol_pct = ((df["Volume"] == 0).sum() / len(df) * 100) if "Volume" in df.columns else 100.0

    if interval == "1d":
        limitation_text = "Daily history provided by Yahoo Finance (~10+ years). Full sample for long-term statistics."
    elif interval in ["5m", "10m", "15m", "30m"]:
        limitation_text = (
            f"Intraday {interval} history is restricted by free Yahoo Finance APIs to the most recent ~60 calendar days. "
            "No synthetic/fake data is used."
        )
    elif interval in ["60m", "1h"]:
        limitation_text = "Hourly intraday history is available for ~730 days (2 years) via Yahoo Finance."
    else:
        limitation_text = "Historical intraday resolution limited to available API depth."

    # Instrument notes
    if ticker == INDEX_TICKER:
        instrument_note = "NASDAQ-100 Index (^NDX): Direct index price, returns, volatility, gaps, and drawdowns."
    elif ticker == ETF_TICKER:
        instrument_note = "Invesco QQQ Trust (QQQ): Tradable ETF proxy for high-resolution microstructure & traded volume."
    else:
        instrument_note = f"Instrument: {ticker}"

    return {
        "ticker": ticker,
        "interval": interval,
        "total_bars": len(df),
        "total_sessions": unique_sessions,
        "start_myt": format_myt_timestamp(start_dt, include_date=True),
        "end_myt": format_myt_timestamp(end_dt, include_date=True),
        "start_date_myt": start_dt.strftime("%Y-%m-%d"),
        "end_date_myt": end_dt.strftime("%Y-%m-%d"),
        "has_volume": has_volume,
        "zero_vol_pct": zero_vol_pct,
        "instrument_note": instrument_note,
        "limitation_text": limitation_text,
    }
