"""
Unit tests for market session identification, opening windows, and resampling.
"""
from datetime import datetime
import pandas as pd
import numpy as np
import pytest

from src.market_sessions import (
    tag_session_phase,
    filter_regular_session,
    tag_opening_windows,
    resample_intraday,
)


def _create_synthetic_day(ny_date_str="2024-06-03"):
    """Generate 1-day 5-minute synthetic OHLCV data spanning 04:00 to 20:00 ET."""
    times = pd.date_range(f"{ny_date_str} 04:00", f"{ny_date_str} 20:00", freq="5min", tz="America/New_York")
    n = len(times)
    np.random.seed(42)
    prices = 18000 + np.cumsum(np.random.randn(n) * 5)

    df = pd.DataFrame({
        "Open": prices,
        "High": prices + 2,
        "Low": prices - 2,
        "Close": prices + 1,
        "Volume": np.random.randint(1000, 50000, n),
    }, index=times)
    return df


def test_session_phase_tagging():
    """Verify bars are correctly tagged as Pre-Market, Regular Session, Post-Market."""
    df = _create_synthetic_day("2024-06-03")
    tagged = tag_session_phase(df)

    # 04:00 NY -> Pre-Market
    assert tagged.loc["2024-06-03 04:00:00-04:00", "session_phase"] == "Pre-Market"
    assert not tagged.loc["2024-06-03 04:00:00-04:00", "is_regular_session"]

    # 09:30 NY -> Regular Session Open
    assert tagged.loc["2024-06-03 09:30:00-04:00", "session_phase"] == "Regular Session"
    assert tagged.loc["2024-06-03 09:30:00-04:00", "is_regular_session"]
    assert tagged.loc["2024-06-03 09:30:00-04:00", "minutes_from_open"] == 0

    # 15:55 NY -> Regular Session
    assert tagged.loc["2024-06-03 15:55:00-04:00", "session_phase"] == "Regular Session"
    assert tagged.loc["2024-06-03 15:55:00-04:00", "is_regular_session"]

    # 16:00 NY -> Regular Session Close
    assert tagged.loc["2024-06-03 16:00:00-04:00", "session_phase"] == "Regular Session"

    # 16:05 NY -> Post-Market
    assert tagged.loc["2024-06-03 16:05:00-04:00", "session_phase"] == "Post-Market"
    assert not tagged.loc["2024-06-03 16:05:00-04:00", "is_regular_session"]


def test_opening_windows_tagging():
    """Verify first 5m, 15m, 30m, 60m flags."""
    df = _create_synthetic_day("2024-06-03")
    tagged = tag_opening_windows(df)

    # 09:30 is in 5m, 15m, 30m, 60m
    bar_930 = tagged.loc["2024-06-03 09:30:00-04:00"]
    assert bar_930["is_first_5m"]
    assert bar_930["is_first_15m"]
    assert bar_930["is_first_30m"]
    assert bar_930["is_first_60m"]

    # 09:40 is NOT in 5m, IS in 15m, 30m, 60m
    bar_940 = tagged.loc["2024-06-03 09:40:00-04:00"]
    assert not bar_940["is_first_5m"]
    assert bar_940["is_first_15m"]
    assert bar_940["is_first_30m"]
    assert bar_940["is_first_60m"]

    # 10:15 is ONLY in 60m
    bar_1015 = tagged.loc["2024-06-03 10:15:00-04:00"]
    assert not bar_1015["is_first_5m"]
    assert not bar_1015["is_first_15m"]
    assert not bar_1015["is_first_30m"]
    assert bar_1015["is_first_60m"]

    # 11:00 is in NONE
    bar_1100 = tagged.loc["2024-06-03 11:00:00-04:00"]
    assert not bar_1100["is_first_60m"]


def test_resample_intraday():
    """Test resampling from 5m to 15m and 60m."""
    df = _create_synthetic_day("2024-06-03")
    res_15m = resample_intraday(df, interval_str="15m")
    assert not res_15m.empty
    # Regular session 09:30 to 16:00 is 390 minutes -> 26 15m bars + 1 at 16:00
    assert len(res_15m) >= 26
