"""
Unit tests for timezone conversion between America/New_York and Asia/Kuala_Lumpur (MYT).
Validates US Standard Time (EST), US Daylight Time (EDT), DST transition dates,
and timezone-aware datetime operations.
"""
from datetime import datetime, date
from zoneinfo import ZoneInfo
import pandas as pd
import pytest

from src.timezone_utils import (
    convert_to_new_york,
    convert_to_myt,
    is_us_dst,
    get_myt_session_times,
    format_myt_timestamp,
    ensure_timezone_aware,
)


def test_us_standard_time_est_to_myt():
    """
    US Standard Time (EST = UTC-5):
    - Market Open 09:30 ET -> 22:30 MYT (UTC+8, offset diff = +13 hours)
    - Market Close 16:00 ET -> 05:00 (+1 day) MYT
    """
    est_date = date(2024, 1, 15)  # January is US Standard Time
    session = get_myt_session_times(est_date)

    assert session["is_dst"] is False
    assert session["tz_abbr"] == "EST"
    assert session["myt_open"].hour == 22
    assert session["myt_open"].minute == 30
    assert session["myt_close"].hour == 5
    assert session["myt_close"].minute == 0
    assert session["myt_open_str"] == "22:30 MYT"
    assert session["myt_close_str"] == "05:00 MYT"
    assert "22:30 – 05:00+1d MYT" in session["myt_session_label"]


def test_us_daylight_time_edt_to_myt():
    """
    US Daylight Time (EDT = UTC-4):
    - Market Open 09:30 ET -> 21:30 MYT (UTC+8, offset diff = +12 hours)
    - Market Close 16:00 ET -> 04:00 (+1 day) MYT
    """
    edt_date = date(2024, 6, 15)  # June is US Daylight Time
    session = get_myt_session_times(edt_date)

    assert session["is_dst"] is True
    assert session["tz_abbr"] == "EDT"
    assert session["myt_open"].hour == 21
    assert session["myt_open"].minute == 30
    assert session["myt_close"].hour == 4
    assert session["myt_close"].minute == 0
    assert session["myt_open_str"] == "21:30 MYT"
    assert session["myt_close_str"] == "04:00 MYT"
    assert "21:30 – 04:00+1d MYT" in session["myt_session_label"]


def test_dst_transition_boundaries():
    """
    Verify behavior on and around US DST transition dates in 2024:
    - DST starts: Sunday, March 10, 2024
    - DST ends: Sunday, November 3, 2024
    """
    # March 8, 2024 (Friday before switch) -> EST
    session_pre_spring = get_myt_session_times(date(2024, 3, 8))
    assert session_pre_spring["is_dst"] is False
    assert session_pre_spring["myt_open"].hour == 22

    # March 11, 2024 (Monday after switch) -> EDT
    session_post_spring = get_myt_session_times(date(2024, 3, 11))
    assert session_post_spring["is_dst"] is True
    assert session_post_spring["myt_open"].hour == 21

    # November 1, 2024 (Friday before switch) -> EDT
    session_pre_fall = get_myt_session_times(date(2024, 11, 1))
    assert session_pre_fall["is_dst"] is True
    assert session_pre_fall["myt_open"].hour == 21

    # November 4, 2024 (Monday after switch) -> EST
    session_post_fall = get_myt_session_times(date(2024, 11, 4))
    assert session_post_fall["is_dst"] is False
    assert session_post_fall["myt_open"].hour == 22


def test_series_timezone_conversion():
    """Test pandas DatetimeIndex and Series conversion to MYT."""
    dt_index = pd.date_range("2024-06-03 09:30", periods=5, freq="5min", tz="America/New_York")
    s = pd.Series(range(5), index=dt_index)

    myt_idx = convert_to_myt(dt_index)
    assert str(myt_idx.tz) == "Asia/Kuala_Lumpur"
    assert myt_idx[0].hour == 21
    assert myt_idx[0].minute == 30

    ny_idx = convert_to_new_york(myt_idx)
    assert str(ny_idx.tz) == "America/New_York"
    assert ny_idx[0].hour == 9
    assert ny_idx[0].minute == 30


def test_ensure_timezone_aware():
    """Verify naive timestamp localization."""
    naive_dt = datetime(2024, 6, 3, 9, 30)
    aware_dt = ensure_timezone_aware(naive_dt, default_tz="America/New_York")
    assert aware_dt.tzinfo is not None
    assert str(aware_dt.tzinfo) == "America/New_York"
