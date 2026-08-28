"""
Timezone utilities for converting between America/New_York (US Eastern Time)
and Asia/Kuala_Lumpur (Malaysia Time, MYT / UTC+8).

Strictly enforces timezone-aware datetime operations without manual offset hardcoding,
properly adapting to US Daylight Saving Time (EDT vs EST) changes.
"""
from datetime import datetime, date, time
from typing import Union, Tuple, Dict, Any
from zoneinfo import ZoneInfo
import pandas as pd

from config.settings import (
    SOURCE_TZ,
    TARGET_TZ,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
)

NY_TZ = ZoneInfo(SOURCE_TZ)
MYT_TZ = ZoneInfo(TARGET_TZ)


def ensure_timezone_aware(
    dt_input: Union[datetime, pd.Timestamp, pd.DatetimeIndex, pd.Series],
    default_tz: str = SOURCE_TZ
) -> Union[datetime, pd.Timestamp, pd.DatetimeIndex, pd.Series]:
    """
    Ensure input datetime object or series is timezone-aware.
    If naive, localize to default_tz. If already aware, keep as-is.
    """
    if isinstance(dt_input, (datetime, pd.Timestamp)):
        if dt_input.tzinfo is None:
            return dt_input.replace(tzinfo=ZoneInfo(default_tz))
        return dt_input
    elif isinstance(dt_input, pd.DatetimeIndex):
        if dt_input.tz is None:
            return dt_input.tz_localize(default_tz)
        return dt_input
    elif isinstance(dt_input, pd.Series):
        if pd.api.types.is_datetime64_any_dtype(dt_input):
            if dt_input.dt.tz is None:
                return dt_input.dt.tz_localize(default_tz)
            return dt_input
        return dt_input
    return dt_input


def convert_to_new_york(
    dt_input: Union[datetime, pd.Timestamp, pd.DatetimeIndex, pd.Series]
) -> Union[datetime, pd.Timestamp, pd.DatetimeIndex, pd.Series]:
    """
    Convert a timezone-aware datetime object, Index, or Series to America/New_York.
    """
    dt_aware = ensure_timezone_aware(dt_input, default_tz=SOURCE_TZ)
    if isinstance(dt_aware, (datetime, pd.Timestamp)):
        return dt_aware.astimezone(NY_TZ)
    elif isinstance(dt_aware, pd.DatetimeIndex):
        return dt_aware.tz_convert(SOURCE_TZ)
    elif isinstance(dt_aware, pd.Series):
        return dt_aware.dt.tz_convert(SOURCE_TZ)
    return dt_aware


def convert_to_myt(
    dt_input: Union[datetime, pd.Timestamp, pd.DatetimeIndex, pd.Series]
) -> Union[datetime, pd.Timestamp, pd.DatetimeIndex, pd.Series]:
    """
    Convert a timezone-aware datetime object, Index, or Series to Asia/Kuala_Lumpur (MYT).
    """
    dt_aware = ensure_timezone_aware(dt_input, default_tz=SOURCE_TZ)
    if isinstance(dt_aware, (datetime, pd.Timestamp)):
        return dt_aware.astimezone(MYT_TZ)
    elif isinstance(dt_aware, pd.DatetimeIndex):
        return dt_aware.tz_convert(TARGET_TZ)
    elif isinstance(dt_aware, pd.Series):
        return dt_aware.dt.tz_convert(TARGET_TZ)
    return dt_aware


def is_us_dst(dt_val: Union[datetime, date, pd.Timestamp, str]) -> bool:
    """
    Check if a given date or datetime falls within US Daylight Saving Time (EDT).
    Returns True for EDT (UTC-4) and False for EST (UTC-5).
    """
    if isinstance(dt_val, str):
        dt_val = pd.to_datetime(dt_val)
    if isinstance(dt_val, (datetime, pd.Timestamp)):
        d = dt_val.date() if hasattr(dt_val, "date") else dt_val
    else:
        d = dt_val

    # Noon in New York on that date to inspect DST status
    test_dt = datetime(d.year, d.month, d.day, 12, 0, 0, tzinfo=NY_TZ)
    # dst() returns non-zero timedelta if DST is in effect
    dst_offset = test_dt.dst()
    return dst_offset is not None and dst_offset.total_seconds() > 0


def get_myt_session_times(
    calendar_date: Union[date, datetime, pd.Timestamp, str]
) -> Dict[str, Any]:
    """
    Given a calendar date, calculate the exact NASDAQ regular trading session
    (09:30–16:00 America/New_York) converted dynamically to Asia/Kuala_Lumpur (MYT).

    Returns a dictionary with:
    - ny_open: datetime in America/New_York
    - ny_close: datetime in America/New_York
    - myt_open: datetime in Asia/Kuala_Lumpur
    - myt_close: datetime in Asia/Kuala_Lumpur
    - is_dst: bool (True = EDT, False = EST)
    - ny_session_label: str (e.g. "09:30 – 16:00 EDT" or "09:30 – 16:00 EST")
    - myt_session_label: str (e.g. "21:30 – 04:00+1 MYT" or "22:30 – 05:00+1 MYT")
    - myt_open_str: str (e.g. "21:30 MYT" or "22:30 MYT")
    - myt_close_str: str (e.g. "04:00 MYT" or "05:00 MYT")
    """
    if isinstance(calendar_date, str):
        calendar_date = pd.to_datetime(calendar_date).date()
    elif isinstance(calendar_date, (datetime, pd.Timestamp)):
        calendar_date = calendar_date.date()

    # Define 09:30 and 16:00 in America/New_York
    ny_open = datetime(
        calendar_date.year,
        calendar_date.month,
        calendar_date.day,
        MARKET_OPEN_HOUR,
        MARKET_OPEN_MINUTE,
        0,
        tzinfo=NY_TZ,
    )
    ny_close = datetime(
        calendar_date.year,
        calendar_date.month,
        calendar_date.day,
        MARKET_CLOSE_HOUR,
        MARKET_CLOSE_MINUTE,
        0,
        tzinfo=NY_TZ,
    )

    myt_open = ny_open.astimezone(MYT_TZ)
    myt_close = ny_close.astimezone(MYT_TZ)

    dst_active = is_us_dst(calendar_date)
    tz_abbr = "EDT" if dst_active else "EST"

    myt_open_str = myt_open.strftime("%H:%M MYT")
    myt_close_str = myt_close.strftime("%H:%M MYT")
    next_day_marker = "+1d" if myt_close.date() > myt_open.date() else ""

    myt_session_label = f"{myt_open.strftime('%H:%M')} – {myt_close.strftime('%H:%M')}{next_day_marker} MYT"
    ny_session_label = f"09:30 – 16:00 {tz_abbr}"

    return {
        "date": calendar_date,
        "ny_open": ny_open,
        "ny_close": ny_close,
        "myt_open": myt_open,
        "myt_close": myt_close,
        "is_dst": dst_active,
        "tz_abbr": tz_abbr,
        "ny_session_label": ny_session_label,
        "myt_session_label": myt_session_label,
        "myt_open_str": myt_open_str,
        "myt_close_str": myt_close_str,
    }


def format_myt_timestamp(
    dt_val: Union[datetime, pd.Timestamp],
    include_date: bool = True,
    include_seconds: bool = False
) -> str:
    """Format a timestamp in MYT format for dashboard display."""
    if dt_val is None or pd.isna(dt_val):
        return "N/A"
    myt_dt = convert_to_myt(dt_val)
    time_fmt = "%H:%M:%S" if include_seconds else "%H:%M"
    if include_date:
        return myt_dt.strftime(f"%Y-%m-%d {time_fmt} MYT")
    return myt_dt.strftime(f"{time_fmt} MYT")


def get_current_market_clock_info() -> Dict[str, Any]:
    """
    Get live market session status based on the current real-time clock.
    Calculates current time in both MYT and ET, session status, and session hours.
    """
    now_utc = datetime.now(ZoneInfo("UTC"))
    now_ny = now_utc.astimezone(NY_TZ)
    now_myt = now_utc.astimezone(MYT_TZ)

    today_session = get_myt_session_times(now_ny.date())
    is_weekday = now_ny.weekday() < 5

    # Check if currently inside regular trading hours (09:30–16:00 NY time on a weekday)
    is_regular_open = (
        is_weekday and (today_session["ny_open"] <= now_ny <= today_session["ny_close"])
    )
    is_pre_market = (
        is_weekday and (now_ny < today_session["ny_open"]) and (now_ny.hour >= 4)
    )
    is_post_market = (
        is_weekday and (now_ny > today_session["ny_close"]) and (now_ny.hour < 20)
    )

    if is_regular_open:
        status_text = "REGULAR SESSION OPEN"
        status_color = "#10b981"  # Emerald green
    elif is_pre_market:
        status_text = "PRE-MARKET"
        status_color = "#f59e0b"  # Amber
    elif is_post_market:
        status_text = "AFTER-HOURS"
        status_color = "#3b82f6"  # Blue
    elif not is_weekday:
        status_text = "WEEKEND CLOSED"
        status_color = "#64748b"  # Slate
    else:
        status_text = "MARKET CLOSED"
        status_color = "#ef4444"  # Red

    return {
        "now_myt": now_myt,
        "now_ny": now_ny,
        "status_text": status_text,
        "status_color": status_color,
        "is_regular_open": is_regular_open,
        "today_session": today_session,
        "dst_active": today_session["is_dst"],
        "tz_abbr": today_session["tz_abbr"],
    }
