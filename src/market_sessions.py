"""
Market session segmentation and boundary identification.
Defines US regular trading session (09:30–16:00 ET), opening windows (5m, 15m, 30m, 60m),
overnight periods, and resampling logic with dynamic MYT representation.
"""
from typing import List, Optional
import pandas as pd
import numpy as np

from config.settings import (
    SOURCE_TZ,
    TARGET_TZ,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
)
from src.timezone_utils import (
    convert_to_new_york,
    convert_to_myt,
    ensure_timezone_aware,
)


def tag_session_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each intraday bar with its market session phase:
    - 'Regular Session': 09:30 - 16:00 America/New_York
    - 'Pre-Market': 04:00 - 09:29 America/New_York
    - 'Post-Market': 16:01 - 20:00 America/New_York
    - 'Overnight': Outside 09:30-16:00
    Adds columns:
    - 'session_phase': categorical phase
    - 'is_regular_session': bool
    - 'session_date': date in America/New_York
    - 'timestamp_ny': timestamp in America/New_York
    - 'timestamp_myt': timestamp in Asia/Kuala_Lumpur
    - 'time_myt_str': HH:MM in MYT
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pd.DatetimeIndex")

    # Ensure timezone aware in New York
    idx_ny = convert_to_new_york(df.index)
    idx_myt = convert_to_myt(df.index)

    df["timestamp_ny"] = idx_ny
    df["timestamp_myt"] = idx_myt
    df["session_date"] = idx_ny.date
    df["time_myt_str"] = idx_myt.strftime("%H:%M")
    df["time_ny_str"] = idx_ny.strftime("%H:%M")

    # Calculate minute of day in NY time
    ny_minutes = idx_ny.hour * 60 + idx_ny.minute
    open_min = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MINUTE  # 570
    close_min = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE  # 960

    # Regular session: >= 09:30 and <= 16:00
    is_reg = (ny_minutes >= open_min) & (ny_minutes <= close_min)
    is_pre = (ny_minutes >= 240) & (ny_minutes < open_min)
    is_post = (ny_minutes > close_min) & (ny_minutes <= 1200)

    phases = np.where(
        is_reg, "Regular Session",
        np.where(is_pre, "Pre-Market", np.where(is_post, "Post-Market", "Overnight"))
    )

    df["session_phase"] = phases
    df["is_regular_session"] = is_reg
    # Minute offset from regular open (0 at 09:30 NY)
    df["minutes_from_open"] = np.where(is_reg, ny_minutes - open_min, np.nan)

    return df


def filter_regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only include regular trading hours (09:30–16:00 ET)."""
    tagged = tag_session_phase(df) if "is_regular_session" not in df.columns else df
    return tagged[tagged["is_regular_session"]].copy()


def tag_opening_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag bars falling into opening analysis intervals:
    - 'first_5m': 09:30 to 09:35 ET
    - 'first_15m': 09:30 to 09:45 ET
    - 'first_30m': 09:30 to 10:00 ET
    - 'first_60m': 09:30 to 10:30 ET
    """
    df = tag_session_phase(df) if "minutes_from_open" not in df.columns else df.copy()
    m = df["minutes_from_open"]

    df["is_first_5m"] = df["is_regular_session"] & (m >= 0) & (m < 5)
    df["is_first_15m"] = df["is_regular_session"] & (m >= 0) & (m < 15)
    df["is_first_30m"] = df["is_regular_session"] & (m >= 0) & (m < 30)
    df["is_first_60m"] = df["is_regular_session"] & (m >= 0) & (m < 60)

    return df


def resample_intraday(
    df: pd.DataFrame,
    interval_str: str = "15m"
) -> pd.DataFrame:
    """
    Resample intraday data (e.g. 5m) to higher timeframes (10m, 15m, 30m, 60m/1h)
    aligned strictly per session date from market open.
    """
    if interval_str == "5m":
        return tag_session_phase(df)

    resample_rules = {
        "10m": "10min",
        "15m": "15min",
        "30m": "30min",
        "60m": "60min",
        "1h": "60min",
    }
    rule = resample_rules.get(interval_str, interval_str)

    # Filter to regular session first for clean interval boundaries
    reg_df = filter_regular_session(df)
    if reg_df.empty:
        return reg_df

    # Group by session_date and resample OHLCV
    resampled_list = []
    for s_date, group in reg_df.groupby("session_date"):
        # Resample within this trading session
        agg_dict = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
        # Add any other numeric columns if present
        for col in group.columns:
            if col not in agg_dict and col not in [
                "timestamp_ny", "timestamp_myt", "session_date", "time_myt_str",
                "time_ny_str", "session_phase", "is_regular_session", "minutes_from_open"
            ]:
                if pd.api.types.is_numeric_dtype(group[col]):
                    agg_dict[col] = "mean"

        bars = group.resample(rule, closed="left", label="left").agg(agg_dict).dropna(subset=["Open", "Close"])
        resampled_list.append(bars)

    if not resampled_list:
        return pd.DataFrame()

    out_df = pd.concat(resampled_list)
    out_df = tag_session_phase(out_df)
    return out_df
