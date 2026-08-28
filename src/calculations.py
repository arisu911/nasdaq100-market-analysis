"""
Core financial and statistical calculations for NASDAQ-100 analytics:
- Daily, overnight, and regular session returns
- Opening gap % and gap-fill detection
- MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion)
- High-Low Range, True Range, Parkinson and Garman-Klass Volatilities
- Drawdown series, Maximum Drawdown, and Drawdown Episode durations
- Rolling return, volatility, and range statistics
"""
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np

from src.timezone_utils import (
    convert_to_new_york,
    convert_to_myt,
    format_myt_timestamp,
)


def compute_daily_returns_and_sessions(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily close-to-close, overnight, and regular-session metrics on daily OHLCV data.

    Outputs dataframe indexed by date with:
    - Open, High, Low, Close, Volume
    - prev_close: Previous regular session close
    - total_return: (Close - prev_close) / prev_close
    - total_return_pct: total_return * 100
    - overnight_return: (Open - prev_close) / prev_close
    - overnight_return_pct: overnight_return * 100
    - regular_return: (Close - Open) / Open
    - regular_return_pct: regular_return * 100
    - abs_total_return_pct: abs(total_return_pct)
    - abs_overnight_return_pct: abs(overnight_return_pct)
    - abs_regular_return_pct: abs(regular_return_pct)
    - gap_pct: (Open - prev_close) / prev_close * 100
    - gap_filled: bool (Low <= prev_close for up gaps; High >= prev_close for down gaps)
    - daily_range: High - Low
    - daily_range_pct: (High - Low) / Open * 100
    - mfe_pct: (High - Open) / Open * 100 (Maximum Favorable Excursion)
    - mae_pct: (Low - Open) / Open * 100 (Maximum Adverse Excursion)
    - weekday: Monday..Friday
    - weekday_num: 0..4
    - month: 1..12
    - year: int
    """
    df = df_daily.copy().sort_index()

    # Shift Close for previous close
    df["prev_close"] = df["Close"].shift(1)

    # Returns
    df["total_return"] = (df["Close"] - df["prev_close"]) / df["prev_close"]
    df["total_return_pct"] = df["total_return"] * 100.0

    df["overnight_return"] = (df["Open"] - df["prev_close"]) / df["prev_close"]
    df["overnight_return_pct"] = df["overnight_return"] * 100.0

    df["regular_return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["regular_return_pct"] = df["regular_return"] * 100.0

    df["abs_total_return_pct"] = df["total_return_pct"].abs()
    df["abs_overnight_return_pct"] = df["overnight_return_pct"].abs()
    df["abs_regular_return_pct"] = df["regular_return_pct"].abs()

    # Opening Gap
    df["gap_pct"] = df["overnight_return_pct"]
    df["gap_direction"] = np.where(df["gap_pct"] > 0, "Up", np.where(df["gap_pct"] < 0, "Down", "Flat"))

    # Gap fill logic:
    # Up gap (Open > prev_close): filled if Low <= prev_close
    # Down gap (Open < prev_close): filled if High >= prev_close
    # Flat gap (Open == prev_close): filled = True
    up_filled = (df["gap_pct"] > 0) & (df["Low"] <= df["prev_close"])
    down_filled = (df["gap_pct"] < 0) & (df["High"] >= df["prev_close"])
    flat_filled = (df["gap_pct"] == 0)
    df["gap_filled"] = up_filled | down_filled | flat_filled

    # Range
    df["daily_range"] = df["High"] - df["Low"]
    df["daily_range_pct"] = (df["High"] - df["Low"]) / df["Open"] * 100.0

    # Path statistics (MFE and MAE)
    df["mfe_pct"] = (df["High"] - df["Open"]) / df["Open"] * 100.0
    df["mae_pct"] = (df["Low"] - df["Open"]) / df["Open"] * 100.0

    # Calendar metadata (based on New York trading day)
    idx_ny = convert_to_new_york(df.index)
    df["weekday_num"] = idx_ny.weekday
    df["weekday"] = idx_ny.strftime("%A")
    df["month"] = idx_ny.month
    df["month_name"] = idx_ny.strftime("%B")
    df["year"] = idx_ny.year

    return df


def compute_cumulative_performance(df_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative equity curves for:
    - Full Market (Total close-to-close)
    - Overnight Only
    - Regular Session Only
    """
    df = df_returns.copy().dropna(subset=["total_return", "overnight_return", "regular_return"])

    df["cum_total"] = (1.0 + df["total_return"]).cumprod() - 1.0
    df["cum_overnight"] = (1.0 + df["overnight_return"]).cumprod() - 1.0
    df["cum_regular"] = (1.0 + df["regular_return"]).cumprod() - 1.0

    return df


def compute_drawdown_series(returns_series: pd.Series) -> Tuple[pd.Series, pd.Series, float]:
    """
    Calculate wealth index, rolling high water mark, drawdown series, and maximum drawdown.
    """
    clean_ret = returns_series.dropna()
    wealth_index = (1.0 + clean_ret).cumprod()
    high_water_mark = wealth_index.cummax()
    drawdown_series = (wealth_index - high_water_mark) / high_water_mark * 100.0
    max_dd = float(drawdown_series.min()) if not drawdown_series.empty else 0.0

    return wealth_index, drawdown_series, max_dd


def compute_drawdown_episodes(
    drawdown_series: pd.Series,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Identify and extract top drawdown episodes (trough depth, peak date, trough date, recovery date, duration).
    """
    if drawdown_series.empty:
        return pd.DataFrame()

    episodes = []
    in_dd = False
    peak_date = None
    trough_date = None
    trough_val = 0.0
    trough_idx = 0
    start_idx = 0

    idx = drawdown_series.index

    for i, (dt, val) in enumerate(drawdown_series.items()):
        if val < 0:
            if not in_dd:
                in_dd = True
                start_idx = i - 1 if i > 0 else 0
                peak_date = idx[start_idx]
                trough_date = dt
                trough_val = val
                trough_idx = i
            else:
                if val < trough_val:
                    trough_val = val
                    trough_date = dt
                    trough_idx = i
        else:
            if in_dd:
                # Recovered
                recovery_date = dt
                duration_to_trough = trough_idx - start_idx
                recovery_duration = i - trough_idx
                total_duration = i - start_idx
                episodes.append({
                    "peak_date": peak_date,
                    "trough_date": trough_date,
                    "recovery_date": recovery_date,
                    "max_drawdown_pct": trough_val,
                    "days_to_trough": duration_to_trough,
                    "recovery_days": recovery_duration,
                    "total_days": total_duration,
                    "is_recovered": True,
                })
                in_dd = False

    # If still in drawdown at end of series
    if in_dd:
        episodes.append({
            "peak_date": peak_date,
            "trough_date": trough_date,
            "recovery_date": None,
            "max_drawdown_pct": trough_val,
            "days_to_trough": trough_idx - start_idx,
            "recovery_days": np.nan,
            "total_days": len(drawdown_series) - start_idx,
            "is_recovered": False,
        })

    if not episodes:
        return pd.DataFrame()

    df_ep = pd.DataFrame(episodes)
    df_ep = df_ep.sort_values(by="max_drawdown_pct", ascending=True).head(top_n).reset_index(drop=True)

    # Format dates in MYT for display
    df_ep["Peak Date (MYT)"] = df_ep["peak_date"].apply(lambda d: format_myt_timestamp(d, include_date=True))
    df_ep["Trough Date (MYT)"] = df_ep["trough_date"].apply(lambda d: format_myt_timestamp(d, include_date=True))
    df_ep["Recovery Date (MYT)"] = df_ep["recovery_date"].apply(
        lambda d: format_myt_timestamp(d, include_date=True) if pd.notna(d) else "Active / In Recovery"
    )
    df_ep["Max Drawdown"] = df_ep["max_drawdown_pct"].apply(lambda x: f"{x:.2f}%")
    df_ep["Decline (Sessions)"] = df_ep["days_to_trough"].astype(int)
    df_ep["Recovery (Sessions)"] = df_ep["recovery_days"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "—")
    df_ep["Total Duration"] = df_ep["total_days"].astype(int)

    return df_ep


def compute_rolling_metrics(
    df_daily: pd.DataFrame,
    windows: List[int] = [20, 60, 120]
) -> pd.DataFrame:
    """
    Compute rolling return, rolling annualized volatility, and rolling range.
    """
    df = df_daily.copy()
    for w in windows:
        # Annualized Volatility: rolling std * sqrt(252) * 100
        df[f"rolling_vol_{w}d"] = df["total_return"].rolling(w).std() * np.sqrt(252) * 100.0
        # Rolling Mean Return
        df[f"rolling_ret_{w}d"] = df["total_return_pct"].rolling(w).mean()
        # Rolling Average Daily Range %
        df[f"rolling_range_{w}d"] = df["daily_range_pct"].rolling(w).mean()

    return df


def compute_parkinson_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Compute Parkinson High-Low Realized Volatility:
    sigma = sqrt( (1 / (4 * ln(2))) * sum(ln(High/Low)^2) / N ) * sqrt(252) * 100
    """
    hl_ratio = np.log(df["High"] / df["Low"]) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    rolling_val = hl_ratio.rolling(window).mean()
    parkinson = np.sqrt(factor * rolling_val) * np.sqrt(252) * 100.0
    return parkinson


def compute_opening_window_metrics(
    df_intraday_5m: pd.DataFrame,
    df_daily: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate opening performance for 5m, 15m, 30m, 60m windows for each trading session.
    Compares opening range against full day high-low range.
    """
    from src.market_sessions import filter_regular_session

    reg_bars = filter_regular_session(df_intraday_5m)
    if reg_bars.empty:
        return pd.DataFrame()

    results = []
    for s_date, day_bars in reg_bars.groupby("session_date"):
        day_bars = day_bars.sort_index()
        day_open = day_bars["Open"].iloc[0]
        day_high = day_bars["High"].max()
        day_low = day_bars["Low"].min()
        day_close = day_bars["Close"].iloc[-1]
        day_range = day_high - day_low

        row = {
            "session_date": s_date,
            "day_open": day_open,
            "day_high": day_high,
            "day_low": day_low,
            "day_close": day_close,
            "day_range": day_range,
            "day_range_pct": (day_range / day_open) * 100.0 if day_open > 0 else 0.0,
        }

        # Check windows
        for mins in [5, 15, 30, 60]:
            # Number of 5m bars = mins // 5
            bars_count = max(1, mins // 5)
            w_bars = day_bars.iloc[:bars_count]
            if len(w_bars) > 0:
                w_close = w_bars["Close"].iloc[-1]
                w_high = w_bars["High"].max()
                w_low = w_bars["Low"].min()
                w_range = w_high - w_low
                w_ret = (w_close - day_open) / day_open * 100.0
                w_range_ratio = (w_range / day_range * 100.0) if day_range > 0 else 0.0

                row[f"ret_{mins}m_pct"] = w_ret
                row[f"range_{mins}m"] = w_range
                row[f"range_{mins}m_pct"] = (w_range / day_open) * 100.0
                row[f"range_ratio_{mins}m_pct"] = w_range_ratio
            else:
                row[f"ret_{mins}m_pct"] = np.nan
                row[f"range_{mins}m"] = np.nan
                row[f"range_{mins}m_pct"] = np.nan
                row[f"range_ratio_{mins}m_pct"] = np.nan

        results.append(row)

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df["session_date"] = pd.to_datetime(res_df["session_date"])
        res_df = res_df.set_index("session_date")
    return res_df


def compute_gap_fill_intraday(
    df_intraday_5m: pd.DataFrame,
    df_daily: pd.DataFrame
) -> pd.DataFrame:
    """
    For days with intraday data, determine whether the opening gap was filled
    and the exact time (in minutes from market open 09:30 ET) when gap filled.
    """
    from src.market_sessions import filter_regular_session

    reg_bars = filter_regular_session(df_intraday_5m)
    if reg_bars.empty:
        return pd.DataFrame()

    results = []
    # Map prev_close from daily
    prev_close_map = df_daily["prev_close"].to_dict()

    for s_date, day_bars in reg_bars.groupby("session_date"):
        # Match session date
        dt_key = pd.to_datetime(s_date).date()
        # Find prev close
        matching_prev = None
        for k, v in prev_close_map.items():
            if hasattr(k, "date") and k.date() == dt_key:
                matching_prev = v
                break

        if matching_prev is None or pd.isna(matching_prev):
            continue

        day_bars = day_bars.sort_index()
        day_open = day_bars["Open"].iloc[0]
        gap_pct = (day_open - matching_prev) / matching_prev * 100.0

        if abs(gap_pct) < 0.01:
            # Flat gap
            continue

        is_up_gap = gap_pct > 0
        gap_filled = False
        minutes_to_fill = np.nan
        fill_time_myt = None

        for idx, bar in day_bars.iterrows():
            m_open = bar.get("minutes_from_open", 0)
            if is_up_gap and bar["Low"] <= matching_prev:
                gap_filled = True
                minutes_to_fill = m_open
                fill_time_myt = bar.get("timestamp_myt", idx)
                break
            elif not is_up_gap and bar["High"] >= matching_prev:
                gap_filled = True
                minutes_to_fill = m_open
                fill_time_myt = bar.get("timestamp_myt", idx)
                break

        results.append({
            "session_date": s_date,
            "day_open": day_open,
            "prev_close": matching_prev,
            "gap_pct": gap_pct,
            "gap_type": "Up" if is_up_gap else "Down",
            "gap_filled": gap_filled,
            "minutes_to_fill": minutes_to_fill,
            "fill_time_myt": fill_time_myt,
        })

    return pd.DataFrame(results)
