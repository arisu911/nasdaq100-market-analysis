"""
Statistical aggregation and descriptive metrics module:
- Comprehensive distribution statistics (mean, median, std, percentiles, skew, kurtosis)
- Historical event frequency counters (positive %, thresholds, gap fills)
- Weekday, Monthly, and Yearly performance aggregations
- Intraday time-of-day statistical profiles with MYT labels
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats

from src.timezone_utils import (
    convert_to_new_york,
    convert_to_myt,
    format_myt_timestamp,
)


def compute_distribution_summary(
    series: pd.Series,
    label: str = "Metric",
    is_percentage: bool = True
) -> Dict[str, Any]:
    """
    Compute comprehensive descriptive distribution statistics for a numeric series:
    - Sample size (n)
    - Mean, Median, Standard Deviation
    - Min, Max
    - 10th, 25th, 50th, 75th, 90th, 95th Percentiles
    - Skewness, Kurtosis
    - Positive %, Negative %, Zero %
    """
    clean_s = series.dropna()
    n = len(clean_s)
    if n == 0:
        return {
            "metric": label,
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "positive_pct": np.nan,
            "negative_pct": np.nan,
            "zero_pct": np.nan,
        }

    p10, p25, p50, p75, p90, p95 = np.percentile(clean_s, [10, 25, 50, 75, 90, 95])
    pos_pct = (clean_s > 0).sum() / n * 100.0
    neg_pct = (clean_s < 0).sum() / n * 100.0
    zero_pct = (clean_s == 0).sum() / n * 100.0

    skew_val = float(stats.skew(clean_s, bias=False)) if n > 2 else 0.0
    kurt_val = float(stats.kurtosis(clean_s, bias=False)) if n > 3 else 0.0

    return {
        "metric": label,
        "count": n,
        "mean": float(clean_s.mean()),
        "median": float(clean_s.median()),
        "std": float(clean_s.std()),
        "min": float(clean_s.min()),
        "max": float(clean_s.max()),
        "p10": float(p10),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p90": float(p90),
        "p95": float(p95),
        "skewness": round(skew_val, 3),
        "kurtosis": round(kurt_val, 3),
        "positive_pct": round(pos_pct, 2),
        "negative_pct": round(neg_pct, 2),
        "zero_pct": round(zero_pct, 2),
        "is_percentage": is_percentage,
    }


def compute_historical_frequencies(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate descriptive historical event frequencies.
    Explicitly labeled as Historical Frequencies (not probabilities/predictions).
    """
    df = df_daily.dropna(subset=["total_return_pct", "overnight_return_pct", "regular_return_pct"])
    n = len(df)
    if n == 0:
        return pd.DataFrame()

    freqs = [
        ("Positive Close-to-Close Sessions", f"{(df['total_return_pct'] > 0).sum() / n * 100:.2f}%", f"{(df['total_return_pct'] > 0).sum()} / {n}"),
        ("Negative Close-to-Close Sessions", f"{(df['total_return_pct'] < 0).sum() / n * 100:.2f}%", f"{(df['total_return_pct'] < 0).sum()} / {n}"),
        ("Absolute Daily Return > 0.25%", f"{(df['abs_total_return_pct'] > 0.25).sum() / n * 100:.2f}%", f"{(df['abs_total_return_pct'] > 0.25).sum()} / {n}"),
        ("Absolute Daily Return > 0.50%", f"{(df['abs_total_return_pct'] > 0.50).sum() / n * 100:.2f}%", f"{(df['abs_total_return_pct'] > 0.50).sum()} / {n}"),
        ("Absolute Daily Return > 1.00%", f"{(df['abs_total_return_pct'] > 1.00).sum() / n * 100:.2f}%", f"{(df['abs_total_return_pct'] > 1.00).sum()} / {n}"),
        ("Absolute Daily Return > 2.00%", f"{(df['abs_total_return_pct'] > 2.00).sum() / n * 100:.2f}%", f"{(df['abs_total_return_pct'] > 2.00).sum()} / {n}"),
        ("Positive Opening Gap (Up Gap)", f"{(df['gap_pct'] > 0).sum() / n * 100:.2f}%", f"{(df['gap_pct'] > 0).sum()} / {n}"),
        ("Negative Opening Gap (Down Gap)", f"{(df['gap_pct'] < 0).sum() / n * 100:.2f}%", f"{(df['gap_pct'] < 0).sum()} / {n}"),
        ("Absolute Opening Gap > 0.50%", f"{(df['gap_pct'].abs() > 0.50).sum() / n * 100:.2f}%", f"{(df['gap_pct'].abs() > 0.50).sum()} / {n}"),
        ("Absolute Opening Gap > 1.00%", f"{(df['gap_pct'].abs() > 1.00).sum() / n * 100:.2f}%", f"{(df['gap_pct'].abs() > 1.00).sum()} / {n}"),
        ("Gap Fill Frequency (All Gaps)", f"{(df['gap_filled']).sum() / n * 100:.2f}%", f"{(df['gap_filled']).sum()} / {n}"),
        ("Daily Range > 1.00%", f"{(df['daily_range_pct'] > 1.00).sum() / n * 100:.2f}%", f"{(df['daily_range_pct'] > 1.00).sum()} / {n}"),
        ("Daily Range > 2.00%", f"{(df['daily_range_pct'] > 2.00).sum() / n * 100:.2f}%", f"{(df['daily_range_pct'] > 2.00).sum()} / {n}"),
    ]

    return pd.DataFrame(freqs, columns=["Historical Behavior Event", "Historical Frequency", "Sample Count (Occurrences / Total)"])


def compute_weekday_statistics(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate day-of-week descriptive statistics (Monday through Friday).
    """
    df = df_daily.dropna(subset=["total_return_pct"]).copy()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    rows = []
    for day in weekday_order:
        subset = df[df["weekday"] == day]
        n = len(subset)
        if n == 0:
            continue

        ret = subset["total_return_pct"]
        rng = subset["daily_range_pct"]
        vol_pct = ret.std() * np.sqrt(252)

        p25_vol, p75_vol, p90_vol, p95_vol = np.percentile(ret.abs(), [25, 75, 90, 95])

        has_vol = "Volume" in subset.columns and (subset["Volume"] > 0).sum() > 0
        avg_vol = subset["Volume"].mean() if has_vol else np.nan
        med_vol = subset["Volume"].median() if has_vol else np.nan

        rows.append({
            "Day of Week": day,
            "Sample Size (n)": n,
            "Mean Return": f"{ret.mean():.3f}%",
            "Median Return": f"{ret.median():.3f}%",
            "Std Dev": f"{ret.std():.3f}%",
            "Annualized Vol": f"{vol_pct:.2f}%",
            "Positive %": f"{(ret > 0).sum() / n * 100:.1f}%",
            "Negative %": f"{(ret < 0).sum() / n * 100:.1f}%",
            "Avg Range %": f"{rng.mean():.2f}%",
            "Median Range %": f"{rng.median():.2f}%",
            "75th Pct Abs Move": f"{p75_vol:.2f}%",
            "90th Pct Abs Move": f"{p90_vol:.2f}%",
            "95th Pct Abs Move": f"{p95_vol:.2f}%",
            "Avg Volume": f"{avg_vol:,.0f}" if pd.notna(avg_vol) else "N/A",
            "_raw_mean_ret": ret.mean(),
            "_raw_median_ret": ret.median(),
            "_raw_std": ret.std(),
            "_raw_pos_pct": (ret > 0).sum() / n * 100,
            "_raw_avg_range": rng.mean(),
        })

    return pd.DataFrame(rows)


def compute_monthly_statistics(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate month-of-year descriptive statistics (January through December).
    """
    df = df_daily.dropna(subset=["total_return_pct"]).copy()
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    rows = []
    for m_idx, m_name in enumerate(month_order, 1):
        subset = df[df["month"] == m_idx]
        n = len(subset)
        if n == 0:
            continue

        ret = subset["total_return_pct"]
        rng = subset["daily_range_pct"]
        ann_vol = ret.std() * np.sqrt(252)

        rows.append({
            "Month": m_name,
            "Sample Sessions (n)": n,
            "Mean Daily Return": f"{ret.mean():.3f}%",
            "Median Daily Return": f"{ret.median():.3f}%",
            "Annualized Volatility": f"{ann_vol:.2f}%",
            "Positive Day %": f"{(ret > 0).sum() / n * 100:.1f}%",
            "Average Range %": f"{rng.mean():.2f}%",
            "_raw_mean_ret": ret.mean(),
            "_raw_pos_pct": (ret > 0).sum() / n * 100,
            "_raw_vol": ann_vol,
        })

    return pd.DataFrame(rows)


def compute_yearly_statistics(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate calendar year descriptive statistics.
    """
    from src.calculations import compute_drawdown_series

    df = df_daily.dropna(subset=["total_return_pct"]).copy()
    rows = []

    for year, group in df.groupby("year"):
        n = len(group)
        if n == 0:
            continue

        ret = group["total_return"]
        cum_ret = (1.0 + ret).cumprod().iloc[-1] - 1.0
        ann_vol = ret.std() * np.sqrt(252) * 100.0

        _, dd_series, max_dd = compute_drawdown_series(ret)

        # Monthly positive count
        month_rets = group.groupby("month")["total_return"].apply(lambda r: (1.0 + r).cumprod().iloc[-1] - 1.0)
        pos_months = (month_rets > 0).sum()
        total_months = len(month_rets)

        rows.append({
            "Year": str(year),
            "Annual Return": f"{cum_ret * 100:.2f}%",
            "Avg Daily Return": f"{group['total_return_pct'].mean():.3f}%",
            "Annualized Vol": f"{ann_vol:.2f}%",
            "Max Drawdown": f"{max_dd:.2f}%",
            "Positive Months": f"{pos_months} / {total_months}",
            "Trading Sessions": n,
            "_raw_cum_ret": cum_ret * 100,
            "_raw_max_dd": max_dd,
        })

    return pd.DataFrame(rows).sort_values(by="Year", ascending=False).reset_index(drop=True)


def compute_intraday_time_bucket_statistics(
    df_intraday: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate intraday performance by regular-session time bucket (e.g. 5m, 15m, 30m, 60m).
    Computes returns, absolute returns, volatility, range, and relative volume.
    X-axis and display labels are strictly formatted in MYT.
    """
    from src.market_sessions import filter_regular_session

    reg = filter_regular_session(df_intraday).copy()
    if reg.empty:
        return pd.DataFrame()

    # Calculate bar returns per day
    reg["bar_return_pct"] = reg.groupby("session_date")["Close"].pct_change() * 100.0
    # First bar return from Open
    first_bar_mask = reg.groupby("session_date").cumcount() == 0
    reg.loc[first_bar_mask, "bar_return_pct"] = (
        (reg.loc[first_bar_mask, "Close"] - reg.loc[first_bar_mask, "Open"]) /
        reg.loc[first_bar_mask, "Open"] * 100.0
    )

    reg["bar_range_pct"] = (reg["High"] - reg["Low"]) / reg["Open"] * 100.0
    reg["bar_abs_return_pct"] = reg["bar_return_pct"].abs()

    # Mean volume across all regular session bars for relative volume calculation
    has_vol = "Volume" in reg.columns and (reg["Volume"] > 0).sum() > 0
    overall_mean_vol = reg["Volume"].mean() if has_vol else 1.0

    # Group by minutes_from_open so bars from EST and EDT line up at identical session progress
    results = []
    for m_open, group in reg.groupby("minutes_from_open"):
        n = len(group)
        if n == 0:
            continue

        ret = group["bar_return_pct"].dropna()
        rng = group["bar_range_pct"].dropna()
        abs_ret = group["bar_abs_return_pct"].dropna()

        # Represent representative MYT time (e.g. modal time_myt_str)
        myt_label = group["time_myt_str"].mode().iloc[0] if "time_myt_str" in group.columns else f"+{int(m_open)}m"
        ny_label = group["time_ny_str"].mode().iloc[0] if "time_ny_str" in group.columns else f"ET+{int(m_open)}m"

        avg_vol = group["Volume"].mean() if has_vol else np.nan
        med_vol = group["Volume"].median() if has_vol else np.nan
        rel_vol = (avg_vol / overall_mean_vol) if has_vol and overall_mean_vol > 0 else np.nan

        results.append({
            "minutes_from_open": int(m_open),
            "time_myt": myt_label,
            "time_ny": ny_label,
            "display_label_myt": f"{myt_label} MYT",
            "sample_size": n,
            "mean_return_pct": ret.mean(),
            "median_return_pct": ret.median(),
            "mean_abs_return_pct": abs_ret.mean(),
            "realized_vol_pct": ret.std() * np.sqrt(252 * 78),  # annualized approx for 5m bars
            "mean_range_pct": rng.mean(),
            "median_range_pct": rng.median(),
            "positive_pct": (ret > 0).sum() / len(ret) * 100.0 if len(ret) > 0 else 0.0,
            "negative_pct": (ret < 0).sum() / len(ret) * 100.0 if len(ret) > 0 else 0.0,
            "avg_volume": avg_vol,
            "median_volume": med_vol,
            "relative_volume": rel_vol,
        })

    out_df = pd.DataFrame(results).sort_values(by="minutes_from_open").reset_index(drop=True)
    return out_df
