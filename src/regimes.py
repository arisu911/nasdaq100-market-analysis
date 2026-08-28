"""
Volatility regime segmentation and cross-regime comparative analytics.
Uses configurable percentile-based methodologies (e.g. Low <25th, Normal 25th-75th, High >75th)
to evaluate how NASDAQ-100 behavior changes across market environments.
"""
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from config.settings import DEFAULT_REGIME_LOW_PCT, DEFAULT_REGIME_HIGH_PCT


def classify_volatility_regimes(
    df_daily: pd.DataFrame,
    metric_col: str = "daily_range_pct",
    low_pct: float = DEFAULT_REGIME_LOW_PCT,
    high_pct: float = DEFAULT_REGIME_HIGH_PCT
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Classify trading sessions into volatility regimes based on empirical distribution percentiles.

    Returns:
    - df with added 'vol_regime' column ('Low Volatility', 'Normal Volatility', 'High Volatility')
    - thresholds: {'low_threshold': float, 'high_threshold': float}
    """
    df = df_daily.copy()

    # If rolling_vol metric chosen but missing, calculate 20d vol
    if metric_col == "rolling_vol_20d" and "rolling_vol_20d" not in df.columns:
        df["rolling_vol_20d"] = df["total_return"].rolling(20).std() * np.sqrt(252) * 100.0

    target_series = df[metric_col].dropna()
    if target_series.empty:
        df["vol_regime"] = "Normal Volatility"
        return df, {"low_threshold": 0.0, "high_threshold": 0.0}

    low_val = float(np.percentile(target_series, low_pct))
    high_val = float(np.percentile(target_series, high_pct))

    conditions = [
        df[metric_col] < low_val,
        (df[metric_col] >= low_val) & (df[metric_col] <= high_val),
        df[metric_col] > high_val,
    ]
    choices = ["Low Volatility", "Normal Volatility", "High Volatility"]

    df["vol_regime"] = np.select(conditions, choices, default="Normal Volatility")

    thresholds = {
        "metric_col": metric_col,
        "low_pct": low_pct,
        "high_pct": high_pct,
        "low_threshold": low_val,
        "high_threshold": high_val,
    }

    return df, thresholds


def compute_regime_comparison_statistics(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comparative behavioral statistics grouped by volatility regime:
    - Sample size (n)
    - Mean total return %
    - Median total return %
    - Total return volatility %
    - Positive sessions %
    - Mean daily range %
    - Mean overnight return %
    - Mean regular session return %
    - Mean absolute gap %
    - Gap fill rate %
    - Mean MFE % & MAE %
    """
    if "vol_regime" not in df_daily.columns:
        df_daily, _ = classify_volatility_regimes(df_daily)

    regime_order = ["Low Volatility", "Normal Volatility", "High Volatility"]
    rows = []

    for regime in regime_order:
        subset = df_daily[df_daily["vol_regime"] == regime].dropna(subset=["total_return_pct"])
        n = len(subset)
        if n == 0:
            continue

        tot_ret = subset["total_return_pct"]
        overnight = subset["overnight_return_pct"]
        regular = subset["regular_return_pct"]
        rng = subset["daily_range_pct"]
        gap_fill_rate = (subset["gap_filled"].sum() / n * 100.0) if "gap_filled" in subset.columns else np.nan

        rows.append({
            "Volatility Regime": regime,
            "Sessions (n)": n,
            "Sample Share": f"{n / len(df_daily) * 100:.1f}%",
            "Mean Daily Return": f"{tot_ret.mean():.3f}%",
            "Median Daily Return": f"{tot_ret.median():.3f}%",
            "Daily Std Dev": f"{tot_ret.std():.2f}%",
            "Annualized Vol": f"{tot_ret.std() * np.sqrt(252):.2f}%",
            "Positive Days %": f"{(tot_ret > 0).sum() / n * 100:.1f}%",
            "Avg Daily Range %": f"{rng.mean():.2f}%",
            "Avg Overnight Return": f"{overnight.mean():.3f}%",
            "Avg Regular Return": f"{regular.mean():.3f}%",
            "Avg Abs Gap %": f"{subset['gap_pct'].abs().mean():.2f}%",
            "Gap Fill Rate": f"{gap_fill_rate:.1f}%" if pd.notna(gap_fill_rate) else "N/A",
            "Avg MFE %": f"{subset['mfe_pct'].mean():.2f}%" if "mfe_pct" in subset.columns else "N/A",
            "Avg MAE %": f"{subset['mae_pct'].mean():.2f}%" if "mae_pct" in subset.columns else "N/A",
            "_raw_mean_ret": tot_ret.mean(),
            "_raw_vol": tot_ret.std() * np.sqrt(252),
            "_raw_range": rng.mean(),
        })

    return pd.DataFrame(rows)
