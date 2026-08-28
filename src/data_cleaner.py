"""
Data validation, cleaning, and quality audit module.
Detects anomalies, duplicate timestamps, invalid price structures, and missing values,
producing a transparent data-quality audit report.
"""
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

from src.timezone_utils import (
    convert_to_new_york,
    convert_to_myt,
    format_myt_timestamp,
)


def clean_market_data(
    df: pd.DataFrame,
    interval: str = "1d"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate and clean OHLCV financial time series:
    1. Check index type and timezone.
    2. Identify and drop duplicate timestamps.
    3. Sort chronological order.
    4. Validate OHLC logic: High >= Low, High >= Open/Close, Low <= Open/Close, Price > 0.
    5. Filter out weekend timestamps.
    6. Audit missing values and anomalies.

    Returns:
    - cleaned_df: pd.DataFrame
    - audit_report: Dict with counts and details of all exclusions and data quality metrics.
    """
    if df.empty:
        return df, {"status": "empty", "initial_rows": 0, "cleaned_rows": 0}

    initial_count = len(df)
    audit = {
        "initial_rows": initial_count,
        "duplicate_timestamps_dropped": 0,
        "negative_or_zero_prices": 0,
        "invalid_ohlc_bars": 0,
        "nan_rows_dropped": 0,
        "weekend_bars_dropped": 0,
        "zero_volume_bars": 0,
        "cleaned_rows": 0,
        "data_quality_score": 100.0,
        "audit_messages": [],
    }

    df_clean = df.copy()

    # 1. Ensure sorted datetime index
    df_clean = df_clean.sort_index()

    # 2. Check duplicate timestamps
    dup_mask = df_clean.index.duplicated(keep="first")
    dup_count = dup_mask.sum()
    if dup_count > 0:
        audit["duplicate_timestamps_dropped"] = int(dup_count)
        audit["audit_messages"].append(f"Dropped {dup_count} duplicate timestamps.")
        df_clean = df_clean[~dup_mask]

    # 3. Check for NaN values in OHLC
    ohlc_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df_clean.columns]
    nan_mask = df_clean[ohlc_cols].isna().any(axis=1)
    nan_count = nan_mask.sum()
    if nan_count > 0:
        audit["nan_rows_dropped"] = int(nan_count)
        audit["audit_messages"].append(f"Dropped {nan_count} rows with missing OHLC values.")
        df_clean = df_clean[~nan_mask]

    # 4. Check for zero or negative prices
    pos_mask = (df_clean[ohlc_cols] <= 0).any(axis=1)
    neg_count = pos_mask.sum()
    if neg_count > 0:
        audit["negative_or_zero_prices"] = int(neg_count)
        audit["audit_messages"].append(f"Dropped {neg_count} rows with zero or negative price values.")
        df_clean = df_clean[~pos_mask]

    # 5. Check OHLC consistency: High >= Low, High >= max(Open, Close), Low <= min(Open, Close)
    high_valid = (df_clean["High"] >= df_clean["Low"]) & \
                 (df_clean["High"] >= df_clean["Open"] * 0.999) & \
                 (df_clean["High"] >= df_clean["Close"] * 0.999)
    low_valid = (df_clean["Low"] <= df_clean["Open"] * 1.001) & \
                (df_clean["Low"] <= df_clean["Close"] * 1.001)
    invalid_ohlc = ~(high_valid & low_valid)
    inv_count = invalid_ohlc.sum()
    if inv_count > 0:
        audit["invalid_ohlc_bars"] = int(inv_count)
        audit["audit_messages"].append(f"Fixed/Filtered {inv_count} bars with invalid OHLC geometry (High < Low).")
        df_clean = df_clean[~invalid_ohlc]

    # 6. Filter out weekend timestamps based on America/New_York date
    idx_ny = convert_to_new_york(df_clean.index)
    weekend_mask = idx_ny.weekday >= 5
    weekend_count = weekend_mask.sum()
    if weekend_count > 0:
        audit["weekend_bars_dropped"] = int(weekend_count)
        audit["audit_messages"].append(f"Dropped {weekend_count} weekend observations.")
        df_clean = df_clean[~weekend_mask]

    # 7. Check zero volume bars
    if "Volume" in df_clean.columns:
        zero_vol = (df_clean["Volume"] == 0).sum()
        audit["zero_volume_bars"] = int(zero_vol)

    final_count = len(df_clean)
    audit["cleaned_rows"] = final_count

    # Calculate quality score (percentage of valid rows retained)
    if initial_count > 0:
        retained_ratio = final_count / initial_count
        audit["data_quality_score"] = round(retained_ratio * 100.0, 2)

    if not audit["audit_messages"]:
        audit["audit_messages"].append("All OHLC records passed structural integrity checks with 0 anomalies.")

    return df_clean, audit


def format_audit_report_table(audit_report: Dict[str, Any]) -> pd.DataFrame:
    """Format audit dictionary into a clean pandas DataFrame for display."""
    items = [
        ("Initial Records Downloaded", f"{audit_report.get('initial_rows', 0):,}"),
        ("Cleaned Records Retained", f"{audit_report.get('cleaned_rows', 0):,}"),
        ("Data Quality Score", f"{audit_report.get('data_quality_score', 100.0):.2f}%"),
        ("Duplicate Timestamps Removed", f"{audit_report.get('duplicate_timestamps_dropped', 0):,}"),
        ("Missing / NaN Bars Dropped", f"{audit_report.get('nan_rows_dropped', 0):,}"),
        ("Zero or Negative Prices Dropped", f"{audit_report.get('negative_or_zero_prices', 0):,}"),
        ("Invalid OHLC Relationships Dropped", f"{audit_report.get('invalid_ohlc_bars', 0):,}"),
        ("Weekend Observations Dropped", f"{audit_report.get('weekend_bars_dropped', 0):,}"),
        ("Zero Volume Bars Flagged", f"{audit_report.get('zero_volume_bars', 0):,}"),
    ]
    return pd.DataFrame(items, columns=["Data Quality Audit Metric", "Status / Count"])
