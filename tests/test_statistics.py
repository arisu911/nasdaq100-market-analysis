"""
Unit tests for statistical distributions, historical frequencies, and aggregations.
"""
import pandas as pd
import numpy as np
import pytest

from src.statistics import (
    compute_distribution_summary,
    compute_historical_frequencies,
    compute_weekday_statistics,
)


def test_distribution_summary():
    """Verify descriptive statistics on a known symmetric array."""
    data = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
    summary = compute_distribution_summary(data, label="Test")

    assert summary["count"] == 5
    assert summary["mean"] == 0.0
    assert summary["median"] == 0.0
    assert summary["min"] == -2.0
    assert summary["max"] == 2.0
    assert summary["positive_pct"] == 40.0
    assert summary["negative_pct"] == 40.0
    assert summary["zero_pct"] == 20.0


def test_historical_frequencies():
    """Test frequency table generation."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="America/New_York")
    df = pd.DataFrame({
        "total_return_pct": [1.0, -0.5, 0.2, 0.8, -1.2, 0.4, -0.1, 1.5, -2.5, 0.6],
        "abs_total_return_pct": [1.0, 0.5, 0.2, 0.8, 1.2, 0.4, 0.1, 1.5, 2.5, 0.6],
        "overnight_return_pct": [0.2, -0.1, 0.0, 0.3, -0.4, 0.1, 0.0, 0.5, -0.8, 0.2],
        "regular_return_pct": [0.8, -0.4, 0.2, 0.5, -0.8, 0.3, -0.1, 1.0, -1.7, 0.4],
        "gap_pct": [0.2, -0.1, 0.0, 0.3, -0.4, 0.1, 0.0, 0.5, -0.8, 0.2],
        "gap_filled": [True, True, True, False, True, False, True, True, False, True],
        "daily_range_pct": [1.2, 0.8, 0.4, 1.0, 1.8, 0.7, 0.3, 2.1, 3.2, 0.9],
    }, index=dates)

    freqs = compute_historical_frequencies(df)
    assert not freqs.empty
    assert len(freqs) >= 10
    # 6 positive out of 10 = 60%
    pos_row = freqs[freqs["Historical Behavior Event"] == "Positive Close-to-Close Sessions"]
    assert "60.00%" in pos_row["Historical Frequency"].iloc[0]
