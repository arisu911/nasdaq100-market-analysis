"""
Unit tests for percentile-based volatility regimes segmentation.
"""
import pandas as pd
import numpy as np
import pytest

from src.regimes import (
    classify_volatility_regimes,
    compute_regime_comparison_statistics,
)


def test_regime_classification_percentiles():
    """Verify 25th / 75th percentile classification split."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="America/New_York")
    # Linear series from 1 to 100
    df = pd.DataFrame({
        "daily_range_pct": np.linspace(1, 100, 100),
        "total_return_pct": np.random.randn(100),
        "overnight_return_pct": np.random.randn(100) * 0.5,
        "regular_return_pct": np.random.randn(100) * 0.5,
        "gap_pct": np.random.randn(100) * 0.5,
        "gap_filled": np.random.choice([True, False], 100),
    }, index=dates)

    classified, thresholds = classify_volatility_regimes(
        df,
        metric_col="daily_range_pct",
        low_pct=25.0,
        high_pct=75.0,
    )

    counts = classified["vol_regime"].value_counts()
    # In a 100-point uniform series, approx 24-25 Low, 50-51 Normal, 24-25 High
    assert "Low Volatility" in counts
    assert "Normal Volatility" in counts
    assert "High Volatility" in counts
    assert counts["Low Volatility"] >= 20
    assert counts["Normal Volatility"] >= 45
    assert counts["High Volatility"] >= 20


def test_regime_comparison_statistics():
    """Verify cross-regime comparison table generation."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D", tz="America/New_York")
    df = pd.DataFrame({
        "daily_range_pct": np.linspace(0.5, 5.0, 60),
        "total_return_pct": np.linspace(-1.0, 1.0, 60),
        "overnight_return_pct": np.linspace(-0.5, 0.5, 60),
        "regular_return_pct": np.linspace(-0.5, 0.5, 60),
        "gap_pct": np.linspace(-0.5, 0.5, 60),
        "gap_filled": [True] * 60,
    }, index=dates)

    classified, _ = classify_volatility_regimes(df, metric_col="daily_range_pct")
    stats_df = compute_regime_comparison_statistics(classified)

    assert not stats_df.empty
    assert len(stats_df) == 3
    assert set(stats_df["Volatility Regime"]) == {"Low Volatility", "Normal Volatility", "High Volatility"}
