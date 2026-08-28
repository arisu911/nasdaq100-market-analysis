"""
Unit tests for financial calculations:
- Returns (total, overnight, regular)
- Opening gap % and gap fill logic
- MFE / MAE formulas
- Drawdowns and drawdown episodes
"""
import pandas as pd
import numpy as np
import pytest

from src.calculations import (
    compute_daily_returns_and_sessions,
    compute_drawdown_series,
    compute_drawdown_episodes,
    compute_cumulative_performance,
)


def _create_sample_daily_data():
    """Create deterministic 4-day daily OHLCV dataset."""
    dates = pd.date_range("2024-06-03", periods=4, freq="D", tz="America/New_York")
    data = {
        # Day 0: Baseline Close = 100
        # Day 1: Gap Up Open 102 (from 100), High 105, Low 99 (fills gap to 100), Close 104
        # Day 2: Gap Down Open 100 (from 104), High 102, Low 98, Close 99 (High 102 does NOT reach 104, not filled)
        # Day 3: Open 100 (from 99), High 103, Low 97, Close 101
        "Open": [100.0, 102.0, 100.0, 100.0],
        "High": [102.0, 105.0, 102.0, 103.0],
        "Low": [98.0, 99.0, 98.0, 97.0],
        "Close": [100.0, 104.0, 99.0, 101.0],
        "Volume": [10000, 15000, 12000, 14000],
    }
    return pd.DataFrame(data, index=dates)


def test_return_calculations():
    """Verify hand-calculated return formulas."""
    df = _create_sample_daily_data()
    res = compute_daily_returns_and_sessions(df)

    # Day 1 (2024-06-04):
    # prev_close = 100, Open = 102, Close = 104
    # total_return = (104 - 100) / 100 = +4.0%
    # overnight_return = (102 - 100) / 100 = +2.0%
    # regular_return = (104 - 102) / 102 = +1.96078%
    d1 = res.iloc[1]
    assert d1["prev_close"] == 100.0
    assert pytest.approx(d1["total_return_pct"], 0.001) == 4.0
    assert pytest.approx(d1["overnight_return_pct"], 0.001) == 2.0
    assert pytest.approx(d1["regular_return_pct"], 0.001) == (104 - 102) / 102 * 100.0


def test_gap_and_gap_fill_logic():
    """Verify gap detection and fill logic."""
    df = _create_sample_daily_data()
    res = compute_daily_returns_and_sessions(df)

    # Day 1: Gap Up (+2.0%), Low is 99 <= prev_close 100 -> gap_filled = True
    d1 = res.iloc[1]
    assert d1["gap_pct"] == 2.0
    assert d1["gap_direction"] == "Up"
    assert d1["gap_filled"] is True or d1["gap_filled"] == 1

    # Day 2: Gap Down (-3.846%), High is 102 < prev_close 104 -> gap_filled = False
    d2 = res.iloc[2]
    assert pytest.approx(d2["gap_pct"], 0.001) == (100 - 104) / 104 * 100.0
    assert d2["gap_direction"] == "Down"
    assert not bool(d2["gap_filled"])


def test_mfe_and_mae():
    """
    Verify MFE (Max Favorable Excursion) and MAE (Max Adverse Excursion):
    Day 1: Open 102, High 105, Low 99, Close 104
    MFE = (105 - 102) / 102 * 100 = +2.941%
    MAE = (99 - 102) / 102 * 100 = -2.941%
    """
    df = _create_sample_daily_data()
    res = compute_daily_returns_and_sessions(df)
    d1 = res.iloc[1]
    assert pytest.approx(d1["mfe_pct"], 0.001) == (105 - 102) / 102 * 100.0
    assert pytest.approx(d1["mae_pct"], 0.001) == (99 - 102) / 102 * 100.0


def test_drawdown_calculations():
    """Test drawdown curve and maximum drawdown calculation."""
    rets = pd.Series([0.10, -0.20, 0.05, -0.10])
    wealth, dd_series, max_dd = compute_drawdown_series(rets)

    # Wealth index:
    # 0: 1.10 (HWM 1.10, DD 0.0%)
    # 1: 1.10 * 0.8 = 0.88 (HWM 1.10, DD = (0.88 - 1.10)/1.10 = -20.0%)
    # 2: 0.88 * 1.05 = 0.924 (HWM 1.10, DD = (0.924 - 1.10)/1.10 = -16.0%)
    # 3: 0.924 * 0.90 = 0.8316 (HWM 1.10, DD = (0.8316 - 1.10)/1.10 = -24.4%)
    assert pytest.approx(dd_series.iloc[0], 0.001) == 0.0
    assert pytest.approx(dd_series.iloc[1], 0.001) == -20.0
    assert pytest.approx(dd_series.iloc[3], 0.001) == -24.4
    assert pytest.approx(max_dd, 0.001) == -24.4
