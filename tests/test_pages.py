"""
Unit tests for Streamlit page rendering logic to prevent KeyErrors and indexing bugs.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.calculations import compute_daily_returns_and_sessions
from pages.extreme_days import render_extreme_days_page


def _create_sample_daily_data():
    """Create sample daily OHLCV dataset."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B", tz="America/New_York")
    np.random.seed(42)
    base_price = 100.0
    opens, highs, lows, closes, vols = [], [], [], [], []
    cur = base_price
    for _ in range(30):
        o = cur * (1 + np.random.normal(0, 0.005))
        h = o * (1 + abs(np.random.normal(0, 0.01)))
        l = o * (1 - abs(np.random.normal(0, 0.01)))
        c = (h + l) / 2 + np.random.normal(0, 0.002) * o
        v = int(np.random.uniform(10000, 50000))
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        vols.append(v)
        cur = c

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": vols,
    }, index=dates)
    return compute_daily_returns_and_sessions(df)


@patch("streamlit.dataframe")
@patch("streamlit.plotly_chart")
@patch("streamlit.markdown")
@patch("streamlit.caption")
@patch("streamlit.info")
@patch("streamlit.warning")
@patch("streamlit.slider", return_value=95.0)
@patch("streamlit.columns")
@patch("streamlit.tabs")
def test_render_extreme_days_page_no_keyerror(
    mock_tabs, mock_cols, mock_slider, mock_warning, mock_info, mock_caption, mock_markdown, mock_plotly, mock_df
):
    """Verify render_extreme_days_page runs without KeyError on valid calculated data."""
    mock_col = MagicMock()
    mock_cols.return_value = [mock_col, mock_col]
    mock_tab = MagicMock()
    mock_tabs.return_value = [mock_tab, mock_tab]

    df_calc = _create_sample_daily_data()
    # Should execute without throwing KeyError
    render_extreme_days_page(df_calc)

    # Verify st.dataframe was called
    assert mock_df.called
