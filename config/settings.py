"""
Global configuration and settings for NASDAQ-100 market analysis.
Centralizes timezone definitions, instrument symbols, session boundaries,
and analytical parameters.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DATA_DIR = DATA_DIR / "cache"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Timezones
# Primary display timezone is Malaysia Time (MYT / Asia/Kuala_Lumpur, UTC+8)
# Source exchange timezone is US Eastern Time (America/New_York)
SOURCE_TZ = "America/New_York"
TARGET_TZ = "Asia/Kuala_Lumpur"
UTC_TZ = "UTC"

# Instrument Tickers
# NASDAQ-100 Index: used for index price, returns, volatility, range, gaps, drawdowns
INDEX_TICKER = "^NDX"
# Invesco QQQ Trust: used for tradable proxy, volume, and high-frequency microstructure
ETF_TICKER = "QQQ"
# S&P 500 benchmark for cross-market correlation analysis
BENCHMARK_INDEX = "^GSPC"
BENCHMARK_ETF = "SPY"

# Market Session in New York Time (ET)
# Regular trading hours: 09:30 - 16:00 America/New_York
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Opening Analysis Windows (in minutes from regular open 09:30 ET)
OPENING_WINDOWS = [5, 15, 30, 60]

# Volatility Regime Percentile Thresholds
DEFAULT_REGIME_LOW_PCT = 25.0
DEFAULT_REGIME_HIGH_PCT = 75.0

# Rolling Analysis Windows (in trading sessions)
DEFAULT_ROLLING_WINDOWS = [20, 60, 120]

# Intraday Supported Resolutions
SUPPORTED_RESOLUTIONS = ["5m", "10m", "15m", "30m", "60m"]

# Analysis Period Presets
PERIOD_PRESETS = ["1Y", "3Y", "5Y", "10Y", "Custom"]

# Extreme Day Default Percentile
DEFAULT_EXTREME_PERCENTILE = 95.0
