# NASDAQ-100 Historical Market Behavior Analytics Terminal

A modular, descriptive quantitative research application built with **Python**, **Streamlit**, **Pandas**, and **Plotly** to analyze how the **NASDAQ-100 Index (`^NDX`)** and its tradable ETF proxy **Invesco QQQ Trust (`QQQ`)** have historically behaved across intraday trading sessions, calendar periods, volatility regimes, overnight holding windows, and market stress drawdowns.

All user-facing timestamps, time axes, tables, and session labels are displayed dynamically in **Malaysia Time (MYT / Asia/Kuala_Lumpur, UTC+8)**.

---

## 1. Project Objective

The primary objective is to provide institutional-grade descriptive statistical analysis of historical NASDAQ-100 behavior without introducing speculative trading rules or predictive bias.

### Key Research Questions
1. **Intraday Profiles**: How do returns, realized volatility, high-low range, and relative volume distribute across different 5-minute to 60-minute windows of the US regular trading session when viewed in MYT?
2. **Overnight vs. Regular Session**: Where does the majority of the NASDAQ-100's cumulative drift and price variance originate—during the overnight non-trading hours (Close$_{t-1} \rightarrow$ Open$_t$) or during the regular session (Open$_t \rightarrow$ Close$_t$)?
3. **Opening & Gap Dynamics**: What is the historical magnitude and direction of opening gaps, what percentage are subsequently filled during the regular session, and what fraction of the daily high-low range is established within the first 5, 15, 30, and 60 minutes?
4. **Day-of-Week Seasonality**: Does the NASDAQ-100 exhibit statistically distinct mean returns, volatility, or positive-day frequencies between Mondays, Tuesdays, Wednesdays, Thursdays, and Fridays?
5. **Volatility Regimes**: How do opening returns, gap fills, and intraday amplitudes transform during Low, Normal, and High volatility market regimes?
6. **Drawdown Structure**: What are the historical drawdown depths, peak-to-trough durations, and recovery paths across market cycles?
7. **Cross-Market Co-Movement**: How does NASDAQ-100 daily return correlation and beta against the S&P 500 fluctuate over rolling 60-day horizons and across volatility environments?

---

## 2. Scope & Ethical Boundaries

### What the Project Does:
- Descriptive statistical aggregation (Mean, Median, Standard Deviation, Skewness, Kurtosis).
- Empirical distribution percentile analysis (10th, 25th, 50th, 75th, 90th, 95th percentiles).
- Timezone-aware conversion from `America/New_York` to `Asia/Kuala_Lumpur` (MYT / UTC+8).
- Multi-resolution intraday resampling (5m, 10m, 15m, 30m, 60m).
- Path analysis using Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE).
- Cross-market correlation and beta estimation against the S&P 500 (`^GSPC`).
- Comprehensive data quality audit and data coverage transparency disclosures.

### What the Project Does NOT Do:
- ❌ **No trading strategies or signal generation** (no buy/sell indicators).
- ❌ **No backtesting or strategy optimization** (no profit curves or Sharpe ratio optimization).
- ❌ **No trade execution or entry/exit rules**.
- ❌ **No predictive claims or forward-looking price forecasts**.
- ❌ **No synthetic or fabricated market data**.

---

## 3. Primary Timezone & Dynamic New York $\rightarrow$ MYT Conversion

The primary user timezone is **Malaysia Time (MYT / Asia/Kuala_Lumpur, UTC+8)**. 

### Why Session Times Vary in MYT
NASDAQ regular trading hours are fixed in New York local time:
$$\text{09:30 – 16:00 America/New\_York}$$

Because the United States observes **Daylight Saving Time (DST)** while Malaysia does not:
- **US Daylight Saving Time (EDT, UTC-4)** (Mid-March to Early November):
  - NASDAQ Open: `09:30 EDT` $\rightarrow$ **`21:30 MYT`**
  - NASDAQ Close: `16:00 EDT` $\rightarrow$ **`04:00 (+1 day) MYT`**
  - Session Hours: **`21:30 – 04:00+1d MYT`** (Offset difference = +12 hours)
- **US Standard Time (EST, UTC-5)** (Early November to Mid-March):
  - NASDAQ Open: `09:30 EST` $\rightarrow$ **`22:30 MYT`**
  - NASDAQ Close: `16:00 EST` $\rightarrow$ **`05:00 (+1 day) MYT`**
  - Session Hours: **`22:30 – 05:00+1d MYT`** (Offset difference = +13 hours)

### Implementation Mechanism
All datetime operations in `src/timezone_utils.py` utilize Python's standard `zoneinfo.ZoneInfo` library. Offsets are **never manually hard-coded**. Every timestamp retains full timezone awareness, dynamically evaluating historical DST boundaries per calendar date.

---

## 4. Data Source & Historical Limitations

- **Provider**: Yahoo Finance (`yfinance`).
- **Instruments**:
  - `^NDX` (NASDAQ-100 Index): Evaluates official index price, percentage returns, volatility, gaps, and drawdowns.
  - `QQQ` (Invesco QQQ Trust ETF): Serves as tradable proxy for share volume and liquid execution microstructure.
  - `^GSPC` (S&P 500 Index): Used for benchmark co-movement and rolling correlation.
- **Coverage & Depth**:
  - **Daily History**: 10+ years (~2,513+ trading sessions).
  - **Intraday 5-Minute Data**: Up to ~60 calendar days (standard free API limit).
  - **Intraday 1-Hour Data**: Up to ~730 calendar days (2 years).
- **Data Integrity & Zero-Fake-Data Rule**:
  - The application strictly reports available historical sample sizes ($n$) and never synthesizes or interpolates missing intraday bars.

---

## 5. Statistical & Calculation Methodology

### 1. Return Decomposition
- **Total Close-to-Close Return**:
  $$R_{\text{total}, t} = \frac{\text{Close}_t - \text{Close}_{t-1}}{\text{Close}_{t-1}}$$
- **Overnight Return**:
  $$R_{\text{overnight}, t} = \frac{\text{Open}_t - \text{Close}_{t-1}}{\text{Close}_{t-1}}$$
- **Regular Session Return**:
  $$R_{\text{regular}, t} = \frac{\text{Close}_t - \text{Open}_t}{\text{Open}_t}$$

### 2. Opening Gap & Gap-Fill Condition
- **Gap Percentage**:
  $$\text{Gap } \% = \frac{\text{Open}_t - \text{Close}_{t-1}}{\text{Close}_{t-1}} \times 100$$
- **Gap Filled Logic**:
  - **Up Gap** ($\text{Gap} > 0$): Filled if regular-session $\text{Low}_t \le \text{Close}_{t-1}$.
  - **Down Gap** ($\text{Gap} < 0$): Filled if regular-session $\text{High}_t \ge \text{Close}_{t-1}$.

### 3. Path Dynamics (MFE / MAE)
- **Maximum Favorable Excursion (MFE)**:
  $$\text{MFE } \% = \frac{\text{High}_t - \text{Open}_t}{\text{Open}_t} \times 100$$
- **Maximum Adverse Excursion (MAE)**:
  $$\text{MAE } \% = \frac{\text{Low}_t - \text{Open}_t}{\text{Open}_t} \times 100$$

### 4. Volatility Regimes
Trading sessions are categorized using empirical distribution percentiles on daily High-Low Range % or trailing 20-day realized volatility:
- **Low Volatility**: $< 25\text{th}$ percentile.
- **Normal Volatility**: $25\text{th} \le x \le 75\text{th}$ percentile.
- **High Volatility**: $> 75\text{th}$ percentile.

---

## 6. Project Architecture

```
nasdaq100_market_analysis/
├── app.py                      # Main Streamlit dashboard & page router
├── requirements.txt            # Project dependencies
├── README.md                   # Complete research documentation
├── .gitignore                  # Ignore rules
│
├── config/
│   ├── __init__.py
│   └── settings.py             # Timezone, ticker, session, and parameter constants
│
├── data/
│   ├── raw/                    # Raw downloaded Parquet files
│   ├── processed/              # Cleaned datasets
│   └── cache/                  # Computation cache
│
├── src/
│   ├── __init__.py
│   ├── timezone_utils.py       # Timezone conversion (ET <-> MYT), DST, market clock
│   ├── market_sessions.py      # Session boundary tagging (09:30-16:00 ET), opening windows
│   ├── data_loader.py          # Data ingestion, Parquet caching, coverage auditor
│   ├── data_cleaner.py         # Data validation, duplicate filter, quality report
│   ├── calculations.py         # Return formulas, gaps, MFE/MAE, volatility, drawdowns
│   ├── statistics.py           # Descriptive statistics, distributions, weekday/monthly stats
│   ├── regimes.py              # Percentile-based volatility regime classification
│   └── charts.py               # Modern Plotly visualization library (Dark theme, MYT axes)
│
├── pages/
│   ├── overview.py             # Page 1: Overview, KPI cards, dynamic MYT market clock
│   ├── intraday_analysis.py    # Page 2: Intraday 5m-60m session profiles in MYT
│   ├── weekday_analysis.py     # Page 3: Monday-Friday, monthly, and annual seasonality
│   ├── opening_analysis.py     # Page 4: Opening windows (5m/15m/30m/60m) & gap-fill stats
│   ├── overnight_analysis.py   # Page 5: Overnight vs Regular session return decomposition
│   ├── volatility_analysis.py  # Page 6: Session volatility profiles, rolling vol & regimes
│   ├── extreme_days.py         # Page 7: Tail event analysis & MFE/MAE path scatter
│   ├── drawdowns.py            # Page 8: Underwater drawdown curve & major episodes
│   └── correlations.py         # Page 9: NASDAQ-100 vs S&P 500 cross-market correlation
│
└── tests/
    ├── __init__.py
    ├── test_timezone.py        # EST, EDT, and DST transition boundary tests
    ├── test_sessions.py        # Regular session and opening window tests
    ├── test_calculations.py    # Return, gap-fill, MFE/MAE, and drawdown tests
    ├── test_statistics.py      # Descriptive distribution and frequency tests
    ├── test_regimes.py         # Volatility regime segmentation tests
    └── verify_pipeline.py      # Full end-to-end integration test
```

---

## 7. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Virtual environment recommended

### Installation Steps
```bash
# 1. Navigate to the project directory
cd nasdaq100_market_analysis

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 8. Running the Application

Launch the Streamlit research dashboard:
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 9. Running Automated Tests

Run the complete test suite with `pytest`:
```bash
pytest tests/
```

Run the end-to-end integration pipeline verification:
```bash
python tests/verify_pipeline.py
```

---

## 10. Known Limitations & Future Improvements

1. **Intraday Free API Depth**: Free Yahoo Finance data limits 5-minute bars to the preceding 60 calendar days. Future versions can connect to institutional data vendors (e.g., Polygon, Databento, FirstRate Data) for multi-year tick/1-minute depth.
2. **Exchange-Wide Market Breadth**: Current volume analysis uses composite index / ETF volume. Future extensions could incorporate Advance/Decline ratios and McClellan Oscillator metrics.
