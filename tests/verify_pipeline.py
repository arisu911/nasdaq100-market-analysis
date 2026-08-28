"""
End-to-end integration and pipeline verification script for NASDAQ-100 analysis.
Tests data downloading, cleaning, calculations, statistics, regimes, and chart generation.
"""
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_loader import (
    load_daily_data,
    load_intraday_data,
    load_benchmark_data,
    get_data_coverage_info,
)
from src.data_cleaner import clean_market_data
from src.calculations import (
    compute_daily_returns_and_sessions,
    compute_cumulative_performance,
    compute_drawdown_series,
    compute_drawdown_episodes,
    compute_rolling_metrics,
    compute_opening_window_metrics,
    compute_gap_fill_intraday,
)
from src.statistics import (
    compute_distribution_summary,
    compute_historical_frequencies,
    compute_weekday_statistics,
    compute_monthly_statistics,
    compute_yearly_statistics,
    compute_intraday_time_bucket_statistics,
)
from src.regimes import (
    classify_volatility_regimes,
    compute_regime_comparison_statistics,
)
from src.timezone_utils import (
    get_current_market_clock_info,
    get_myt_session_times,
)
from src.charts import (
    plot_cumulative_performance,
    plot_drawdown_curve,
    plot_intraday_profile,
    plot_weekday_comparison,
)


def run_verification():
    print("=== 1. Testing Timezone & Clock Helpers ===")
    clock = get_current_market_clock_info()
    print(f"Current MYT: {clock['now_myt']}, Status: {clock['status_text']}")
    session_est = get_myt_session_times("2024-01-15")
    session_edt = get_myt_session_times("2024-06-15")
    print(f"EST Session: {session_est['myt_session_label']}, EDT Session: {session_edt['myt_session_label']}")

    print("\n=== 2. Testing Daily Data Loader & Cleaning ===")
    raw_daily = load_daily_data(ticker="^NDX", period="10y")
    clean_daily, audit_daily = clean_market_data(raw_daily, interval="1d")
    print(f"Daily Clean Rows: {len(clean_daily)}, Quality Score: {audit_daily['data_quality_score']}%")

    print("\n=== 3. Testing Calculations & Returns ===")
    calc_daily = compute_daily_returns_and_sessions(clean_daily)
    print(f"Daily Calculations Columns: {calc_daily.columns.tolist()[:10]}")
    cum_perf = compute_cumulative_performance(calc_daily)
    print(f"Cum Total End: {cum_perf['cum_total'].iloc[-1]*100:.2f}%")

    print("\n=== 4. Testing Drawdowns & Episodes ===")
    wealth, dd_series, max_dd = compute_drawdown_series(calc_daily["total_return"])
    episodes = compute_drawdown_episodes(dd_series, top_n=5)
    print(f"Max Drawdown: {max_dd:.2f}%, Top Episodes Count: {len(episodes)}")

    print("\n=== 5. Testing Statistics & Aggregations ===")
    freqs = compute_historical_frequencies(calc_daily)
    weekdays = compute_weekday_statistics(calc_daily)
    months = compute_monthly_statistics(calc_daily)
    years = compute_yearly_statistics(calc_daily)
    print(f"Freqs: {len(freqs)} items, Weekdays: {len(weekdays)} rows, Years: {len(years)} rows")

    print("\n=== 6. Testing Volatility Regimes ===")
    classified, thresh = classify_volatility_regimes(calc_daily)
    reg_stats = compute_regime_comparison_statistics(classified)
    print(f"Regime Stats Count: {len(reg_stats)}")

    print("\n=== 7. Testing Intraday 5m Loader & Resampling ===")
    raw_intra = load_intraday_data(ticker="^NDX", interval="5m")
    clean_intra, audit_intra = clean_market_data(raw_intra, interval="5m")
    intra_stats = compute_intraday_time_bucket_statistics(clean_intra)
    print(f"Intraday Bars: {len(clean_intra)}, Time Buckets in MYT: {len(intra_stats)}")

    print("\n=== 8. Testing Opening Windows & Gap Fills ===")
    open_win = compute_opening_window_metrics(clean_intra, calc_daily)
    gap_fills = compute_gap_fill_intraday(clean_intra, calc_daily)
    print(f"Opening Windows Days: {len(open_win)}, Gap Fills Analyzed: {len(gap_fills)}")

    print("\n=== 9. Testing Benchmark Correlation ===")
    spx_raw = load_benchmark_data(ticker="^GSPC", period="10y")
    spx_calc = compute_daily_returns_and_sessions(spx_raw)
    corr = calc_daily["total_return_pct"].corr(spx_calc["total_return_pct"])
    print(f"NASDAQ-100 vs S&P 500 Daily Return Correlation: {corr:.4f}")

    print("\n=== 10. Testing Chart Generation ===")
    fig1 = plot_cumulative_performance(cum_perf)
    fig2 = plot_drawdown_curve(dd_series)
    fig3 = plot_intraday_profile(intra_stats, metric="mean_return_pct")
    fig4 = plot_weekday_comparison(weekdays)
    print("All Plotly figures constructed successfully.")

    print("\n[SUCCESS] PIPELINE FULLY VERIFIED WITH ZERO ERRORS!")


if __name__ == "__main__":
    run_verification()
