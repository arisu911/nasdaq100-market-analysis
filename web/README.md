# NASDAQ-100 Quantitative Behavior Web Terminal (FastAPI + Vanilla Web Stack)

This directory contains the production decoupled web architecture for the **NASDAQ-100 Historical Market Behavior Analytics Terminal**.

It replaces the legacy Streamlit execution model with a high-performance, asynchronous **FastAPI backend** and a reactive **Vanilla HTML5/CSS3/ES6 JavaScript frontend** powered by `Plotly.react()`.

---

## 1. Directory Structure

```
web/
├── main.py                  # Asynchronous FastAPI backend exposing 11 REST API endpoints
├── requirements.txt         # Production web runtime dependencies
├── run_web.bat              # One-click Windows runner (launches Microsoft Edge + Uvicorn)
└── static/
    ├── index.html           # Single-Page Application interface covering all 9 analytical views
    ├── css/
    │   └── dashboard.css    # Dark financial terminal design system (#0e1117 / #161b22)
    └── js/
        └── app.js           # Central state controller, Plotly.react client, and safe table renderer
```

---

## 2. API Endpoints Mounted (`main.py`)

All 9 research modules are served via JSON REST endpoints:

- **Overview & Macro Seasonality:** `GET /api/overview` (Cumulative returns, monthly seasonality, yearly performance, distribution stats)
- **Weekday Seasonality:** `GET /api/weekday` (Monday–Friday win rates, daily range amplitude, monthly/yearly breakdowns)
- **Intraday Profiling (RTH):** `GET /api/intraday` (09:30–16:00 ET regular trading hours progression, volume curves, MYT time buckets)
- **Opening Dynamics & Gaps:** `GET /api/opening` (Cash open gap distributions, gap fill probabilities, 5m/15m/30m/60m opening range share)
- **Overnight vs Regular Session:** `GET /api/overnight` (ETH electronic trading hours vs RTH cash session drift and variance share)
- **Volatility & Regimes:** `GET /api/volatility` (Rolling realized volatility 20d/60d/120d, percentile regime classification)
- **Drawdown & Stress History:** `GET /api/drawdowns` (Underwater drawdown curve, episode durations, recovery paths)
- **Extreme Day Profiles:** `GET /api/extreme-days` (Tail event filtering, MFE vs. MAE scatter paths, top gainers/losers)
- **Cross-Market Correlations:** `GET /api/correlations` (Rolling correlations, OLS regression beta against ^GSPC, SPY, QQQ, TQQQ)
- **Real-Time Market Clock:** `GET /api/market-clock` (Live MYT clock, market open/closed state, US DST status)
- **Configuration & Metadata:** `GET /api/config` (Instruments, periods, weekdays, regimes, resolutions)

---

## 3. Quickstart & Execution

### One-Click Launch (Recommended)
Double-click or execute from terminal:
```bat
.\web\run_web.bat
```
This automatically boots the Uvicorn server on `127.0.0.1:8000` and launches Microsoft Edge.

### Manual Launch via Uvicorn
```bash
cd web
py -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at **`http://127.0.0.1:8000`** (optimized for Microsoft Edge and Chromium browsers).
