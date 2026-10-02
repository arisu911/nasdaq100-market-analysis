/**
 * NASDAQ-100 Quantitative Behavior Terminal — Frontend Application Logic
 * Implements centralized state management, Plotly.react() rendering,
 * real-time MYT market clock, safe table rendering, and decoupled navigation.
 */

(() => {
  'use strict';

  // ---------------------------------------------------------------------------
  // Application State
  // ---------------------------------------------------------------------------
  const state = {
    currentView: 'overview',
    instrument: '^NDX',
    period: '10Y',
    weekday: 'All Days',
    regime: 'All Regimes',
    intradayResolution: '5m',
    intradayMetric: 'mean_abs_return_pct',
    volLowPct: 25.0,
    volHighPct: 75.0,
    volMetric: 'daily_range_pct',
    extremePercentile: 95.0,
    benchmark: '^GSPC',
    isLoading: false,
    cachedViewData: {},
  };

  const VIEW_TITLES = {
    overview: {
      title: 'Market Behavior Overview',
      subtitle: 'Descriptive statistical summary and session structure in Asia/Kuala_Lumpur (MYT)',
    },
    weekday: {
      title: 'Day-of-Week & Calendar Seasonality',
      subtitle: 'Comparative performance, volatility, and range across Monday–Friday, monthly and yearly cycles',
    },
    intraday: {
      title: 'Intraday Time-of-Day Profiles (RTH)',
      subtitle: 'Multi-resolution regular trading hours (09:30–16:00 ET) progression in Asia/Kuala_Lumpur (MYT)',
    },
    opening: {
      title: 'Opening Dynamics & Gap Behavior',
      subtitle: 'Cash open gap distributions, gap fill probabilities, and opening range expansion (15m/30m/60m)',
    },
    overnight: {
      title: 'Overnight vs Regular Session Dynamics',
      subtitle: 'Attribution analysis: Electronic trading hours (ETH) drift vs Regular session (RTH) return',
    },
    volatility: {
      title: 'Volatility Dynamics & Regime Shifts',
      subtitle: 'Rolling realized volatility horizons (20d/60d/120d) and percentile-based regime segmentation',
    },
    drawdowns: {
      title: 'Drawdown Dynamics & Stress History',
      subtitle: 'Historical peak-to-trough drawdowns, underwater duration, and recovery trajectories',
    },
    'extreme-days': {
      title: 'Extreme Historical Days & Tail Events',
      subtitle: 'Outlier return sessions, gap days, and path excursion dynamics (MFE vs MAE)',
    },
    correlations: {
      title: 'Cross-Market Correlation Dynamics',
      subtitle: 'NASDAQ-100 / QQQ rolling correlation and beta regression against benchmark instruments',
    },
  };

  // ---------------------------------------------------------------------------
  // DOM Elements
  // ---------------------------------------------------------------------------
  const DOM = {
    clockDigits: document.getElementById('clock-digits'),
    statusDot: document.getElementById('status-dot'),
    statusText: document.getElementById('status-text'),
    sessionHours: document.getElementById('session-hours'),
    dstInfo: document.getElementById('dst-info'),
    pageTitle: document.getElementById('page-title'),
    pageSubtitle: document.getElementById('page-subtitle'),
    filterInstrument: document.getElementById('filter-instrument'),
    filterPeriod: document.getElementById('filter-period'),
    filterWeekday: document.getElementById('filter-weekday'),
    filterRegime: document.getElementById('filter-regime'),
    btnRefresh: document.getElementById('btn-refresh'),
    loadingOverlay: document.getElementById('loading-overlay'),
    loadingText: document.getElementById('loading-text'),
    navItems: document.querySelectorAll('.nav-item'),
    viewPanels: document.querySelectorAll('.view-panel'),
    sidebar: document.getElementById('sidebar'),
    sidebarToggle: document.getElementById('sidebar-toggle'),
  };

  // ---------------------------------------------------------------------------
  // Utility Functions
  // ---------------------------------------------------------------------------
  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function showLoading(message = 'FETCHING QUANTITATIVE DATA...') {
    state.isLoading = true;
    if (DOM.loadingText) DOM.loadingText.textContent = message;
    if (DOM.loadingOverlay) DOM.loadingOverlay.classList.add('active');
    if (DOM.btnRefresh) DOM.btnRefresh.classList.add('loading');
  }

  function hideLoading() {
    state.isLoading = false;
    if (DOM.loadingOverlay) DOM.loadingOverlay.classList.remove('active');
    if (DOM.btnRefresh) DOM.btnRefresh.classList.remove('loading');
  }

  // ---------------------------------------------------------------------------
  // Safe Table Renderer
  // Conforms strictly to constraint:
  // Inject cell values with key fallbacks:
  // row.label || row.series || row.metric_label || Object.values(row)[0]
  // ---------------------------------------------------------------------------
  function renderSafeTable(containerId, records) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!records || !Array.isArray(records) || records.length === 0) {
      container.innerHTML = '<div class="empty-table-msg">No data available for current selection.</div>';
      return;
    }

    const headers = Object.keys(records[0]);
    let html = '<div class="table-responsive"><table class="data-table"><thead><tr>';

    headers.forEach((h) => {
      html += `<th>${escapeHtml(h)}</th>`;
    });

    html += '</tr></thead><tbody>';

    records.forEach((row) => {
      html += '<tr>';
      headers.forEach((h) => {
        let val = row[h];
        if (val === undefined || val === null || val === '') {
          val = row.label || row.series || row.metric_label || Object.values(row)[0] || '—';
        }
        let strVal = String(val);
        let cellClass = '';
        if (strVal.startsWith('+')) cellClass = 'style="color: #3fb950;"';
        else if (strVal.startsWith('-') && !strVal.startsWith('--') && !strVal.startsWith('—')) cellClass = 'style="color: #f85149;"';

        html += `<td ${cellClass}>${escapeHtml(strVal)}</td>`;
      });
      html += '</tr>';
    });

    html += '</tbody></table></div>';
    container.innerHTML = html;
  }

  // ---------------------------------------------------------------------------
  // Plotly React Renderer
  // Flicker-free chart re-rendering with enforced terminal theme overrides
  // ---------------------------------------------------------------------------
  function renderPlotlyChart(containerId, figureObj) {
    const el = document.getElementById(containerId);
    if (!el || !figureObj || !window.Plotly) return;

    const data = figureObj.data || [];
    const layout = figureObj.layout || {};

    // Enforce dark terminal palette
    layout.paper_bgcolor = '#161b22';
    layout.plot_bgcolor = '#0e1117';
    layout.autosize = true;

    if (!layout.font) layout.font = {};
    layout.font.family = 'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif';
    layout.font.color = '#c9d1d9';
    layout.font.size = 11.5;

    const config = {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      toImageButtonOptions: {
        format: 'png',
        filename: `nasdaq100_${containerId}`,
        height: 600,
        width: 1000,
        scale: 2,
      },
    };

    Plotly.react(el, data, layout, config);
  }

  // ---------------------------------------------------------------------------
  // API Fetch Helper
  // ---------------------------------------------------------------------------
  async function fetchApi(endpoint, params = {}) {
    const url = new URL(endpoint, window.location.origin);
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null) {
        url.searchParams.append(k, String(v));
      }
    });

    try {
      const resp = await fetch(url.toString(), {
        headers: { Accept: 'application/json' },
      });
      if (!resp.ok) {
        const errText = await resp.text();
        throw new Error(`API error (${resp.status}): ${errText}`);
      }
      return await resp.json();
    } catch (err) {
      console.error(`Fetch failed for ${endpoint}:`, err);
      throw err;
    }
  }

  // ---------------------------------------------------------------------------
  // Real-Time Market Clock Manager
  // ---------------------------------------------------------------------------
  let currentServerTimeStr = null;
  let serverSyncDate = null;

  async function syncMarketClock() {
    try {
      const clock = await fetchApi('/api/market-clock');
      currentServerTimeStr = clock.now_myt;
      serverSyncDate = new Date();

      if (DOM.clockDigits) DOM.clockDigits.textContent = clock.now_myt;
      if (DOM.statusText) DOM.statusText.textContent = clock.status_text;
      if (DOM.statusDot) {
        DOM.statusDot.style.backgroundColor = clock.status_color || '#10b981';
        DOM.statusDot.style.boxShadow = `0 0 8px ${clock.status_color || '#10b981'}`;
      }
      if (DOM.sessionHours) {
        DOM.sessionHours.textContent = `Regular Session: ${clock.session_label}`;
      }
      if (DOM.dstInfo) {
        DOM.dstInfo.textContent = `US: ${clock.tz_abbr} (${clock.dst_description})`;
      }
    } catch (e) {
      console.warn('Clock sync warning:', e);
    }
  }

  function startLocalClockTick() {
    setInterval(() => {
      if (!serverSyncDate || !currentServerTimeStr) return;
      const elapsedMs = Date.now() - serverSyncDate.getTime();
      const [h, m, s] = currentServerTimeStr.split(':').map(Number);
      const totalSec = h * 3600 + m * 60 + s + Math.floor(elapsedMs / 1000);
      const nh = Math.floor((totalSec / 3600) % 24).toString().padStart(2, '0');
      const nm = Math.floor((totalSec / 60) % 60).toString().padStart(2, '0');
      const ns = (totalSec % 60).toString().padStart(2, '0');
      if (DOM.clockDigits) DOM.clockDigits.textContent = `${nh}:${nm}:${ns}`;
    }, 1000);

    // Sync from server every 60s
    setInterval(syncMarketClock, 60000);
  }

  // ---------------------------------------------------------------------------
  // View Renderers
  // ---------------------------------------------------------------------------

  // 1. Overview View
  async function loadOverviewView(forceRefresh = false) {
    showLoading('COMPUTING MACRO OVERVIEW & SEASONALITY...');
    try {
      const data = await fetchApi('/api/overview', {
        instrument: state.instrument,
        period: state.period,
        weekday: state.weekday,
        regime: state.regime,
        force_refresh: forceRefresh,
      });

      // KPIs
      document.getElementById('kpi-overview-sessions').textContent = data.kpis.trading_sessions;
      document.getElementById('kpi-overview-mean-ret').textContent = data.kpis.mean_daily_return;
      document.getElementById('kpi-overview-vol').textContent = data.kpis.annualized_volatility;
      document.getElementById('kpi-overview-range').textContent = data.kpis.avg_daily_range;
      document.getElementById('kpi-overview-win-rate').textContent = data.kpis.positive_sessions_pct;
      document.getElementById('kpi-overview-max-dd').textContent = data.kpis.max_drawdown;

      // Charts
      renderPlotlyChart('chart-overview-cum', data.charts.cumulative);
      renderPlotlyChart('chart-overview-dist', data.charts.distribution);
      renderPlotlyChart('chart-overview-month', data.charts.monthly);
      renderPlotlyChart('chart-overview-year', data.charts.yearly);

      // Tables
      renderSafeTable('table-overview-freq', data.tables.historical_frequencies);
      renderSafeTable('table-overview-dist', data.tables.distribution_summary);
      renderSafeTable('table-overview-audit', data.tables.audit_report);
    } catch (err) {
      alert(`Failed to load Overview data: ${err.message}`);
    } finally {
      hideLoading();
    }
  }

  // 2. Weekday View
  async function loadWeekdayView(forceRefresh = false) {
    showLoading('COMPUTING WEEKDAY & CALENDAR SEASONALITY...');
    try {
      const data = await fetchApi('/api/weekday', {
        instrument: state.instrument,
        period: state.period,
        regime: state.regime,
        force_refresh: forceRefresh,
      });

      renderPlotlyChart('chart-weekday-comp', data.charts.weekday_comparison);
      renderPlotlyChart('chart-weekday-range', data.charts.weekday_range_vol);
      renderPlotlyChart('chart-weekday-month', data.charts.monthly_seasonality);
      renderPlotlyChart('chart-weekday-year', data.charts.weekday_box);

      renderSafeTable('table-weekday-summary', data.tables.weekday_summary);
      renderSafeTable('table-weekday-month', data.tables.monthly_summary);
      renderSafeTable('table-weekday-year', data.tables.yearly_summary);
    } catch (err) {
      alert(`Failed to load Weekday data: ${err.message}`);
    } finally {
      hideLoading();
    }
  }

  // 3. Intraday View
  async function loadIntradayView(forceRefresh = false) {
    showLoading('ANALYZING INTRADAY MICROSTRUCTURE...');
    try {
      const data = await fetchApi('/api/intraday', {
        instrument: state.instrument,
        resolution: state.intradayResolution,
        metric: state.intradayMetric,
        force_refresh: forceRefresh,
      });

      const titleEl = document.getElementById('intraday-prof-title');
      if (titleEl) {
        titleEl.textContent = `📈 Intraday ${state.intradayResolution} Profile across Regular Trading Hours (MYT)`;
      }

      renderPlotlyChart('chart-intraday-prof', data.charts.profile);
      renderPlotlyChart('chart-intraday-vol', data.charts.volatility);
      renderPlotlyChart('chart-intraday-scatter', data.charts.volume_scatter);

      renderSafeTable('table-intraday-buckets', data.tables.time_buckets);
    } catch (err) {
      alert(`Failed to load Intraday data: ${err.message}`);
    } finally {
      hideLoading();
    }
  }

  // 4. Opening View
  async function loadOpeningView(forceRefresh = false) {
    showLoading('EVALUATING CASH OPEN & GAP EXPANSION...');
    try {
      const data = await fetchApi('/api/opening', {
        instrument: state.instrument,
        period: state.period,
        weekday: state.weekday,
        regime: state.regime,
        force_refresh: forceRefresh,
      });

      // KPIs
      document.getElementById('kpi-opening-mean').textContent = data.kpis.mean_gap;
      document.getElementById('kpi-opening-median').textContent = data.kpis.median_gap;
      document.getElementById('kpi-opening-up').textContent = data.kpis.up_gaps_pct;
      document.getElementById('kpi-opening-down').textContent = data.kpis.down_gaps_pct;
      document.getElementById('kpi-opening-fill-rate').textContent = data.kpis.gap_fill_rate;
      document.getElementById('kpi-opening-fill-time').textContent = data.kpis.median_time_to_fill || 'N/A';

      // Charts
      renderPlotlyChart('chart-opening-gap', data.charts.gap_distribution);
      if (data.charts.time_to_fill) {
        renderPlotlyChart('chart-opening-fill', data.charts.time_to_fill);
      } else {
        const el = document.getElementById('chart-opening-fill');
        if (el) el.innerHTML = '<div class="empty-table-msg">Intraday gap-fill distribution requires active intraday dataset.</div>';
      }

      if (data.charts.opening_range_share) {
        renderPlotlyChart('chart-opening-share', data.charts.opening_range_share);
      }

      // Tables
      renderSafeTable('table-opening-stats', data.tables.gap_statistics);
      renderSafeTable('table-opening-windows', data.tables.opening_windows);
    } catch (err) {
      alert(`Failed to load Opening data: ${err.message}`);
    } finally {
      hideLoading();
    }
  }

  // 5. Overnight View
  async function loadOvernightView(forceRefresh = false) {
    showLoading('DECOMPOSING OVERNIGHT VS REGULAR SESSION...');
    try {
      const data = await fetchApi('/api/overnight', {
        instrument: state.instrument,
        period: state.period,
        weekday: state.weekday,
        regime: state.regime,
        force_refresh: forceRefresh,
      });

      // KPIs
      document.getElementById('kpi-overnight-mean').textContent = data.kpis.overnight_mean;
      document.getElementById('kpi-overnight-vol').textContent = `Vol: ${data.kpis.overnight_vol}`;
      document.getElementById('kpi-overnight-reg-mean').textContent = data.kpis.regular_mean;
      document.getElementById('kpi-overnight-reg-vol').textContent = `Vol: ${data.kpis.regular_vol}`;
      document.getElementById('kpi-overnight-var-share').textContent = data.kpis.overnight_var_share;
      document.getElementById('kpi-overnight-pos-drift').textContent = data.kpis.overnight_pos_pct;

      // Charts
      renderPlotlyChart('chart-overnight-cum', data.charts.cumulative);
      renderPlotlyChart('chart-overnight-dist', data.charts.overnight_dist);
      renderPlotlyChart('chart-overnight-reg', data.charts.regular_dist);

      // Table
      renderSafeTable('table-overnight-attrib', data.tables.attribution);
    } catch (err) {
      alert(`Failed to load Overnight data: ${err.message}`);
    } finally {
      hideLoading();
    }
  }

  // 6. Volatility View
  async function loadVolatilityView(forceRefresh = false) {
    showLoading('MODELING VOLATILITY DYNAMICS & REGIMES...');
    try {
      const data = await fetchApi('/api/volatility', {
        instrument: state.instrument,
        period: state.period,
        low_pct: state.volLowPct,
        high_pct: state.volHighPct,
        metric: state.volMetric,
        force_refresh: forceRefresh,
      });

      renderPlotlyChart('chart-vol-rolling', data.charts.rolling_vol);
      renderPlotlyChart('chart-vol-regimes', data.charts.regime_comparison);
      if (data.charts.session_volatility) {
        renderPlotlyChart('chart-vol-session', data.charts.session_volatility);
      }

      renderSafeTable('table-vol-regimes', data.tables.regime_summary);
    } catch (err) {
      alert(`Failed to load Volatility data: ${err.message}`);
    } finally {
      hideLoading();
    }
  }

  // 7. Drawdowns View
  async function loadDrawdownsView(forceRefresh = false) {
    showLoading('ANALYZING STRESS & DRAWDOWN EPISODES...');
    try {
      const data = await fetchApi('/api/drawdowns', {
        instrument: state.instrument,
        period: state.period,
        weekday: state.weekday,
        regime: state.regime,
        force_refresh: forceRefresh,
      });

      document.getElementById('kpi-dd-max').textContent = data.kpis.max_drawdown;
      document.getElementById('kpi-dd-curr').textContent = data.kpis.current_drawdown;
      document.getElementById('kpi-dd-avg').textContent = data.kpis.avg_in_drawdown;
      document.getElementById('kpi-dd-sessions').textContent = data.kpis.total_sessions;

      renderPlotlyChart('chart-dd-underwater', data.charts.underwater);
      renderPlotlyChart('chart-dd-bar', data.charts.episodes_bar);

      renderSafeTable('table-dd-episodes', data.tables.episodes);
    } catch (err) {
      alert(`Failed to load Drawdown data: ${err.message}`);
    } finally {
      hideLoading();
    }
  }

  // 8. Extreme Days View
  async function loadExtremeDaysView(forceRefresh = false) {
    showLoading('ISOLATING TAIL EVENTS & EXCURSION PATHS...');
    try {
      const data = await fetchApi('/api/extreme-days', {
        instrument: state.instrument,
        period: state.period,
        percentile: state.extremePercentile,
        force_refresh: forceRefresh,
      });

      const infoEl = document.getElementById('extreme-cutoff-info');
      if (infoEl) {
        infoEl.textContent = `Cutoff: |Return| ≥ ${data.cutoff_info.cutoff_abs_return} (${data.cutoff_info.sessions_count} sessions of ${data.cutoff_info.total_sessions})`;
      }

      renderPlotlyChart('chart-extreme-mfe', data.charts.mfe_mae);
      renderPlotlyChart('chart-extreme-dist', data.charts.tail_distribution);

      renderSafeTable('table-extreme-gainers', data.tables.top_gainers);
      renderSafeTable('table-extreme-losers', data.tables.top_losers);
      renderSafeTable('table-extreme-range', data.tables.top_range);
      renderSafeTable('table-extreme-gaps', data.tables.top_gaps);
    } catch (err) {
      alert(`Failed to load Extreme Days data: ${err.message}`);
    } finally {
      hideLoading();
    }
  }

  // 9. Correlations View
  async function loadCorrelationsView(forceRefresh = false) {
    showLoading('ESTIMATING CROSS-MARKET BETA & CORRELATION...');
    try {
      const data = await fetchApi('/api/correlations', {
        instrument: state.instrument,
        benchmark: state.benchmark,
        period: state.period,
        force_refresh: forceRefresh,
      });

      document.getElementById('kpi-corr-r').textContent = data.kpis.correlation;
      document.getElementById('kpi-corr-beta').textContent = data.kpis.beta;
      document.getElementById('kpi-corr-r2').textContent = data.kpis.r_squared;
      document.getElementById('kpi-corr-sessions').textContent = data.kpis.sessions_count;

      const titleEl = document.getElementById('corr-rolling-title');
      if (titleEl) {
        titleEl.textContent = `🔗 Rolling 60-Session Return Correlation (${state.instrument} vs ${state.benchmark})`;
      }

      renderPlotlyChart('chart-corr-rolling', data.charts.rolling_correlation);
      renderPlotlyChart('chart-corr-scatter', data.charts.scatter);
      renderPlotlyChart('chart-corr-regime', data.charts.regime_correlation);

      renderSafeTable('table-corr-regimes', data.tables.regime_summary);
    } catch (err) {
      alert(`Failed to load Correlations data: ${err.message}`);
    } finally {
      hideLoading();
    }
  }

  // ---------------------------------------------------------------------------
  // Centralized View Dispatcher
  // ---------------------------------------------------------------------------
  function loadActiveView(forceRefresh = false) {
    switch (state.currentView) {
      case 'overview':
        return loadOverviewView(forceRefresh);
      case 'weekday':
        return loadWeekdayView(forceRefresh);
      case 'intraday':
        return loadIntradayView(forceRefresh);
      case 'opening':
        return loadOpeningView(forceRefresh);
      case 'overnight':
        return loadOvernightView(forceRefresh);
      case 'volatility':
        return loadVolatilityView(forceRefresh);
      case 'drawdowns':
        return loadDrawdownsView(forceRefresh);
      case 'extreme-days':
        return loadExtremeDaysView(forceRefresh);
      case 'correlations':
        return loadCorrelationsView(forceRefresh);
      default:
        return loadOverviewView(forceRefresh);
    }
  }

  // ---------------------------------------------------------------------------
  // Navigation Router & Event Listeners
  // ---------------------------------------------------------------------------
  function setupNavigation() {
    DOM.navItems.forEach((item) => {
      item.addEventListener('click', () => {
        const viewId = item.getAttribute('data-view');
        if (viewId === state.currentView) return;

        // Switch active nav item
        DOM.navItems.forEach((n) => n.classList.remove('active'));
        item.classList.add('active');

        // Switch active panel
        DOM.viewPanels.forEach((p) => p.classList.remove('active'));
        const activePanel = document.getElementById(`view-${viewId}`);
        if (activePanel) activePanel.classList.add('active');

        // Update topbar headers
        state.currentView = viewId;
        const meta = VIEW_TITLES[viewId] || { title: viewId, subtitle: '' };
        if (DOM.pageTitle) DOM.pageTitle.textContent = meta.title;
        if (DOM.pageSubtitle) DOM.pageSubtitle.textContent = meta.subtitle;

        // Trigger view load
        loadActiveView();
      });
    });

    // Subtabs switcher (e.g. inside Weekday view)
    document.querySelectorAll('.view-tab-btn').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const parent = btn.closest('.view-panel');
        if (!parent) return;

        parent.querySelectorAll('.view-tab-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');

        const targetSubtab = btn.getAttribute('data-subtab');
        parent.querySelectorAll('.subtab-content').forEach((c) => {
          c.style.display = c.id === `subtab-${targetSubtab}` ? 'block' : 'none';
        });

        // Trigger window resize to adjust Plotly charts
        window.dispatchEvent(new Event('resize'));
      });
    });
  }

  function setupFilterListeners() {
    // Global Instrument
    if (DOM.filterInstrument) {
      DOM.filterInstrument.addEventListener('change', (e) => {
        state.instrument = e.target.value;
        loadActiveView();
      });
    }

    // Global Period
    if (DOM.filterPeriod) {
      DOM.filterPeriod.addEventListener('change', (e) => {
        state.period = e.target.value;
        loadActiveView();
      });
    }

    // Global Weekday
    if (DOM.filterWeekday) {
      DOM.filterWeekday.addEventListener('change', (e) => {
        state.weekday = e.target.value;
        loadActiveView();
      });
    }

    // Global Regime
    if (DOM.filterRegime) {
      DOM.filterRegime.addEventListener('change', (e) => {
        state.regime = e.target.value;
        loadActiveView();
      });
    }

    // Refresh Button
    if (DOM.btnRefresh) {
      DOM.btnRefresh.addEventListener('click', () => {
        loadActiveView(true);
      });
    }

    // Intraday sub-controls
    const ctrlIntraRes = document.getElementById('ctrl-intraday-res');
    if (ctrlIntraRes) {
      ctrlIntraRes.addEventListener('change', (e) => {
        state.intradayResolution = e.target.value;
        if (state.currentView === 'intraday') loadIntradayView();
      });
    }

    const ctrlIntraMetric = document.getElementById('ctrl-intraday-metric');
    if (ctrlIntraMetric) {
      ctrlIntraMetric.addEventListener('change', (e) => {
        state.intradayMetric = e.target.value;
        if (state.currentView === 'intraday') loadIntradayView();
      });
    }

    // Volatility sub-controls
    const ctrlVolLow = document.getElementById('ctrl-vol-low');
    const valVolLow = document.getElementById('val-vol-low');
    if (ctrlVolLow) {
      ctrlVolLow.addEventListener('input', (e) => {
        state.volLowPct = parseFloat(e.target.value);
        if (valVolLow) valVolLow.textContent = `${state.volLowPct}%`;
      });
      ctrlVolLow.addEventListener('change', () => {
        if (state.currentView === 'volatility') loadVolatilityView();
      });
    }

    const ctrlVolHigh = document.getElementById('ctrl-vol-high');
    const valVolHigh = document.getElementById('val-vol-high');
    if (ctrlVolHigh) {
      ctrlVolHigh.addEventListener('input', (e) => {
        state.volHighPct = parseFloat(e.target.value);
        if (valVolHigh) valVolHigh.textContent = `${state.volHighPct}%`;
      });
      ctrlVolHigh.addEventListener('change', () => {
        if (state.currentView === 'volatility') loadVolatilityView();
      });
    }

    const ctrlVolMetric = document.getElementById('ctrl-vol-metric');
    if (ctrlVolMetric) {
      ctrlVolMetric.addEventListener('change', (e) => {
        state.volMetric = e.target.value;
        if (state.currentView === 'volatility') loadVolatilityView();
      });
    }

    // Extreme days sub-control
    const ctrlExtPct = document.getElementById('ctrl-extreme-pct');
    const valExtPct = document.getElementById('val-extreme-pct');
    if (ctrlExtPct) {
      ctrlExtPct.addEventListener('input', (e) => {
        state.extremePercentile = parseFloat(e.target.value);
        if (valExtPct) valExtPct.textContent = `${state.extremePercentile}%`;
      });
      ctrlExtPct.addEventListener('change', () => {
        if (state.currentView === 'extreme-days') loadExtremeDaysView();
      });
    }

    // Correlations benchmark selector
    const ctrlCorrBm = document.getElementById('ctrl-corr-bm');
    if (ctrlCorrBm) {
      ctrlCorrBm.addEventListener('change', (e) => {
        state.benchmark = e.target.value;
        if (state.currentView === 'correlations') loadCorrelationsView();
      });
    }

    // Resize event listener for responsive Plotly charts
    window.addEventListener('resize', () => {
      resizeCharts();
    });
  }

  // ---------------------------------------------------------------------------
  // Responsive Plotly Resize Helper
  // ---------------------------------------------------------------------------
  function resizeCharts() {
    window.dispatchEvent(new Event('resize'));
    const charts = document.querySelectorAll('.view-panel.active .chart-wrapper, .chart-wrapper');
    charts.forEach((c) => {
      if (c && c.data && window.Plotly) {
        try {
          Plotly.Plots.resize(c);
        } catch (err) {
          // ignore unrendered containers
        }
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Sidebar Collapse / Toggle Manager (LocalStorage Persisted)
  // ---------------------------------------------------------------------------
  function setupSidebarToggle() {
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('sidebar-toggle');
    if (!toggleBtn || !sidebar) return;

    // Restore persistent state from localStorage
    const savedState = localStorage.getItem('nasdaq_sidebar_collapsed');
    if (savedState === 'true') {
      sidebar.classList.add('collapsed');
      toggleBtn.classList.add('active');
    }

    toggleBtn.addEventListener('click', () => {
      const isNowCollapsed = sidebar.classList.toggle('collapsed');
      toggleBtn.classList.toggle('active', isNowCollapsed);
      localStorage.setItem('nasdaq_sidebar_collapsed', isNowCollapsed ? 'true' : 'false');

      // Trigger resize immediately and after CSS transition duration
      resizeCharts();
      setTimeout(resizeCharts, 260);
    });
  }

  // ---------------------------------------------------------------------------
  // Application Bootstrap
  // ---------------------------------------------------------------------------
  async function init() {
    setupSidebarToggle();
    setupNavigation();
    setupFilterListeners();
    await syncMarketClock();
    startLocalClockTick();
    await loadActiveView();
  }

  // Start when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
