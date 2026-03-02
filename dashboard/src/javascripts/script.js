const API = '';
const DARK = { background: '#161b22', text: '#c9d1d9', grid: '#21262d' };

/* -- Bootstrap tooltips init --------------------------------------- */
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => new bootstrap.Tooltip(el));
});

/* -- helpers ------------------------------------------------------- */
function $(id) { return document.getElementById(id); }
function fmt$(v) { return '$' + v.toLocaleString('en-US', {minimumFractionDigits: 0}); }
function fmtPct(v, dec=2) { return (v >= 0 ? '+' : '') + v.toFixed(dec) + '%'; }
function fmtTime(iso) {
    if (!iso) return '—';
    try { return new Date(iso).toLocaleString('it-IT', {day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'}); }
    catch { return iso; }
}

/* -- Dashboard Candlestick chart ----------------------------------- */
const candleChart = LightweightCharts.createChart($('chart-candles'), {
    layout: { background: { color: DARK.background }, textColor: DARK.text },
    grid: { vertLines: { color: DARK.grid }, horzLines: { color: DARK.grid } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: DARK.grid },
    timeScale: { borderColor: DARK.grid, timeVisible: false },
});
const candleSeries = candleChart.addCandlestickSeries({
    upColor: '#26a69a', downColor: '#ef5350',
    borderUpColor: '#26a69a', borderDownColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
});
const smaLine = candleChart.addLineSeries({ color: '#2196F3', lineWidth: 1, title: 'SMA 50' });
const emaLine = candleChart.addLineSeries({ color: '#FF9800', lineWidth: 1, title: 'EMA 20' });

/* -- Backtest Candlestick chart (with signal markers) — lazy init -- */
let btCandleChart = null, btCandleSeries = null, btSmaLine = null, btEmaLine = null;
let _btCandles = [];
function initBtCandleChart() {
    if (btCandleChart) return;
    btCandleChart = LightweightCharts.createChart($('chart-bt-candles'), {
        layout: { background: { color: DARK.background }, textColor: DARK.text },
        grid: { vertLines: { color: DARK.grid }, horzLines: { color: DARK.grid } },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: DARK.grid },
        timeScale: { borderColor: DARK.grid, timeVisible: false },
    });
    btCandleSeries = btCandleChart.addCandlestickSeries({
        upColor: '#26a69a', downColor: '#ef5350',
        borderUpColor: '#26a69a', borderDownColor: '#ef5350',
        wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    });
    btSmaLine = btCandleChart.addLineSeries({ color: '#2196F3', lineWidth: 1, title: 'SMA 50' });
    btEmaLine = btCandleChart.addLineSeries({ color: '#FF9800', lineWidth: 1, title: 'EMA 20' });
}

/* -- Backtest equity chart — lazy init ----------------------------- */
let eqChart = null, eqSeries = null, eqBaseline = null;
function initBacktestEquityChart() {
    if (eqChart) return;
    eqChart = LightweightCharts.createChart($('chart-equity'), {
        layout: { background: { color: DARK.background }, textColor: DARK.text },
        grid: { vertLines: { color: DARK.grid }, horzLines: { color: DARK.grid } },
        rightPriceScale: { borderColor: DARK.grid },
        timeScale: { borderColor: DARK.grid, timeVisible: false },
    });
    eqSeries = eqChart.addAreaSeries({
        topColor: 'rgba(38,166,154,0.4)', bottomColor: 'rgba(38,166,154,0.0)',
        lineColor: '#26a69a', lineWidth: 2,
    });
    eqBaseline = eqChart.addLineSeries({ color: '#ef5350', lineWidth: 1, lineStyle: 2, title: 'Break Even' });
}

/* -- Live/Real equity chart — lazy init ---------------------------- */
let eqLiveChart = null, eqLiveSeries = null, eqLiveBaseline = null;
function initLiveEquityChart() {
    if (eqLiveChart) return;
    eqLiveChart = LightweightCharts.createChart($('chart-equity-live'), {
        layout: { background: { color: DARK.background }, textColor: DARK.text },
        grid: { vertLines: { color: DARK.grid }, horzLines: { color: DARK.grid } },
        rightPriceScale: { borderColor: DARK.grid },
        timeScale: { borderColor: DARK.grid, timeVisible: true },
    });
    eqLiveSeries = eqLiveChart.addAreaSeries({
        topColor: 'rgba(240,185,11,0.4)', bottomColor: 'rgba(240,185,11,0.0)',
        lineColor: '#f0b90b', lineWidth: 2,
    });
    eqLiveBaseline = eqLiveChart.addLineSeries({ color: '#ef5350', lineWidth: 1, lineStyle: 2, title: 'Initial' });
}

let _allCandles = [];

/* -- helper: fetch with error detection ---------------------------- */
async function apiFetch(url, opts) {
    const r = await fetch(url, opts);

    if (!r.ok) {
        const body = await r.json().catch(() => ({ error: r.statusText }));
        throw new Error(body.error || `HTTP ${r.status}`);
    }
    return r.json();
}

/* ======================================================================
   DASHBOARD: Load candles, indicators, prediction, live data
   ====================================================================== */
async function loadDashboard() {
    try {
        const [candles, indicators, prediction, liveSignals] = await Promise.all([
            apiFetch(API + '/api/candles?last_n=3000'),
            apiFetch(API + '/api/indicators?last_n=3000'),
            apiFetch(API + '/api/prediction'),
            apiFetch(API + '/api/live/signals'),
        ]);

        _allCandles = candles;
        candleSeries.setData(candles);
        smaLine.setData(indicators.sma50);
        emaLine.setData(indicators.ema20);
        if (liveSignals.length) candleSeries.setMarkers(liveSignals.sort((a,b) => a.time - b.time));
        setRange(365);

        // Prediction card
        const box = $('prediction-box');
        box.classList.remove('long','short');
        if (prediction.action==='LONG') box.classList.add('long');
        else if (prediction.action==='SHORT') box.classList.add('short');
        const badge = prediction.action==='LONG'?'badge-long':prediction.action==='SHORT'?'badge-short':'badge-hold';
        $('pred-action').innerHTML = `<span class="badge ${badge} fs-5">${prediction.action}</span>`;
        $('pred-conf').textContent = (prediction.confidence*100).toFixed(1)+'%';
        $('pred-hold').textContent = (prediction.hold*100).toFixed(1)+'%';
        $('pred-long').textContent = (prediction.long*100).toFixed(1)+'%';
        $('pred-short').textContent = (prediction.short*100).toFixed(1)+'%';
        $('bar-hold').style.width = (prediction.hold*100)+'%';
        $('bar-long').style.width = (prediction.long*100)+'%';
        $('bar-short').style.width = (prediction.short*100)+'%';
        $('pred-price').textContent = fmt$(prediction.last_close);
        $('pred-time').textContent = prediction.last_time;

        $('last-update').textContent = 'Updated: ' + new Date().toLocaleString('it-IT');
        $('pipeline-error').classList.add('d-none');

        loadLiveData();
    } catch (e) {
        console.error('Pipeline load error:', e);
        $('pipeline-error').classList.remove('d-none');
        $('pipeline-error-text').textContent = e.message || 'Unknown error';
        loadLiveData();
    }
}

/* -- Load live trading data ---------------------------------------- */
async function loadLiveData() {
    try {
        const [status, liveTrades, liveEquity] = await Promise.all([
            fetch(API+'/api/live/status').then(r=>r.json()),
            fetch(API+'/api/live/trades?last_n=20').then(r=>r.json()),
            fetch(API+'/api/live/equity').then(r=>r.json()),
        ]);
        applyLiveData(status, liveTrades, liveEquity);
    } catch(e) { console.warn('Live data load error:', e); }
}

/* -- Apply live data to DOM (used by both REST fetch and WebSocket) -- */
function applyLiveData(status, liveTrades, liveEquity) {
    $('scheduler-dot').className = 'status-dot '+(status.scheduler_running?'online':'offline');
    $('scheduler-label').textContent = status.scheduler_running?'Scheduler ON':'Scheduler OFF';
    $('mt5-dot').className = 'status-dot '+(status.mt5_connected?'online':'offline');
    $('mt5-label').textContent = status.mt5_connected?'MT5 ON':'MT5 OFF';

    $('live-next-pred').textContent = fmtTime(status.next_prediction);
    $('live-next-exec').textContent = fmtTime(status.next_execution);
    $('live-open-count').textContent = status.open_trades;
    $('live-pending-count').textContent = status.pending_trades;
    $('live-total-count').textContent = status.total_trades;

    if (status.last_prediction) {
        const lp = status.last_prediction;
        const b2 = lp.action==='LONG'?'badge-long':lp.action==='SHORT'?'badge-short':'badge-hold';
        $('live-last-signal').innerHTML = `<span class="badge ${b2}">${lp.action}</span> ${(lp.confidence*100).toFixed(0)}%`;
    }

    const errBox = $('live-error-box');
    if (status.last_error) { errBox.classList.remove('d-none'); $('live-error-text').textContent = status.last_error; }
    else { errBox.classList.add('d-none'); }

    $('live-trade-count').textContent = liveTrades.length;
    const lb = $('live-trades-body');
    if (!liveTrades.length) {
        lb.innerHTML = '<tr><td colspan="7" class="text-center text-muted py-3">No live trades yet</td></tr>';
    } else {
        lb.innerHTML = liveTrades.map(t => {
            const d = (t.exec_time||t.signal_time||'').split('T')[0];
            const sb = t.status==='open'?'badge-open':t.status==='pending'?'badge-pending':t.status==='closed'?'badge-closed':'badge-cancelled';
            const pnl = t.pnl_pct!==null?fmtPct(t.pnl_pct*100,2):'—';
            const pc = t.pnl_pct>0?'text-success':t.pnl_pct<0?'text-danger':'';
            const canClose = t.status==='open'||t.status==='pending';
            const closeBtn = canClose
                ? `<button class="btn btn-sm btn-outline-danger p-0 px-1" onclick="closePosition('${t.id||t.mt5_ticket}')" title="Close position"><i class="bi bi-trash"></i></button>`
                : '';
            return `<tr>
                <td class="text-nowrap">${d}</td>
                <td><span class="badge ${t.direction==='LONG'?'badge-long':'badge-short'}">${t.direction}</span></td>
                <td>${t.entry_price||'—'}</td>
                <td>${t.tp?t.tp+' / '+t.sl:'—'}</td>
                <td><span class="badge ${sb}">${t.status}</span></td>
                <td class="${pc} fw-bold">${pnl}</td>
                <td>${closeBtn}</td>
            </tr>`;
        }).join('');
    }

    if (liveEquity.length > 0) {
        initLiveEquityChart();
        eqLiveSeries.setData(liveEquity);
        if (liveEquity.length > 1) {
            const bv = liveEquity[0].value;
            eqLiveBaseline.setData([{time:liveEquity[0].time,value:bv},{time:liveEquity[liveEquity.length-1].time,value:bv}]);
        }
        eqLiveChart.timeScale().fitContent();
    }
}

/* ======================================================================
   BACKTEST: On-demand from the Backtest tab
   ====================================================================== */

/* Load default params into the form on startup */
async function loadBacktestDefaults() {
    try {
        const d = await apiFetch(API + '/api/bt/defaults');
        $('bt-backtest-window').value = d.backtest_window;
        $('bt-predict-window').value = d.predict_window;
        $('bt-initial-capital').value = d.initial_capital;
        $('bt-lookahead').value = d.lookahead;
        $('bt-atr-mult').value = d.atr_mult;
        $('bt-threshold').value = d.threshold;
        $('bt-model-type').value = d.model_type;
        // Confidence tiers
        if (d.confidence_thresholds) {
            $('bt-conf-low-max').value = d.confidence_thresholds.low_max;
            $('bt-conf-avg-max').value = d.confidence_thresholds.avg_max;
        }
        if (d.position_sizes) {
            $('bt-ps-low').value = +(d.position_sizes.low * 100).toFixed(4);
            $('bt-ps-avg').value = +(d.position_sizes.avg * 100).toFixed(4);
            $('bt-ps-high').value = +(d.position_sizes.high * 100).toFixed(4);
        }
    } catch(e) { console.warn('Could not load backtest defaults:', e); }
}

/* Run backtest with form parameters */
async function runBacktest() {
    const btn = $('btn-run-backtest');
    const progressBox = $('bt-progress-box');
    const errorBox = $('bt-error-box');

    // Gather params from form
    const params = {
        backtest_window: parseInt($('bt-backtest-window').value),
        predict_window: parseInt($('bt-predict-window').value),
        initial_capital: parseFloat($('bt-initial-capital').value),
        lookahead: parseInt($('bt-lookahead').value),
        atr_mult: parseFloat($('bt-atr-mult').value),
        threshold: parseFloat($('bt-threshold').value),
        model_type: $('bt-model-type').value,
        confidence_thresholds: {
            low_max: parseFloat($('bt-conf-low-max').value),
            avg_max: parseFloat($('bt-conf-avg-max').value),
        },
        position_sizes: {
            low: parseFloat($('bt-ps-low').value) / 100,
            avg: parseFloat($('bt-ps-avg').value) / 100,
            high: parseFloat($('bt-ps-high').value) / 100,
        },
    };

    // UI: show progress
    btn.disabled = true;
    progressBox.classList.remove('d-none');
    errorBox.classList.add('d-none');
    $('bt-progress-text').textContent = 'Running backtest... this may take a few minutes.';

    try {
        // POST to run backtest
        const result = await apiFetch(API + '/api/bt/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });

        // Fetch results + candle data for the backtest chart
        const [equity, trades, stats, signals, candles, indicators] = await Promise.all([
            apiFetch(API + '/api/bt/equity'),
            apiFetch(API + '/api/bt/trades'),
            apiFetch(API + '/api/bt/stats'),
            apiFetch(API + '/api/bt/signals'),
            apiFetch(API + '/api/candles?last_n=3000'),
            apiFetch(API + '/api/indicators?last_n=3000'),
        ]);

        // Show results section
        $('bt-results').classList.remove('d-none');
        $('bt-placeholder').classList.add('d-none');

        // Stats cards
        $('st-equity').textContent = fmt$(stats.final_equity);
        $('st-equity').style.color = stats.total_return_pct >= 0 ? '#26a69a' : '#ef5350';
        $('st-return').textContent = fmtPct(stats.total_return_pct);
        $('st-return').style.color = stats.total_return_pct >= 0 ? '#26a69a' : '#ef5350';
        $('st-trades').textContent = `${stats.wins}W / ${stats.losses}L (${stats.trades})`;
        $('st-hitrate').textContent = stats.hit_rate + '%';
        $('st-hitrate').style.color = stats.hit_rate >= 50 ? '#26a69a' : '#ef5350';
        $('st-dd').textContent = fmtPct(stats.max_drawdown);
        $('st-dd').style.color = '#ef5350';
        $('st-avgret').textContent = fmtPct(stats.avg_return, 3);
        $('st-avgret').style.color = stats.avg_return >= 0 ? '#26a69a' : '#ef5350';

        // Backtest candlestick chart with signal markers
        initBtCandleChart();
        _btCandles = candles;
        btCandleSeries.setData(candles);
        btSmaLine.setData(indicators.sma50);
        btEmaLine.setData(indicators.ema20);
        if (signals.length) btCandleSeries.setMarkers(signals.sort((a,b) => a.time - b.time));
        setBtRange(365);

        // Equity chart
        initBacktestEquityChart();
        eqSeries.setData(equity);
        if (equity.length > 1) {
            const bv = equity[0].value;
            eqBaseline.setData([{time:equity[0].time,value:bv},{time:equity[equity.length-1].time,value:bv}]);
        }
        eqChart.timeScale().fitContent();

        // Trades table
        $('trade-count').textContent = trades.length;
        const tbody = $('trades-body');
        if (!trades.length) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted py-3">No trades</td></tr>';
        } else {
            tbody.innerHTML = trades.slice().reverse().map(t => `
                <tr>
                    <td class="text-nowrap">${t.time.split('T')[0]||t.time.split(' ')[0]}</td>
                    <td><span class="badge ${t.direction==='LONG'?'badge-long':'badge-short'}">${t.direction}</span></td>
                    <td>${t.entry}</td>
                    <td class="text-success">${t.tp}</td>
                    <td class="text-danger">${t.sl}</td>
                    <td class="${t.win?'text-success':'text-danger'} fw-bold">${fmtPct(t.return_pct,1)}</td>
                </tr>
            `).join('');
        }

        // Resize after rendering
        setTimeout(handleResize, 100);

    } catch (e) {
        console.error('Backtest error:', e);
        errorBox.classList.remove('d-none');
        $('bt-error-text').textContent = e.message || 'Unknown error';
    } finally {
        btn.disabled = false;
        progressBox.classList.add('d-none');
    }
}

/* -- Action buttons ------------------------------------------------ */
async function runPredictNow() {
    if (!confirm('Execute prediction now?')) return;
    try {
        const r = await fetch(API+'/api/live/predict-now',{method:'POST'});
        const d = await r.json();
        alert('Prediction completed: '+(d.prediction?.action||'see dashboard'));
        loadLiveData();
    } catch(e) { alert('Error: '+e); }
}
async function runExecuteNow() {
    if (!confirm('Execute pending trade now?')) return;
    try {
        await fetch(API+'/api/live/execute-now',{method:'POST'});
        alert('Execution completed');
        loadLiveData();
    } catch(e) { alert('Error: '+e); }
}
async function runAllNow() {
    if (!confirm('Execute Prediction + Execution now?\n(Requires MT5 connection for trade execution)')) return;
    try {
        await fetch(API+'/api/live/run-now',{method:'POST'});
        alert('Prediction + Execution completed!');
        loadLiveData();
    } catch(e) { alert('Error: '+e); }
}
async function recordEquityNow() {
    try {
        const r = await apiFetch(API+'/api/live/record-equity', {method:'POST'});
        loadLiveData();
    } catch(e) { alert('Error: '+e.message); }
}

/* -- Range selectors ----------------------------------------------- */
function setRange(days, evt) {
    if (days === 0) { candleChart.timeScale().fitContent(); }
    else {
        const total = _allCandles.length;
        if (!total) return;
        const from = total > days ? _allCandles[total-days].time : _allCandles[0].time;
        const to = _allCandles[total-1].time;
        candleChart.timeScale().setVisibleRange({ from, to });
    }
    document.querySelectorAll('#tab-dashboard .card-header .btn-outline-secondary').forEach(b => b.classList.remove('active'));
    evt && evt.target && evt.target.classList.add('active');
}

function setBtRange(days, evt) {
    if (!btCandleChart || !_btCandles.length) return;
    if (days === 0) { btCandleChart.timeScale().fitContent(); }
    else {
        const total = _btCandles.length;
        const from = total > days ? _btCandles[total-days].time : _btCandles[0].time;
        const to = _btCandles[total-1].time;
        btCandleChart.timeScale().setVisibleRange({ from, to });
    }
    document.querySelectorAll('#tab-backtest .card-header .btn-outline-secondary').forEach(b => b.classList.remove('active'));
    evt && evt.target && evt.target.classList.add('active');
}

/* -- Resize -------------------------------------------------------- */
function handleResize() {
    candleChart.resize($('chart-candles').clientWidth, 420);
    if (btCandleChart) btCandleChart.resize($('chart-bt-candles').clientWidth, 380);
    if (eqChart) eqChart.resize($('chart-equity').clientWidth, 300);
    if (eqLiveChart) eqLiveChart.resize($('chart-equity-live').clientWidth, 220);
}
window.addEventListener('resize', handleResize);

/* Persist active tab via URL hash (e.g. #backtest, #live-config) */
document.querySelectorAll('#mainTabs button[data-bs-toggle="pill"]').forEach(btn => {
    btn.addEventListener('shown.bs.tab', () => {
        const hash = btn.dataset.bsTarget.replace('#tab-', '');
        history.replaceState(null, '', '#' + hash);
    });
});
(function restoreTab() {
    const hash = location.hash.replace('#', '');
    if (hash) {
        const btn = document.getElementById('tab-' + hash + '-btn');
        if (btn) new bootstrap.Tab(btn).show();
    }
})();

/* Resize backtest charts when switching to Backtest tab */
document.getElementById('tab-backtest-btn').addEventListener('shown.bs.tab', () => {
    setTimeout(handleResize, 50);
});

/* Resize dashboard charts when switching back */
document.getElementById('tab-dashboard-btn').addEventListener('shown.bs.tab', () => {
    initLiveEquityChart();
    setTimeout(handleResize, 50);
});

/* -- WebSocket for live data (replaces polling) -------------------- */
let _ws = null;
let _wsRetry = null;

function connectLiveWS() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    _ws = new WebSocket(`${proto}//${location.host}/ws/live`);

    _ws.onopen = () => {
        console.log('[WS] connected');
        if (_wsRetry) { clearTimeout(_wsRetry); _wsRetry = null; }
    };

    _ws.onmessage = (e) => {
        try {
            const d = JSON.parse(e.data);
            applyLiveData(d.status, d.trades, d.equity);
        } catch (err) { console.warn('[WS] parse error:', err); }
    };

    _ws.onclose = () => {
        console.log('[WS] closed — reconnecting in 5 s');
        _ws = null;
        _wsRetry = setTimeout(connectLiveWS, 5000);
    };

    _ws.onerror = () => { if (_ws) _ws.close(); };
}

/* ======================================================================
   LIVE CONFIG: Load / Save / Close Position
   ====================================================================== */

// --- Unsaved-change tracking ------------------------------------------------
const LC_IDS = [
    'lc-symbol','lc-lookahead','lc-timeframe','lc-threshold','lc-model-type',
    'lc-conf-low-max','lc-conf-avg-max',
    'lc-ps-low','lc-ps-avg','lc-ps-high',
    'lc-volume-min','lc-volume-max',
    'lc-pred-time','lc-exec-time','lc-check-min',
];
let _lcSaved = {};                       // snapshot of last-saved values

function _lcSnapshot() {
    const snap = {};
    LC_IDS.forEach(id => { const el = $(id); if (el) snap[id] = el.value; });
    return snap;
}

function _lcMarkDirty() {
    LC_IDS.forEach(id => {
        const el = $(id);
        if (!el) return;
        if (el.value !== (_lcSaved[id] ?? '')) el.classList.add('lc-changed');
        else el.classList.remove('lc-changed');
    });
}

function _lcClearDirty() {
    LC_IDS.forEach(id => { const el = $(id); if (el) el.classList.remove('lc-changed'); });
}

// Attach listeners once DOM is ready
LC_IDS.forEach(id => {
    const el = $(id);
    if (el) el.addEventListener('input', _lcMarkDirty);
    if (el) el.addEventListener('change', _lcMarkDirty);
});
async function validateSymbol() {
    const input = $('lc-symbol');
    const feedback = $('lc-symbol-feedback');
    const btn = $('btn-validate-symbol');
    const symbol = input.value.trim().toUpperCase();

    if (!symbol) {
        feedback.className = 'small mt-1 text-danger';
        feedback.textContent = 'Symbol cannot be empty.';
        feedback.classList.remove('d-none');
        return;
    }

    btn.disabled = true;
    feedback.className = 'small mt-1 text-muted';
    feedback.innerHTML = '<i class="bi bi-hourglass-split me-1"></i>Checking...';
    feedback.classList.remove('d-none');

    try {
        const r = await apiFetch(API + '/api/live/validate-symbol', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ symbol }),
        });
        if (r.valid) {
            feedback.className = 'small mt-1 text-success';
            feedback.innerHTML = '<i class="bi bi-check-circle me-1"></i>Valid symbol in MT5.';
            input.value = r.symbol;
        } else {
            feedback.className = 'small mt-1 text-danger';
            feedback.innerHTML = '<i class="bi bi-x-circle me-1"></i>' + (r.error || 'Invalid symbol.');
        }
    } catch(e) {
        feedback.className = 'small mt-1 text-danger';
        feedback.innerHTML = '<i class="bi bi-x-circle me-1"></i>' + (e.message || 'Validation failed.');
    } finally {
        btn.disabled = false;
        setTimeout(() => feedback.classList.add('d-none'), 5000);
    }
}

async function loadLiveConfig() {
    try {
        const c = await apiFetch(API + '/api/live/config');
        $('lc-symbol').value = c.symbol || '';
        $('lc-lookahead').value = c.lookahead;
        $('lc-timeframe').value = c.timeframe;
        $('lc-threshold').value = c.threshold;
        $('lc-model-type').value = c.model_type || 'lstm';
        if (c.confidence_thresholds) {
            $('lc-conf-low-max').value = c.confidence_thresholds.low_max;
            $('lc-conf-avg-max').value = c.confidence_thresholds.avg_max;
        }
        if (c.position_sizes) {
            $('lc-ps-low').value = +(c.position_sizes.low * 100).toFixed(4);
            $('lc-ps-avg').value = +(c.position_sizes.avg * 100).toFixed(4);
            $('lc-ps-high').value = +(c.position_sizes.high * 100).toFixed(4);
        }
        $('lc-volume-min').value = c.volume_min ?? 0.01;
        $('lc-volume-max').value = c.volume_max ?? 30;
        $('lc-pred-time').value = String(c.prediction_hour).padStart(2,'0')+':'+String(c.prediction_minute).padStart(2,'0');
        $('lc-exec-time').value = String(c.execution_hour).padStart(2,'0')+':'+String(c.execution_minute).padStart(2,'0');
        $('lc-check-min').value = c.check_positions_minute;
        _lcSaved = _lcSnapshot();
        _lcClearDirty();
    } catch(e) { console.warn('Could not load live config:', e); }
}

async function saveLiveConfig() {
    const btn = $('btn-save-live-config');
    const banner = $('lc-save-result');
    btn.disabled = true;
    banner.classList.add('d-none');

    const payload = {
        symbol: $('lc-symbol').value.trim().toUpperCase(),
        lookahead: parseInt($('lc-lookahead').value),
        timeframe: $('lc-timeframe').value,
        threshold: parseFloat($('lc-threshold').value),
        model_type: $('lc-model-type').value,
        confidence_thresholds: {
            low_max: parseFloat($('lc-conf-low-max').value),
            avg_max: parseFloat($('lc-conf-avg-max').value),
        },
        position_sizes: {
            low: parseFloat($('lc-ps-low').value) / 100,
            avg: parseFloat($('lc-ps-avg').value) / 100,
            high: parseFloat($('lc-ps-high').value) / 100,
        },
        volume_min: parseFloat($('lc-volume-min').value),
        volume_max: parseFloat($('lc-volume-max').value),
        prediction_hour: parseInt($('lc-pred-time').value.split(':')[0]),
        prediction_minute: parseInt($('lc-pred-time').value.split(':')[1]),
        execution_hour: parseInt($('lc-exec-time').value.split(':')[0]),
        execution_minute: parseInt($('lc-exec-time').value.split(':')[1]),
        check_positions_minute: parseInt($('lc-check-min').value),
    };

    try {
        const r = await apiFetch(API + '/api/live/config', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload),
        });
        banner.className = 'mt-2 alert alert-success py-2 mb-0';
        banner.innerHTML = '<i class="bi bi-check-circle me-1"></i>' + (r.message || 'Configuration saved.');
        banner.classList.remove('d-none');
        _lcSaved = _lcSnapshot();
        _lcClearDirty();
    } catch(e) {
        banner.className = 'mt-2 alert alert-danger py-2 mb-0';
        banner.innerHTML = '<i class="bi bi-exclamation-triangle me-1"></i>' + (e.message || 'Failed to save.');
        banner.classList.remove('d-none');
    } finally {
        btn.disabled = false;
        setTimeout(() => banner.classList.add('d-none'), 5000);
    }
}

async function resetLiveConfig() {
    if (!confirm('Reset all configuration to defaults? This will overwrite your current config.json with config.default.json.')) return;

    const btn = $('btn-reset-live-config');
    const banner = $('lc-save-result');
    const symFeedback = $('lc-symbol-feedback');
    btn.disabled = true;
    banner.classList.add('d-none');
    if (symFeedback) symFeedback.classList.add('d-none');

    try {
        const r = await apiFetch(API + '/api/live/config/reset', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
        });
        banner.className = 'mt-2 alert alert-success py-2 mb-0';
        banner.innerHTML = '<i class="bi bi-check-circle me-1"></i>' + (r.message || 'Configuration reset to defaults.');
        banner.classList.remove('d-none');
        // Reload the form with the fresh default values
        await loadLiveConfig();
    } catch(e) {
        banner.className = 'mt-2 alert alert-danger py-2 mb-0';
        banner.innerHTML = '<i class="bi bi-exclamation-triangle me-1"></i>' + (e.message || 'Failed to reset.');
        banner.classList.remove('d-none');
    } finally {
        btn.disabled = false;
        setTimeout(() => banner.classList.add('d-none'), 5000);
    }
}

async function closePosition(id) {
    if (!id) { alert('No trade identifier.'); return; }
    if (!confirm(`Close position "${id}"? This action cannot be undone.`)) return;

    try {
        const r = await apiFetch(API + '/api/live/close-position', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({identifier: id}),
        });
        alert(r.message || 'Position closed.');
        loadLiveData();
    } catch(e) {
        alert('Error: ' + (e.message || 'Failed to close position.'));
    }
}

/* -- Init ------------------------------------------------------------ */
loadBacktestDefaults();
loadLiveConfig();
loadDashboard().then(() => {
    initLiveEquityChart();
    handleResize();
    connectLiveWS();
});