const API = '';
const DARK = { background: '#161b22', text: '#c9d1d9', grid: '#21262d' };

/* ── helpers ─────────────────────────────────────────────────────── */
function $(id) { return document.getElementById(id); }
function fmt$(v) { return '$' + v.toLocaleString('en-US', {minimumFractionDigits: 0}); }
function fmtPct(v, dec=2) { return (v >= 0 ? '+' : '') + v.toFixed(dec) + '%'; }
function fmtTime(iso) {
    if (!iso) return '—';
    try { return new Date(iso).toLocaleString('it-IT', {day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'}); }
    catch { return iso; }
}

/* ── Candlestick chart ───────────────────────────────────────────── */
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

/* ── Equity chart (backtest) ─────────────────────────────────────── */
const eqChart = LightweightCharts.createChart($('chart-equity'), {
    layout: { background: { color: DARK.background }, textColor: DARK.text },
    grid: { vertLines: { color: DARK.grid }, horzLines: { color: DARK.grid } },
    rightPriceScale: { borderColor: DARK.grid },
    timeScale: { borderColor: DARK.grid, timeVisible: false },
});
const eqSeries = eqChart.addAreaSeries({
    topColor: 'rgba(38,166,154,0.4)', bottomColor: 'rgba(38,166,154,0.0)',
    lineColor: '#26a69a', lineWidth: 2,
});
const eqBaseline = eqChart.addLineSeries({ color: '#ef5350', lineWidth: 1, lineStyle: 2, title: 'Break Even' });

/* ── Equity chart (live) — lazy init ─────────────────────────────── */
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
    eqLiveBaseline = eqLiveChart.addLineSeries({ color: '#ef5350', lineWidth: 1, lineStyle: 2, title: 'Break Even' });
}

let _allCandles = [];

/* ── helper: fetch with error detection ──────────────────────────── */
async function apiFetch(url) {
    const r = await fetch(url);
    
    if (!r.ok) {
        const body = await r.json().catch(() => ({ error: r.statusText }));
        throw new Error(body.error || `HTTP ${r.status}`);
    }
    return r.json();
}

/* ── Fetch all data ──────────────────────────────────────────────── */
async function loadAll() {
    try {
        const [candles, indicators, equity, trades, stats, signals, prediction] = await Promise.all([
            apiFetch(API + '/api/candles?last_n=3000'),
            apiFetch(API + '/api/indicators?last_n=3000'),
            apiFetch(API + '/api/bt/equity'),
            apiFetch(API + '/api/bt/trades'),
            apiFetch(API + '/api/bt/stats'),
            apiFetch(API + '/api/bt/signals'),
            apiFetch(API + '/api/prediction'),
        ]);

        _allCandles = candles;
        candleSeries.setData(candles);
        smaLine.setData(indicators.sma50);
        emaLine.setData(indicators.ema20);
        if (signals.length) candleSeries.setMarkers(signals.sort((a,b) => a.time - b.time));
        setRange(365);

        eqSeries.setData(equity);
        if (equity.length > 1) {
            const bv = equity[0].value;
            eqBaseline.setData([{time:equity[0].time,value:bv},{time:equity[equity.length-1].time,value:bv}]);
        }
        eqChart.timeScale().fitContent();

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
        $('p-window').textContent = stats.backtest_window + ' giorni';
        $('p-lookahead').textContent = stats.lookahead + ' candele';
        $('p-atr').textContent = stats.atr_mult + 'x';
        $('p-thresh').textContent = (stats.threshold * 100) + '%';

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

        $('trade-count').textContent = trades.length;
        const tbody = $('trades-body');
        if (!trades.length) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted py-3">Nessun trade</td></tr>';
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

        $('last-update').textContent = 'Aggiornato: ' + new Date().toLocaleString('it-IT');
        $('pipeline-error').classList.add('d-none');
        loadLiveData();
    } catch (e) {
        console.error('Pipeline load error:', e);
        $('pipeline-error').classList.remove('d-none');
        $('pipeline-error-text').textContent = e.message || 'Unknown error';
        loadLiveData();
    }
}

/* ── Load live trading data ──────────────────────────────────────── */
async function loadLiveData() {
    try {
        const [status, liveTrades, liveEquity] = await Promise.all([
            fetch(API+'/api/live/status').then(r=>r.json()),
            fetch(API+'/api/live/trades?last_n=20').then(r=>r.json()),
            fetch(API+'/api/live/equity').then(r=>r.json()),
        ]);

        $('scheduler-dot').className = 'status-dot '+(status.scheduler_running?'online':'offline');
        $('scheduler-label').textContent = status.scheduler_running?'Scheduler ON':'Scheduler OFF';
        $('mt5-dot').className = 'status-dot '+(status.mt5_connected?'online':'warning');
        $('mt5-label').textContent = status.mt5_connected?'MT5 ON':'Paper Mode';

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
            lb.innerHTML = '<tr><td colspan="6" class="text-center text-muted py-3">Nessun trade live ancora</td></tr>';
        } else {
            lb.innerHTML = liveTrades.map(t => {
                const d = (t.exec_time||t.signal_time||'').split('T')[0];
                const sb = t.status==='open'?'badge-open':t.status==='pending'?'badge-pending':t.status==='closed'?'badge-closed':'badge-cancelled';
                const pnl = t.pnl_pct!==null?fmtPct(t.pnl_pct*100,2):'—';
                const pc = t.pnl_pct>0?'text-success':t.pnl_pct<0?'text-danger':'';
                return `<tr>
                    <td class="text-nowrap">${d}</td>
                    <td><span class="badge ${t.direction==='LONG'?'badge-long':'badge-short'}">${t.direction}</span></td>
                    <td>${t.entry_price||'—'}</td>
                    <td>${t.tp?t.tp+' / '+t.sl:'—'}</td>
                    <td><span class="badge ${sb}">${t.status}</span></td>
                    <td class="${pc} fw-bold">${pnl}</td>
                </tr>`;
            }).join('');
        }

        if (liveEquity.length > 1) {
            initLiveEquityChart();
            eqLiveSeries.setData(liveEquity);
            const bv = liveEquity[0].value;
            eqLiveBaseline.setData([{time:liveEquity[0].time,value:bv},{time:liveEquity[liveEquity.length-1].time,value:bv}]);
            eqLiveChart.timeScale().fitContent();
        }
    } catch(e) { console.warn('Live data load error:', e); }
}

/* ── Action buttons ──────────────────────────────────────────────── */
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
    if (!confirm('Execute Prediction + Execution now?\n(REAL TRADE if MT5 is connected, otherwise Paper)')) return;
    try {
        await fetch(API+'/api/live/run-now',{method:'POST'});
        alert('Prediction + Execution completed!');
        loadLiveData();
    } catch(e) { alert('Error: '+e); }
}

/* ── Range selector ──────────────────────────────────────────────── */
function setRange(days, evt) {
    if (days === 0) { candleChart.timeScale().fitContent(); }
    else {
        const total = _allCandles.length;
        const from = total > days ? _allCandles[total-days].time : _allCandles[0].time;
        const to = _allCandles[total-1].time;
        candleChart.timeScale().setVisibleRange({ from, to });
    }
    document.querySelectorAll('.card-header .btn-outline-secondary').forEach(b => b.classList.remove('active'));
    evt && evt.target && evt.target.classList.add('active');
}

/* ── Resize ──────────────────────────────────────────────────────── */
function handleResize() {
    candleChart.resize($('chart-candles').clientWidth, 420);
    eqChart.resize($('chart-equity').clientWidth, 220);
    if (eqLiveChart) eqLiveChart.resize($('chart-equity-live').clientWidth, 220);
}
window.addEventListener('resize', handleResize);
document.querySelector('[data-bs-target="#tab-eq-live"]').addEventListener('shown.bs.tab', () => {
    initLiveEquityChart(); handleResize(); loadLiveData();
});

/* ── Auto-refresh live data every 60s ────────────────────────────── */
setInterval(loadLiveData, 60000);

/* ── Init ────────────────────────────────────────────────────────── */
loadAll().then(handleResize);