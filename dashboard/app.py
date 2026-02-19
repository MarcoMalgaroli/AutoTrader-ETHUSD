"""
ETHUSD AI Trading Dashboard — FastAPI Backend
Serves OHLC data, backtest trades, equity curve, AI predictions,
and LIVE trading data with scheduled execution.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path so we can import project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import machine_learning.lstm as lstm
from dataset_utils import feature_engineering
from backtest import backtest as bt
from live_trading import trader as live_trader
from live_trading import trade_logger
from models.MT5Services import MT5Services

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SYMBOL = "ETHUSD"
DATASET_RAW = PROJECT_ROOT / "datasets" / "raw" / "ETHUSD_D1_3082.csv"
DATASET_FINAL = PROJECT_ROOT / "datasets" / "final" / "ETHUSD_D1_3082.csv"
LOOKAHEAD = 10
ATR_MULT = 2.0
BACKTEST_WINDOW = 365
PREDICT_WINDOW = 30
INITIAL_CAPITAL = 100_000
THRESHOLD = 0.40
POSITION_SIZE = 0.1

# ─── APP ─────────────────────────────────────────────────────────────────────
app = FastAPI(title=f"{SYMBOL} AI Dashboard")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# ─── MT5 CONNECTION (opzionale) ──────────────────────────────────────────────
_mt5_service = None

def _try_connect_mt5():
    global _mt5_service
    try:
        _mt5_service = MT5Services(SYMBOL)
        print("[Dashboard] ✅ Connesso a MetaTrader 5")
    except Exception as e:
        print(f"[Dashboard] ⚠ MT5 non disponibile: {e}")
        _mt5_service = None

# ─── SCHEDULER ───────────────────────────────────────────────────────────────
_scheduler = None

# ─── CACHED STATE ────────────────────────────────────────────────────────────
_cache: dict = {}


def _ensure_data():
    """Load (or reload) dataset, run backtest & live prediction once, cache results."""
    if _cache:
        return

    print("[Dashboard] Loading dataset and running pipeline...")

    # 1. Feature engineering (uses raw → final)
    path_list_final = feature_engineering.calculate_features(
        [DATASET_RAW], lookahead=LOOKAHEAD, atr_mult=ATR_MULT
    )
    final_path = path_list_final[0]

    df_full = pd.read_csv(final_path)
    df_full["time"] = pd.to_datetime(df_full["time"])

    # 2. Train model on full data for live prediction
    model, scaler, feature_cols = lstm.train_lstm_model(
        df_full, lookahead_days=LOOKAHEAD, plot_results=False
    )
    probs = lstm.predict_next_move(model, df_full, feature_cols, scaler)

    _cache["live_prediction"] = {
        "hold": float(probs[0]),
        "long": float(probs[1]),
        "short": float(probs[2]),
        "action": ["HOLD", "LONG", "SHORT"][int(np.argmax(probs))],
        "confidence": float(np.max(probs)),
        "last_close": float(df_full.iloc[-1]["close"]),
        "last_time": str(df_full.iloc[-1]["time"]),
    }

    # 3. Backtest
    res = bt.backtest_triple_barrier(
        df_full,
        BACKTEST_WINDOW,
        PREDICT_WINDOW,
        INITIAL_CAPITAL,
        LOOKAHEAD,
        ATR_MULT,
        threshold=THRESHOLD,
        position_size=POSITION_SIZE,
    )
    _cache["backtest_result"] = res
    _cache["df"] = df_full
    _cache["backtest_window"] = BACKTEST_WINDOW

    # 4. Rebuild detailed trade list from backtest equity engine
    n = len(df_full)
    start_point = n - BACKTEST_WINDOW
    bt_df = df_full.iloc[start_point:].copy().reset_index(drop=True)
    pred_arr = np.zeros(len(bt_df), dtype=int)

    # Re-derive signals from equity curve changes
    eq = res.equity_curve.values
    trades = []
    close_arr = bt_df["close"].values
    open_arr = bt_df["open"].values
    atr_arr = bt_df["ATR_14"].values
    time_arr = bt_df["time"].values

    # Walk through backtest result: reconstruct from trade_returns
    tr = res.trade_returns.values
    trade_idx = 0
    for i in range(len(bt_df) - 1):
        if trade_idx < len(tr) and tr[trade_idx] != 0.0:
            ret = tr[trade_idx]
            signal_price = close_arr[i]
            atr = atr_arr[i]
            entry = open_arr[i + 1] if i + 1 < len(open_arr) else signal_price
            upper_b = signal_price + ATR_MULT * atr
            lower_b = signal_price - ATR_MULT * atr

            # Determine direction from return sign and price relationship
            direction = "LONG" if ret > 0 and entry < upper_b or ret < 0 and entry > lower_b else "SHORT"
            # Simpler: use equity change direction with entry price
            if abs(ret) > 0.0001:
                trades.append({
                    "time": str(pd.Timestamp(time_arr[i])),
                    "direction": direction,
                    "entry": round(float(entry), 2),
                    "tp": round(float(upper_b if direction == "LONG" else lower_b), 2),
                    "sl": round(float(lower_b if direction == "LONG" else upper_b), 2),
                    "return_pct": round(float(ret) * 100, 3),
                    "win": bool(ret > 0),
                })
        trade_idx += 1

    _cache["trades"] = trades
    print(f"[Dashboard] Ready — {len(trades)} trades cached.")


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _scheduler
    _ensure_data()
    # Connetti a MT5 (se disponibile)
    _try_connect_mt5()
    # Avvia lo scheduler per il live trading
    _scheduler = live_trader.setup_scheduler(mt5_service=_mt5_service)
    _scheduler.start()
    print("[Dashboard] 🚀 Scheduler avviato")


@app.on_event("shutdown")
async def shutdown():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        print("[Dashboard] Scheduler fermato")
    if _mt5_service:
        _mt5_service.shutdown()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "symbol": SYMBOL})


@app.get("/api/candles")
async def api_candles(last_n: int = Query(400, ge=30, le=5000)):
    """OHLC candlestick data for the chart."""
    _ensure_data()
    df = _cache["df"]
    subset = df.tail(last_n)
    records = []
    for _, r in subset.iterrows():
        records.append({
            "time": int(pd.Timestamp(r["time"]).timestamp()),
            "open": round(float(r["open"]), 2),
            "high": round(float(r["high"]), 2),
            "low": round(float(r["low"]), 2),
            "close": round(float(r["close"]), 2),
        })
    return records


@app.get("/api/indicators")
async def api_indicators(last_n: int = Query(400, ge=30, le=5000)):
    """SMA / EMA overlay data."""
    _ensure_data()
    df = _cache["df"].tail(last_n)
    sma = []
    ema = []
    for _, r in df.iterrows():
        ts = int(pd.Timestamp(r["time"]).timestamp())
        if not np.isnan(r.get("SMA_50", np.nan)):
            sma.append({"time": ts, "value": round(float(r["SMA_50"]), 2)})
        if not np.isnan(r.get("EMA_20", np.nan)):
            ema.append({"time": ts, "value": round(float(r["EMA_20"]), 2)})
    return {"sma50": sma, "ema20": ema}


@app.get("/api/equity")
async def api_equity():
    """Equity curve data."""
    _ensure_data()
    res = _cache["backtest_result"]
    records = []
    for ts, val in zip(res.equity_curve.index, res.equity_curve.values):
        records.append({
            "time": int(pd.Timestamp(ts).timestamp()),
            "value": round(float(val), 2),
        })
    return records


@app.get("/api/trades")
async def api_trades():
    """List of backtest trades with TP/SL levels."""
    _ensure_data()
    return _cache["trades"]


@app.get("/api/stats")
async def api_stats():
    """Backtest summary statistics."""
    _ensure_data()
    s = _cache["backtest_result"].summary
    return {
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": round(s["final_equity"], 2),
        "total_return_pct": round((s["final_equity"] / INITIAL_CAPITAL - 1) * 100, 2),
        "trades": s["trades"],
        "wins": s["wins"],
        "losses": s["losses"],
        "hit_rate": round(s["hit_rate"] * 100, 2),
        "avg_return": round(s["avg_return"] * 100, 3),
        "max_drawdown": round(s["max_drawdown"] * 100, 2),
        "backtest_window": BACKTEST_WINDOW,
        "lookahead": LOOKAHEAD,
        "atr_mult": ATR_MULT,
        "threshold": THRESHOLD,
    }


@app.get("/api/prediction")
async def api_prediction():
    """Current AI prediction for the next candle."""
    _ensure_data()
    return _cache["live_prediction"]


@app.get("/api/signals")
async def api_signals(last_n: int = Query(400, ge=30, le=5000)):
    """Trade signal markers for the chart (from backtest)."""
    _ensure_data()
    df = _cache["df"]
    n = len(df)
    start = n - BACKTEST_WINDOW

    # Re-derive from backtest_result trade_returns
    res = _cache["backtest_result"]
    bt_df = df.iloc[start:].copy()
    tr = res.trade_returns.values
    markers = []

    for i in range(min(len(bt_df) - 1, len(tr))):
        ret = tr[i]
        if abs(ret) < 1e-8:
            continue
        row = bt_df.iloc[i]
        ts = int(pd.Timestamp(row["time"]).timestamp())
        close = float(row["close"])
        atr = float(row["ATR_14"])

        # Infer direction: if return is positive, it could be either direction.
        # Use the simpler heuristic: match with target label if available
        target = int(row["target"])
        direction = "long" if target == 1 else ("short" if target == -1 else ("long" if ret > 0 else "short"))

        markers.append({
            "time": ts,
            "position": "belowBar" if direction == "long" else "aboveBar",
            "color": "#26a69a" if ret > 0 else "#ef5350",
            "shape": "arrowUp" if direction == "long" else "arrowDown",
            "text": f"{'W' if ret > 0 else 'L'} {ret*100:+.1f}%",
            "size": 2,
        })

    return markers


# ─── LIVE TRADING ROUTES ─────────────────────────────────────────────────────

@app.get("/api/live/status")
async def api_live_status():
    """Stato corrente del live trader: scheduler, MT5, ultimo segnale."""
    status = live_trader.get_status()
    # Aggiungi info dal scheduler sui prossimi run
    if _scheduler and _scheduler.running:
        jobs = {j.id: j for j in _scheduler.get_jobs()}
        pred_job = jobs.get("daily_prediction")
        exec_job = jobs.get("daily_execution")
        status["next_prediction"] = pred_job.next_run_time.isoformat() if pred_job and pred_job.next_run_time else None
        status["next_execution"] = exec_job.next_run_time.isoformat() if exec_job and exec_job.next_run_time else None
    return status


@app.get("/api/live/trades")
async def api_live_trades(last_n: int = Query(20, ge=1, le=200)):
    """Ultimi N trade live (più recenti prima)."""
    return trade_logger.get_last_n(last_n)


@app.get("/api/live/equity")
async def api_live_equity():
    """Equity curve basata sui trade live chiusi."""
    curve = trade_logger.compute_equity_curve(INITIAL_CAPITAL)
    result = []
    for pt in curve:
        if pt["time"]:
            try:
                ts = int(pd.Timestamp(pt["time"]).timestamp())
            except Exception:
                ts = 0
            result.append({"time": ts, "value": pt["value"]})
    return result


@app.post("/api/live/run-now")
async def api_live_run_now():
    """Trigger manuale: esegue prediction + execution immediatamente."""
    live_trader.run_now(mt5_service=_mt5_service)
    return {"status": "ok", "message": "Prediction + Execution completati"}


@app.post("/api/live/predict-now")
async def api_live_predict_now():
    """Trigger manuale: esegue solo prediction."""
    live_trader.job_predict(mt5_service=_mt5_service)
    return {"status": "ok", "prediction": live_trader.state.get("last_prediction")}


@app.post("/api/live/execute-now")
async def api_live_execute_now():
    """Trigger manuale: esegue il trade pending."""
    live_trader.job_execute(mt5_service=_mt5_service)
    return {"status": "ok", "pending": live_trader.state.get("pending_trade_id")}


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.app:app", host="127.0.0.1", port=8050, reload=False)
