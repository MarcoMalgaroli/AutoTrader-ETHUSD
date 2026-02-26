"""
ETHUSD AI Trading Dashboard — FastAPI Backend
Serves OHLC data, backtest trades, equity curve, AI predictions,
and LIVE trading data with scheduled execution.
"""

import asyncio
import json
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root to path so we can import project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

with open(PROJECT_ROOT / "config.json", "r") as f:
    CONFIG = json.load(f)

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

import machine_learning.lstm_classifier as lstm
from dataset_utils import dataset_utils, feature_engineering
from backtest import backtest_triple_barrier as bt
from live_trading import trader as live_trader
from live_trading import trade_logger
from models.MT5Services import MT5Services

def log(msg: str) -> None:
    print(f"[\x1b[44mDashboard\x1b[0m] {msg}")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SYMBOL = CONFIG["symbol"]
DATASET_RAW = PROJECT_ROOT / CONFIG["paths"]["dataset_raw"]
DATASET_FINAL = PROJECT_ROOT / CONFIG["paths"]["dataset_final"]
LOOKAHEAD = CONFIG["trading"]["lookahead"]
ATR_MULT = CONFIG["trading"]["atr_mult"]
BACKTEST_WINDOW = CONFIG["backtest"]["backtest_window"]
PREDICT_WINDOW = CONFIG["backtest"]["predict_window"]
INITIAL_CAPITAL = CONFIG["trading"]["initial_capital"]
THRESHOLD = CONFIG["trading"]["threshold"]
POSITION_SIZE = CONFIG["trading"]["position_size"]

# ─── MT5 CONNECTION ──────────────────────────────────────────────────────────
_mt5_service = None

def _try_connect_mt5():
    global _mt5_service
    try:
        _mt5_service = MT5Services(SYMBOL)
        log("\x1b[32mConnect to MetaTrader 5\x1b[0m")
    except Exception as e:
        log(f"\x1b[31mMT5 is unavailable: {e}\x1b[0m")
        _mt5_service = None


def _record_equity_snapshot():
    """Record today's account equity from MT5 into the DB.
    If MT5 is not connected, skip silently."""
    if _mt5_service is None:
        return
    try:
        info = _mt5_service.get_account_info()
        if info:
            trade_logger.record_equity_snapshot(
                equity=info["equity"],
                balance=info.get("balance"),
                margin=info.get("margin"),
                free_margin=info.get("margin_free"),
            )
            log(f"Equity snapshot recorded: ${info['equity']:,.2f}")
    except Exception as e:
        log(f"\x1b[31mFailed to record equity snapshot: {e}\x1b[0m")

# ─── WEBSOCKET LIVE PUSH ────────────────────────────────────────────────────
_ws_clients: set = set()


def _gather_live_data() -> dict:
    """Collect status + trades + equity in a single payload for WS push."""
    status = live_trader.get_status()
    status["mt5_connected"] = _mt5_service is not None
    if _scheduler and _scheduler.running:
        status["scheduler_running"] = True
        jobs = {j.id: j for j in _scheduler.get_jobs()}
        pred_job = jobs.get("daily_prediction")
        exec_job = jobs.get("daily_execution")
        status["next_prediction"] = (
            pred_job.next_run_time.isoformat()
            if pred_job and pred_job.next_run_time
            else None
        )
        status["next_execution"] = (
            exec_job.next_run_time.isoformat()
            if exec_job and exec_job.next_run_time
            else None
        )

    trades = trade_logger.get_last_n(20)

    snapshots = trade_logger.get_equity_snapshots()
    if snapshots:
        equity = []
        for s in snapshots:
            try:
                equity.append(
                    {"time": int(pd.Timestamp(s["date"]).timestamp()), "value": s["equity"]}
                )
            except Exception:
                continue
    else:
        curve = trade_logger.compute_equity_curve(INITIAL_CAPITAL)
        equity = []
        for pt in curve:
            if pt["time"]:
                try:
                    equity.append(
                        {"time": int(pd.Timestamp(pt["time"]).timestamp()), "value": pt["value"]}
                    )
                except Exception:
                    pass

    return {"status": status, "trades": trades, "equity": equity}


async def _broadcast_live():
    """Push live data to all connected WebSocket clients immediately."""
    if not _ws_clients:
        return
    data = _gather_live_data()
    dead = set()
    for ws in list(_ws_clients):
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    _ws_clients -= dead


async def _periodic_broadcast():
    """Background task: push live data to all WS clients every 10 seconds."""
    while True:
        await asyncio.sleep(10)
        await _broadcast_live()


# ─── SCHEDULER ───────────────────────────────────────────────────────────────
_scheduler = None

# ─── CACHED STATE ────────────────────────────────────────────────────────────
_cache: dict = {}


def _ensure_data():
    """Load (or reload) dataset, run backtest & live prediction once, cache results.
    Stores any error in _cache['_error'] so API routes can return it to clients.
    """
    if "_error" in _cache:
        return  # already failed — don't retry automatically
    if _cache:
        return

    try:
        log("\x1b[36mLoading dataset and running pipeline...\x1b[0m")

        # 1. Generate dataset from MT5 (if connected) or load existing raw datasets
        if _mt5_service:
            # load from MT5 and save to raw
            path_list_raw = dataset_utils.generate_dataset(_mt5_service)
        else:
            # fallback: load existing raw datasets if MT5 not connected
            log("\x1b[33mUsing existing dataset files (MT5 not connected)\x1b[0m")
            path_list_raw = list(DATASET_RAW.glob("*.csv"))
            if not path_list_raw:
                raise RuntimeError(f"No dataset found at {DATASET_RAW}")

        # 2. Validate raw datasets before proceeding
        if not dataset_utils.validate_dataset(path_list_raw):
            raise RuntimeError("Dataset validation failed. Check server logs for details.")

        # 3. Feature engineering (uses raw → final)
        path_list_final = feature_engineering.calculate_features(
            path_list_raw, lookahead=LOOKAHEAD, atr_mult=ATR_MULT
        )

        # 4. Cache final datasets
        final_path = None
        for path in path_list_final:
            tf = path.stem.split('_')[-1]
            _cache["df_" + tf] = pd.read_csv(path)
            if tf == "D1":
                final_path = path

        if not final_path:
            final_path = path_list_final[0]

        df_full = pd.read_csv(final_path)
        df_full["time"] = pd.to_datetime(df_full["time"])

        # 5. Train model on full data for live prediction
        model, scaler, feature_cols = lstm.train_lstm_classifier(
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

        _cache["df"] = df_full

        log("Ready — data loaded (backtest available on demand).")

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _cache["_error"] = str(exc)
        log(f"\x1b[31mPipeline failed: {exc}\x1b[0m")


def _check_data_ready():
    """Call after _ensure_data(); returns a JSONResponse with the error if the
    pipeline failed, or None if everything is loaded and ready."""
    if "_error" in _cache:
        return JSONResponse(
            status_code=503,
            content={"error": _cache["_error"]},
        )
    if "df" not in _cache:
        return JSONResponse(
            status_code=503,
            content={"error": "Data not loaded yet."},
        )
    return None



# ─── APP ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log("\x1b[32mSTARTUP...\x1b[0m")
    global _scheduler
    # Connect to MT5 (if available)
    _try_connect_mt5()
    _ensure_data()
    # Record today's equity snapshot on startup (if MT5 available)
    _record_equity_snapshot()
    # Startup live trading scheduler
    _scheduler = live_trader.setup_scheduler(mt5_service=_mt5_service)
    # Add daily equity snapshot job
    from apscheduler.triggers.cron import CronTrigger
    _scheduler.add_job(
        _record_equity_snapshot,
        CronTrigger(hour=23, minute=59),
        id="equity_snapshot",
        name="Daily Equity Snapshot (23:59 UTC)",
        replace_existing=True,
    )
    _scheduler.start()
    log("\x1b[32mScheduler started\x1b[0m")

    # Start periodic WebSocket broadcast
    _broadcast_task = asyncio.create_task(_periodic_broadcast())
    
    yield

    _broadcast_task.cancel()
    if _scheduler:
        _scheduler.shutdown(wait=False)
        log("\x1b[31mScheduler stopped\x1b[0m")
    if _mt5_service:
        _mt5_service.shutdown()
        log("\x1b[31mMT5 connection closed\x1b[0m")
    
    log("\x1b[31mSHUTDOWN complete\x1b[0m")

app = FastAPI(title=f"{SYMBOL} AI Dashboard", lifespan=lifespan)
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
app.mount("/public", StaticFiles(directory=Path(__file__).parent / "src"), name="public")

# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "symbol": SYMBOL})


@app.get("/api/candles")
async def api_candles(last_n: int = Query(400, ge=30, le=5000)):
    """OHLC candlestick data for the chart."""
    _ensure_data()
    if err := _check_data_ready():
        return err
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
    if err := _check_data_ready():
        return err
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


# ─── BACKTEST HELPERS ─────────────────────────────────────────────────────────

def _check_backtest_ready():
    """Return a JSONResponse with an error if backtest hasn't been run yet."""
    if "backtest_result" not in _cache:
        return JSONResponse(
            status_code=404,
            content={"error": "Backtest not run yet. Configure parameters and launch it."},
        )
    return None


@app.post("/api/bt/run")
async def api_bt_run(
    backtest_window: int = Body(BACKTEST_WINDOW),
    predict_window: int = Body(PREDICT_WINDOW),
    initial_capital: float = Body(INITIAL_CAPITAL),
    lookahead: int = Body(LOOKAHEAD),
    atr_mult: float = Body(ATR_MULT),
    threshold: float = Body(THRESHOLD),
    position_size: float = Body(POSITION_SIZE),
    model_type: str = Body("lstm"),
):
    """Run backtest on demand with user-specified parameters."""
    _ensure_data()
    if err := _check_data_ready():
        return err

    try:
        df_full = _cache["df"]
        res, trades = bt.backtest_triple_barrier(
            df_full,
            backtest_window,
            predict_window,
            initial_capital,
            lookahead,
            atr_mult,
            threshold=threshold,
            position_size=position_size,
            model_type=model_type,
        )
        _cache["backtest_result"] = res
        _cache["backtest_window"] = backtest_window
        _cache["backtest_trades"] = trades
        _cache["backtest_params"] = {
            "backtest_window": backtest_window,
            "predict_window": predict_window,
            "initial_capital": initial_capital,
            "lookahead": lookahead,
            "atr_mult": atr_mult,
            "threshold": threshold,
            "position_size": position_size,
            "model_type": model_type,
        }
        log(f"Backtest complete — {len(trades)} trades.")
        return {"status": "ok", "trades": len(trades)}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/api/bt/defaults")
async def api_bt_defaults():
    """Return default backtest parameters from config."""
    return {
        "backtest_window": BACKTEST_WINDOW,
        "predict_window": PREDICT_WINDOW,
        "initial_capital": INITIAL_CAPITAL,
        "lookahead": LOOKAHEAD,
        "atr_mult": ATR_MULT,
        "threshold": THRESHOLD,
        "position_size": POSITION_SIZE,
        "model_type": "lstm",
    }


# API endpoints for backtest results (available after /api/bt/run)
@app.get("/api/bt/equity")
async def api_equity():
    """Equity curve data."""
    if err := _check_backtest_ready():
        return err
    res = _cache["backtest_result"]
    records = []
    for ts, val in zip(res.equity_curve.index, res.equity_curve.values):
        records.append({
            "time": int(pd.Timestamp(ts).timestamp()),
            "value": round(float(val), 2),
        })
    return records


@app.get("/api/bt/trades")
async def api_trades():
    """List of backtest trades with TP/SL levels."""
    if err := _check_backtest_ready():
        return err
    return _cache["backtest_trades"]


@app.get("/api/bt/stats")
async def api_stats():
    """Backtest summary statistics."""
    if err := _check_backtest_ready():
        return err
    s = _cache["backtest_result"].summary
    p = _cache["backtest_params"]
    return {
        "initial_capital": p["initial_capital"],
        "final_equity": round(s["final_equity"], 2),
        "total_return_pct": round((s["final_equity"] / p["initial_capital"] - 1) * 100, 2),
        "trades": s["trades"],
        "wins": s["wins"],
        "losses": s["losses"],
        "hit_rate": round(s["hit_rate"] * 100, 2),
        "avg_return": round(s["avg_return"] * 100, 3),
        "max_drawdown": round(s["max_drawdown"] * 100, 2),
        "backtest_window": p["backtest_window"],
        "lookahead": p["lookahead"],
        "atr_mult": p["atr_mult"],
        "threshold": p["threshold"],
        "model_type": p["model_type"],
    }

@app.get("/api/bt/signals")
async def api_signals(last_n: int = Query(400, ge=30, le=5000)):
    """Trade signal markers for the chart (from backtest)."""
    if err := _check_backtest_ready():
        return err
    df = _cache["df"]
    n = len(df)
    bw = _cache["backtest_params"]["backtest_window"]
    start = n - bw

    trades = _cache["backtest_trades"]
    markers = []

    for t in trades:
        ret = t["return_pct"]
        direction = t["direction"]

        try:
            ts = int(pd.Timestamp(t["time"]).timestamp())
        except Exception:
            continue
        
        markers.append({
            "time": ts,
            "position": "belowBar" if direction == "LONG" else "aboveBar",
            "color": "#26a69a" if ret > 0 else "#ef5350",
            "shape": "arrowUp" if direction == "LONG" else "arrowDown",
            "text": f"{'W' if ret > 0 else 'L'} {ret*100:+.1f}%",
            "size": 2,
        })

    return markers


# ─── LIVE TRADING ROUTES ─────────────────────────────────────────────────────

@app.get("/api/prediction")
async def api_prediction():
    """Current AI prediction for the next candle."""
    _ensure_data()
    if err := _check_data_ready():
        return err
    return _cache["live_prediction"]

@app.get("/api/live/status")
async def api_live_status():
    """Current live trader status: scheduler, MT5, last signal."""
    status = live_trader.get_status()
    # Override MT5 status from actual connection (trader.state is only
    # updated when a job fires, so on fresh startup it would be False)
    status["mt5_connected"] = _mt5_service is not None
    if _scheduler and _scheduler.running:
        status["scheduler_running"] = True
        jobs = {j.id: j for j in _scheduler.get_jobs()}
        pred_job = jobs.get("daily_prediction")
        exec_job = jobs.get("daily_execution")
        status["next_prediction"] = pred_job.next_run_time.isoformat() if pred_job and pred_job.next_run_time else None
        status["next_execution"] = exec_job.next_run_time.isoformat() if exec_job and exec_job.next_run_time else None
    return status


@app.get("/api/live/trades")
async def api_live_trades(last_n: int = Query(20, ge=1, le=200)):
    """Last N live trades (most recent first)."""
    return trade_logger.get_last_n(last_n)


@app.get("/api/live/equity")
async def api_live_equity():
    """Real equity curve from daily MT5 account snapshots.
    Falls back to trade-based equity if no snapshots exist."""
    snapshots = trade_logger.get_equity_snapshots()
    if snapshots:
        result = []
        for s in snapshots:
            try:
                ts = int(pd.Timestamp(s["date"]).timestamp())
            except Exception:
                continue
            result.append({"time": ts, "value": s["equity"]})
        return result
    # Fallback: compute from closed trades
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


@app.post("/api/live/record-equity")
async def api_record_equity_now():
    """Manual trigger: record current MT5 equity snapshot."""
    if _mt5_service is None:
        return JSONResponse(status_code=503, content={"error": "MT5 not connected"})
    _record_equity_snapshot()
    await _broadcast_live()
    return {"status": "ok"}


@app.post("/api/live/run-now")
async def api_live_run_now():
    """Manual trigger: execute prediction + immediate execution."""
    live_trader.run_now(mt5_service=_mt5_service)
    await _broadcast_live()
    return {"status": "ok", "message": "Prediction + Execution completed"}


@app.post("/api/live/predict-now")
async def api_live_predict_now():
    """Manual trigger: execute solo prediction."""
    live_trader.job_predict(mt5_service=_mt5_service)
    await _broadcast_live()
    return {"status": "ok", "prediction": live_trader.state.get("last_prediction")}


@app.post("/api/live/execute-now")
async def api_live_execute_now():
    """Manual trigger: execute il trade pending."""
    live_trader.job_execute(mt5_service=_mt5_service)
    await _broadcast_live()
    return {"status": "ok", "pending": live_trader.state.get("pending_trade_id")}


# ─── WEBSOCKET ───────────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    """Stream live status, trades & equity to the frontend."""
    await ws.accept()
    _ws_clients.add(ws)
    try:
        await ws.send_json(_gather_live_data())
        while True:
            await ws.receive_text()  # keep-alive; raises on disconnect
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _ws_clients.discard(ws)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.app:app", host=CONFIG["dashboard"]["host"], port=CONFIG["dashboard"]["port"], reload=False)
