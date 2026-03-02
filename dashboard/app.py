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
import machine_learning.mlp as mlp
from dataset_utils import dataset_utils, feature_engineering
from backtest import backtest_triple_barrier as bt
from live_trading import trader as live_trader
from live_trading import trade_logger
from models.MT5Services import MT5Services
from config_utils import get_trading_config, save_config, reset_to_defaults, load_default_config

def log(msg: str) -> None:
    print(f"[\x1b[44mDashboard\x1b[0m] {msg}")

# --- CONFIG ------------------------------------------------------------------
_TR_CFG = get_trading_config(CONFIG)
SYMBOL = CONFIG["symbol"]
DATASET_RAW = PROJECT_ROOT / CONFIG["paths"]["dataset_raw"]
DATASET_FINAL = PROJECT_ROOT / CONFIG["paths"]["dataset_final"]
LOOKAHEAD = _TR_CFG["lookahead"]
ATR_MULT = _TR_CFG["atr_mult"]
MODEL_TYPE = CONFIG["live_trading"].get("model_type", "lstm")
BACKTEST_WINDOW = CONFIG["backtest"]["backtest_window"]
PREDICT_WINDOW = CONFIG["backtest"]["predict_window"]
INITIAL_CAPITAL = CONFIG["backtest"]["initial_capital"]
COMMISSION = CONFIG["backtest"]["commission"]
THRESHOLD = _TR_CFG["threshold"]
CONFIDENCE_THRESHOLDS = _TR_CFG.get("confidence_thresholds", {"low_max": 0.45, "avg_max": 0.55})
POSITION_SIZES = _TR_CFG.get("position_sizes", {"low": 0.005, "avg": 0.01, "high": 0.04})

# --- MT5 CONNECTION ----------------------------------------------------------
_mt5_service = None

def _try_connect_mt5():
    global _mt5_service
    try:
        _mt5_service = MT5Services(SYMBOL)
        log("\x1b[32mConnect to MetaTrader 5\x1b[0m")
    except Exception as e:
        log(f"\x1b[31mMT5 is unavailable: {e}\x1b[0m")
        _mt5_service = None
    trade_logger.set_mt5_service(_mt5_service)


# --- WEBSOCKET LIVE PUSH ----------------------------------------------------
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
    _ws_clients.difference_update(dead)


async def _periodic_broadcast():
    """Background task: push live data to all WS clients every 10 seconds."""
    while True:
        await asyncio.sleep(10)
        await _broadcast_live()


# --- SCHEDULER ---------------------------------------------------------------
_scheduler = None

# --- CACHED STATE ------------------------------------------------------------
_cache: dict = {}


def _ensure_data():
    """Load (or reload) dataset, run live prediction once, cache results.
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
        if MODEL_TYPE == "mlp":
            model, scaler, feature_cols = mlp.train_mlp_model(
                df_full, lookahead_days=LOOKAHEAD, plot_results=False
            )
            probs = mlp.predict_next_move(model, df_full, feature_cols, scaler)
        else:
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



# --- APP ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log("\x1b[32mSTARTUP...\x1b[0m")
    global _scheduler
    # Connect to MT5 (if available)
    _try_connect_mt5()
    _ensure_data()
    # Record today's equity snapshot on startup (if MT5 available)
    trade_logger.record_equity_snapshot()
    # Startup live trading scheduler
    _scheduler = live_trader.setup_scheduler(mt5_service=_mt5_service)
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

# --- ROUTES ------------------------------------------------------------------

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


# --- BACKTEST HELPERS ---------------------------------------------------------

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
    model_type: str = Body("lstm"),
    confidence_thresholds: dict = Body(None),
    position_sizes: dict = Body(None),
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
            model_type=model_type,
            confidence_thresholds=confidence_thresholds or CONFIDENCE_THRESHOLDS,
            position_sizes=position_sizes or POSITION_SIZES,
            commission=COMMISSION,
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
            "model_type": model_type,
            "confidence_thresholds": confidence_thresholds,
            "position_sizes": position_sizes,
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
        "model_type": "lstm",
        "confidence_thresholds": CONFIDENCE_THRESHOLDS,
        "position_sizes": POSITION_SIZES,
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


# --- LIVE TRADING ROUTES -----------------------------------------------------

@app.get("/api/prediction")
async def api_prediction():
    """Current AI prediction for the next candle."""
    _ensure_data()
    if err := _check_data_ready():
        return err
    return _cache["live_prediction"]

@app.get("/api/live/signals")
async def api_live_signals():
    """Trade signal markers for the dashboard chart (from live trades).
    Excludes cancelled trades."""
    all_trades = trade_logger.get_all()
    markers = []
    for t in all_trades:
        if t["status"] == "cancelled":
            continue
        direction = t["direction"]
        # Use exec_time for open/closed trades, signal_time for pending
        time_str = t.get("exec_time") or t.get("signal_time")
        if not time_str:
            continue
        try:
            ts = int(pd.Timestamp(time_str).timestamp())
        except Exception:
            continue

        # Determine color/text based on status
        status = t["status"]
        pnl = t.get("pnl_pct")
        if status == "closed" and pnl is not None:
            color = "#26a69a" if pnl > 0 else "#ef5350"
            label = f"{'W' if pnl > 0 else 'L'} {pnl*100:+.1f}%"
        elif status == "open":
            color = "#f0b90b"  # yellow for open
            label = "OPEN"
        elif status == "pending":
            color = "#42a5f5"  # blue for pending
            label = "PENDING"
        else:
            color = "#888"
            label = status.upper()

        markers.append({
            "time": ts,
            "position": "belowBar" if direction == "LONG" else "aboveBar",
            "color": color,
            "shape": "arrowUp" if direction == "LONG" else "arrowDown",
            "text": label,
            "size": 2,
        })

    return markers


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
    trade_logger.record_equity_snapshot()
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
    """Manual trigger: execute the pending trade."""
    live_trader.job_execute(mt5_service=_mt5_service)
    await _broadcast_live()
    return {"status": "ok", "pending": live_trader.state.get("pending_trade_id")}


# --- LIVE CONFIG ROUTES ------------------------------------------------------

@app.post("/api/live/validate-symbol")
async def api_validate_symbol(symbol: str = Body(..., embed=True)):
    """Check whether a symbol exists in MT5 Market Watch."""
    symbol = symbol.strip().upper()
    if not symbol:
        return JSONResponse(status_code=400, content={"valid": False, "error": "Symbol cannot be empty."})
    if _mt5_service is None:
        return JSONResponse(status_code=503, content={"valid": False, "error": "MT5 is not connected — cannot validate symbol."})
    try:
        _mt5_service.get_symbol_info(symbol)
        return {"valid": True, "symbol": symbol}
    except Exception:
        return {"valid": False, "error": f"Symbol '{symbol}' not found in MT5."}


@app.get("/api/live/config")
async def api_live_config_get():
    """Return current live trading configuration."""
    _la_key = f"trading_{LOOKAHEAD}"
    _la_cfg = CONFIG.get(_la_key, {})
    return {
        "symbol": SYMBOL,
        "lookahead": LOOKAHEAD,
        "timeframe": CONFIG["live_trading"]["timeframe"],
        "model_type": CONFIG["live_trading"].get("model_type", "lstm"),
        "threshold": _la_cfg.get("threshold", 0.35),
        "confidence_thresholds": _la_cfg.get("confidence_thresholds", {"low_max": 0.45, "avg_max": 0.55}),
        "position_sizes": _la_cfg.get("position_sizes", {"low": 0.005, "avg": 0.01, "high": 0.04}),
        "volume_min": CONFIG["live_trading"].get("volume_min", 0.01),
        "volume_max": CONFIG["live_trading"].get("volume_max", 30.0),
        "prediction_hour": CONFIG["live_trading"]["prediction_hour"],
        "prediction_minute": CONFIG["live_trading"]["prediction_minute"],
        "execution_hour": CONFIG["live_trading"]["execution_hour"],
        "execution_minute": CONFIG["live_trading"]["execution_minute"],
        "check_positions_minute": CONFIG["live_trading"]["check_positions_minute"],
    }


@app.post("/api/live/config")
async def api_live_config_save(
    symbol: str = Body(...),
    lookahead: int = Body(...),
    timeframe: str = Body(...),
    model_type: str = Body("lstm"),
    threshold: float = Body(...),
    confidence_thresholds: dict = Body(...),
    position_sizes: dict = Body(...),
    volume_min: float = Body(...),
    volume_max: float = Body(...),
    prediction_hour: int = Body(...),
    prediction_minute: int = Body(...),
    execution_hour: int = Body(...),
    execution_minute: int = Body(...),
    check_positions_minute: int = Body(...),
):
    """Save live trading config to config.json and reschedule jobs."""
    global SYMBOL, LOOKAHEAD, ATR_MULT, THRESHOLD, CONFIDENCE_THRESHOLDS, POSITION_SIZES
    global BACKTEST_WINDOW, PREDICT_WINDOW, INITIAL_CAPITAL, COMMISSION, _mt5_service

    symbol = symbol.strip().upper()

    # If symbol changed, validate via MT5 and reconnect
    if symbol != SYMBOL:
        if _mt5_service is None:
            return JSONResponse(status_code=503, content={"error": "MT5 not connected — cannot change symbol."})
        try:
            _mt5_service.get_symbol_info(symbol)
        except Exception:
            return JSONResponse(status_code=400, content={"error": f"Symbol '{symbol}' not found in MT5."})
        # Reconnect MT5 service with the new symbol
        try:
            _mt5_service = MT5Services(symbol)
            trade_logger.set_mt5_service(_mt5_service)
            log(f"\x1b[32mMT5 reconnected with symbol {symbol}\x1b[0m")
        except Exception as e:
            log(f"\x1b[31mFailed to reconnect MT5 with {symbol}: {e}\x1b[0m")
            return JSONResponse(status_code=500, content={"error": f"MT5 reconnection failed: {e}"})
        SYMBOL = symbol
        CONFIG["symbol"] = symbol
        live_trader.SYMBOL = symbol

    # Update in-memory config
    CONFIG["live_trading"]["timeframe"] = timeframe
    CONFIG["live_trading"]["model_type"] = model_type
    CONFIG["trading"]["lookahead"] = lookahead
    LOOKAHEAD = lookahead
    live_trader.LOOKAHEAD = lookahead
    _la_key = f"trading_{LOOKAHEAD}"
    if _la_key not in CONFIG:
        CONFIG[_la_key] = {}
    CONFIG[_la_key]["threshold"] = threshold
    CONFIG[_la_key]["confidence_thresholds"] = confidence_thresholds
    CONFIG[_la_key]["position_sizes"] = position_sizes
    CONFIG["live_trading"]["volume_min"] = volume_min
    CONFIG["live_trading"]["volume_max"] = volume_max
    CONFIG["live_trading"]["prediction_hour"] = prediction_hour
    CONFIG["live_trading"]["prediction_minute"] = prediction_minute
    CONFIG["live_trading"]["execution_hour"] = execution_hour
    CONFIG["live_trading"]["execution_minute"] = execution_minute
    CONFIG["live_trading"]["check_positions_minute"] = check_positions_minute

    # Update module-level variables
    THRESHOLD = threshold
    CONFIDENCE_THRESHOLDS = confidence_thresholds
    POSITION_SIZES = position_sizes

    # Refresh derived config that the save endpoint previously missed
    _tr = get_trading_config(CONFIG)
    ATR_MULT = _tr["atr_mult"]
    BACKTEST_WINDOW = CONFIG["backtest"]["backtest_window"]
    PREDICT_WINDOW = CONFIG["backtest"]["predict_window"]
    INITIAL_CAPITAL = CONFIG["backtest"]["initial_capital"]
    COMMISSION = CONFIG["backtest"]["commission"]

    # Update trader module globals
    live_trader.THRESHOLD = threshold
    live_trader.TIMEFRAME = timeframe
    live_trader.VOLUME_MIN = volume_min
    live_trader.VOLUME_MAX = volume_max
    live_trader.CONFIDENCE_THRESHOLDS = confidence_thresholds
    live_trader.POSITION_SIZES = position_sizes

    # Persist to disk
    try:
        save_config(CONFIG)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to write config: {e}"})

    # Reschedule jobs with new times
    if _scheduler and _scheduler.running:
        from apscheduler.triggers.cron import CronTrigger
        try:
            _scheduler.reschedule_job(
                "daily_prediction",
                trigger=CronTrigger(hour=prediction_hour, minute=prediction_minute),
            )
            _scheduler.reschedule_job(
                "daily_execution",
                trigger=CronTrigger(hour=execution_hour, minute=execution_minute),
            )
            _scheduler.reschedule_job(
                "check_positions",
                trigger=CronTrigger(minute=check_positions_minute),
            )
            log(f"Scheduler rescheduled: pred@{prediction_hour}:{prediction_minute:02d}, exec@{execution_hour}:{execution_minute:02d}, check@xx:{check_positions_minute:02d}")
        except Exception as e:
            log(f"\x1b[31mFailed to reschedule: {e}\x1b[0m")

    await _broadcast_live()
    return {"status": "ok", "message": "Configuration saved and scheduler updated."}


@app.post("/api/live/config/reset")
async def api_live_config_reset():
    """Reset config.json to config.default.json and reload in-memory state."""
    global CONFIG, _TR_CFG, SYMBOL, THRESHOLD, CONFIDENCE_THRESHOLDS, POSITION_SIZES
    global LOOKAHEAD, ATR_MULT, INITIAL_CAPITAL, COMMISSION
    global BACKTEST_WINDOW, PREDICT_WINDOW, _mt5_service
    try:
        CONFIG.update(reset_to_defaults())
        _TR_CFG = get_trading_config(CONFIG)

        new_symbol = CONFIG["symbol"]
        if new_symbol != SYMBOL:
            # Reconnect MT5 with the default symbol
            try:
                _mt5_service = MT5Services(new_symbol)
                trade_logger.set_mt5_service(_mt5_service)
                log(f"\x1b[32mMT5 reconnected with default symbol {new_symbol}\x1b[0m")
            except Exception as e:
                log(f"\x1b[31mMT5 reconnection failed for {new_symbol}: {e}\x1b[0m")
                _mt5_service = None
                trade_logger.set_mt5_service(None)
            SYMBOL = new_symbol
            live_trader.SYMBOL = new_symbol

        LOOKAHEAD = _TR_CFG["lookahead"]
        ATR_MULT = _TR_CFG["atr_mult"]
        BACKTEST_WINDOW = CONFIG["backtest"]["backtest_window"]
        PREDICT_WINDOW = CONFIG["backtest"]["predict_window"]
        INITIAL_CAPITAL = CONFIG["backtest"]["initial_capital"]
        COMMISSION = CONFIG["backtest"]["commission"]
        THRESHOLD = _TR_CFG["threshold"]
        CONFIDENCE_THRESHOLDS = _TR_CFG.get("confidence_thresholds", {"low_max": 0.45, "avg_max": 0.55})
        POSITION_SIZES = _TR_CFG.get("position_sizes", {"low": 0.005, "avg": 0.01, "high": 0.04})
        live_trader.THRESHOLD = THRESHOLD
        live_trader.TIMEFRAME = CONFIG["live_trading"]["timeframe"]
        live_trader.VOLUME_MIN = CONFIG["live_trading"].get("volume_min", 0.01)
        live_trader.VOLUME_MAX = CONFIG["live_trading"].get("volume_max", 30.0)
        live_trader.CONFIDENCE_THRESHOLDS = CONFIDENCE_THRESHOLDS
        live_trader.POSITION_SIZES = POSITION_SIZES
        log("Config reset to defaults.")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to reset config: {e}"})
    await _broadcast_live()
    return {"status": "ok", "message": "Configuration reset to defaults."}


@app.get("/api/live/config/defaults")
async def api_live_config_defaults():
    """Return the default config values (from config.default.json) for display."""
    try:
        defaults = load_default_config()
        la = defaults["trading"]["lookahead"]
        la_key = f"trading_{la}"
        la_cfg = defaults.get(la_key, {})
        return {
            "symbol": defaults.get("symbol", "ETHUSD"),
            "lookahead": la,
            "timeframe": defaults["live_trading"]["timeframe"],
            "threshold": la_cfg.get("threshold", 0.35),
            "confidence_thresholds": la_cfg.get("confidence_thresholds", {"low_max": 0.45, "avg_max": 0.55}),
            "position_sizes": la_cfg.get("position_sizes", {"low": 0.005, "avg": 0.01, "high": 0.04}),
            "volume_min": defaults["live_trading"].get("volume_min", 0.01),
            "volume_max": defaults["live_trading"].get("volume_max", 30.0),
            "prediction_hour": defaults["live_trading"]["prediction_hour"],
            "prediction_minute": defaults["live_trading"]["prediction_minute"],
            "execution_hour": defaults["live_trading"]["execution_hour"],
            "execution_minute": defaults["live_trading"]["execution_minute"],
            "check_positions_minute": defaults["live_trading"]["check_positions_minute"],
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to load defaults: {e}"})


@app.post("/api/live/close-position")
async def api_close_position(identifier: str = Body(..., embed=True)):
    """Manually close an open position by trade ID or MT5 ticket."""
    identifier = identifier.strip()
    if not identifier:
        return JSONResponse(status_code=400, content={"error": "No identifier provided."})

    # Search open AND pending trades
    open_trades = trade_logger.get_open_trades()
    pending_trades = trade_logger.get_pending_trades()
    trade = None
    for t in open_trades + pending_trades:
        if str(t.get("id")) == identifier or str(t.get("mt5_ticket")) == identifier:
            trade = t
            break

    if not trade:
        return JSONResponse(status_code=404, content={"error": f"No open/pending trade found for '{identifier}'."})

    # Pending trades can simply be cancelled (no MT5 interaction needed)
    if trade["status"] == "pending":
        trade_logger.mark_cancelled(trade["id"], comment="Manually cancelled")
        log(f"Pending trade {trade['id']} cancelled manually.")
        await _broadcast_live()
        return {"status": "ok", "message": f"Pending trade {trade['id']} cancelled."}

    # Open trade — requires MT5 if it has a real ticket
    ticket = trade.get("mt5_ticket", 0)
    direction = trade["direction"]
    entry = trade["entry_price"]

    if ticket and _mt5_service is None:
        return JSONResponse(status_code=503, content={"error": "MT5 not connected — cannot close a live position."})

    # Close on MT5 if connected and has a real ticket
    if _mt5_service is not None and ticket:
        try:
            _mt5_service.close_position(ticket)
            log(f"MT5 position {ticket} closed manually.")
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"MT5 close failed: {e}"})

    # Mark trade as closed in DB
    try:
        exit_price = entry  # rough; a real implementation would read last price
        if _mt5_service:
            try:
                tick = _mt5_service.get_last_tick(SYMBOL)
                if tick:
                    exit_price = tick.get("bid", entry) if direction == "LONG" else tick.get("ask", entry)
            except Exception:
                pass
        pnl = (exit_price - entry) / entry if direction == "LONG" else (entry - exit_price) / entry
        trade_logger.mark_closed(trade["id"], exit_price, pnl)
        log(f"Trade {trade['id']} manually closed. Exit={exit_price:.2f}, PnL={pnl:.4%}")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"DB update failed: {e}"})

    await _broadcast_live()
    return {"status": "ok", "message": f"Position {trade['id']} closed.", "exit_price": exit_price, "pnl": pnl}


# --- WEBSOCKET ---------------------------------------------------------------

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


# --- ENTRY POINT -------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.app:app", host=CONFIG["dashboard"]["host"], port=CONFIG["dashboard"]["port"], reload=False)
