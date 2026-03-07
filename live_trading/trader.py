"""
Live Trader - Scheduler + Pipeline for automatic trading ETHUSD.

Daily workflow:
  23:55 (configurable)  -->  Download data from MT5, feature engineering, LSTM prediction
  00:05 (configurable)  -->  If segnal is not HOLD, send order to MT5 with TP/SL (triple barrier)

Everything is logged in data/live_trades.db (SQLite) with trade_logger.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_CONFIG_PATH = PROJECT_ROOT / "config.json"

with open(_CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

import machine_learning.lstm_classifier as lstm
import machine_learning.mlp as mlp
from dataset_utils import dataset_utils, feature_engineering
from live_trading import trade_logger
from config_utils import get_trading_config

# --- CONFIG ------------------------------------------------------------------
_TR_CFG = get_trading_config(CONFIG)
SYMBOL = CONFIG["symbol"]
TIMEFRAME = CONFIG["live_trading"]["timeframe"]
LOOKAHEAD = _TR_CFG["lookahead"]
ATR_MULT = _TR_CFG["atr_mult"]
THRESHOLD = _TR_CFG["threshold"]
VOLUME_MIN = CONFIG["live_trading"].get("volume_min", 0.01)
VOLUME_MAX = CONFIG["live_trading"].get("volume_max", 30.0)
MODEL_TYPE = CONFIG["live_trading"].get("model_type", "lstm")
CONFIDENCE_THRESHOLDS = _TR_CFG.get("confidence_thresholds", {"low_max": 0.45, "avg_max": 0.55})
POSITION_SIZES = _TR_CFG.get("position_sizes", {"low": 0.005, "avg": 0.01, "high": 0.04})

# Paths
DATASET_RAW = PROJECT_ROOT / CONFIG["paths"]["dataset_raw"] / f"{SYMBOL}_{TIMEFRAME}.csv"
DATASET_FINAL = PROJECT_ROOT / CONFIG["paths"]["dataset_final"] / f"{SYMBOL}_{TIMEFRAME}.csv"


def _reload_config():
    """Re-read config.json from disk and refresh all module-level settings.

    Called at the top of every scheduled job so that config changes made
    via the dashboard are picked up without restarting the process.
    """
    global CONFIG, _TR_CFG, SYMBOL, TIMEFRAME, LOOKAHEAD, ATR_MULT
    global THRESHOLD, VOLUME_MIN, VOLUME_MAX, CONFIDENCE_THRESHOLDS, POSITION_SIZES
    global DATASET_RAW, DATASET_FINAL, MODEL_TYPE

    with open(_CONFIG_PATH, "r") as f:
        CONFIG = json.load(f)

    _TR_CFG = get_trading_config(CONFIG)
    SYMBOL = CONFIG["symbol"]
    TIMEFRAME = CONFIG["live_trading"]["timeframe"]
    LOOKAHEAD = _TR_CFG["lookahead"]
    ATR_MULT = _TR_CFG["atr_mult"]
    THRESHOLD = _TR_CFG["threshold"]
    VOLUME_MIN = CONFIG["live_trading"].get("volume_min", 0.01)
    VOLUME_MAX = CONFIG["live_trading"].get("volume_max", 30.0)
    MODEL_TYPE = CONFIG["live_trading"].get("model_type", "lstm")
    CONFIDENCE_THRESHOLDS = _TR_CFG.get("confidence_thresholds", {"low_max": 0.45, "avg_max": 0.55})
    POSITION_SIZES = _TR_CFG.get("position_sizes", {"low": 0.005, "avg": 0.01, "high": 0.04})
    DATASET_RAW = PROJECT_ROOT / CONFIG["paths"]["dataset_raw"] / f"{SYMBOL}_{TIMEFRAME}.csv"
    DATASET_FINAL = PROJECT_ROOT / CONFIG["paths"]["dataset_final"] / f"{SYMBOL}_{TIMEFRAME}.csv"


def _get_position_size_pct(confidence: float) -> float:
    """Return the position-size fraction (e.g. 0.01 = 1 %) for a given confidence
    using the configurable tier thresholds (same logic as backtest)."""
    low_max = CONFIDENCE_THRESHOLDS.get("low_max", 0.45)
    avg_max = CONFIDENCE_THRESHOLDS.get("avg_max", 0.55)

    if confidence < low_max:
        tier, pct = "LOW", POSITION_SIZES.get("low", 0.005)
    elif confidence < avg_max:
        tier, pct = "AVG", POSITION_SIZES.get("avg", 0.01)
    else:
        tier, pct = "HIGH", POSITION_SIZES.get("high", 0.04)

    log(f"Confidence {confidence:.2%} → {tier} tier → bet {pct:.2%} of equity")
    return pct


def _get_confidence_tier(confidence: float) -> str:
    """Return the confidence tier label (snapshot at prediction time)."""
    low_max = CONFIDENCE_THRESHOLDS.get("low_max", 0.45)
    avg_max = CONFIDENCE_THRESHOLDS.get("avg_max", 0.55)
    if confidence >= avg_max:
        return "HIGH"
    elif confidence >= low_max:
        return "AVG"
    return "LOW"


def calculate_volume(confidence: float, direction: str, mt5_service=None) -> float:
    """Convert a confidence-based capital percentage into an MT5 volume (lots).

    Steps:
      1. Determine bet-fraction from confidence tier.
      2. Get account equity (MT5 if connected, else INITIAL_CAPITAL).
      3. dollar_risk = equity * bet_fraction.
      4. volume = dollar_risk / (price * contract_size).
      5. Round to the symbol's volume step and clamp to [VOLUME_MIN, VOLUME_MAX].
    """
    pct = _get_position_size_pct(confidence)

    # --- Account equity ---
    equity = CONFIG["backtest"].get("initial_capital", 100_000)
    if mt5_service is not None:
        try:
            info = mt5_service.get_account_info()
            if info and info.get("equity"):
                equity = info["equity"]
        except Exception:
            pass

    dollar_risk = equity * pct

    # --- Symbol info for conversion ---
    price = 1.0
    contract_size = 1.0
    volume_step = 0.01
    volume_min_sym = 0.01
    volume_max_sym = 500.0

    if mt5_service is not None:
        try:
            tick = mt5_service.get_last_tick(SYMBOL)
            price = tick.get("ask", 1.0) if direction == "LONG" else tick.get("bid", 1.0)
        except Exception:
            pass
        try:
            sym_info = mt5_service.get_symbol_info(SYMBOL)
            contract_size = sym_info.get("trade_contract_size", 1.0) or 1.0
            volume_step = sym_info.get("volume_step", 0.01) or 0.01
            volume_min_sym = sym_info.get("volume_min", 0.01) or 0.01
            volume_max_sym = sym_info.get("volume_max", 500.0) or 500.0
        except Exception:
            pass

    raw_volume = dollar_risk / (price * contract_size) if (price * contract_size) else 0.01

    # Round to nearest volume step
    if volume_step > 0:
        raw_volume = round(raw_volume / volume_step) * volume_step

    # Clamp: user limits (config) intersected with broker limits (symbol)
    effective_min = max(VOLUME_MIN, volume_min_sym)
    effective_max = min(VOLUME_MAX, volume_max_sym)
    volume = max(effective_min, min(raw_volume, effective_max))

    # Final rounding (avoid floating-point dust)
    decimals = max(0, len(str(volume_step).rstrip('0').split('.')[-1]))
    volume = round(volume, decimals)

    log(f"Volume calc: equity=${equity:,.0f} × {pct:.2%} = ${dollar_risk:,.0f} "
        f"→ {raw_volume:.4f} lots → clamped {volume:.2f} "
        f"[{effective_min}-{effective_max}]")
    return volume


# Global shared state (accessible from the dashboard)
state = {
    "last_prediction": None,       # {action, probs, confidence, time}
    "last_signal_time": None,
    "pending_trade_id": None,      # id of pending trade to be executed
    "mt5_connected": False,
    "scheduler_running": False,
    "next_prediction": None,       # next run prediction
    "next_execution": None,        # next run execution
    "last_error": None,
    "model": None,                 # LSTM model loaded
    "scaler": None,
    "feature_cols": None,
}


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[\x1b[45mLiveTrader\x1b[0m {ts}] {msg}")

def recover_pending():
    """Recover any pending trade from a previous session."""
    pending = trade_logger.get_pending_trades()
    if pending:
        latest = pending[-1]
        state["pending_trade_id"] = latest["id"]
        log(f"Recovered pending trade from previous session: {latest['id']}")

# --- JOB 1: PREDICTION (23:55) ----------------------------------------------

def job_predict(mt5_service=None):
    """
    Executed each day at 23:55.
    1. Download updated data from MT5 (or use CSV if MT5 is unavailable)
    2. Feature engineering
    3. LSTM prediction
    4. If LONG/SHORT signal above threshold → create pending trade
    """
    _reload_config()
    log("=== JOB PREDICTION START ===")

    try:
        # -- 1. Data download --
        df_raw = None
        if mt5_service is not None:
            try:
                log("Download live data from MT5...")
                raw_path = dataset_utils.generate_dataset(mt5_service, symbol=SYMBOL, timeframes=[TIMEFRAME])
                if not dataset_utils.validate_dataset(raw_path):
                    raise ValueError("Dataset validation failed")
                log("Data downloaded successfully")
                state["mt5_connected"] = True
                df_raw = pd.read_csv(raw_path[0])
            except Exception as e:
                log(f"\x1b[91mMT5 download failed: {e}. Using local CSV.\x1b[0m")
                state["mt5_connected"] = False
                df_raw = None

        if df_raw is None:
            log(f"Use local dataset: {DATASET_RAW}")
            df_raw = pd.read_csv(DATASET_RAW)

        # -- 2. Feature engineering --
        log("Feature engineering...")
        path_list_final = feature_engineering.calculate_features(
            [DATASET_RAW], lookahead=LOOKAHEAD, atr_mult=ATR_MULT
        )
        df = pd.read_csv(path_list_final[0])
        df["time"] = pd.to_datetime(df["time"])

        # -- 3. Train model + prediction --
        log(f"Training {MODEL_TYPE.upper()} model...")
        if MODEL_TYPE == "mlp":
            model, scaler, feature_cols = mlp.train_mlp_model(
                df, lookahead_days=LOOKAHEAD, plot_results=False
            )
            probs = mlp.predict_next_move(model, df, feature_cols, scaler)
        else:
            model, scaler, feature_cols = lstm.train_lstm_classifier(
                df, lookahead_days=LOOKAHEAD, plot_results=False
            )
            probs = lstm.predict_next_move(model, df, feature_cols, scaler)

        # save model in state
        state["model"] = model
        state["scaler"] = scaler
        state["feature_cols"] = feature_cols

        action_idx = int(np.argmax(probs))
        actions = ["HOLD", "LONG", "SHORT"]
        action = actions[action_idx]
        confidence = float(probs[action_idx])

        prediction = {
            "action": action,
            "hold": float(probs[0]),
            "long": float(probs[1]),
            "short": float(probs[2]),
            "confidence": confidence,
            "confidence_tier": _get_confidence_tier(confidence),
            "last_close": float(df.iloc[-1]["close"]),
            "atr": float(df.iloc[-1]["ATR_14"]),
            "time": datetime.now().isoformat(),
        }
        state["last_prediction"] = prediction
        state["last_signal_time"] = datetime.now().isoformat()
        state["last_error"] = None

        log(f"Prediction: {action} (conf={confidence:.2%})")
        log(f"  HOLD={probs[0]:.4f}  LONG={probs[1]:.4f}  SHORT={probs[2]:.4f}")

        # -- 4. Create pending trade if above threshold --
        if action in ("LONG", "SHORT") and confidence >= THRESHOLD:
            trade = trade_logger.create_trade(
                direction=action,
                signal_time=datetime.now().isoformat(),
                predicted_probs={
                    "hold": float(probs[0]),
                    "long": float(probs[1]),
                    "short": float(probs[2]),
                },
                confidence=confidence,
            )
            state["pending_trade_id"] = trade["id"]
            log(f"PENDING Trade created: {trade['id']} → {action}")
        else:
            state["pending_trade_id"] = None
            log(f"No trade: signal={action}, conf={confidence:.2%} (threshold={THRESHOLD:.0%})")

        log("=== JOB PREDICTION END ===\n")

    except Exception as e:
        state["last_error"] = str(e)
        log(f"\x1b[91mERROR in job_predict: {e}\x1b[0m")
        import traceback
        traceback.print_exc()


# --- JOB 2: EXECUTION (00:05) -----------------------------------------------

def job_execute(mt5_service=None):
    """
    Executed each day at 00:05.
    Take pending order and execute on MT5.
    If MT5 is unavailable, the trade is cancelled.
    """
    _reload_config()
    log("=== JOB EXECUTION START ===")

    trade_id = state.get("pending_trade_id")
    if not trade_id:
        log("No pending trade to execute.")
        log("=== JOB EXECUTION END ===\n")
        return

    pending = trade_logger.get_pending_trades()
    trade = next((t for t in pending if t["id"] == trade_id), None)
    if not trade:
        log(f"Trade {trade_id} not found among pending trades.")
        state["pending_trade_id"] = None
        log("=== JOB EXECUTION END ===\n")
        return

    direction = trade["direction"]
    pred = state.get("last_prediction", {})
    last_close = pred.get("last_close", 0)
    atr = pred.get("atr", 0)

    # Calculate TP/SL with triple barrier
    if direction == "LONG":
        tp = last_close + ATR_MULT * atr
        sl = last_close - ATR_MULT * atr
    else:  # SHORT
        tp = last_close - ATR_MULT * atr
        sl = last_close + ATR_MULT * atr

    log(f"Trade execution {trade_id}: {direction} @ ~{last_close:.2f}")
    log(f"  TP={tp:.2f}  SL={sl:.2f}  ATR={atr:.2f}")

    # --- Dynamic volume based on model confidence ---
    confidence = trade.get("confidence", 0.0)
    volume = calculate_volume(confidence, direction, mt5_service)
    position_size_pct = _get_position_size_pct(confidence)
    log(f"  Confidence={confidence:.2%}  Volume={volume:.2f} lots  "
        f"Position%={position_size_pct:.2%}")

    # -- Send to MT5 --
    if mt5_service is not None:
        try:
            # Calculate sl_mult and tp_mult in points
            symbol_info = mt5_service.get_symbol_info(SYMBOL)
            point = symbol_info["point"]
            tp_points = abs(tp - last_close) / point
            sl_points = abs(sl - last_close) / point

            result = mt5_service.place_order(
                order_type="BUY" if direction == "LONG" else "SELL",
                symbol=SYMBOL,
                volume=volume,
                sl_mult=sl_points,
                tp_mult=tp_points,
                comment=f"AI-{trade_id}",
            )

            ticket = result.order if hasattr(result, "order") else 0
            entry = result.price if hasattr(result, "price") else last_close

            # Read back actual TP/SL set by MT5 (based on live bid/ask)
            actual_tp, actual_sl = tp, sl
            if ticket and mt5_service is not None:
                try:
                    pos = mt5_service.get_active_positions(ticket=ticket)
                    if pos and len(pos) > 0:
                        actual_tp = pos[0].tp
                        actual_sl = pos[0].sl
                        log(f"  MT5 actual TP={actual_tp:.2f}  SL={actual_sl:.2f}")
                except Exception as e2:
                    log(f"  Could not read back MT5 TP/SL: {e2}")

            trade_logger.mark_open(
                trade_id=trade_id,
                entry_price=entry,
                tp=actual_tp,
                sl=actual_sl,
                mt5_ticket=ticket,
                volume=volume,
            )
            state["pending_trade_id"] = None
            log(f"Order executed! Ticket={ticket}, Entry={entry:.2f}")

        except Exception as e:
            trade_logger.mark_cancelled(trade_id, comment=f"MT5 error: {e}")
            state["pending_trade_id"] = None
            state["last_error"] = str(e)
            log(f"\x1b[91mMT5 error: {e}\x1b[0m")

    else:
        # MT5 not connected - cancel the pending trade
        log("\x1b[93mMT5 not connected - trade cancelled\x1b[0m")
        trade_logger.mark_cancelled(trade_id, comment="MT5 not connected")
        state["pending_trade_id"] = None

    log("=== JOB EXECUTION END ===\n")





# --- SCHEDULER SETUP --------------------------------------------------------

def setup_scheduler(mt5_service=None):
    """
    Configure APScheduler with daily jobs.
    Return scheduler instance (call .start() from dashboard).
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BackgroundScheduler(timezone="UTC")

    recover_pending()

    # Job 1: Prediction at configured time (UTC)
    scheduler.add_job(
        job_predict,
        CronTrigger(hour=CONFIG["live_trading"]["prediction_hour"], minute=CONFIG["live_trading"]["prediction_minute"]),
        kwargs={"mt5_service": mt5_service},
        id="daily_prediction",
        name=f"Daily Prediction ({CONFIG['live_trading']['prediction_hour']}:{CONFIG['live_trading']['prediction_minute']:02d} UTC)",
        replace_existing=True,
    )

    # Job 2: Execution at configured time (UTC)
    scheduler.add_job(
        job_execute,
        CronTrigger(hour=CONFIG["live_trading"]["execution_hour"], minute=CONFIG["live_trading"]["execution_minute"]),
        kwargs={"mt5_service": mt5_service},
        id="daily_execution",
        name=f"Daily Execution ({CONFIG['live_trading']['execution_hour']}:{CONFIG['live_trading']['execution_minute']:02d} UTC)",
        replace_existing=True,
    )

    # Job 3: Daily equity snapshot at 23:59 UTC
    scheduler.add_job(
        trade_logger.record_equity_snapshot,
        CronTrigger(hour=23, minute=59),
        id="equity_snapshot",
        name="Daily Equity Snapshot (23:59 UTC)",
        replace_existing=True,
    )

    state["scheduler_running"] = True
    log("Scheduler setup: prediction@23:55, execution@00:05")

    return scheduler


def get_status() -> dict:
    """Return trader status for the dashboard."""

    pred = state.get("last_prediction")
    pending = trade_logger.get_pending_trades()
    open_trades = trade_logger.get_open_trades()

    return {
        "scheduler_running": state["scheduler_running"],
        "mt5_connected": state["mt5_connected"],
        "last_prediction": pred,
        "last_signal_time": state.get("last_signal_time"),
        "pending_trades": len(pending),
        "open_trades": len(open_trades),
        "total_trades": len(trade_logger.get_all()),
        "last_error": state.get("last_error"),
        "next_prediction": state.get("next_prediction"),
        "next_execution": state.get("next_execution"),
    }


# --- MANUAL TRIGGER (for testing) -------------------------------------------

def run_now(mt5_service=None):
    """Execute manual prediction + execution (test / debug)."""
    log("\x1b[36mMANUAL RUN - Prediction\x1b[0m")
    job_predict(mt5_service)
    log("\x1b[36mMANUAL RUN - Execution\x1b[0m")
    job_execute(mt5_service)
    log("\x1b[36mMANUAL RUN completed\x1b[0m")
