"""
Live Trader — Scheduler + Pipeline for automatic trading ETHUSD.

Daily workflow:
  23:55  ➜  Download data from MT5, feature engineering, LSTM prediction
  00:05  ➜  If segnal is not HOLD, send order to MT5 with TP/SL (triple barrier)

Everything is loggato in data/live_trades.db (SQLite) with trade_logger.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

with open(PROJECT_ROOT / "config.json", "r") as f:
    CONFIG = json.load(f)

import machine_learning.lstm_classifier as lstm
from dataset_utils import dataset_utils, feature_engineering
from live_trading import trade_logger

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SYMBOL = CONFIG["symbol"]
TIMEFRAME = CONFIG["live_trading"]["timeframe"]
LOOKAHEAD = CONFIG["trading"]["lookahead"]
ATR_MULT = CONFIG["trading"]["atr_mult"]
THRESHOLD = CONFIG["trading"]["threshold"]
VOLUME = CONFIG["live_trading"]["volume"]
POSITION_SIZE = CONFIG["live_trading"]["position_size"]
INITIAL_CAPITAL = CONFIG["trading"]["initial_capital"]

# Paths
DATASET_RAW = PROJECT_ROOT / CONFIG["paths"]["dataset_raw"] / f"{SYMBOL}_{TIMEFRAME}.csv"
DATASET_FINAL = PROJECT_ROOT / CONFIG["paths"]["dataset_final"] / f"{SYMBOL}_{TIMEFRAME}.csv"

# Stato globale condiviso (accessibile dalla dashboard)
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

# ─── JOB 1: PREDICTION (23:55) ──────────────────────────────────────────────

def job_predict(mt5_service=None):
    """
    Executed each day at 23:55.
    1. Download updated data from MT5 (or usa CSV se MT5 non disponibile)
    2. Feature engineering
    3. LSTM prediction
    4. Se segnale LONG/SHORT sopra soglia → crea trade pending
    """
    log("═══ JOB PREDICTION START ═══")

    try:
        # ── 1. Data download ──
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

        # ── 2. Feature engineering ──
        log("Feature engineering...")
        path_list_final = feature_engineering.calculate_features(
            [DATASET_RAW], lookahead=LOOKAHEAD, atr_mult=ATR_MULT
        )
        df = pd.read_csv(path_list_final[0])
        df["time"] = pd.to_datetime(df["time"])

        # ── 3. Train model + prediction ──
        log("Training LSTM model...")
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
            "last_close": float(df.iloc[-1]["close"]),
            "atr": float(df.iloc[-1]["ATR_14"]),
            "time": datetime.now().isoformat(),
        }
        state["last_prediction"] = prediction
        state["last_signal_time"] = datetime.now().isoformat()
        state["last_error"] = None

        log(f"Prediction: {action} (conf={confidence:.2%})")
        log(f"  HOLD={probs[0]:.4f}  LONG={probs[1]:.4f}  SHORT={probs[2]:.4f}")

        # ── 4. Create pending trade if above threshold ──
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

        log("═══ JOB PREDICTION END ═══\n")

    except Exception as e:
        state["last_error"] = str(e)
        log(f"\x1b[91mERROR in job_predict: {e}\x1b[0m")
        import traceback
        traceback.print_exc()


# ─── JOB 2: EXECUTION (00:05) ───────────────────────────────────────────────

def job_execute(mt5_service=None):
    """
    Executed each day at 00:05.
    Take pending order and execute on MT5.
    If MT5 is unavailable, the trade is marked as cancelled
    """
    log("═══ JOB EXECUTION START ═══")

    trade_id = state.get("pending_trade_id")
    if not trade_id:
        log("No pending trade to execute.")
        log("═══ JOB EXECUTION END ═══\n")
        return

    pending = trade_logger.get_pending_trades()
    trade = next((t for t in pending if t["id"] == trade_id), None)
    if not trade:
        log(f"Trade {trade_id} not found among pending trades.")
        state["pending_trade_id"] = None
        log("═══ JOB EXECUTION END ═══\n")
        return

    direction = trade["direction"]
    pred = state.get("last_prediction", {})
    last_close = pred.get("last_close", 0)
    atr = pred.get("atr", 0)

    # Calculate TP/SL con triple barrier
    if direction == "LONG":
        tp = last_close + ATR_MULT * atr
        sl = last_close - ATR_MULT * atr
    else:  # SHORT
        tp = last_close - ATR_MULT * atr
        sl = last_close + ATR_MULT * atr

    log(f"Trade execution {trade_id}: {direction} @ ~{last_close:.2f}")
    log(f"  TP={tp:.2f}  SL={sl:.2f}  ATR={atr:.2f}")

    # ── Send to MT5 ──
    if mt5_service is not None:
        try:
            # Calculate sl_mult e tp_mult in points
            symbol_info = mt5_service.get_symbol_info(SYMBOL)
            point = symbol_info["point"]
            tp_points = abs(tp - last_close) / point
            sl_points = abs(sl - last_close) / point

            result = mt5_service.place_order(
                order_type="BUY" if direction == "LONG" else "SELL",
                symbol=SYMBOL,
                volume=VOLUME,
                sl_mult=sl_points,
                tp_mult=tp_points,
                comment=f"AI-{trade_id}",
            )

            ticket = result.order if hasattr(result, "order") else 0
            entry = result.price if hasattr(result, "price") else last_close

            trade_logger.mark_open(
                trade_id=trade_id,
                entry_price=entry,
                tp=tp,
                sl=sl,
                mt5_ticket=ticket,
            )
            state["pending_trade_id"] = None
            log(f"Order executed! Ticket={ticket}, Entry={entry:.2f}")

        except Exception as e:
            trade_logger.mark_cancelled(trade_id, comment=f"MT5 error: {e}")
            state["pending_trade_id"] = None
            state["last_error"] = str(e)
            log(f"\x1b[91mMT5 error: {e}\x1b[0m")

    else:
        # Simulation without MT5 (paper trading)
        log("\x1b[93mMT5 not connected — Paper Trading mode\x1b[0m")
        trade_logger.mark_open(
            trade_id=trade_id,
            entry_price=last_close,
            tp=tp,
            sl=sl,
            mt5_ticket=0,
        )
        state["pending_trade_id"] = None
        log(f"Trade simulato aperto: entry={last_close:.2f}")

    log("═══ JOB EXECUTION END ═══\n")


# ─── JOB 3: CHECK OPEN POSITIONS (each hour) ─────────────────────────────────

def job_check_positions(mt5_service=None):
    """
    Check if open trades have been closed (TP/SL raggiunto).
    Update trade_logger accordingly.
    """
    open_trades = trade_logger.get_open_trades()
    if not open_trades:
        return

    log(f"Check {len(open_trades)} open trades...")

    for trade in open_trades:
        ticket = trade.get("mt5_ticket", 0)

        # ── Check with MT5 ──
        if mt5_service is not None and ticket:
            try:
                positions = mt5_service.get_active_positions()
                if positions is not None:
                    pos = positions[positions["ticket"] == ticket]
                    if pos.empty:
                        # Closed position — get results from history deals
                        _close_from_history(mt5_service, trade)
                else:
                    # No open positions, may be closed
                    _close_from_history(mt5_service, trade)
            except Exception as e:
                log(f"\x1b[91mError checking ticket {ticket}: {e}\x1b[0m")

        # ── Simulation: check with data from CSV ──
        elif ticket == 0:
            _check_simulated_trade(trade)


def _close_from_history(mt5_service, trade):
    """Search in history deals the trade result."""
    try:
        from datetime import timezone
        exec_time = datetime.fromisoformat(trade["exec_time"]) if trade["exec_time"] else datetime.now() - timedelta(days=30)
        deals = mt5_service.get_history_deals(
            from_date=exec_time,
            to_date=datetime.now(timezone.utc),
        )
        if deals is not None and not deals.empty:
            # Search closing deal
            entry = trade["entry_price"]
            direction = trade["direction"]
            deal = deals[deals["comment"].str.contains(trade["id"], na=False)]
            if not deal.empty:
                exit_price = float(deal.iloc[-1]["price"])
                if direction == "LONG":
                    pnl = (exit_price - entry) / entry * POSITION_SIZE
                else:
                    pnl = (entry - exit_price) / entry * POSITION_SIZE
                trade_logger.mark_closed(trade["id"], exit_price, pnl)
                log(f"\x1b[92mTrade {trade['id']} closed with MT5: exit={exit_price:.2f}, PnL={pnl:.4%}\x1b[0m")
    except Exception as e:
        log(f"\x1b[91mError retrieving history: {e}\x1b[0m")


def _check_simulated_trade(trade):
    """
    Simulated check: use last price from CSV to see if TP or SL hit.
    """
    try:
        df = pd.read_csv(DATASET_RAW)
        if df.empty:
            return
        current_price = float(df.iloc[-1]["close"])
        entry = trade["entry_price"]
        tp = trade["tp"]
        sl = trade["sl"]
        direction = trade["direction"]

        hit_tp = (direction == "LONG" and current_price >= tp) or \
                 (direction == "SHORT" and current_price <= tp)
        hit_sl = (direction == "LONG" and current_price <= sl) or \
                 (direction == "SHORT" and current_price >= sl)

        if hit_tp or hit_sl:
            exit_price = tp if hit_tp else sl
            if direction == "LONG":
                pnl = (exit_price - entry) / entry * POSITION_SIZE
            else:
                pnl = (entry - exit_price) / entry * POSITION_SIZE
            trade_logger.mark_closed(trade["id"], exit_price, pnl)
            result = "\x1b[92mTP\x1b[0m" if hit_tp else "\x1b[91mSL ❌\x1b[0m"
            log(f"Simulated trade {trade['id']} close ({result}): PnL={pnl:.4%}")

    except Exception:
        pass


# ─── SCHEDULER SETUP ────────────────────────────────────────────────────────

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

    # Job 3: Check positions each hour
    scheduler.add_job(
        job_check_positions,
        CronTrigger(minute=CONFIG["live_trading"]["check_positions_minute"]),
        kwargs={"mt5_service": mt5_service},
        id="check_positions",
        name="Check Open Positions (each hour)",
        replace_existing=True,
    )

    state["scheduler_running"] = True
    log("Scheduler setup: prediction@23:55, execution@00:05, check@xx:30")

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


# ─── MANUAL TRIGGER (per testing) ───────────────────────────────────────────

def run_now(mt5_service=None):
    """Execute manual prediction + execution (test / debug)."""
    log("\x1b[36mMANUAL RUN — Prediction\x1b[0m")
    job_predict(mt5_service)
    log("\x1b[36mMANUAL RUN — Execution\x1b[0m")
    job_execute(mt5_service)
    log("\x1b[36mMANUAL RUN completed\x1b[0m")
