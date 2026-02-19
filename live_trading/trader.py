"""
Live Trader — Scheduler + Pipeline per il trading automatico ETHUSD.

Workflow giornaliero:
  23:55  ➜  Download dati da MT5, feature engineering, LSTM prediction
  00:05  ➜  Se il segnale non è HOLD, invia ordine a MT5 con TP/SL (triple barrier)

Tutto viene loggato in data/live_trades.json tramite trade_logger.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Aggiungi root al path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import machine_learning.lstm as lstm
from dataset_utils import feature_engineering
from live_trading import trade_logger

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SYMBOL = "ETHUSD"
TIMEFRAME = "D1"
LOOKAHEAD = 10
ATR_MULT = 2.0
THRESHOLD = 0.40            # confidenza minima per aprire un trade
VOLUME = 0.01               # lotti da tradare
POSITION_SIZE = 0.1         # 10% del capitale per trade (usato per il log)
INITIAL_CAPITAL = 100_000

# Paths
DATASET_RAW = PROJECT_ROOT / "datasets" / "raw" / "ETHUSD_D1_3082.csv"
DATASET_FINAL = PROJECT_ROOT / "datasets" / "final" / "ETHUSD_D1_3082.csv"

# Stato globale condiviso (accessibile dalla dashboard)
state = {
    "last_prediction": None,       # {action, probs, confidence, time}
    "last_signal_time": None,
    "pending_trade_id": None,      # id del trade in attesa di esecuzione
    "mt5_connected": False,
    "scheduler_running": False,
    "next_prediction": None,       # prossimo run prediction
    "next_execution": None,        # prossimo run execution
    "last_error": None,
    "model": None,                 # modello LSTM caricato
    "scaler": None,
    "feature_cols": None,
}


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[LiveTrader {ts}] {msg}")


# ─── JOB 1: PREDICTION (23:55) ──────────────────────────────────────────────

def job_predict(mt5_service=None):
    """
    Eseguito ogni giorno alle 23:55.
    1. Scarica dati aggiornati da MT5 (o usa CSV se MT5 non disponibile)
    2. Feature engineering
    3. LSTM prediction
    4. Se segnale LONG/SHORT sopra soglia → crea trade pending
    """
    log("═══ JOB PREDICTION START ═══")

    try:
        # ── 1. Acquisizione dati ──
        df_raw = None
        if mt5_service is not None:
            try:
                log("Scaricamento dati live da MT5...")
                df_raw = mt5_service.get_historical_data_pos(
                    symbol=SYMBOL, timeframe=TIMEFRAME, pos=0, count=3500
                )
                # Salva come CSV aggiornato
                df_raw.to_csv(DATASET_RAW, index=False)
                log(f"Dati MT5 scaricati: {len(df_raw)} candele")
                state["mt5_connected"] = True
            except Exception as e:
                log(f"⚠ MT5 download fallito: {e}. Uso CSV locale.")
                state["mt5_connected"] = False
                df_raw = None

        if df_raw is None:
            log(f"Uso dataset locale: {DATASET_RAW}")
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
        model, scaler, feature_cols = lstm.train_lstm_model(
            df, lookahead_days=LOOKAHEAD, plot_results=False
        )
        probs = lstm.predict_next_move(model, df, feature_cols, scaler)

        # Salva modello nello stato
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

        # ── 4. Crea trade pending se sopra soglia ──
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
            log(f"✅ Trade PENDING creato: {trade['id']} → {action}")
        else:
            state["pending_trade_id"] = None
            log(f"⏸ Nessun trade: segnale={action}, conf={confidence:.2%} (soglia={THRESHOLD:.0%})")

        log("═══ JOB PREDICTION END ═══\n")

    except Exception as e:
        state["last_error"] = str(e)
        log(f"❌ ERRORE in job_predict: {e}")
        import traceback
        traceback.print_exc()


# ─── JOB 2: EXECUTION (00:05) ───────────────────────────────────────────────

def job_execute(mt5_service=None):
    """
    Eseguito ogni giorno alle 00:05.
    Prende il trade pending e lo esegue su MT5.
    Se MT5 non è disponibile, segna il trade come cancelled.
    """
    log("═══ JOB EXECUTION START ═══")

    trade_id = state.get("pending_trade_id")
    if not trade_id:
        log("Nessun trade pending da eseguire.")
        log("═══ JOB EXECUTION END ═══\n")
        return

    pending = trade_logger.get_pending_trades()
    trade = next((t for t in pending if t["id"] == trade_id), None)
    if not trade:
        log(f"Trade {trade_id} non trovato tra i pending.")
        state["pending_trade_id"] = None
        log("═══ JOB EXECUTION END ═══\n")
        return

    direction = trade["direction"]
    pred = state.get("last_prediction", {})
    last_close = pred.get("last_close", 0)
    atr = pred.get("atr", 0)

    # Calcola TP/SL con triple barrier
    if direction == "LONG":
        tp = last_close + ATR_MULT * atr
        sl = last_close - ATR_MULT * atr
    else:  # SHORT
        tp = last_close - ATR_MULT * atr
        sl = last_close + ATR_MULT * atr

    log(f"Esecuzione trade {trade_id}: {direction} @ ~{last_close:.2f}")
    log(f"  TP={tp:.2f}  SL={sl:.2f}  ATR={atr:.2f}")

    # ── Invio a MT5 ──
    if mt5_service is not None:
        try:
            # Calcola sl_mult e tp_mult in punti
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
            log(f"✅ Ordine eseguito! Ticket={ticket}, Entry={entry:.2f}")

        except Exception as e:
            trade_logger.mark_cancelled(trade_id, comment=f"MT5 error: {e}")
            state["pending_trade_id"] = None
            state["last_error"] = str(e)
            log(f"❌ Errore MT5: {e}")

    else:
        # Simulazione senza MT5 (paper trading)
        log("⚠ MT5 non connesso — Paper Trading mode")
        trade_logger.mark_open(
            trade_id=trade_id,
            entry_price=last_close,
            tp=tp,
            sl=sl,
            mt5_ticket=0,
        )
        state["pending_trade_id"] = None
        log(f"📝 Trade simulato aperto: entry={last_close:.2f}")

    log("═══ JOB EXECUTION END ═══\n")


# ─── JOB 3: CHECK OPEN POSITIONS (ogni ora) ─────────────────────────────────

def job_check_positions(mt5_service=None):
    """
    Controlla se i trade aperti sono stati chiusi (TP/SL raggiunto).
    Aggiorna trade_logger di conseguenza.
    """
    open_trades = trade_logger.get_open_trades()
    if not open_trades:
        return

    log(f"Controllo {len(open_trades)} trade aperti...")

    for trade in open_trades:
        ticket = trade.get("mt5_ticket", 0)

        # ── Verifica tramite MT5 ──
        if mt5_service is not None and ticket:
            try:
                positions = mt5_service.get_active_positions()
                if positions is not None:
                    pos = positions[positions["ticket"] == ticket]
                    if pos.empty:
                        # Posizione chiusa — recupera risultato dagli history deals
                        _close_from_history(mt5_service, trade)
                else:
                    # Nessuna posizione aperta, potrebbe essersi chiusa
                    _close_from_history(mt5_service, trade)
            except Exception as e:
                log(f"  ⚠ Errore check ticket {ticket}: {e}")

        # ── Simulazione: controlla con dati dal CSV ──
        elif ticket == 0:
            _check_simulated_trade(trade)


def _close_from_history(mt5_service, trade):
    """Cerca nelle history deals il risultato del trade."""
    try:
        from datetime import timezone
        exec_time = datetime.fromisoformat(trade["exec_time"]) if trade["exec_time"] else datetime.now() - timedelta(days=30)
        deals = mt5_service.get_history_deals(
            from_date=exec_time,
            to_date=datetime.now(timezone.utc),
        )
        if deals is not None and not deals.empty:
            # Cerca il deal di chiusura
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
                log(f"  ✅ Trade {trade['id']} chiuso via MT5: exit={exit_price:.2f}, PnL={pnl:.4%}")
    except Exception as e:
        log(f"  ⚠ Errore recupero history: {e}")


def _check_simulated_trade(trade):
    """
    Controllo simulato: guarda l'ultimo prezzo dal CSV per vedere se TP o SL è stato toccato.
    In produzione con MT5, questo non serve.
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
            result = "TP ✅" if hit_tp else "SL ❌"
            log(f"  📝 Trade simulato {trade['id']} chiuso ({result}): PnL={pnl:.4%}")

    except Exception:
        pass


# ─── SCHEDULER SETUP ────────────────────────────────────────────────────────

def setup_scheduler(mt5_service=None):
    """
    Configura APScheduler con i job giornalieri.
    Ritorna l'istanza scheduler (da fare .start() nella dashboard).
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BackgroundScheduler(timezone="UTC")

    # Job 1: Prediction alle 23:55 UTC
    scheduler.add_job(
        job_predict,
        CronTrigger(hour=23, minute=55),
        kwargs={"mt5_service": mt5_service},
        id="daily_prediction",
        name="Daily Prediction (23:55 UTC)",
        replace_existing=True,
    )

    # Job 2: Execution alle 00:05 UTC
    scheduler.add_job(
        job_execute,
        CronTrigger(hour=0, minute=5),
        kwargs={"mt5_service": mt5_service},
        id="daily_execution",
        name="Daily Execution (00:05 UTC)",
        replace_existing=True,
    )

    # Job 3: Check posizioni ogni ora
    scheduler.add_job(
        job_check_positions,
        CronTrigger(minute=30),  # ogni ora al minuto 30
        kwargs={"mt5_service": mt5_service},
        id="check_positions",
        name="Check Open Positions (ogni ora)",
        replace_existing=True,
    )

    state["scheduler_running"] = True
    log("Scheduler configurato: prediction@23:55, execution@00:05, check@xx:30")

    return scheduler


def get_status() -> dict:
    """Ritorna lo stato del trader per la dashboard."""
    from apscheduler.schedulers.background import BackgroundScheduler

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
    """Esegue manualmente prediction + execution (per test / debug)."""
    log("🔧 MANUAL RUN — Prediction")
    job_predict(mt5_service)
    log("🔧 MANUAL RUN — Execution")
    job_execute(mt5_service)
    log("🔧 MANUAL RUN completato")
