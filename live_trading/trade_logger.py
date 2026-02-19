"""
Trade Logger — persiste i trade live su file JSON.
Ogni trade ha: id, datetime_signal, datetime_exec, direction, entry_price,
tp, sl, status (pending | open | closed | cancelled), exit_price, pnl_pct, close_time.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRADES_FILE = DATA_DIR / "live_trades.json"

_lock = Lock()


def _ensure_file() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not TRADES_FILE.exists():
        TRADES_FILE.write_text(json.dumps([], indent=2))


def _read_all() -> List[dict]:
    _ensure_file()
    with open(TRADES_FILE, "r") as f:
        return json.load(f)


def _write_all(trades: List[dict]) -> None:
    _ensure_file()
    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def create_trade(
    direction: str,
    signal_time: str,
    predicted_probs: dict,
    confidence: float,
) -> dict:
    """Registra un nuovo trade in stato *pending* (in attesa di esecuzione alle 00:05)."""
    trade = {
        "id": str(uuid.uuid4())[:8],
        "direction": direction.upper(),        # LONG | SHORT
        "signal_time": signal_time,            # quando la prediction è avvenuta
        "exec_time": None,                     # quando l'ordine MT5 è stato inviato
        "entry_price": None,
        "tp": None,
        "sl": None,
        "exit_price": None,
        "pnl_pct": None,
        "status": "pending",                   # pending → open → closed / cancelled
        "mt5_ticket": None,
        "confidence": round(confidence, 4),
        "probs": predicted_probs,
        "close_time": None,
        "comment": "",
    }
    with _lock:
        trades = _read_all()
        trades.append(trade)
        _write_all(trades)
    return trade


def update_trade(trade_id: str, **kwargs) -> Optional[dict]:
    """Aggiorna uno o più campi di un trade esistente."""
    with _lock:
        trades = _read_all()
        for t in trades:
            if t["id"] == trade_id:
                t.update(kwargs)
                _write_all(trades)
                return t
    return None


def mark_open(
    trade_id: str,
    entry_price: float,
    tp: float,
    sl: float,
    mt5_ticket: int,
    exec_time: Optional[str] = None,
) -> Optional[dict]:
    """Segna il trade come aperto dopo l'esecuzione su MT5."""
    return update_trade(
        trade_id,
        status="open",
        entry_price=round(entry_price, 2),
        tp=round(tp, 2),
        sl=round(sl, 2),
        mt5_ticket=mt5_ticket,
        exec_time=exec_time or datetime.utcnow().isoformat(),
    )


def mark_closed(
    trade_id: str,
    exit_price: float,
    pnl_pct: float,
    close_time: Optional[str] = None,
) -> Optional[dict]:
    """Segna il trade come chiuso."""
    return update_trade(
        trade_id,
        status="closed",
        exit_price=round(exit_price, 2),
        pnl_pct=round(pnl_pct, 4),
        close_time=close_time or datetime.utcnow().isoformat(),
    )


def mark_cancelled(trade_id: str, comment: str = "") -> Optional[dict]:
    """Annulla un trade pending (ad es. se MT5 non è disponibile)."""
    return update_trade(trade_id, status="cancelled", comment=comment)


def get_last_n(n: int = 10) -> List[dict]:
    """Ritorna gli ultimi *n* trade (più recenti prima)."""
    trades = _read_all()
    return list(reversed(trades[-n:]))


def get_open_trades() -> List[dict]:
    """Ritorna tutti i trade attualmente aperti."""
    return [t for t in _read_all() if t["status"] == "open"]


def get_pending_trades() -> List[dict]:
    """Ritorna tutti i trade in attesa di esecuzione."""
    return [t for t in _read_all() if t["status"] == "pending"]


def get_all() -> List[dict]:
    """Ritorna tutti i trade."""
    return _read_all()


def compute_equity_curve(initial_capital: float = 100_000) -> List[dict]:
    """
    Calcola la equity curve basata sui trade *chiusi* (in ordine cronologico).
    Ritorna lista di {time, value}.
    """
    trades = _read_all()
    closed = sorted(
        [t for t in trades if t["status"] == "closed" and t["pnl_pct"] is not None],
        key=lambda t: t["close_time"] or "",
    )
    equity = initial_capital
    curve = [{"time": None, "value": equity}]  # placeholder, will be set from first trade

    for t in closed:
        equity *= 1 + t["pnl_pct"]
        curve.append({
            "time": t["close_time"],
            "value": round(equity, 2),
        })

    # Set first point time from the first trade's exec_time
    if closed:
        curve[0]["time"] = closed[0].get("exec_time", closed[0].get("signal_time"))
    else:
        curve[0]["time"] = datetime.utcnow().isoformat()

    return curve
