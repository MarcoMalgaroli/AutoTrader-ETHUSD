"""
Trade Logger — persist live trades in a SQLite database.
Each trade has: id, direction, signal_time, exec_time, entry_price,
tp, sl, status (pending | open | closed | cancelled), exit_price, pnl_pct,
close_time, mt5_ticket, confidence, probs, comment.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_FILE = DATA_DIR / "live_trades.db"

_lock = Lock()

# Column order used when converting rows to dicts
_COLUMNS = [
    "id", "direction", "signal_time", "exec_time", "entry_price",
    "tp", "sl", "exit_price", "pnl_pct", "status", "mt5_ticket",
    "confidence", "probs", "close_time", "comment",
]


def _get_conn() -> sqlite3.Connection:
    """Return a connection to the SQLite database (creates it if needed)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id          TEXT PRIMARY KEY,
            direction   TEXT NOT NULL,
            signal_time TEXT,
            exec_time   TEXT,
            entry_price REAL,
            tp          REAL,
            sl          REAL,
            exit_price  REAL,
            pnl_pct     REAL,
            status      TEXT NOT NULL DEFAULT 'pending',
            mt5_ticket  INTEGER,
            confidence  REAL,
            probs       TEXT,
            close_time  TEXT,
            comment     TEXT DEFAULT ''
        )
    """)
    conn.commit()
    return conn


def _row_to_dict(row: tuple) -> dict:
    """Convert a database row to a dict, deserialising the probs JSON."""
    d = dict(zip(_COLUMNS, row))
    if d["probs"] is not None:
        try:
            d["probs"] = json.loads(d["probs"])
        except (json.JSONDecodeError, TypeError):
            pass
    return d


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def create_trade(
    direction: str,
    signal_time: str,
    predicted_probs: dict,
    confidence: float,
) -> dict:
    """Register a new trade in *pending* state (waiting to be executed at 00:05)."""
    trade = {
        "id": str(uuid.uuid4())[:8],
        "direction": direction.upper(),
        "signal_time": signal_time,
        "exec_time": None,
        "entry_price": None,
        "tp": None,
        "sl": None,
        "exit_price": None,
        "pnl_pct": None,
        "status": "pending",
        "mt5_ticket": None,
        "confidence": round(confidence, 4),
        "probs": predicted_probs,
        "close_time": None,
        "comment": "",
    }
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                """INSERT INTO trades
                   (id, direction, signal_time, exec_time, entry_price, tp, sl,
                    exit_price, pnl_pct, status, mt5_ticket, confidence, probs,
                    close_time, comment)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trade["id"], trade["direction"], trade["signal_time"],
                    trade["exec_time"], trade["entry_price"], trade["tp"],
                    trade["sl"], trade["exit_price"], trade["pnl_pct"],
                    trade["status"], trade["mt5_ticket"], trade["confidence"],
                    json.dumps(trade["probs"], default=str),
                    trade["close_time"], trade["comment"],
                ),
            )
            conn.commit()
        finally:
            conn.close()
    return trade


def update_trade(trade_id: str, **kwargs) -> Optional[dict]:
    """Update one or more fields of an existing trade."""
    if not kwargs:
        return None
    # Serialise probs if present
    if "probs" in kwargs and not isinstance(kwargs["probs"], str):
        kwargs["probs"] = json.dumps(kwargs["probs"], default=str)

    set_clause = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [trade_id]

    with _lock:
        conn = _get_conn()
        try:
            conn.execute(f"UPDATE trades SET {set_clause} WHERE id = ?", values)
            conn.commit()
            cur = conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM trades WHERE id = ?",
                (trade_id,),
            )
            row = cur.fetchone()
        finally:
            conn.close()
    return _row_to_dict(row) if row else None


def mark_open(
    trade_id: str,
    entry_price: float,
    tp: float,
    sl: float,
    mt5_ticket: int,
    exec_time: Optional[str] = None,
) -> Optional[dict]:
    """Mark a trade as open after execution on MT5."""
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
    """Mark a trade as closed."""
    return update_trade(
        trade_id,
        status="closed",
        exit_price=round(exit_price, 2),
        pnl_pct=round(pnl_pct, 4),
        close_time=close_time or datetime.utcnow().isoformat(),
    )


def mark_cancelled(trade_id: str, comment: str = "") -> Optional[dict]:
    """Cancel a pending trade (e.g. if MT5 is not available)."""
    return update_trade(trade_id, status="cancelled", comment=comment)


def get_last_n(n: int = 10) -> List[dict]:
    """Return the last *n* trades (most recent first)."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM trades ORDER BY rowid DESC LIMIT ?",
            (n,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return [_row_to_dict(r) for r in rows]


def get_open_trades() -> List[dict]:
    """Return all currently open trades."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM trades WHERE status = 'open'"
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return [_row_to_dict(r) for r in rows]


def get_pending_trades() -> List[dict]:
    """Return all pending trades (waiting to be executed)."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM trades WHERE status = 'pending'"
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return [_row_to_dict(r) for r in rows]


def get_all() -> List[dict]:
    """Return all trades."""
    conn = _get_conn()
    try:
        cur = conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM trades ORDER BY rowid")
        rows = cur.fetchall()
    finally:
        conn.close()
    return [_row_to_dict(r) for r in rows]


def compute_equity_curve(initial_capital: float = 100_000) -> List[dict]:
    """
    Compute the equity curve based on *closed* trades (in chronological order).
    Returns a list of {time, value}.
    """
    conn = _get_conn()
    try:
        cur = conn.execute(
            f"""SELECT {', '.join(_COLUMNS)} FROM trades
                WHERE status = 'closed' AND pnl_pct IS NOT NULL
                ORDER BY close_time""",
        )
        closed = [_row_to_dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    equity = initial_capital
    curve = [{"time": None, "value": equity}]

    for t in closed:
        equity *= 1 + t["pnl_pct"]
        curve.append({
            "time": t["close_time"],
            "value": round(equity, 2),
        })

    if closed:
        curve[0]["time"] = closed[0].get("exec_time") or closed[0].get("signal_time")
    else:
        curve[0]["time"] = datetime.now(timezone.utc).isoformat()

    return curve
