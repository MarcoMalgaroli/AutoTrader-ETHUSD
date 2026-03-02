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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS equity_snapshots (
            date        TEXT PRIMARY KEY,
            equity      REAL NOT NULL,
            balance     REAL,
            margin      REAL,
            free_margin REAL
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


# --- MT5 live data enrichment --------------------------------------------
_mt5_service = None

def set_mt5_service(service) -> None:
    """Set the MT5 service instance for live data enrichment.
    Read functions will prefer MT5 live data over the DB when available."""
    global _mt5_service
    _mt5_service = service


def _enrich_with_mt5(trades: List[dict]) -> List[dict]:
    """Enrich open trades with real-time data from MT5 active positions.
    Falls back to DB data silently if MT5 is unavailable or fails."""
    if _mt5_service is None:
        return trades
    try:
        positions_df = _mt5_service.get_active_positions()
        if positions_df is None or positions_df.empty:
            return trades
        mt5_pos = {int(row["ticket"]): row for _, row in positions_df.iterrows()}
    except Exception:
        return trades

    enriched = []
    for t in trades:
        t = dict(t)  # shallow copy — don't mutate original
        ticket = t.get("mt5_ticket", 0)
        if t["status"] == "open" and ticket and ticket in mt5_pos:
            pos = mt5_pos[ticket]
            t["entry_price"] = round(float(pos["price_open"]), 2)
            if float(pos["tp"]) != 0:
                t["tp"] = round(float(pos["tp"]), 2)
            if float(pos["sl"]) != 0:
                t["sl"] = round(float(pos["sl"]), 2)
            t["current_price"] = round(float(pos["price_current"]), 2)
            t["unrealized_pnl"] = round(float(pos["profit"]), 2)
            # Live pnl_pct from MT5 prices
            price_open = float(pos["price_open"])
            price_current = float(pos["price_current"])
            if price_open != 0:
                if t["direction"] == "LONG":
                    t["pnl_pct"] = round((price_current - price_open) / price_open, 6)
                else:
                    t["pnl_pct"] = round((price_open - price_current) / price_open, 6)
        enriched.append(t)
    return enriched


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

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
        exec_time=exec_time or datetime.now(timezone.utc).isoformat(),
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
        close_time=close_time or datetime.now(timezone.utc).isoformat(),
    )


def mark_cancelled(trade_id: str, comment: str = "") -> Optional[dict]:
    """Cancel a pending trade (e.g. if MT5 is not available)."""
    return update_trade(trade_id, status="cancelled", comment=comment)


def get_last_n(n: int = 10) -> List[dict]:
    """Return the last *n* trades (most recent first).
    Open trades are enriched with live MT5 data when available."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM trades ORDER BY rowid DESC LIMIT ?",
            (n,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return _enrich_with_mt5([_row_to_dict(r) for r in rows])


def get_open_trades() -> List[dict]:
    """Return all currently open trades.
    Enriched with live MT5 position data (TP/SL/current price/PnL) when available."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM trades WHERE status = 'open'"
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return _enrich_with_mt5([_row_to_dict(r) for r in rows])


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
    """Return all trades.
    Open trades are enriched with live MT5 data when available."""
    conn = _get_conn()
    try:
        cur = conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM trades ORDER BY rowid")
        rows = cur.fetchall()
    finally:
        conn.close()
    return _enrich_with_mt5([_row_to_dict(r) for r in rows])


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


# ----------------------------------------------------------------------
# Equity Snapshots (daily real account equity from MT5)
# ----------------------------------------------------------------------

def record_equity_snapshot() -> bool:
    """Record (or update) a daily equity snapshot.
    If no equity value is provided, fetches it from MT5 automatically.
    Uses today's date by default. Upserts so only one row per date.
    Returns True if a snapshot was recorded, False otherwise."""
    # Auto-fetch from MT5
    if _mt5_service is None:
        return False
    try:
        info = _mt5_service.get_account_info()
        if not info or "equity" not in info:
            return False
        equity = info["equity"]
        balance = info.get("balance")
        margin = info.get("margin")
        free_margin = info.get("margin_free")
    except Exception:
        return False

    date = datetime.now().strftime("%Y-%m-%d")
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                """INSERT INTO equity_snapshots (date, equity, balance, margin, free_margin)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(date) DO UPDATE SET
                       equity=excluded.equity,
                       balance=excluded.balance,
                       margin=excluded.margin,
                       free_margin=excluded.free_margin""",
                (date, round(equity, 2), balance and round(balance, 2),
                 margin and round(margin, 2), free_margin and round(free_margin, 2)),
            )
            conn.commit()
        finally:
            conn.close()
    return True


def get_equity_snapshots() -> List[dict]:
    """Return all equity snapshots ordered by date.
    If MT5 is available and today has no snapshot yet, appends live account equity."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            "SELECT date, equity, balance, margin, free_margin "
            "FROM equity_snapshots ORDER BY date"
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    snapshots = [
        {"date": r[0], "equity": r[1], "balance": r[2],
         "margin": r[3], "free_margin": r[4]}
        for r in rows
    ]

    # Append today's live MT5 equity if not already snapshotted
    if _mt5_service is not None:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            if not any(s["date"] == today for s in snapshots):
                info = _mt5_service.get_account_info()
                if info and "equity" in info:
                    snapshots.append({
                        "date": today,
                        "equity": round(float(info["equity"]), 2),
                        "balance": round(float(info.get("balance", 0)), 2),
                        "margin": round(float(info.get("margin", 0)), 2),
                        "free_margin": round(float(info.get("margin_free", 0)), 2),
                        "source": "mt5_live",
                    })
        except Exception:
            pass

    return snapshots
