"""
Triple-Barrier Method Visualization
Uses real ETHUSD D1 candlestick data from datasets/final/ETHUSD_D1.csv
Demonstrates the three possible outcomes:
  1. Upper barrier hit  → label +1 (take profit)
  2. Lower barrier hit  → label -1 (stop loss)
  3. Vertical barrier hit → label  0 (time expiry)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplfinance as mpf

# ── Configuration ─────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "final", "ETHUSD_D1.csv")

N_STEPS  = 10    # vertical barrier — number of candles in each window
CONTEXT  = 3     # extra context candles shown before and after the active window
PT_WIDTH = 0.1  # take-profit distance as a fraction of the entry close
SL_WIDTH = 0.1  # stop-loss distance  as a fraction of the entry close

# ── Load OHLCV ────────────────────────────────────────────────────────────
raw = pd.read_csv(DATA_PATH, parse_dates=["time"], index_col="time")
raw.index.name = "Date"
ohlcv = raw[["open", "high", "low", "close", "tick_volume"]].rename(
    columns={"open": "Open", "high": "High", "low": "Low",
             "close": "Close", "tick_volume": "Volume"}
)

# ── Window scanner ────────────────────────────────────────────────────────
def find_example(df: pd.DataFrame, n: int, pt: float, sl: float,
                 target: int, start: int = 0) -> tuple:
    """Return (window_start_idx, hit_local_idx) for the first window
    producing the requested barrier outcome (+1, -1, or 0)."""
    for i in range(start, len(df) - n - 1):
        entry = df["Close"].iloc[i]
        upper = entry * (1 + pt)
        lower = entry * (1 - sl)
        label, hit = 0, n          # default: vertical barrier
        for j in range(1, n + 1):
            if df["High"].iloc[i + j] >= upper:
                label, hit = +1, j
                break
            if df["Low"].iloc[i + j] <= lower:
                label, hit = -1, j
                break
        if label == target:
            return i, hit
    raise RuntimeError(f"No window found for label={target} — "
                       "try adjusting PT_WIDTH / SL_WIDTH / N_STEPS")

# ── Colour palette ────────────────────────────────────────────────────────
C_UP   = "#26a69a"   # bullish candle body
C_DOWN = "#ef5350"   # bearish candle body
C_UB   = "#2DC653"   # upper barrier  (green)
C_LB   = "#E63946"   # lower barrier  (red)
C_VB   = "#FF9F1C"   # vertical barrier (orange)
C_ENTRY = "#2176AE"  # entry marker

mc    = mpf.make_marketcolors(up=C_UP, down=C_DOWN,
                               edge="inherit", wick="inherit",
                               volume="in", ohlc="inherit")
style = mpf.make_mpf_style(marketcolors=mc,
                            gridstyle=":", gridcolor="#dddddd",
                            facecolor="white", figcolor="white",
                            y_on_right=False)

SCENARIOS = [
    (+1, "Upper Barrier Hit\n(Take Profit  |  label = +1)", C_UB,  "triple_barrier_upper.png"),
    (-1, "Lower Barrier Hit\n(Stop Loss    |  label = −1)", C_LB,  "triple_barrier_lower.png"),
    ( 0, "Vertical Barrier Hit\n(Time Expiry  |  label = 0)",  C_VB, "triple_barrier_vertical.png"),
]

# ── Build combined figure ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Triple-Barrier Method  —  Three Possible Outcomes  (ETHUSD D1)",
    fontsize=14, fontweight="bold", y=1.02
)

for col, (target_label, title, hit_color, _fname) in enumerate(SCENARIOS):
    # Ensure there are enough candles before the window for context
    start_idx, hit_local = find_example(
        ohlcv, N_STEPS, PT_WIDTH, SL_WIDTH, target_label, start=CONTEXT
    )
    # Full slice: CONTEXT candles before + active window + CONTEXT candles after
    slice_start = start_idx - CONTEXT
    slice_end   = start_idx + N_STEPS + 1 + CONTEXT
    window = ohlcv.iloc[slice_start:slice_end].copy()

    # x-positions within the extended window
    x_entry  = CONTEXT                  # first candle of the active window
    x_last   = CONTEXT + N_STEPS        # vertical barrier (end of active window)
    x_hit    = CONTEXT + hit_local      # candle where barrier was hit

    entry = window["Close"].iloc[x_entry]
    upper = entry * (1 + PT_WIDTH)
    lower = entry * (1 - SL_WIDTH)
    ax    = axes[col]

    # ── candlestick chart via mplfinance ──────────────────────────────────
    mpf.plot(window, type="candle", style=style, ax=ax, volume=False)

    # ── shade the active window to distinguish it from context candles ────
    ax.axvspan(x_entry - 0.5, x_last + 0.5,
               color="#e8f4fd", alpha=0.45, zorder=0)
    # Faint vertical separators
    ax.axvline(x_entry - 0.5, color="#aaaaaa", linewidth=0.8,
               linestyle="-", zorder=1)
    ax.axvline(x_last  + 0.5, color="#aaaaaa", linewidth=0.8,
               linestyle="-", zorder=1)

    # ── three barriers (confined to the active window only) ───────────────
    x_left  = x_entry - 0.5
    x_right = x_last  + 0.5
    # Horizontal barriers: span only from window start to window end
    ax.hlines(upper, x_left, x_right, colors=C_UB, linestyles="--", linewidth=1.6, zorder=2)
    ax.hlines(lower, x_left, x_right, colors=C_LB, linestyles="--", linewidth=1.6, zorder=2)
    # Vertical barrier: span only between the two horizontal barriers
    ax.vlines(x_last, lower, upper, colors=C_VB, linestyles="--", linewidth=1.6, zorder=2)

    # ── entry marker (open circle at first active candle) ─────────────────
    ax.scatter(x_entry, entry, color="white", edgecolors=C_ENTRY,
               s=80, zorder=6, linewidths=2)

    # ── barrier-hit marker ────────────────────────────────────────────────
    if target_label == +1:
        hit_y = upper
    elif target_label == -1:
        hit_y = lower
    else:
        hit_y = window["Close"].iloc[x_hit]

    ax.scatter(x_hit, hit_y, color=hit_color,
               s=150, zorder=7, marker="*")

    # ── label badge ───────────────────────────────────────────────────────
    lbl_txt = f"label = {'+1' if target_label == 1 else target_label}"
    ax.text(0.97, 0.04, lbl_txt,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, fontweight="bold", color=hit_color,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=hit_color, lw=1.5))

    # ── manual legend ─────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=C_UB, linestyle="--", linewidth=1.5,
               label=f"Barriera superiore"),
        Line2D([0], [0], color=C_LB, linestyle="--", linewidth=1.5,
               label=f"Barriera inferiore"),
        Line2D([0], [0], color=C_VB, linestyle="--", linewidth=1.5,
               label=f"Barriera verticale ({N_STEPS} candele)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=hit_color,
               markersize=10, label="Hit della barriera"),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5,
              loc="upper left", framealpha=0.85)

    # ── axis formatting ───────────────────────────────────────────────────
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.set_xlabel("Candle index", fontsize=9, labelpad=20)
    ax.set_ylabel("Price (USD)", fontsize=9)
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()

# ── Save each subplot as a separate file ─────────────────────────────────
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for ax, (*_, fname) in zip(axes, SCENARIOS):
    bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(fname, dpi=150, bbox_inches=bbox)
    print(f"Saved → {fname}")

plt.show()
