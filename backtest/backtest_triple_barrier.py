import json
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import machine_learning.lstm_classifier as lstm
import machine_learning.mlp as mlp
from pathlib import Path

with open(Path(__file__).resolve().parent.parent / "config.json", "r") as f:
    CONFIG = json.load(f)

PRINT_WIDTH = CONFIG["print_width"]


class BacktestResult:
    def __init__(self, equity_curve: pd.Series, trade_returns: pd.Series, summary: dict):
        self.equity_curve = equity_curve
        self.trade_returns = trade_returns
        self.summary = summary


def backtest_triple_barrier(df: pd.DataFrame, backtest_window: int, predict_window: int, initial_capital: float, lookahead: int, atr_mult: float = 1.0, threshold: float = 0.34, position_size: float = 0.01, model_type: str = "lstm"):
    """
    Backtest a classifier trained with Triple Barrier labels on the last `backtest_window` samples, simulating predictions for the next `predict_window` samples.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing OHLC data, ATR_14, and target columns.
        Required columns: 'open', 'high', 'low', 'close', 'ATR_14', 'target'.
    backtest_window : int
        Number of most recent samples to include in the backtest (e.g. 365 for 1 year of daily data).
    predict_window : int
        Number of future samples to simulate predictions for (e.g. 30 for 1 month).
    initial_capital : float
        Initial capital for the backtest.
    lookahead : int
        Lookahead window (number of bars) used to create the labels.
    atr_mult : float
        ATR multiplier for both take profit and stop loss (default 1.0).
    threshold : float
        Minimum predicted probability required to take a trade (default 0.34).
    position_size : float
        Proportion of capital to risk per trade (default 0.01 for 1%).
    model_type : str
        Model to use: 'lstm' or 'mlp' (default 'lstm').
    """


    print("\n" + f" WALK-FORWARD BACKTEST ({model_type.upper()}) ".center(PRINT_WIDTH, "="))
    n = len(df)
    start_point = n - backtest_window
    
    min_context = lstm.SEQ_LEN if model_type == "lstm" else 1
    if start_point < min_context + lookahead:
        raise ValueError("Dataset too short for the backtest_window and lookahead requested.")

    all_preds = np.zeros(n, dtype=int)
    all_probs = np.zeros((n, 3), dtype=float)
    prediction_mask = np.zeros(n, dtype=bool)

    # --- WALK-FORWARD CICLE ---
    for candle in range(start_point, n, predict_window):
        end_window = min(candle + predict_window, n)
        print(f"\n  -> Simulation: Train on data until index {candle} | Test: {candle}->{end_window}")
        
        # 1. Train on "Known past"
        current_train_df = df.iloc[:candle].copy()
        
        # Train internally manage Train/Val split using tail of current_train_df
        if model_type == "mlp":
            model, scaler, feature_cols = mlp.train_mlp_model(
                current_train_df,
                lookahead_days=lookahead,
                plot_results=False
            )
        else:
            model, scaler, feature_cols = lstm.train_lstm_classifier(
                current_train_df, 
                lookahead_days=lookahead,
                plot_results=False
            )
        
        # 2. Predict on "Future unseen"
        if model_type == "mlp":
            preds, probs, indices = mlp.batch_predict_mlp(
                model, df, feature_cols, scaler,
                start_idx=candle, end_idx=end_window
            )
        else:
            preds, probs, indices = lstm.batch_predict_lstm(
                model, df, feature_cols, scaler, lstm.SEQ_LEN, 
                start_idx=candle, end_idx=end_window
            )
        
        if len(indices) > 0:
            all_preds[indices] = preds
            all_probs[indices] = probs
            prediction_mask[indices] = True
            
    # --- EQUITY CALCULATION ---
    final_preds = np.zeros(n, dtype=int)
    
    for i in np.where(prediction_mask)[0]:
        cls = all_preds[i]
        prob = all_probs[i, cls]
        
        if prob < threshold:
            final_preds[i] = 0 # Hold if low confidence
        else:
            # Mapping from LSTM Classes (0:Hold, 1:Long, 2:Short) to Signals (-1, 0, 1)
            if cls == 1: final_preds[i] = 1
            elif cls == 2: final_preds[i] = -1
            else: final_preds[i] = 0

    # --- WALK-FORWARD DIAGNOSTIC ---
    pred_indices = np.where(prediction_mask)[0]
    if len(pred_indices) > 0:
        n_long = (final_preds[pred_indices] == 1).sum()
        n_short = (final_preds[pred_indices] == -1).sum()
        n_hold = (final_preds[pred_indices] == 0).sum()
        print(f"\n  -> Prediction Distribution: LONG={n_long}, SHORT={n_short}, HOLD={n_hold}")
        
        # Compare predictions vs actual targets
        trade_indices = pred_indices[final_preds[pred_indices] != 0]
        if len(trade_indices) > 0:
            actual_targets = df['target'].values[trade_indices]
            correct = (final_preds[trade_indices] == actual_targets).sum()
            print(f"  -> Signal Accuracy (pred vs label): {correct}/{len(trade_indices)} = {correct/len(trade_indices):.2%}")
            
            # Per-class accuracy
            for signal, name in [(1, 'LONG'), (-1, 'SHORT')]:
                mask = final_preds[trade_indices] == signal
                if mask.sum() > 0:
                    cls_correct = (actual_targets[mask] == signal).sum()
                    print(f"     {name}: {cls_correct}/{mask.sum()} correct = {cls_correct/mask.sum():.2%}")

    backtest_slice = slice(start_point, n)
    
    return backtest_calc_equity(
        df.iloc[backtest_slice].copy(),
        final_preds[backtest_slice],
        all_probs[backtest_slice],
        initial_capital,
        lookahead,
        atr_mult,
        position_size=position_size
    )

def backtest_calc_equity(df: pd.DataFrame, pred_arr: np.ndarray, probs_arr: np.ndarray, initial_capital: float, lookahead: int, atr_mult: float, position_size: float):
    """Calculate equity curve given the signals."""
    required_cols = {'open', 'high', 'low', 'close', 'ATR_14', 'target'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"\x1b[91;1m X-> DataFrame missing required columns: {missing_cols}\x1b[0m")
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    if lookahead <= 0:
        print(f"\x1b[91;1m X-> lookahead must be a positive integer\x1b[0m")
        raise ValueError("lookahead must be a positive integer")

    print(f"  -> Initial Capital: ${initial_capital}")

    n = len(df)
    # Extract arrays from DataFrame
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    close_arr = df['close'].values
    atr_arr = df['ATR_14'].values
    time_arr = df['time'].values
    true_arr = np.asarray(df['target'].values.astype(int))

    if pred_arr.shape[0] != n:
        print(f"\x1b[91;1m X-> y_pred must have the same length as the DataFrame\x1b[0m")
        raise ValueError("y_pred must have the same length as the DataFrame")

    valid_labels = {-1, 0, 1}
    if not set(np.unique(true_arr)).issubset(valid_labels):
        print(f"\x1b[91;1m X-> target must contain only -1, 0, 1 labels\x1b[0m")
        raise ValueError("target must contain only -1, 0, 1 labels")
    if not set(np.unique(pred_arr)).issubset(valid_labels):
        print(f"\x1b[91;1m X-> y_pred must contain only -1, 0, 1 labels\x1b[0m")
        raise ValueError("y_pred must contain only -1, 0, 1 labels")

    # Calculate actual returns based on Triple Barrier logic
    equity = [initial_capital]
    trade_returns = [] # percentage returns per trade
    trades = [] # list of dicts with trade details

    current_capital = initial_capital

    for i in range(n-1):
        if pred_arr[i] == 0:
            # No trade taken
            equity.append(current_capital)
            trade_returns.append(0.0)
            continue
        
        # The signal is generated at close[i], barriers are calculated from close[i]
        signal_price = close_arr[i]
        atr = atr_arr[i]
        
        # Entry price is the open of the next candle (realistic execution)
        entry_price = open_arr[i+1]

        if atr <= 0 or np.isnan(atr):
            equity.append(current_capital)
            trade_returns.append(0.0)
            continue

        # Calculate TP/SL levels based on signal_price (close[i]) - MATCHES LABELING LOGIC
        # Note: barriers are SYMMETRIC and based on signal_price, not entry_price
        upper_barrier = signal_price + atr_mult * atr
        lower_barrier = signal_price - atr_mult * atr
        
        # TP/SL depend on direction
        if pred_arr[i] == 1:  # Long position
            tp_price = upper_barrier
            sl_price = lower_barrier
        else:  # Short position (pred_arr[i] == -1)
            tp_price = lower_barrier
            sl_price = upper_barrier
            
        # Simulate trade over lookahead window (j starts at i+1, same as labeling uses range(1, len(df)))
        exit_price = None
        for j in range(i + 1, min(i + lookahead + 1, len(df))):
            high = high_arr[j]
            low = low_arr[j]
            open_price = open_arr[j]
            
            upper_hit = high >= upper_barrier
            lower_hit = low <= lower_barrier
            
            if pred_arr[i] == 1:  # Long
                if upper_hit and lower_hit:
                    # Both barriers hit in same candle - use open price to infer direction (same as labeling)
                    if open_price >= signal_price:
                        exit_price = sl_price  # Started high, likely went down first
                    else:
                        exit_price = tp_price  # Started low, likely went up first
                    break
                elif upper_hit:
                    exit_price = tp_price
                    break
                elif lower_hit:
                    exit_price = sl_price
                    break
            else:  # Short
                if upper_hit and lower_hit:
                    # Both barriers hit in same candle - use open price to infer direction (same as labeling)
                    if open_price >= signal_price:
                        exit_price = tp_price  # Started high, likely went down first (good for short)
                    else:
                        exit_price = sl_price  # Started low, likely went up first (bad for short)
                    break
                elif lower_hit:
                    exit_price = tp_price
                    break
                elif upper_hit:
                    exit_price = sl_price
                    break
        
        if exit_price is None:
            # Time barrier hit - exit at close of last bar in window
            if i + lookahead < n:
                exit_price = close_arr[i + lookahead]
            else:
                exit_price = close_arr[-1]

        # Calculate return based on entry_price (realistic P&L from actual entry)
        raw_return = 0.0
        if pred_arr[i] == 1:
            raw_return = (exit_price - entry_price) / entry_price
        else:
            raw_return = (entry_price - exit_price) / entry_price

        net_return = raw_return - CONFIG["trading"]["commission"]  # Subtract commission
        
        # Adjust position size based on confidence (probs_arr[i])
        confidence = probs_arr[i, 1] if pred_arr[i] == 1 else probs_arr[i, 2]  # LONG prob if long, SHORT prob if short
        if confidence < 0.45:
            adjusted_position_size = position_size * 0.5 # reduce position size for low confidence (half of base position size)
            print(f"  \x1b[91;1m-> Low confidence ({confidence:.2%})\x1b[0m")
        elif confidence < 0.50:
            adjusted_position_size = position_size # base position size (1% of capital)
            print(f"  \x1b[93;1m-> Medium confidence ({confidence:.2%})\x1b[0m")
        else:
            adjusted_position_size = position_size * 4 # increase position size for high confidence (double base position size)
            print(f"  \x1b[92;1m-> HIGH confidence ({confidence:.2%})\x1b[0m")

        equity_change = current_capital * net_return * adjusted_position_size
        current_capital += equity_change

        trade_returns.append(net_return)
        equity.append(current_capital)
        
        trades.append({
            "time": str(pd.Timestamp(time_arr[i])),
            "direction": "LONG" if pred_arr[i] == 1 else "SHORT",
            "entry": round(float(entry_price), 2),
            "tp": round(float(tp_price), 2),
            "sl": round(float(sl_price), 2),
            "return_pct": net_return,
            "win": bool(net_return > 0),
            "confidence": confidence,
        })

    # Build date index for equity curve
    # equity has n entries (one per candle from i=0..n-2, plus initial capital at position 0)
    dates = pd.to_datetime(df['time'].values)
    # equity[0] = initial capital (before first candle), equity[1..n-1] = after candle 0..n-2
    equity_dates = dates[:len(equity)]

    trade_returns = pd.Series(trade_returns, name="trade_returns")

    n_trades = int((pred_arr != 0).sum())
    n_wins = int((trade_returns > 0).sum())
    n_losses = int((trade_returns < 0).sum())
    hit_rate = float(n_wins / n_trades) if n_trades > 0 else 0.0
    avg_return = float(trade_returns[pred_arr[:-1] != 0].mean()) if n_trades > 0 else 0.0
    volatility = float(trade_returns[pred_arr[:-1] != 0].std(ddof=0)) if n_trades > 0 else 0.0

    running_max = pd.Series(equity).cummax()
    drawdown = (pd.Series(equity) / running_max) - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    summary = {
        "samples": n,
        "trades": n_trades,
        "wins": n_wins,
        "losses": n_losses,
        "hit_rate": hit_rate,
        "avg_return": avg_return,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "final_equity": float(equity[-1]) if len(equity) else initial_capital,
    }

    return BacktestResult(
        equity_curve=pd.Series(equity, index=equity_dates),
        trade_returns=trade_returns,
        summary=summary,
    ), trades