from typing import Iterable

import numpy as np
import pandas as pd

class BacktestResult:
    def __init__(self, equity_curve: pd.Series, trade_returns: pd.Series, summary: dict):
        self.equity_curve = equity_curve
        self.trade_returns = trade_returns
        self.summary = summary


def backtest_triple_barrier(df: pd.DataFrame, y_pred: Iterable[int], lookahead: int, atr_mult: float = 1.0) -> BacktestResult:
	"""
	Backtest a classifier trained with Triple Barrier labels.

	Parameters
	----------
	df : pd.DataFrame
		DataFrame containing OHLC data, ATR_14, and target columns.
		Required columns: 'open', 'high', 'low', 'close', 'ATR_14', 'target'.
	y_pred : Iterable[int]
		Predicted labels: -1, 0, +1.
	lookahead : int
		Lookahead window (number of bars) used to create the labels.
	atr_mult : float
		ATR multiplier for both take profit and stop loss (default 1.0).

	Returns
	-------
	BacktestResult
		Equity curve, per-trade returns, and a summary dictionary.
	"""

	required_cols = {'open', 'high', 'low', 'close', 'ATR_14', 'target'}
	missing_cols = required_cols - set(df.columns)
	if missing_cols:
		raise ValueError(f"DataFrame missing required columns: {missing_cols}")

	if lookahead <= 0:
		raise ValueError("lookahead must be a positive integer")

	# Extract arrays from DataFrame
	open_arr = df['open'].values
	high_arr = df['high'].values
	low_arr = df['low'].values
	close_arr = df['close'].values
	atr_arr = df['ATR_14'].values
	true_arr = df['target'].values.astype(int)

	pred_arr = np.asarray(list(y_pred), dtype=int)

	n = len(df)
	if pred_arr.shape[0] != n:
		raise ValueError("y_pred must have the same length as the DataFrame")

	valid_labels = {-1, 0, 1}
	if not set(np.unique(true_arr)).issubset(valid_labels):
		raise ValueError("target must contain only -1, 0, 1 labels")
	if not set(np.unique(pred_arr)).issubset(valid_labels):
		raise ValueError("y_pred must contain only -1, 0, 1 labels")

	# Calculate actual returns based on Triple Barrier logic
	trade_returns = np.zeros(n, dtype=float)

	for i in range(n):
		if pred_arr[i] == 0:
			# No trade taken
			trade_returns[i] = 0.0
			continue

		entry_price = close_arr[i]
		atr = atr_arr[i]

		if atr <= 0 or np.isnan(atr):
			trade_returns[i] = 0.0
			continue

		# Calculate TP/SL levels based on prediction direction
		if pred_arr[i] == 1:  # Long position
			tp_price = entry_price + atr_mult * atr
			sl_price = entry_price - atr_mult * atr
		else:  # Short position (pred_arr[i] == -1)
			tp_price = entry_price - atr_mult * atr
			sl_price = entry_price + atr_mult * atr
		# Simulate trade over lookahead window
		exit_price = entry_price
		for j in range(i + 1, min(i + lookahead + 1, n)):
			if pred_arr[i] == 1:  # Long
				if high_arr[j] >= tp_price:
					exit_price = tp_price
					break
				elif low_arr[j] <= sl_price:
					exit_price = sl_price
					break
			else:  # Short
				if low_arr[j] <= tp_price:
					exit_price = tp_price
					break
				elif high_arr[j] >= sl_price:
					exit_price = sl_price
					break
		else:
			# Time barrier hit - exit at close of last bar in window
			if i + lookahead < n:
				exit_price = close_arr[i + lookahead]
			else:
				exit_price = close_arr[-1]

		# Calculate return
		if pred_arr[i] == 1:
			trade_returns[i] = (exit_price - entry_price) / entry_price
		else:
			trade_returns[i] = (entry_price - exit_price) / entry_price

	trade_returns = pd.Series(trade_returns, name="trade_return")
	equity_curve = (1.0 + trade_returns).cumprod().rename("equity")

	n_trades = int((pred_arr != 0).sum())
	n_wins = int((trade_returns > 0).sum())
	n_losses = int((trade_returns < 0).sum())
	hit_rate = float(n_wins / n_trades) if n_trades > 0 else 0.0
	avg_return = float(trade_returns[pred_arr != 0].mean()) if n_trades > 0 else 0.0
	volatility = float(trade_returns[pred_arr != 0].std(ddof=0)) if n_trades > 0 else 0.0

	running_max = equity_curve.cummax()
	drawdown = (equity_curve / running_max) - 1.0
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
		"final_equity": float(equity_curve.iloc[-1]) if len(equity_curve) else 1.0,
	}

	return BacktestResult(
		equity_curve=equity_curve,
		trade_returns=trade_returns,
		summary=summary,
	)
