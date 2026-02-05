from typing import Iterable

import numpy as np
import pandas as pd

PRINT_WIDTH = 100


class BacktestResult:
    def __init__(self, equity_curve: pd.Series, trade_returns: pd.Series, summary: dict):
        self.equity_curve = equity_curve
        self.trade_returns = trade_returns
        self.summary = summary


def backtest_triple_barrier(df: pd.DataFrame, y_pred: Iterable[int], initial_capital: float, bet: float, lookahead: int, atr_mult: float = 1.0) -> BacktestResult:
	"""
	Backtest a classifier trained with Triple Barrier labels.

	Parameters
	----------
	df : pd.DataFrame
		DataFrame containing OHLC data, ATR_14, and target columns.
		Required columns: 'open', 'high', 'low', 'close', 'ATR_14', 'target'.
	y_pred : Iterable[int]
		Predicted labels: -1, 0, +1.
	initial_capital : float
		Initial capital for the backtest.
	bet : float
		Fixed bet size per trade (as a fraction of initial capital).
	lookahead : int
		Lookahead window (number of bars) used to create the labels.
	atr_mult : float
		ATR multiplier for both take profit and stop loss (default 1.0).

	Returns
	-------
	BacktestResult
		Equity curve, per-trade returns, and a summary dictionary.
	"""

	print("\n" + " BACKTEST ".center(PRINT_WIDTH, "="))

	required_cols = {'open', 'high', 'low', 'close', 'ATR_14', 'target'}
	missing_cols = required_cols - set(df.columns)
	if missing_cols:
		print(f"\x1b[91;1m X-> DataFrame missing required columns: {missing_cols}\x1b[0m")
		raise ValueError(f"DataFrame missing required columns: {missing_cols}")

	if lookahead <= 0:
		print(f"\x1b[91;1m X-> lookahead must be a positive integer\x1b[0m")
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

	current_capital = initial_capital

	for i in range(n-1):
		if pred_arr[i] == 0:
			# No trade taken
			equity.append(current_capital)
			trade_returns.append(0.0)
			continue
		
		# IMPORTANT: Use close price as reference for barriers (same as feature_engineering labeling)
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
		for j in range(i + 1, min(i + lookahead + 1, n)):
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
		if pred_arr[i] == 1:
			pct_change = (exit_price - entry_price) / entry_price
			trade_returns.append(pct_change)
			equity_change = current_capital * bet * pct_change
		else:
			pct_change = (entry_price - exit_price) / entry_price
			trade_returns.append(pct_change)
			equity_change = current_capital * bet * pct_change
		current_capital += equity_change
		equity.append(current_capital)

	trade_returns = pd.Series(trade_returns, name="trade_return")

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

	print("\x1b[92m\n  -> Backtest completed successfully\x1b[0m")

	return BacktestResult(
		equity_curve=equity,
		trade_returns=trade_returns,
		summary=summary,
	)
