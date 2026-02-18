from typing import Iterable

import numpy as np
import pandas as pd
import machine_learning.lstm as lstm

PRINT_WIDTH = 100


class BacktestResult:
    def __init__(self, equity_curve: pd.Series, trade_returns: pd.Series, summary: dict):
        self.equity_curve = equity_curve
        self.trade_returns = trade_returns
        self.summary = summary

def backtest_triple_barrier(df: pd.DataFrame, backtest_window: int, predict_window: int, initial_capital: float, lookahead: int, atr_mult: float = 1.0, threshold: float = 0.4, position_size: float = 0.01) -> BacktestResult:
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
		Minimum predicted probability required to take a trade (default 0.4).
	Returns
	-------
	BacktestResult
		Equity curve, per-trade returns, and a summary dictionary.
	"""
	
	print("\n" + " TRIPLE BARRIER BACKTEST ".center(PRINT_WIDTH, "="))

	if backtest_window <= 0 or predict_window <= 0:
		print(f"\x1b[91;1m X-> backtest_window and predict_window must be positive integers\x1b[0m")
		raise ValueError("backtest_window and predict_window must be positive integers")
	
	n = len(df)
	train_data_end = n - backtest_window

	preds = []
	probs = []

	for candle in range(train_data_end, n + predict_window, predict_window):
		print(f"\n  -> Train on (0 -> {candle}), test on ({candle - predict_window} -> {candle})")
		simulated_df = df.iloc[:candle].copy()
		
		_, targets, p, pr = lstm.train_lstm_model(simulated_df, predict_window=predict_window if candle <= n else (predict_window - (candle - n)), plot_results=False)
		preds.extend(p)
		probs.extend(pr)

	return backtest_triple_barrier_window(df.iloc[-(backtest_window + predict_window):].copy(), preds, probs, initial_capital, lookahead, atr_mult, threshold, position_size=position_size)


def backtest_triple_barrier_window(df: pd.DataFrame, y_pred: Iterable[int], y_prob: Iterable[float], initial_capital: float, lookahead: int, atr_mult: float = 1.0, threshold: float = 0.4, commission: float = 0.001, position_size: float = 0.01) -> BacktestResult:
	"""
	Backtest a classifier trained with Triple Barrier labels.

	Parameters
	----------
	df : pd.DataFrame
		DataFrame containing OHLC data, ATR_14, and target columns.
		Required columns: 'open', 'high', 'low', 'close', 'ATR_14', 'target'.
	y_pred : Iterable[int]
		Predicted labels: -1, 0, +1.
	y_prob : Iterable[float]
		Predicted probabilities for each label.
	initial_capital : float
		Initial capital for the backtest.
	lookahead : int
		Lookahead window (number of bars) used to create the labels.
	atr_mult : float
		ATR multiplier for both take profit and stop loss (default 1.0).
	threshold : float
		Minimum predicted probability required to take a trade (default 0.4).
	commission : float
		Proportional transaction cost per trade (default 0.001 for 0.1%).
	position_size : float
		Proportion of capital to risk per trade (default 1.0 for 100%).
	Returns
	-------
	BacktestResult
		Equity curve, per-trade returns, and a summary dictionary.
	"""

	required_cols = {'open', 'high', 'low', 'close', 'ATR_14', 'target'}
	missing_cols = required_cols - set(df.columns)
	if missing_cols:
		print(f"\x1b[91;1m X-> DataFrame missing required columns: {missing_cols}\x1b[0m")
		raise ValueError(f"DataFrame missing required columns: {missing_cols}")

	if lookahead <= 0:
		print(f"\x1b[91;1m X-> lookahead must be a positive integer\x1b[0m")
		raise ValueError("lookahead must be a positive integer")

	print(f"  -> Capitale Iniziale: ${initial_capital}")
	print(f"  -> Confidence Threshold: {threshold}")

	n = len(df)
	# Extract arrays from DataFrame
	open_arr = df['open'].values
	high_arr = df['high'].values
	low_arr = df['low'].values
	close_arr = df['close'].values
	atr_arr = df['ATR_14'].values
	true_arr = np.asarray(df['target'].values.astype(int))

	pred_arr = np.asarray(list(y_pred), dtype=int)
	probs_arr = np.asarray(list(y_prob), dtype=float)

	if pred_arr.shape[0] != n:
		print(f"\x1b[91;1m X-> y_pred must have the same length as the DataFrame\x1b[0m")
		raise ValueError("y_pred must have the same length as the DataFrame")
	if probs_arr.shape[0] != n:
		print(f"\x1b[91;1m X-> y_prob must have the same length as the DataFrame\x1b[0m")
		raise ValueError("y_prob must have the same length as the DataFrame")

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

		net_return = raw_return - commission  # Subtract commission
		equity_change = current_capital * position_size * net_return
		current_capital += equity_change

		trade_returns.append(net_return)
		equity.append(current_capital)

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
		equity_curve=pd.Series(equity),
		trade_returns=trade_returns,
		summary=summary,
	)
