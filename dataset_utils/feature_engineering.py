import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from typing import List, Optional
from pathlib import Path

PRINT_WIDTH = 100
BASE_PATH_RAW = Path("datasets", "raw")
BASE_PATH_FINAL = Path("datasets", "final")

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical features and cleanup database for ML
    """
    data = df.copy()
    data.sort_values('time', inplace = True)

    # Trend indicators
    # EMA (Exponential Moving Average)
    data['EMA10'] = EMAIndicator(close = data['close'], window = 10).ema_indicator()
    data['EMA20'] = EMAIndicator(close = data['close'], window = 20).ema_indicator()
    data['EMA50'] = EMAIndicator(close = data['close'], window = 50).ema_indicator()

    data['dist_EMA20'] = (data['close'] - data['EMA20']) / data['EMA20']

    # RSI (Relative Strength Index -> Momentum)
    data['RSI'] = RSIIndicator(close = data['close'], window = 14).rsi()

    # ATR (Average True Range -> Volatility)
    data['ATR'] = AverageTrueRange(high = data['high'], low = data['low'], close = data['close'], window = 14).average_true_range()
    data['ATR_norm'] = data['ATR'] / data['close']

    # Log Returns -> Percentual logarithmic variation between candles
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))

    # Volume change
    data['vol_SMA'] = data['tick_volume'].rolling(20).mean()
    data['vol_rel'] = data['tick_volume'] / data['vol_SMA']

    return data

def add_target(df: pd.DataFrame, lookahead: int = 10, atr_mult_tp: float = 0.5, atr_mult_sl: float = 0.5) -> pd.DataFrame:
    """
    Multi-class target:
    0 = no trade / close
    1 = good long
    2 = good short
    """
    data = df.copy()

    highs = data['high'].rolling(lookahead).max().shift(-lookahead) # max high in the next lookahead candles
    lows  = data['low'].rolling(lookahead).min().shift(-lookahead)  # min low in the next lookahead candles

    close = data['close']
    atr   = data['ATR']

    long_tp  = close + atr_mult_tp * atr # take profit for long
    long_sl  = close - atr_mult_sl * atr # stop loss for long
    short_tp = close - atr_mult_tp * atr # take profit for short
    short_sl = close + atr_mult_sl * atr # stop loss for short

    target = np.zeros(len(data), dtype=int) # default 0 = no trade / close
    long_cond = (highs >= long_tp) & (lows > long_sl) # conditions for a good long trade
    short_cond = (lows <= short_tp) & (highs < short_sl) # conditions for a good short trade

    target[long_cond] = 1 # 1 = good long
    target[short_cond] = 2 # 2 = good short

    data['target'] = target
    return data

def calculate_features(path_list: Optional[List[Path]] = None) -> List[Path]:
    print("\n" + " ADDING FEATURES FROM RAW DATASETS ".center(PRINT_WIDTH, "="))

    BASE_PATH_FINAL.mkdir(parents = True, exist_ok = True)    
    if not path_list:
        path_list = list(BASE_PATH_RAW.glob("*.csv"))

    path_list_out = []
    for path in path_list:
        print(f"---> Enhancing {path.name}")
        try:
            df = pd.read_csv(path, parse_dates = True)
            symbol, timeframe, length = path.stem.split("_")

            df_features = add_features(df)
            print("  -> Added indicators")
            
            df_final = add_target(df_features)
            print("  -> Calculated target values")

            # remove rows with incomplete data
            initial_len = len(df_final)
            df_final.dropna(inplace = True)
            print(f"  -> Removed rows (NaN): {initial_len - len(df_final)} ({initial_len} --> {len(df_final)})")

            file_path = BASE_PATH_FINAL / path.name
            df_final.to_csv(file_path, index = False)
            print(f"  -> Enhanced dataset for {symbol} [{timeframe}] saved at {file_path}")
            path_list_out.append(file_path)

            # Target correlation analysis
            print("  -> Target correlation analysis")
            correlation = df_final.corr(numeric_only = True)['target'].sort_values(ascending = False)
            print("      -> Top positive correlations:")
            for idx, val in correlation.head(5).items():
                print(f"         {idx:<15}: {val:>10.5f}")

            print("\n      -> Top negative correlations:")
            for idx, val in correlation.tail(5).items():
                print(f"         {idx:<15}: {val:>10.5f}")

        except Exception as e:
            print(f" X-> Parsing/reading error: {e}")

    return path_list_out
