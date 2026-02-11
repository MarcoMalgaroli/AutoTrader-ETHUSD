import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator, StochRSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, EaseOfMovementIndicator, NegativeVolumeIndexIndicator
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
    
    # Momentum indicators: RSI (Relative Strength Index), ROC (Rate of Change)
    for length in [5, 10, 15, 20]:
        data[f'RSI_{length}'] = RSIIndicator(close = data['close'], window = length).rsi()
        data[f'RSI_{length}_diff'] = data[f'RSI_{length}'].diff()
    data['ROC_10'] = ROCIndicator(close = data['close'], window = 10).roc()
    data['StochRSI_14_k'] = StochRSIIndicator(close = data['close'], window = 14).stochrsi_k()
    data['StochRSI_14_d'] = StochRSIIndicator(close = data['close'], window = 14).stochrsi_d()

    # Trend indicators: SMA (Simple Moving Average), EMA (Exponential Moving Average)
    for length in [5, 10, 20, 50]:
        data[f'SMA_{length}'] = data['close'].rolling(window = length).mean()
        data[f'EMA_{length}'] = EMAIndicator(close = data['close'], window = length).ema_indicator()
        data[f'dist_SMA_{length}'] = (data['close'] - data[f'SMA_{length}']) / data[f'SMA_{length}']
        data[f'dist_SMA_{length}_diff'] = data[f'dist_SMA_{length}'].diff()
        data[f'dist_EMA_{length}'] = (data['close'] - data[f'EMA_{length}']) / data[f'EMA_{length}']
        data[f'dist_EMA_{length}_diff'] = data[f'dist_EMA_{length}'].diff()
    
    data['MACD'] = MACD(close = data['close'], window_slow=26, window_fast=12, window_sign=9).macd()
    data['MACD_norm'] = data['MACD'] / data['close']

    # Volatility indicators: ATR (Average True Range), Bollinger Bands, Keltner Channel
    data['ATR_14'] = AverageTrueRange(high = data['high'], low = data['low'], close = data['close'], window = 14).average_true_range()
    data['ATR_norm'] = data['ATR_14'] / data['close']

    bb_indicator = BollingerBands(close=data['close'], window=20, window_dev=2)
    data['BB_width_pct'] = (bb_indicator.bollinger_hband() - bb_indicator.bollinger_lband()) / data['close']
    data['BB_pct'] = (data['close'] - bb_indicator.bollinger_mavg()) / bb_indicator.bollinger_mavg()

    kc_indicator = KeltnerChannel(high = data['high'], low = data['low'], close = data['close'], window = 20, window_atr = 10)
    data['KC_width_pct'] = (kc_indicator.keltner_channel_hband() - kc_indicator.keltner_channel_lband()) / data['close']
    data['KC_pct'] = (data['close'] - kc_indicator.keltner_channel_mband()) / kc_indicator.keltner_channel_mband()

    # Volume indicators: Volume SMA and Volume relative to its SMA
    data['OBV_pct'] = OnBalanceVolumeIndicator(close = data['close'], volume = data['tick_volume']).on_balance_volume().pct_change()
    data['ADI_diff'] = AccDistIndexIndicator(high = data['high'], low = data['low'], close = data['close'], volume = data['tick_volume']).acc_dist_index().diff()
    data['EMI_norm'] = EaseOfMovementIndicator(high = data['high'], low = data['low'], volume = data['tick_volume']).ease_of_movement() / data['close']
    data['NVI_pct'] = NegativeVolumeIndexIndicator(close = data['close'], volume = data['tick_volume']).negative_volume_index().pct_change()

    data['vol_SMA_20'] = data['tick_volume'].rolling(20).mean()
    data['vol_rel'] = data['tick_volume'] / data['vol_SMA_20']

    # Log Returns -> Percentual logarithmic variation between candles
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))

    return data

# note: each row's target is the action to take at that candle's close price (when the next candle opens)
def add_target(df: pd.DataFrame, lookahead: int = 10, atr_mult: float = 1.0) -> pd.DataFrame:
    """
    Triple-barrier labeling (Marcos López de Prado):
    1 = Long (upper barrier hit first)
    0 = Hold (time barrier hit first)
    -1 = Short (lower barrier hit first)
    """
    data = df.copy()
    n = len(data)
    target = np.zeros(n, dtype=np.int8)

    for i in range(n):
        lookahead_window = data.iloc[i : min(i + lookahead + 1, n)]
        target[i] = calculate_barrier(lookahead_window, atr_mult)

    data['target'] = target
    return data

def calculate_barrier(df: pd.DataFrame, atr_mult: float = 1.0) -> int:
    """
    Calculate which barrier is hit first for a given candle range.
    
    Triple Barrier logic (symmetric barriers):
    - Upper barrier: close + atr_mult * atr
    - Lower barrier: close - atr_mult * atr
    
    Returns:
        +1 if upper barrier hit first (bullish move)
        -1 if lower barrier hit first (bearish move)
         0 if time barrier hit (neither barrier touched)
    """

    close = df['close'].iloc[0]
    atr = df['ATR_14'].iloc[0]
    if atr <= 0 or np.isnan(atr):
        return 0

    upper_barrier = close + atr_mult * atr
    lower_barrier = close - atr_mult * atr

    # Iterate through lookahead window to find which barrier is hit first
    for i in range(1, len(df)):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        open_price = df['open'].iloc[i]
        
        upper_hit = high >= upper_barrier
        lower_hit = low <= lower_barrier
        
        if upper_hit and lower_hit:
            # Both barriers hit in same candle - use open price to infer direction
            # If open is closer to upper, price likely went down first (bearish)
            # If open is closer to lower, price likely went up first (bullish)
            if open_price >= close:
                return -1  # Started high, likely went down first
            else:
                return 1   # Started low, likely went up first
        elif upper_hit:
            return 1   # Bullish - price went up first
        elif lower_hit:
            return -1  # Bearish - price went down first
    
    return 0  # Time barrier hit - neither barrier touched

def calculate_features(path_list: Optional[List[Path]] = None, lookahead: int = 10, atr_mult: float = 1.0) -> List[Path]:
    print("\n" + " ADDING FEATURES FROM RAW DATASETS ".center(PRINT_WIDTH, "="))

    BASE_PATH_FINAL.mkdir(parents = True, exist_ok = True)    
    if not path_list:
        path_list = list(BASE_PATH_RAW.glob("*.csv"))

    path_list_out = []
    for path in path_list:
        print(f"\n---> Enhancing {path.name}")
        try:
            df = pd.read_csv(path, parse_dates = True)
            symbol, timeframe, length = path.stem.split("_")

            df_features = add_features(df)
            print("  -> Added indicators")
            
            df_final = add_target(df_features, lookahead, atr_mult)
            print("  -> Calculated target values. Distribution:")
            targets_dist = df_final['target'].value_counts(normalize = True)
            print(f"     SHORT: {targets_dist.get(-1, 0) * 100:.2f}%")
            print(f"     HOLD: {targets_dist.get(0, 0) * 100:.2f}%")
            print(f"     LONG: {targets_dist.get(1, 0) * 100:.2f}%")

            # remove rows with incomplete data
            initial_len = len(df_final)
            df_final.dropna(inplace = True)
            print(f"  -> Removed rows (NaN): {initial_len - len(df_final)} ({initial_len} --> {len(df_final)})")

            file_path = BASE_PATH_FINAL / path.name
            df_final.to_csv(file_path, index = False)
            print(f"\x1b[92m  -> Enhanced dataset for {symbol} [{timeframe}] saved at {file_path}\x1b[0m")
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
            print(f"\x1b[91;1m X-> Parsing/reading error: {type(e).__name__} -> {e}\x1b[0m")

    return path_list_out
