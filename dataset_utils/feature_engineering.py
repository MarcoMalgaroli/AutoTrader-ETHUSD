import json
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator, StochRSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, EaseOfMovementIndicator, NegativeVolumeIndexIndicator
from typing import List, Optional
from pathlib import Path

with open(Path(__file__).resolve().parent.parent / "config.json", "r") as f:
    CONFIG = json.load(f)

PRINT_WIDTH = CONFIG["print_width"]
BASE_PATH_RAW = Path(CONFIG["paths"]["dataset_raw"])
BASE_PATH_FINAL = Path(CONFIG["paths"]["dataset_final"])

# Non-stationary columns that must never be used as ML features.
# These grow/shrink with the asset's price level and would cause
# distribution shift between training and inference.
NON_STATIONARY_COLS = {
    'time', 'target',
    'open', 'high', 'low', 'close', 'spread',
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
    'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',
    'ATR_14', 'MACD', 'vol_SMA_20', 'tick_volume',
    'ADI_diff',
}

def select_features(df: pd.DataFrame, corr_threshold: float = 0.85) -> List[str]:
    """
    Automatically select all stationary/normalized features, then remove
    inter-feature redundancy using target-aware greedy elimination.

    When two features are correlated above the threshold, the one with
    LOWER absolute correlation to the target is dropped. This preserves
    the most predictive feature from each correlated group.

    Should be called on TRAINING data only to avoid data leakage.

    Args:
        df: DataFrame containing all computed features (output of add_features + add_target).
        corr_threshold: Drop one of any pair of features whose absolute
                        Pearson correlation exceeds this value (default 0.85).
    Returns:
        List of selected feature column names.
    """
    # 1. Identify all stationary candidate columns
    candidates = [c for c in df.columns if c not in NON_STATIONARY_COLS]

    # 2. Compute target relevance for each feature
    target_corr = df[candidates].corrwith(df['target']).abs().fillna(0)

    # 3. Target-aware greedy removal: for each highly-correlated pair,
    #    drop the feature with lower |correlation to target|
    features_df = df[candidates]
    corr_matrix = features_df.corr().abs()
    to_drop = set()

    for i in range(len(candidates)):
        if candidates[i] in to_drop:
            continue
        for j in range(i + 1, len(candidates)):
            if candidates[j] in to_drop:
                continue
            if corr_matrix.iloc[i, j] > corr_threshold:
                # Drop the less target-relevant feature
                if target_corr[candidates[i]] >= target_corr[candidates[j]]:
                    to_drop.add(candidates[j])
                else:
                    to_drop.add(candidates[i])
                    break  # feature i was dropped, move to the next i

    selected = [c for c in candidates if c not in to_drop]

    print(f"  -> Feature Selection: {len(candidates)} candidates -> {len(selected)} selected (removed {len(to_drop)} redundant)")
    if to_drop:
        print(f"     Dropped for inter-correlation > {corr_threshold}: {sorted(to_drop)}")
    print(f"     Selected: {selected}")

    return selected


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical features and cleanup database for ML
    """
    data = df.copy()
    data.sort_values('time', inplace = True)

    data['candle_body'] = (data['close'] - data['open']) / data['open']
    data['candle_shadow_up'] = (data['high'] - data[['close', 'open']].max(axis=1)) / data['close']
    data['candle_shadow_low'] = (data[['close', 'open']].min(axis=1) - data['low']) / data['close']
    
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

    # --- SHORT-TERM MOMENTUM (helps detect regime transitions faster) ---
    data['ROC_1'] = data['close'].pct_change(1)   # 1-day return
    data['ROC_3'] = data['close'].pct_change(3)   # 3-day return
    data['ROC_5'] = data['close'].pct_change(5)   # 5-day return
    data['RSI_5_accel'] = data['RSI_5'].diff().diff()  # RSI acceleration (change of change)

    # --- TREND REGIME SIGNALS (explicit crossover features) ---
    data['SMA_5_20_cross'] = (data['SMA_5'] - data['SMA_20']) / data['close']  # Normalized SMA crossover
    data['SMA_10_50_cross'] = (data['SMA_10'] - data['SMA_50']) / data['close']
    data['EMA_5_20_cross'] = (data['EMA_5'] - data['EMA_20']) / data['close']

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
            symbol, timeframe = path.stem.split("_", maxsplit=1)
            timeframe = timeframe.split("_")[0]  # strip trailing suffixes like _3082

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
