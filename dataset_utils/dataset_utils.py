from pathlib import Path
from typing import List, Optional
import pandas as pd
import mplfinance as mpf
from models.MT5Services import MT5Services

PRINT_WIDTH = 100
BASE_PATH_RAW = Path("datasets", "raw")

TIMEFRAMES = {
    "M1": "1min",
    "M2": "2min",
    "M3": "3min",
    "M4": "4min",
    "M5": "5min",
    "M6": "6min",
    "M10": "10min",
    "M12": "12min",
    "M15": "15min",
    "M20": "20min",
    "M30": "30min",
    "H1": "1h",
    "H2": "2h",
    "H3": "3h",
    "H4": "4h",
    "H6": "6h",
    "H8": "8h",
    "H12": "12h",
    "D1": "1D",
    "W1": "1W",
    "MN1": "1MS",
}

def generate_dataset(mt5: MT5Services, symbol: Optional[str] = None, timeframes: List[str] = ["D1", "H1", "M15", "M5"]) -> List[Path]:
    symbol = symbol or mt5.get_selected_symbol()
    print("\n" + f" CREATING DATASET FOR {symbol} ".center(PRINT_WIDTH, "="))
    
    BASE_PATH_RAW.mkdir(parents = True, exist_ok = True)    
    path_list = []

    for tf in timeframes:
        try:
            df = mt5.get_historical_data_all(symbol, timeframe = tf)

            if df is None:
                raise ValueError(f"No data returned for {symbol} [{tf}]")

            file_path = BASE_PATH_RAW / f"{symbol}_{tf}_{len(df)}.csv"
            df.to_csv(file_path, index = False)

            print(f"\x1b[92m  -> Dataset for {symbol} [{tf}] saved at {file_path}\x1b[0m")
            path_list.append(file_path)
        
        except Exception as e:
            raise Exception(f"Error generating dataset for {symbol} [{tf}]: {type(e).__name__} -> {e}")
    return path_list

def validate_dataset(path_list: Optional[List[Path]] = None, show_graph: Optional[bool] = False) -> bool:
    print("\n" + " VALIDATING DATASETS ".center(PRINT_WIDTH, "="))
    if not path_list:
        path_list = list(BASE_PATH_RAW.glob("*.csv"))

    flag = True
    for path in path_list:
        print(f"\n---> Validating {path.name}")
        ok = True
        try:
            df = pd.read_csv(path, parse_dates = ['time'])
            symbol, timeframe, length = path.stem.split("_")
            length = int(length)

            print(f"  -> Start date: {df['time'].iloc[0]}")
            print(f"  -> End date:   {df['time'].iloc[-1]}")

            if len(df) != length:
                print(f"\x1b[91;1m X-> Length mismatch: {len(df)} != {length}\x1b[0m")
                ok = False
            
            required_cols = {"time", "open", "high", "low", "close", "tick_volume", "spread"}
            missing = required_cols - set(df.columns)
            if missing:
                print(f"\x1b[91;1m X-> Missing columns: {missing}\x1b[0m")
                ok = False
            
            if df.isnull().any().any():
                print("\x1b[91;1m X-> NaN values found\x1b[0m")
                ok = False

            df["time"] = pd.to_datetime(df["time"])
            if not df["time"].is_monotonic_increasing:
                print("\x1b[91;1m X-> Data not sorted chronologically\x1b[0m")
                ok = False
            
            # check if every date and time is present without gaps (for the given timeframe)
            expected_freq = TIMEFRAMES.get(timeframe, "1D")
            expected_times = pd.date_range(start = df["time"].iloc[0], end = df["time"].iloc[-1], freq = expected_freq)
            if not expected_times.isin(df["time"]).all():
                print(f"\x1b[33;1m X-> Missing timestamps detected. Expected: {len(expected_times)}, got {len(df)} (difference: {len(expected_times) - len(df)})\x1b[0m")
            
            if ok:
                print("\x1b[92m  -> Success\x1b[0m")
                if show_graph:
                    df.set_index('time', inplace = True)
                    df.rename(columns = {"tick_volume": "volume"}, inplace = True)
                    mpf.plot(df.tail(100), type = 'candle', mav = (3, 6, 9), title = f'{symbol} [{timeframe}]', volume = True)
            else:
                flag = False

        except Exception as e:
            print(f"\x1b[91;1m X-> Parsing/reading error: {type(e).__name__} -> {e}\x1b[0m")
            flag = False
    return flag

def plot_dataset(path: Path, num_candles: int = 200, atr_mult: float = 1.0):
    print("\n" + f"\x1b[36mPlotting dataset {path.name}\x1b[0m")
    try:
        df = pd.read_csv(path, parse_dates = ['time']).tail(num_candles)
        df.set_index('time', inplace = True)
        df.rename(columns = {"tick_volume": "volume"}, inplace = True)

        longs = df['close'].where(df['target'] == 1)
        shorts = df['close'].where(df['target'] == -1)

        df['upper_barrier']  = df['close'] + atr_mult * df['ATR_14']
        df['lower_barrier']  = df['close'] - atr_mult * df['ATR_14']

        def barrier_segments(df):
            alines = []
            colors = []

            for i in range(len(df)):
                t = df.index[i]
                close = df['close'].iloc[i]

                if df['target'].iloc[i] != 0:  # LONG or SHORT
                    alines.append([(t, close), (t, df['upper_barrier'].iloc[i])]) # plot upper barrier
                    alines.append([(t, close), (t, df['lower_barrier'].iloc[i])]) # plot lower barrier

                    if df['target'].iloc[i] == 1:  # LONG ==> upper green (take profit), lower red (stop loss)
                        colors.append('green')
                        colors.append('red')
                    elif df['target'].iloc[i] == -1:  # SHORT ==> upper red (stop loss), lower green (take profit)
                        colors.append('red')
                        colors.append('green')

            return dict(alines=alines, colors=colors, linestyle = 'dotted', linewidths=3)

        apds = [
            mpf.make_addplot(df['EMA_20'], color='orange', width=1),
            mpf.make_addplot(df['SMA_50'], color='blue', width=1),
            mpf.make_addplot(df['RSI_10'], color='purple', panel=1, ylabel='RSI_10'),
            mpf.make_addplot(df['ATR_14'], panel=2, color='red', ylabel='ATR_14'),
            mpf.make_addplot(longs, type='scatter', marker='^', color='green', markersize=50),
            mpf.make_addplot(shorts, type='scatter', marker='v', color='red', markersize=50),
        ]
        mpf.plot(df, type = 'candle', title = f'{path.stem}', volume = True, addplot = apds, style = 'yahoo', alines = barrier_segments(df), figsize=(12, 8))
    except Exception as e:
        raise Exception(f"Error plotting dataset {path.name}: {type(e).__name__} -> {e}")