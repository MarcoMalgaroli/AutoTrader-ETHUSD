from pathlib import Path
from typing import List, Optional
import pandas as pd
import mplfinance as mpf
from models.MT5Services import MT5Services

PRINT_WIDTH = 100
BASE_PATH_RAW = Path("datasets", "raw")

def generate_dataset(mt5: MT5Services, symbol: str = None, timeframes: List[str] = ["D1", "H1", "M15", "M5"]) -> List[Path]:
    symbol = symbol or mt5.get_selected_symbol()
    print("\n" + f" CREATING DATASET FOR {symbol} ".center(PRINT_WIDTH, "="))
    
    BASE_PATH_RAW.mkdir(parents = True, exist_ok = True)    
    path_list = []

    for tf in timeframes:
        try:
            df = mt5.get_historical_data_all(symbol, timeframe = tf)

            file_path = BASE_PATH_RAW / f"{symbol}_{tf}_{len(df)}.csv"
            df.to_csv(file_path, index = False)

            print(f"  -> Dataset for {symbol} [{tf}] saved at {file_path}")
            path_list.append(file_path)
        
        except Exception as e:
            print(f"Error, {symbol} [{tf}]: {e}")
    return path_list

def validate_dataset(path_list: Optional[List[Path]] = None, show_graph: Optional[bool] = False) -> bool:
    print("\n" + " VALIDATING DATASETS ".center(PRINT_WIDTH, "="))
    if not path_list:
        path_list = list(BASE_PATH_RAW.glob("*.csv"))

    flag = True
    for path in path_list:
        print(f"---> Validating {path.name}")
        ok = True
        try:
            df = pd.read_csv(path, parse_dates = True)
            symbol, timeframe, length = path.stem.split("_")
            length = int(length)

            if len(df) != length:
                print(f" X-> Lenght mismatch: {len(df)} != {length}")
                ok = False
            
            required_cols = {"time", "open", "high", "low", "close", "tick_volume", "spread"}
            missing = required_cols - set(df.columns)
            if missing:
                print(f" X-> Missing columns: {missing}")
                ok = False
            
            if df.isnull().any().any():
                print(" X-> NaN values found")
                ok = False

            df["time"] = pd.to_datetime(df["time"])
            if not df["time"].is_monotonic_increasing:
                print(" X-> Data not sorted chronologically")
                ok = False
            
            if ok:
                print("  -> Success")
                if show_graph:
                    df.set_index('time', inplace = True)
                    df.rename(columns = {"tick_volume": "volume"}, inplace = True)
                    mpf.plot(df.tail(100), type = 'candle', mav = (3, 6, 9), title = f'{symbol} [{timeframe}]', volume = True)
            else:
                flag = False

        except Exception as e:
            print(f" X-> Parsing/reading error: {e}")
            flag = False
    return flag