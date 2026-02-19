import numpy as np
from backtest import backtest
from machine_learning.random_forest import train_random_forest_model as rf_model
import machine_learning.lstm as lstm
# from models.MT5Services import MT5Services
from dataset_utils import dataset_utils, feature_engineering
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# https://www.youtube.com/watch?v=kbBPD1YbYcg&list=PL0iqkWGt8zzlFU_Ds4PEPee6Zg0c11Y-t&index=4

# https://github.com/Quantreo/MetaTrader-5-AUTOMATED-TRADING-using-Python/blob/main/06_money_management.ipynb


PRINT_WIDTH = 100
SYMBOL = "ETHUSD"

def main():

    print("\n" + "=" * PRINT_WIDTH + "\n")
    print("\x1b[32;1m" + f"{SYMBOL} AI".center(PRINT_WIDTH) + "\x1b[0m")
    print("\n" + "=" * PRINT_WIDTH + "\n")

    try:
        # mt5 = MT5Services(SYMBOL) # Setup terminal connection
        print("Connection to MT5 and dataset generation skipped for debugging purposes.")
    except Exception as e:
        print(f"\x1b[91;1mTerminating due to error: {e}\x1b[0m")
        return
    
    path_list = [Path('datasets/raw/ETHUSD_D1_3082.csv'), Path('datasets/raw/ETHUSD_H1_48685.csv')]
    path_list = [Path('datasets/raw/ETHUSD_D1_3082.csv')]
    path_list_final = [Path('datasets/final/ETHUSD_D1_3082.csv'), Path('datasets/final/ETHUSD_H1_48685.csv')]
    path_list_final = [Path('datasets/final/ETHUSD_D1_3082.csv')]
    
    # generate datasets for D1 and H1 timeframes
    # path_list = dataset_utils.generate_dataset(mt5, timeframes = ["D1", "H1"]) # Download datasets
    
    # Validate datasets
    if not dataset_utils.validate_dataset(path_list):
        print("\x1b[91;1mTerminating due to dataset validation failure.\x1b[0m")
        return
    print("\x1b[32;1m\nDatasets validated successfully.\x1b[0m")

    lookahead = 10
    atr_mult = 2.0

    # Feature engineering
    path_list_final = feature_engineering.calculate_features(path_list, lookahead=lookahead, atr_mult=atr_mult)
    
    for path in path_list_final:
        dataset_utils.plot_dataset(path, num_candles = 300, atr_mult=atr_mult)
    
    for path in path_list_final:
        print("\n" + " LIVE PREDICTION ".center(PRINT_WIDTH, "="))
        df = pd.read_csv(path)

        preds = []
        # for i in range(-40, -20):
            # print(f"Candle {i}:")
            # data = df.iloc[:i]
        data = df.copy()
        print(data.tail(5))
        model, scaler, feature_cols = lstm.train_lstm_model(data, lookahead_days=lookahead) # Train LSTM model on each dataset

        # PREDICT NEXT CANDLE
        probs = lstm.predict_next_move(model, data, feature_cols, scaler)

        print(f"Probabilities for next candle (closing 'now' candle):")
        print(f"HOLD:  {probs[0]:.4f}")
        print(f"LONG:  {probs[1]:.4f}")
        print(f"SHORT: {probs[2]:.4f}")

        best_action = np.argmax(probs)
        actions = ["HOLD", "LONG", "SHORT"]
        
        print(f"\n>>> CONSIGLIO AI: {actions[best_action]} (Confidenza: {probs[best_action]:.2%})")
        print(f"  > REAL (may be unreliable): {actions[data.iloc[-1]['target']]}, CLOSE: {data.iloc[-1]['close']}")
        preds.append((data.iloc[-1]['time'], actions[best_action], probs[best_action], actions[data.iloc[-1]['target']]))
        
        print("\n" + " LIVE PREDICTION SUMMARY ".center(PRINT_WIDTH, "="))
        for time, action, conf, real in preds:
            print(f"{time}: AI={action} (Conf: {conf:.2%}), REAL={real}")
    
    backtest_window = 365 # Backtest on last ~1 year of daily data
    predict_window = 30
    initial_capital = 100000
    df = pd.read_csv(path_list_final[0])
    res = backtest.backtest_triple_barrier(df, backtest_window, predict_window, initial_capital, lookahead, atr_mult, threshold=0.40, position_size=0.1)
    
    print("\n" + " BACKTEST RESULTS ".center(PRINT_WIDTH, "="))
    print(res.summary)

    plt.figure(figsize=(12, 6))
    plt.plot(res.equity_curve.index, res.equity_curve.values, label='ML Strategy (Long/Short)', color='green')
    plt.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.3, label='Break Even')
    
    # bh_equity = initial_capital * (1 + (df['close'].values[-(backtest_window + predict_window):] - df['close'].values[-(backtest_window + predict_window)]) / df['close'].values[-(backtest_window + predict_window)])
    # plt.plot(bh_equity, label='Buy & Hold ETH', color='gray', alpha=0.5, linestyle='--')

    plt.title('Equity Curve: ML Model vs Buy & Hold')
    plt.xlabel('Date')
    plt.ylabel('Capital ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()
    
    # mt5.shutdown()

if __name__ == "__main__":
    main()