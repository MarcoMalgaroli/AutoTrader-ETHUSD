from backtest import backtest
from machine_learning.random_forest import train_random_forest_model as rf_model
import machine_learning.lstm as lstm
from models.MT5Services import MT5Services
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
    
    path_list = [Path('datasets/raw/ETHUSD_D1_3078.csv'), Path('datasets/raw/ETHUSD_H1_48685.csv')]
    path_list = [Path('datasets/raw/ETHUSD_D1_3078.csv')]
    path_list_final = [Path('datasets/final/ETHUSD_D1_3078.csv'), Path('datasets/final/ETHUSD_H1_48685.csv')]
    path_list_final = [Path('datasets/final/ETHUSD_D1_3078.csv')]
    
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
    
    # for path in path_list_final:
    #     dataset_utils.plot_dataset(path, num_candles = 300, atr_mult=atr_mult)
    
    for path in path_list_final:
        df = pd.read_csv(path)
        df = df.iloc[:-(lookahead + 23)].copy()
        print(df)
        model, scaler, feature_cols = lstm.train_lstm_model(df, lookahead_days=lookahead) # Train LSTM model on each dataset

        # PREDICT NEXT CANDLE
        print("\n" + " LIVE PREDICTION ".center(PRINT_WIDTH, "="))
        last_probs = lstm.predict_next_move(model, df, feature_cols, scaler)

        print(f"Probabilities for next candle (closing 'now' candle):")
        print(f"HOLD:  {last_probs[0]:.4f}")
        print(f"LONG:  {last_probs[1]:.4f}")
        print(f"SHORT: {last_probs[2]:.4f}")

        threshold = 0.34
        m = max(last_probs)
        if last_probs[1] > threshold and last_probs[1] == m:
            print(">>> ACTION: OPEN LONG")
        elif last_probs[2] > threshold and last_probs[2] == m:
            print(">>> ACTION: OPEN SHORT")
        else:
            print(">>> ACTION: HOLD / NO TRADE")
        print(f"REAL: {df.iloc[-1]['target']}, CLOSE: {df.iloc[-1]['close']}")
    
    
    # backtest_window = 365
    # predict_window = 30
    # initial_capital = 100000
    # df = pd.read_csv(path_list_final[0])
    # res = backtest.backtest_triple_barrier(df, backtest_window, predict_window, initial_capital, lookahead, atr_mult, position_size=0.1)
    

    # plt.figure(figsize=(12, 6))
    # plt.plot(res.equity_curve, label='ML Strategy (Long/Short)', color='green')
    # plt.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.3, label='Break Even')
    
    # bh_equity = initial_capital * (1 + (df['close'].values[-(backtest_window + predict_window):] - df['close'].values[-(backtest_window + predict_window)]) / df['close'].values[-(backtest_window + predict_window)])
    # plt.plot(bh_equity, label='Buy & Hold ETH', color='gray', alpha=0.5, linestyle='--')

    # plt.title('Equity Curve: ML Model vs Buy & Hold')
    # plt.xlabel('Trading Days')
    # plt.ylabel('Capital ($)')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()
    
    # print(res.summary)
    
    # mt5.shutdown()

if __name__ == "__main__":
    main()