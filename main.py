from backtest import equity_curve, backtest
from machine_learning.random_forest import train_random_forest_model as rf_model
from machine_learning.lstm import train_lstm_model as lstm_model
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
        mt5 = MT5Services(SYMBOL) # Setup terminal connection
        # print("Connection to MT5 and dataset generation skipped for debugging purposes.")
    except Exception as e:
        print(f"\x1b[91;1mTerminating due to error: {e}\x1b[0m")
        return
    
    # order = mt5.place_order("BUY")
    # print(f"Placed test order: {order}")

    
    path_list = dataset_utils.generate_dataset(mt5, timeframes = ["D1", "H1"]) # Download datasets
    # path_list = [Path('datasets/raw/ETHUSD_D1_3074.csv')]#, Path('datasets/raw/ETHUSD_H1_48667.csv')]#, Path('datasets/raw/ETHUSD_M15_178208.csv')]
    
    # Validate datasets
    if not dataset_utils.validate_dataset(path_list):
        print("\x1b[91;1mTerminating due to dataset validation failure.\x1b[0m")
        return
    print("\x1b[32;1m\nDatasets validated successfully.\x1b[0m")

    lookahead = 15
    atr_mult = 3.0

    # Feature engineering
    path_list_final = feature_engineering.calculate_features(path_list, lookahead=lookahead, atr_mult=atr_mult)
    # path_list_final = [Path('datasets/final/ETHUSD_D1_3074.csv'), Path('datasets/final/ETHUSD_H1_48667.csv'), Path('datasets/final/ETHUSD_M15_178208.csv')]
    for path in path_list_final:
        dataset_utils.plot_dataset(path, num_candles = 300, atr_mult=atr_mult)
    
    for path in path_list_final:
        lstm_model(path) # Train LSTM model on each dataset

    # Train Random Forest models
    # preds = []
    # for dataset_path in path_list_final:
    #     preds.append(rf_model(dataset_path))
    
    # # for dataset_path in path_list_final:
    # #     equity_curve.backtest(dataset_path)
    
    # initial_capital = 10000

    # for dataset_path in path_list_final:
    #     y_pred = preds.pop(0)
    #     df = pd.read_csv(dataset_path)[-len(y_pred):].reset_index(drop=True)
    #     res = backtest.backtest_triple_barrier(df, y_pred, initial_capital, bet=0.1, lookahead=lookahead, atr_mult=atr_mult)

    #     print(res.summary)
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(res.equity_curve, label='ML Strategy (Long/Short)', color='green')
    #     plt.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.3, label='Break Even')
        
    #     # bh_equity = initial_capital * (1 + (df['close'].values - df['close'].values[0]) / df['close'].values[0])
    #     # plt.plot(bh_equity, label='Buy & Hold ETH', color='gray', alpha=0.5, linestyle='--')

    #     plt.title('Equity Curve: ML Model vs Buy & Hold')
    #     plt.xlabel('Trading Days')
    #     plt.ylabel('Capital ($)')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     plt.show()

    # mt5.shutdown()

if __name__ == "__main__":
    main()