import json
import numpy as np
from backtest import backtest_triple_barrier
import machine_learning.lstm_classifier as lstm
import machine_learning.mlp as mlp
# from models.MT5Services import MT5Services
from dataset_utils import dataset_utils, feature_engineering
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# https://www.youtube.com/watch?v=kbBPD1YbYcg&list=PL0iqkWGt8zzlFU_Ds4PEPee6Zg0c11Y-t&index=4

# https://github.com/Quantreo/MetaTrader-5-AUTOMATED-TRADING-using-Python/blob/main/06_money_management.ipynb

with open(Path(__file__).resolve().parent / "config.json", "r") as f:
    CONFIG = json.load(f)

PRINT_WIDTH = CONFIG["print_width"]
SYMBOL = CONFIG["symbol"]

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
    
    # path_list = [Path('datasets/raw/ETHUSD_D1_3082.csv'), Path('datasets/raw/ETHUSD_H1_48685.csv')]
    path_list = [Path('datasets/raw/ETHUSD_D1.csv')]
    # path_list_final = [Path('datasets/final/ETHUSD_D1_3082.csv'), Path('datasets/final/ETHUSD_H1_48685.csv')]
    # path_list_final = [Path('datasets/final/ETHUSD_D1_3082.csv')]
    
    # generate datasets for D1 and H1 timeframes
    # path_list = dataset_utils.generate_dataset(mt5, timeframes = ["D1", "H1"]) # Download datasets
    
    # Validate datasets
    if not dataset_utils.validate_dataset(path_list):
        print("\x1b[91;1mTerminating due to dataset validation failure.\x1b[0m")
        return
    print("\x1b[32;1m\nDatasets validated successfully.\x1b[0m")

    lookahead = CONFIG["trading"]["lookahead"]
    atr_mult = CONFIG["trading"]["atr_mult"]

    # Feature engineering
    path_list_final = feature_engineering.calculate_features(path_list, lookahead=lookahead, atr_mult=atr_mult)
    
    # for path in path_list_final:
    #     dataset_utils.plot_dataset(path, num_candles = 300, atr_mult=atr_mult)
    
    # for path in path_list_final:
    #     print("\n" + " LIVE PREDICTION ".center(PRINT_WIDTH, "="))
    #     df = pd.read_csv(path)

    #     preds = []
    #     for i in range(-50, -20):
    #         print(f"Candle {i}:")
    #         data = df.iloc[:i]
    #         print(data.tail(5))
    #         model, scaler, feature_cols = lstm.train_lstm_classifier(data, lookahead_days=lookahead, plot_results=False) # Train LSTM model on each dataset

    #         # PREDICT NEXT CANDLE
    #         probs = lstm.predict_next_move(model, data, feature_cols, scaler)

    #         print(f"Probabilities for next candle (closing 'now' candle):")
    #         print(f"HOLD:  {probs[0]:.4f}")
    #         print(f"LONG:  {probs[1]:.4f}")
    #         print(f"SHORT: {probs[2]:.4f}")

    #         best_action = np.argmax(probs)
    #         actions = ["HOLD", "LONG", "SHORT"]
            
    #         print(f"\n>>> CONSIGLIO AI: {actions[best_action]} (Confidenza: {probs[best_action]:.2%})")
    #         print(f"  > REAL (may be unreliable): {actions[data.iloc[-1]['target']]}, CLOSE: {data.iloc[-1]['close']}")
    #         preds.append((data.iloc[-1]['time'], actions[best_action], probs[best_action], actions[data.iloc[-1]['target']]))
        
    #     print("\n" + " LIVE PREDICTION SUMMARY ".center(PRINT_WIDTH, "="))
    #     for time, action, conf, real in preds:
    #         print(f"{time}: AI={action} (Conf: {conf:.2%}), REAL={real}")
    
    backtest_window = CONFIG["backtest"]["backtest_window"]
    predict_window = CONFIG["backtest"]["predict_window"]
    initial_capital = CONFIG["trading"]["initial_capital"]
    df = pd.read_csv(path_list_final[0])

    # --- LSTM Classifier Backtest ---
    res_lstm_class, trades_lstm_class = backtest_triple_barrier.backtest_triple_barrier(
        df, backtest_window, predict_window, initial_capital, lookahead, atr_mult,
        threshold=CONFIG["trading"]["threshold"], position_size=CONFIG["trading"]["position_size"],
        model_type="lstm"
    )

    # --- MLP Backtest ---
    res_mlp, trades_mlp = backtest_triple_barrier.backtest_triple_barrier(
        df, backtest_window, predict_window, initial_capital, lookahead, atr_mult,
        threshold=CONFIG["trading"]["threshold"], position_size=CONFIG["trading"]["position_size"],
        model_type="mlp"
    )

    

    print("\n" + " LSTM CLASSIFIER BACKTEST RESULTS ".center(PRINT_WIDTH, "="))
    print(res_lstm_class.summary)
    
    print("\n" + " MLP BACKTEST RESULTS ".center(PRINT_WIDTH, "="))
    print(res_mlp.summary)


    # --- Comparison Plot ---
    plt.figure(figsize=(12, 6))
    plt.plot(res_lstm_class.equity_curve.index, res_lstm_class.equity_curve.values, label='LSTM Classifier Strategy', color='green')
    plt.plot(res_mlp.equity_curve.index, res_mlp.equity_curve.values, label='MLP Strategy', color='blue')
    
    plt.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.3, label='Break Even')

    plt.title('Equity Curve: LSTM Classifier vs MLP')
    plt.xlabel('Date')
    plt.ylabel('Capital ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()
    
    # print(trades_lstm_class)
    # print(trades_mlp)
    # print(trades_lstm_reg_sltp)
    # print(trades_lstm_reg_eod)
    # mt5.shutdown()

if __name__ == "__main__":
    main()