from backtest import equity_curve
from machine_learning.random_forest import train_random_forest_model as rf_model
from models.MT5Services import MT5Services
from dataset_utils import dataset_utils, feature_engineering
from pathlib import Path

PRINT_WIDTH = 100
SYMBOL = "ETHUSD"

def main():

    print("\n" + "=" * PRINT_WIDTH + "\n")
    print("\x1b[32;1m" + f"{SYMBOL} ORACLE-AI".center(PRINT_WIDTH) + "\x1b[0m")
    print("\n" + "=" * PRINT_WIDTH + "\n")
    path_list = []

    try:
        # mt5 = MT5Services(SYMBOL) # Setup terminal connection
        # path_list = dataset_utils.generate_dataset(mt5, timeframes = ["D1", "H1"]) # Download datasets
        # mt5.shutdown()
        print("Connection to MT5 and dataset generation skipped for debugging purposes.")
    except Exception as e:
        print(f"\x1b[91;1mTerminating due to error: {e}\x1b[0m")
        return
    
    path_list = [Path('datasets/raw/ETHUSD_D1_3065.csv'), Path('datasets/raw/ETHUSD_H1_48449.csv')]
    
    # Validate datasets
    if not dataset_utils.validate_dataset(path_list):
        print("\x1b[91;1mTerminating due to dataset validation failure.\x1b[0m")
        return
    
    print("\x1b[32;1m\nDatasets validated successfully.\x1b[0m")

    lookahead = 10
    atr_mult = 1.0

    # Feature engineering
    path_list_final = feature_engineering.calculate_features(path_list, lookahead=lookahead, atr_mult=atr_mult)
    for path in path_list_final:
        dataset_utils.plot_dataset(path, num_candles = 300, atr_mult=atr_mult)

    # path_list_final = [Path('datasets/final/ETHUSD_D1_3065.csv'), Path('datasets/final/ETHUSD_H1_48449.csv')]
    # Train Random Forest models
    for dataset_path in path_list_final:
        rf_model(dataset_path)

    
    for dataset_path in path_list_final:
        equity_curve.backtest(dataset_path)

if __name__ == "__main__":
    main()