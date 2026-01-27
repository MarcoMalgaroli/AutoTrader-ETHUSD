from machine_learning.random_forest import train_random_forest_model as rf_model
from models.MT5Services import MT5Services
from dataset_utils import dataset_utils, feature_engineering
from pathlib import Path


def main():
    SYMBOL = "ETHUSD"

    # # Setup terminal connection
    # mt5 = MT5Services(SYMBOL)

    # # Download datasets
    # path_list = dataset_utils.generate_dataset(mt5, timeframes = ["D1", "H1", "M15", "M5"])
    # mt5.shutdown()

    # # Validate datasets
    # print("Everything is OK" if dataset_utils.validate_dataset(path_list) else "Problems found")

    # Feature engineering
    path_list = [Path('datasets/final/ETHUSD_D1_3059.csv'), Path('datasets/final/ETHUSD_H1_48306.csv'), Path('datasets/final/ETHUSD_M15_176764.csv'), Path('datasets/final/ETHUSD_M5_518717.csv')]
    print(path_list)
    # feature_engineering.calculate_features(path_list)

    # Train Random Forest models --> accuratezza bassa, come lanciare una moneta
    for i, dataset_path in enumerate(path_list, start=1):
        rf_model(dataset_path, rows = 1000 * i)

if __name__ == "__main__":
    main()