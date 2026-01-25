from models.MT5Services import MT5Services
from dataset_utils import dataset_utils, feature_engineering
from pathlib import Path


def main():
    SYMBOL = "ETHUSD"
    mt5 = MT5Services(SYMBOL)

    path_list = dataset_utils.generate_dataset(mt5, timeframes = ["D1", "H1", "M15", "M5"])
    mt5.shutdown()

    print("Everything is OK" if dataset_utils.validate_dataset(path_list) else "Problems found")

    feature_engineering.calculate_features(path_list)


if __name__ == "__main__":
    main()