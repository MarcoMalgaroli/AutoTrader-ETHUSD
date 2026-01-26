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

    # # Feature engineering
    # feature_engineering.calculate_features(path_list)

    # print("Everything is OK" if dataset_utils.validate_dataset() else "Problems found")
    # validated_datasets = feature_engineering.calculate_features()
    # print(validated_datasets)

    # Train Random Forest models --> accuratezza bassa, come lanciare una moneta :(
    rf_model("datasets\\final\\ETHUSD_D1_3058.csv")
    rf_model("datasets\\final\\ETHUSD_H1_48284.csv")
    rf_model("datasets\\final\\ETHUSD_M15_176673.csv")
    rf_model("datasets\\final\\ETHUSD_M5_518443.csv")

    


if __name__ == "__main__":
    main()