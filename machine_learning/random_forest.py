import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


PRINT_WIDTH = 100


def remove_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """
    Remove the features that have a correlation higher than the threshold.
    """
    features_df = df.drop(columns=['target', 'time'], errors='ignore')
    corr_matrix = features_df.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"  -> Redundant features removed ({len(to_drop)}): {to_drop}")
    return df.drop(columns=to_drop)

def train_random_forest_model(file_name):
    """
    Train a Random Forest model on the dataset located at file_name.
    Args:
        file_name (str): Path to the CSV file containing the dataset.
    """
    print("\n" + " RANDOM FOREST TRAINING ".center(PRINT_WIDTH, "="))

    print(f"---> Loading dataset from {file_name}...")
    df = pd.read_csv(file_name)

    # cols_to_drop = ['time', 'target', 'open', 'high', 'low', 'close', 
    #             'SMA_5', 'EMA_5', 'SMA_10', 'EMA_10', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50', 
    #             'ATR_14', 'MACD', 'vol_SMA_20', 'tick_volume']
    # SELECTED_FEATURES = [c for c in df.columns if c not in cols_to_drop]
    
    # SELECTED_FEATURES = [
    #     'BB_width_pct',
    #     'dist_SMA_20',
    #     'dist_SMA_50',
    #     'KC_width_pct',
    #     'MACD_norm',
    #     'KC_pct',
    #     'dist_EMA_50',
    #     'RSI_20',
    #     'BB_pct',
    #     'ATR_norm',
    #     'vol_rel',
    #     'spread',
    #     'OBV_pct',
    # ]

    SELECTED_FEATURES = remove_highly_correlated_features(df[[c for c in df.columns if c not in ['time', 'target', 'open', 'high', 'low', 'close']]]).columns.tolist()
    
    X = df[SELECTED_FEATURES]
    y = df['target']
    
    print(f"  -> Features used ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")

    # Temporal Split use 85% for training and the rest for testing
    # Gap to avoid Triple Barrier lookahead leakage (default lookahead=10)
    train_size = int(len(df) * 0.85)
    gap = 10  # Should match the lookahead used in labeling
    
    X_train, X_test = X.iloc[:train_size - gap], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size - gap], y.iloc[train_size:]
    
    print(f"  -> Training on {len(X_train)} rows (with gap={gap} to avoid leakage)")
    print(f"  -> Testing on {len(X_test)} rows")

    # Random Forest Training
    # n_estimators = decision trees count
    # min_samples_leaf prevents the model from memorizing noise (overfitting)
    # model = RandomForestClassifier(
    #     n_estimators = 200,
    #     max_depth = 12,
    #     min_samples_split = 2,
    #     min_samples_leaf = 4,
    #     class_weight = 'balanced',
    #     random_state = 42,
    #     n_jobs = -1,
    # )

    model = RandomForestClassifier(
        n_estimators = 150,
        max_depth = 20,
        min_samples_split = 60,
        min_samples_leaf = 2,
        class_weight = 'balanced',
        random_state = 42,
        n_jobs = -1,
    )
    model.fit(X_train, y_train)

    # Evaluation
    print("  -> Results on Test Set")
    preds = model.predict(X_test)
    print("Accuracy:", model.score(X_test, y_test))
    print("\nClassification Report:\n", classification_report(y_test, preds))

    class_names = { -1: 'SHORT', 0: 'HOLD', 1: 'LONG' }
    cm = confusion_matrix(y_test, preds)
    print("  -> Confusion Matrix:")
    print(f"     Was {class_names[-1]} and predicted {class_names[-1]} : {cm[0][0]} ({cm[0][0] * 100 / len(y_test):.2f}%) | Was {class_names[-1]} and predicted {class_names[0]} : {cm[0][1]} ({cm[0][1] * 100 / len(y_test):.2f}%) | Was {class_names[-1]} and predicted {class_names[1]} : {cm[0][2]} ({cm[0][2] * 100 / len(y_test):.2f}%)")
    print(f"     Was {class_names[0]} and predicted {class_names[-1]} : {cm[1][0]} ({cm[1][0] * 100 / len(y_test):.2f}%) | Was {class_names[0]} and predicted {class_names[0]} : {cm[1][1]} ({cm[1][1] * 100 / len(y_test):.2f}%) | Was {class_names[0]} and predicted {class_names[1]} : {cm[1][2]} ({cm[1][2] * 100 / len(y_test):.2f}%)")
    print(f"     Was {class_names[1]} and predicted {class_names[-1]} : {cm[2][0]} ({cm[2][0] * 100 / len(y_test):.2f}%) | Was {class_names[1]} and predicted {class_names[0]} : {cm[2][1]} ({cm[2][1] * 100 / len(y_test):.2f}%) | Was {class_names[1]} and predicted {class_names[1]} : {cm[2][2]} ({cm[2][2] * 100 / len(y_test):.2f}%)")
    print(f"     Overall Accuracy: {(cm[0][0] + cm[1][1] + cm[2][2]) * 100 / len(y_test):.2f}%")

    # --- FEATURE IMPORTANCE (La parte più importante!) ---
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n--- TOP 10 FEATURE VINCENTI ---")
    for f in range(10):
        print(f"{f+1}. {SELECTED_FEATURES[indices[f]]} ({importances[indices[f]]:.4f})")

    # plt.figure(figsize=(12, 6))
    # plt.title("Feature Importances (Cosa guarda il modello?)")
    # plt.bar(range(X.shape[1]), importances[indices], align="center")
    # plt.xticks(range(X.shape[1]), [SELECTED_FEATURES[i] for i in indices], rotation=90)
    # plt.xlabel("Feature")
    # plt.ylabel("Importanza")
    # plt.tight_layout()
    # plt.show()
    return model