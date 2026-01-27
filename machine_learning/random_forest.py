from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

PRINT_WIDTH = 100

def train_random_forest_model(file_name, rows: int = 1000):
    """
    Train a Random Forest model on the dataset located at file_name.
    Args:
        file_name (str): Path to the CSV file containing the dataset.
        rows (int): Number of rows to read from the tail of the dataset. Default is 1000.
    """
    print("\n" + " RANDOM FOREST TRAINING ".center(PRINT_WIDTH, "="))

    print(f"---> Loading dataset from {file_name}...")
    df = pd.read_csv(file_name).tail(rows)

    # Columns to remove in order to train on features only to learn patterns
    drop_cols = ['time', 'target', 'open', 'high', 'low', 'close', 'tick_volume']
    # Building feature matrix X and target vector y
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df['target']
    
    print(f"  -> Features used ({len(feature_cols)}): {feature_cols}")

    # Temporal Split use 85% for training and the rest for testing
    split_point = int(len(df) * 0.95)
    
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]
    
    print(f"  -> Training on {len(X_train)} rows")
    print(f"  -> Testing on {len(X_test)} rows")

    # Random Forest Training
    # n_estimators=100 means 100 decision trees
    # min_samples_leaf=50 prevents the model from memorizing noise (overfitting)
    model = RandomForestClassifier(
        n_estimators = 200,
        max_depth = 8,
        min_samples_leaf = 50,
        max_features = 0.7,
        class_weight = "balanced",
        random_state = 42,
        n_jobs = -1,
    )
    model.fit(X_train, y_train)

    # Evaluation
    print("  -> Results on Test Set")

    # probs = model.predict_proba(X_test)
    probs = model.predict_proba(X_test)
    p_flat  = probs[:, 0]
    p_long  = probs[:, 1]
    p_short = probs[:, 2]


    print("      -> Avg probabilities:")
    print(f"FLAT  : {p_flat.mean():.3f}")
    print(f"LONG  : {p_long.mean():.3f}")
    print(f"SHORT : {p_short.mean():.3f}")

    print("      -> Max probabilities:")
    print(f"FLAT  : {p_flat.max():.3f}")
    print(f"LONG  : {p_long.max():.3f}")
    print(f"SHORT : {p_short.max():.3f}")

    threshold = 0.45

    preds_thr = np.where(
        p_long > threshold, 1,
        np.where(p_short > threshold, 2, 0)
    )
    
    print("  -> Classification Report:")
    print(classification_report(
        y_test,
        preds_thr,
        target_names = ["FLAT", "LONG", "SHORT"],
        zero_division = 0
    ))
        
    # Confusion Matrix
    cm = confusion_matrix(y_test, preds_thr)
    print("  -> Confusion Matrix:")
    print(f"     True FLAT: {cm[0][0]} ({cm[0][0] * 100 / len(y_test):.2f}%) | False FLAT (LONG): {cm[0][1]} ({cm[0][1] * 100 / len(y_test):.2f}%) | False FLAT (SHORT): {cm[0][2]} ({cm[0][2] * 100 / len(y_test):.2f}%)")
    print(f"     False LONG (FLAT): {cm[1][0]} ({cm[1][0] * 100 / len(y_test):.2f}%) | True LONG (Profit): {cm[1][1]} ({cm[1][1] * 100 / len(y_test):.2f}%) | False LONG (SHORT): {cm[1][2]} ({cm[1][2] * 100 / len(y_test):.2f}%)")
    print(f"     False SHORT (FLAT): {cm[2][0]} ({cm[2][0] * 100 / len(y_test):.2f}%) | False SHORT (LONG): {cm[2][1]} ({cm[2][1] * 100 / len(y_test):.2f}%) | True SHORT (Profit): {cm[2][2]} ({cm[2][2] * 100 / len(y_test):.2f}%)")
    print(f"     Overall Accuracy: {(cm[0][0] + cm[1][1] + cm[2][2]) * 100 / len(y_test):.2f}%")
    return model