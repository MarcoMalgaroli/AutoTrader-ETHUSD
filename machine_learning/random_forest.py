from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

PRINT_WIDTH = 100

def train_random_forest_model(file_name):
    """
    Train a Random Forest model on the dataset located at file_name
    """
    print("\n" + " TRAINING ".center(PRINT_WIDTH, "="))

    print(f"---> Loading dataset from {file_name}...")
    df = pd.read_csv(file_name)
    
    # Columns to remove in order to train on features only to learn patterns
    drop_cols = ['time', 'target', 'open', 'high', 'low', 'close', 'tick_volume']
    
    # Building feature matrix X and target vector y
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df['target']
    
    print(f"  -> Features used ({len(feature_cols)}): {feature_cols}")

    # Temporal Split use 70% for training and the rest for testing
    split_point = int(len(df) * 0.70)
    
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]
    
    print(f"  -> Training on {len(X_train)} rows\n  -> Testing on {len(X_test)} rows")

    # Random Forest Training
    # n_estimators=100 means 100 decision trees
    # min_samples_leaf=50 prevents the model from memorizing noise (overfitting)
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluation
    print("  -> Results on Test Set")
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds) # When it says "Up", how often is it correct?
    rec = recall_score(y_test, preds)    # When it says "Up", how often does it miss?
    print(f"     Accuracy:  {acc:.4f} (50% is like flipping a coin)")
    print(f"     Precision: {prec:.4f} (Important to avoid false signals)")
    print(f"     Recall:    {rec:.4f} (Important to not miss opportunities)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    print("  -> Confusion Matrix:")
    print(f"     True Negatives (Avoided crashes): {cm[0][0]} ({cm[0][0] * 100 / len(y_test):.2f}%) | False Positives (Loss): {cm[0][1]} ({cm[0][1] * 100 / len(y_test):.2f}%)")
    print(f"     False Negatives (Missed opportunities): {cm[1][0]} ({cm[1][0] * 100 / len(y_test):.2f}%) | True Positives (Profit): {cm[1][1]} ({cm[1][1] * 100 / len(y_test):.2f}%)")

    # Feature Importance (What does the model look at?)
    print("  -> Feature Importance:")
    importances = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print(importances)
    return model