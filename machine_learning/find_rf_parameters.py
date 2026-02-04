from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

PRINT_WIDTH = 100

def train_random_forest_model(file_name = Path('datasets/final/ETHUSD_D1_3065.csv')):
    """
    Search for Random Forest parameters on the dataset located at file_name.
    Args:
        file_name (str): Path to the CSV file containing the dataset.
    """
    print("\n" + " RANDOM FOREST PARAMETERS SEARCH ".center(PRINT_WIDTH, "="))

    print(f"---> Loading dataset from {file_name}...")
    df = pd.read_csv(file_name)

    cols_to_drop = ['time', 'target', 'open', 'high', 'low', 'close', 
                'SMA_5', 'EMA_5', 'SMA_10', 'EMA_10', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50', 
                'ATR_14', 'MACD', 'vol_SMA_20', 'tick_volume']
    SELECTED_FEATURES = [c for c in df.columns if c not in cols_to_drop]
    
    # SELECTED_FEATURES = [
    #     'BB_width_pct',
    #     'KC_width_pct',
    #     'ATR_norm',
    #     'dist_SMA_50',
    #     'dist_EMA_50',
    #     'RSI_15',
    #     'vol_rel',
    #     'MACD_norm',
    #     'RSI_20',
    #     'EMI_norm'
    # ]

    X = df[SELECTED_FEATURES]
    y = df['target']
    
    print(f"  -> Features used ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")

    # Temporal Split use 85% for training and the rest for testing
    train_size = int(len(df) * 0.85)
    
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    print(f"  -> Training on {len(X_train)} rows")
    print(f"  -> Testing on {len(X_test)} rows")

    param_grid = {
        'n_estimators': [x for x in range(50, 651, 50)],
        'max_depth': [8, 9, 10, 12, 15],
        'min_samples_split': [x for x in range(6, 15, 2)],
        'min_samples_leaf': [x for x in range(1, 5)],
    }
    rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))


train_random_forest_model()