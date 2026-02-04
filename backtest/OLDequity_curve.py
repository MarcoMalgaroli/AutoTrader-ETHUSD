import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

PRINT_WIDTH = 100

# --- CONFIGURAZIONE ---
FILE_NAME = Path("datasets", "final", "ETHUSD_D1_3065.csv")
# FILE_NAME = Path("datasets", "final", "ETHUSD_M15_176764.csv")
# FILE_NAME = Path("datasets", "final", "ETHUSD_M5_518717.csv")

INITIAL_CAPITAL = 10000  # Dollars
FEE_PER_TRADE = 0 # 0.001
BET_SIZE = 0.01  # Not used in this simple backtest



print("\n" + " BACKTESTING ".center(PRINT_WIDTH, "="))
df = pd.read_csv(FILE_NAME)

# cols_to_drop = ['time', 'target', 'open', 'high', 'low', 'close', 
#                 'SMA_5', 'EMA_5', 'SMA_10', 'EMA_10', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50', 
#                 'ATR_14', 'MACD', 'vol_SMA_20', 'tick_volume']
# SELECTED_FEATURES = [c for c in df.columns if c not in cols_to_drop]

SELECTED_FEATURES = [
    'BB_width_pct', 'KC_width_pct', 'ATR_norm', 'dist_SMA_50', 
    'dist_EMA_50', 'RSI_15', 'vol_rel', 'MACD_norm', 'RSI_20', 'EMI_norm'
]

# Prepare data
X = df[SELECTED_FEATURES]
y = df['target']

print(f"  -> Features used ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")

# Temporal Split (85% Train, 15% Backtest)
train_size = int(len(df) * 0.85)
gap = 10 # To avoid Triple Barrier leakage

X_train, X_test = X.iloc[:train_size-gap], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size-gap], y.iloc[train_size:]

print(f"  -> Training on {len(X_train)} rows")
print(f"  -> Backtesting on {len(X_test)} rows")

# Price data for profit calculation (corresponding to the test set)
# Note: the signal is TODAY, we trade the candle of TOMORROW (or today's close)
# Signal calculated at yesterday's close -> Enter at today's open -> Exit at today's close
price_data = df.iloc[train_size:].copy()
price_data = price_data[['open', 'close', 'high', 'low', 'ATR_14']]

# --- MODEL TRAINING ---
# Random Forest Training
# n_estimators = decision trees count
# min_samples_leaf prevents the model from memorizing noise (overfitting)
print("Training model...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_leaf=4, 
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

# Evaluation
print("Running Trading Simulation...")

# Generate signals on the Test Set
signals = rf.predict(X_test)
price_data['signal'] = signals

print("  -> Results on Test Set")
print("Accuracy:", rf.score(X_test, y_test))
print("\nClassification Report:\n", classification_report(y_test, signals))
cm = confusion_matrix(y_test, signals)
print("Confusion Matrix:\n", cm)
print("  -> Confusion Matrix:")
print(f"     Was HOLD and predicted HOLD : {cm[0][0]} ({cm[0][0] * 100 / len(y_test):.2f}%) | Was HOLD and predicted LONG : {cm[0][1]} ({cm[0][1] * 100 / len(y_test):.2f}%) | Was HOLD and predicted SHORT : {cm[0][2]} ({cm[0][2] * 100 / len(y_test):.2f}%)")
print(f"     Was LONG and predicted HOLD : {cm[1][0]} ({cm[1][0] * 100 / len(y_test):.2f}%) | Was LONG and predicted LONG : {cm[1][1]} ({cm[1][1] * 100 / len(y_test):.2f}%) | Was LONG and predicted SHORT : {cm[1][2]} ({cm[1][2] * 100 / len(y_test):.2f}%)")
print(f"     Was SHORT and predicted HOLD : {cm[2][0]} ({cm[2][0] * 100 / len(y_test):.2f}%) | Was SHORT and predicted LONG : {cm[2][1]} ({cm[2][1] * 100 / len(y_test):.2f}%) | Was SHORT and predicted SHORT : {cm[2][2]} ({cm[2][2] * 100 / len(y_test):.2f}%)")
print(f"     Overall Accuracy: {(cm[0][0] + cm[1][1] + cm[2][2]) * 100 / len(y_test):.2f}%")

# --- FEATURE IMPORTANCE (The most important part!) ---
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n--- TOP 10 WINNING FEATURES ---")
for f in range(10):
    print(f"{f+1}. {SELECTED_FEATURES[indices[f]]} ({importances[indices[f]]:.4f})")

plt.figure(figsize=(12, 6))
plt.title("Feature Importances (What the model looks at?)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [SELECTED_FEATURES[i] for i in indices], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


# --- BACKTEST ENGINE ---
capital = INITIAL_CAPITAL
equity = [INITIAL_CAPITAL]

positions = []

# ATR Multipliers (devono essere gli stessi usati nel training!)
ATR_MULT_TP = 1.2
ATR_MULT_SL = 1.0
MAX_HOLD_DAYS = 10


wins = 0
losses = 0
drawdown = []

# Loop day by day
# for each day's signal, calculate profit based on next day's price movement following the predicted signal
for i in range(len(price_data) - 1):
    # Data of the day and future prices
    current_close = price_data.iloc[i]['close']
    current_atr = price_data.iloc[i]['ATR_14']
    next_open = price_data.iloc[i+1]['open']
    next_high = price_data.iloc[i+1]['high']
    next_low = price_data.iloc[i+1]['low']
    next_close = price_data.iloc[i+1]['close']

    sig = price_data.iloc[i]['signal'] # predicted signal for the next day based on the next 10 days
    
    # Daily percentage change
    trade_result = 0

    positions.append({
        position_days = 0,
        entry_price = 0.0,
        position_type = 0, # 1: Long, 2: Short
        stop_loss = 0.0,
        take_profit = 0.0,
    })
    
    # TRADING LOGIC
    if sig == 1: # LONG
        # Profit if price goes up, loss if it goes down - Commission
        trade_result = (capital * BET_SIZE) * (daily_change_pct - FEE_PER_TRADE)
    
    elif sig == 2: # SHORT
        # Profit if price goes down (inverse), loss if it goes up - Commission
        trade_result = (capital * BET_SIZE) * (-daily_change_pct - FEE_PER_TRADE)

    # Capital Update
    capital += trade_result
    equity.append(capital)
    
    # Win/Loss Statistics (only if we traded)
    if sig != 0:
        if trade_result > 0: wins += 1
        else: losses += 1

# --- FINAL METRICS CALCULATION ---
equity = np.array(equity)
total_profit = capital - INITIAL_CAPITAL
total_return = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
max_capital = np.maximum.accumulate(equity)
drawdowns = (equity - max_capital) / max_capital
max_drawdown = drawdowns.min() * 100
avg_profit_per_trade = total_profit / (wins + losses) if (wins + losses) > 0 else 0

print("\n" + "="*30)
print(f" BACKTEST RESULTS (D1)")
print("="*30)
print(f"Initial Capital: ${INITIAL_CAPITAL}")
print(f"Final Capital:   ${capital:.2f}")
print(f"Total Profit:    ${total_profit:.2f}")
print(f"Total Return: {total_return:.2f}%")

print(f"Max Drawdown:      {max_drawdown:.2f}%")
print(f"Trades Executed:    {wins + losses}")
print(f"Win Rate:          {wins/(wins+losses)*100:.1f}%")
print(f"Avg Profit/Trade:  ${avg_profit_per_trade:.2f}")
print("="*30)

# Comparison with Buy & Hold
eth_start = price_data.iloc[0]['open']
eth_end = price_data.iloc[-1]['close']
bh_return = ((eth_end - eth_start) / eth_start) * 100
print(f"Buy & Hold ETH:    {bh_return:.2f}%")

# --- PLOT ---
plt.figure(figsize=(12, 6))
plt.plot(equity, label='ML Strategy (Long/Short)', color='green')
plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', alpha=0.3, label='Break Even')
bh_equity = INITIAL_CAPITAL * (1 + (price_data['close'].values - eth_start) / eth_start)
plt.plot(bh_equity, label='Buy & Hold ETH', color='gray', alpha=0.5, linestyle='--')

plt.title('Equity Curve: ML Model vs Buy & Hold')
plt.xlabel('Trading Days')
plt.ylabel('Capital ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()