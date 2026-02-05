import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

PRINT_WIDTH = 100


# winning features
SELECTED_FEATURES = [
    'BB_width_pct', 'KC_width_pct', 'ATR_norm', 'dist_SMA_50', 
    'dist_EMA_50', 'RSI_15', 'vol_rel', 'MACD_norm', 'RSI_20', 'EMI_norm'
]

def backtest(path: Path, initial_capital: float = 10000):
    print("\n" + " BACKTEST ".center(PRINT_WIDTH, "="))
    df = pd.read_csv(path)

    # Prepare the data
    X = df[SELECTED_FEATURES]
    y = df['target']

    # Temporal Split (85% Train, 15% Backtest)
    train_size = int(len(df) * 0.85)

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:].copy() # .copy() to avoid warning
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    # Price data for profit calculation (corresponding to the test set)
    # Note: shift(-1) because if the signal is TODAY, we trade the candle of TOMORROW (or today's close)
    # For simplicity, we assume:
    # Signal calculated at Yesterday's close -> Enter at Today's Open -> Exit at Today's Close
    price_data = df.iloc[train_size:].copy()
    price_data = price_data[['open', 'close', 'high', 'low']]

    # --- MODEL TRAINING ---
    print("  -> Training model...")

    # best for hourly data
    # rf = RandomForestClassifier(
    #     n_estimators=200, max_depth=12, min_samples_leaf=4, 
    #     class_weight='balanced', random_state=42, n_jobs=-1
    # )

    #best for daily data
    rf = RandomForestClassifier(
        n_estimators = 150,
        max_depth = 20,
        min_samples_split = 60,
        min_samples_leaf = 2,
        class_weight = 'balanced',
        random_state = 42,
        n_jobs = -1,
    )
    rf.fit(X_train, y_train)

    # Generate signals on the Test Set
    signals = rf.predict(X_test)
    price_data['signal'] = signals

    # --- BACKTEST ENGINE ---
    print("  -> Running backtest...")

    equity = [initial_capital]
    capital = initial_capital
    wins = 0
    losses = 0

    # Loop day by day
    for i in range(len(price_data)):
        # Data of the day
        open_price = price_data.iloc[i]['open']
        close_price = price_data.iloc[i]['close']
        sig = price_data.iloc[i]['signal']
        
        # Calculate daily percentage change
        daily_change_pct = (close_price - open_price) / open_price
        
        trade_result = 0
        
        # TRADING LOGIC
        if sig == 1: # LONG
            # Profit if price goes up, loss if it goes down - Commission
            trade_result = capital * daily_change_pct
        
        elif sig == -1: # SHORT
            # Profit if price goes down (inverse), loss if it goes up - Commission
            trade_result = capital * -daily_change_pct

        # Capital Update
        capital += trade_result
        equity.append(capital)
        
        # Win/Loss Statistics (only if we traded)
        if sig != 0:
            if trade_result > 0: wins += 1
            else: losses += 1

    # --- FINAL METRICS CALCULATION ---
    equity = np.array(equity)
    total_profit = capital - initial_capital
    total_return = ((capital - initial_capital) / initial_capital) * 100
    max_capital = np.maximum.accumulate(equity)
    drawdowns = (equity - max_capital) / max_capital
    max_drawdown = drawdowns.min() * 100

    print("\n" + "="*30)
    print(f" BACKTEST RESULTS (D1)")
    print("="*30)
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Capital:   ${capital:.2f}")
    print(f"Total Profit:    ${total_profit:.2f}")
    print(f"Total Return:    {total_return:.2f}%")
    print(f"Max Drawdown:    {max_drawdown:.2f}%")
    print(f"Trades Executed: {wins + losses}")
    print(f"Win Rate:        {wins/(wins+losses)*100:.1f}%" if (wins+losses)>0 else "N/A")
    print("="*30)

    # Comparison with Buy & Hold
    eth_start = price_data.iloc[0]['open']
    eth_end = price_data.iloc[-1]['close']
    bh_return = ((eth_end - eth_start) / eth_start) * 100
    print(f"Buy & Hold ETH:    {bh_return:.2f}%")

    # --- PLOT ---
    plt.figure(figsize=(12, 6))
    plt.plot(equity, label='ML Strategy (Long/Short)', color='green')
    plt.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.3, label='Break Even')
    
    bh_equity = initial_capital * (1 + (price_data['close'].values - eth_start) / eth_start)
    plt.plot(bh_equity, label='Buy & Hold ETH', color='gray', alpha=0.5, linestyle='--')

    plt.title('Equity Curve: ML Model vs Buy & Hold')
    plt.xlabel('Trading Days')
    plt.ylabel('Capital ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()