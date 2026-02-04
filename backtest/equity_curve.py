import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path


# Le tue feature vincenti
SELECTED_FEATURES = [
    'BB_width_pct', 'KC_width_pct', 'ATR_norm', 'dist_SMA_50', 
    'dist_EMA_50', 'RSI_15', 'vol_rel', 'MACD_norm', 'RSI_20', 'EMI_norm'
]

def backtest(path: Path, initial_capital: float = 10000):
    print(f"--- Caricamento e Preparazione Backtest ---")
    df = pd.read_csv(path)

    # Prepariamo i dati
    X = df[SELECTED_FEATURES]
    y = df['target']

    # Split Temporale (85% Train, 15% Backtest)
    train_size = int(len(df) * 0.85)

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:].copy() # .copy() per evitare warning
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    # Dati di prezzo per il calcolo profitti (corrispondenti al test set)
    # Nota: shift(-1) perché se il segnale è OGGI, noi tradiamo la candela di DOMANI (o la chiusura di oggi)
    # Per semplicità, assumiamo:
    # Segnale calcolato alla chiusura di Ieri -> Entro in Open di Oggi -> Esco in Close di Oggi
    price_data = df.iloc[train_size:].copy()
    price_data = price_data[['open', 'close', 'high', 'low']]

    # --- ADDESTRAMENTO MODELLO ---
    print("Addestramento modello in corso...")

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

    # Generiamo i segnali sul Test Set
    signals = rf.predict(X_test)
    price_data['signal'] = signals

    # --- MOTORE DI BACKTEST ---
    print("Simulazione Trading...")

    equity = [initial_capital]
    capital = initial_capital
    wins = 0
    losses = 0

    # Loop giorno per giorno
    for i in range(len(price_data)):
        # Dati del giorno
        open_price = price_data.iloc[i]['open']
        close_price = price_data.iloc[i]['close']
        sig = price_data.iloc[i]['signal']
        
        # Calcolo variazione percentuale della giornata
        daily_change_pct = (close_price - open_price) / open_price
        
        trade_result = 0
        
        # LOGICA DI TRADING
        if sig == 1: # LONG
            # Guadagno se sale, perdo se scende - Commissione
            trade_result = capital * daily_change_pct
        
        elif sig == -1: # SHORT
            # Guadagno se scende (inverso), perdo se sale - Commissione
            trade_result = capital * -daily_change_pct

        # Aggiornamento Capitale
        capital += trade_result
        equity.append(capital)
        
        # Statistiche Win/Loss (solo se abbiamo tradato)
        if sig != 0:
            if trade_result > 0: wins += 1
            else: losses += 1

    # --- CALCOLO METRICHE FINALI ---
    equity = np.array(equity)
    total_profit = capital - initial_capital
    total_return = ((capital - initial_capital) / initial_capital) * 100
    max_capital = np.maximum.accumulate(equity)
    drawdowns = (equity - max_capital) / max_capital
    max_drawdown = drawdowns.min() * 100

    print("\n" + "="*30)
    print(f" RISULTATI BACKTEST (D1)")
    print("="*30)
    print(f"Capitale Iniziale: ${initial_capital:.2f}")
    print(f"Capitale Finale:   ${capital:.2f}")
    print(f"Profitto Totale:  ${total_profit:.2f}")
    print(f"Rendimento Totale: {total_return:.2f}%")
    print(f"Max Drawdown:      {max_drawdown:.2f}%")
    print(f"Trade Eseguiti:    {wins + losses}")
    print(f"Win Rate:          {wins/(wins+losses)*100:.1f}%" if (wins+losses)>0 else "N/A")
    print("="*30)

    # Confronto con Buy & Hold
    eth_start = price_data.iloc[0]['open']
    eth_end = price_data.iloc[-1]['close']
    bh_return = ((eth_end - eth_start) / eth_start) * 100
    print(f"Buy & Hold ETH:    {bh_return:.2f}%")

    # --- GRAFICO ---
    plt.figure(figsize=(12, 6))
    plt.plot(equity, label='Strategia ML (Long/Short)', color='green')
    plt.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.3, label='Break Even')
    
    bh_equity = initial_capital * (1 + (price_data['close'].values - eth_start) / eth_start)
    plt.plot(bh_equity, label='Buy & Hold ETH', color='gray', alpha=0.5, linestyle='--')

    plt.title('Equity Curve: Modello ML vs Buy & Hold')
    plt.xlabel('Giorni di Trading')
    plt.ylabel('Capitale ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()