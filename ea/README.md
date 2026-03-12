# MT5 Strategy Tester - Signal Replay

This folder contains an MQL5 Expert Advisor that replays the pre-computed ML
predictions inside the **MT5 Strategy Tester**, giving you a broker-realistic
backtest with real spread, slippage, and execution.

## How to execute backtest with strategy tester

### Step 1 - Export signals from the Dashboard

1. Open the dashboard and run a backtest with your desired parameters.
2. Click **Download Signals** and save a JSON file with the same format the EA expects.

The dashboard runs the same walk-forward loop as the backtest and produces a
JSON file containing every LONG/SHORT signal with TP, SL, confidence, and
position sizing.

### Step 2 - Place the JSON file

The EA loads signals from the **MT5 common data folder**:

```
<Common Data>\Files\signals_lstm_D1.json
```

To find it in MT5:
1. **File → Open Data Folder** - this opens the local data folder.
2. Go **up one level** in Explorer.
3. Open **Common → Files**.

Copy your downloaded JSON into that `Files` folder.

### Step 3 - Install and compile the EA

1. **Copy the EA** (`AutoTrader_SignalFollower.mq5`) to:
   ```
   <MT5 Data Folder>\MQL5\Experts\AutoTrader_SignalFollower.mq5
   ```

2. **Compile** the EA in MetaEditor (F7).

### Step 4 - Run in MT5 Strategy Tester

1. **Open Strategy Tester** (Ctrl+R) and configure:
   - **Expert**: `AutoTrader_SignalFollower`
   - **Symbol**: `ETHUSD` (or the one selected in the config tab)
   - **Period** (near symbol): must match the timeframe used in the JSON (e.g., Daily)
   - **Date range**: should cover the dates in the selected backtest window (e.g., if backtest windows = 365 and timeframe = daily, you should set the start date back one year.)
   - **Model**: set to *Every tick*
   - **Inputs**:
     - `InpSignalFile`: filename of your JSON (e.g., `signals_lstm_D1.json`)
     - `InpInitialCapital`: must match with the value in the backtest dashboard
     - `InpUseFileSizing`: `true` to use confidence-based position sizing
     - `InpFixedLots`: only used if `InpUseFileSizing` = `false`

2. **Start** the test.

## What the EA does (per new bar)

1. **Checks time barriers** on any open trade - if `lookahead` bars have
   elapsed without TP/SL being hit, the position is force-closed (same as
   the position check in the live dashboard WS loop).

2. **Looks up the signal** for the *previous* bar's datetime in the JSON file
   (binary search, O(log n)).

3. If a LONG or SHORT signal exists:
   - Calculates **volume** from `position_size_pct × equity / (price × contract_size)`,
     rounded to the symbol's volume step, clamped to broker limits - identical
     to `calculate_volume()` in the live trader.
   - Opens a **market order** with the absolute TP and SL from the JSON.


## JSON format reference

```json
{
  "symbol": "ETHUSD",
  "timeframe": "D1",
  "lookahead": 4,
  "atr_mult": 1.5,
  "threshold": 0.35,
  "model_type": "lstm",
  "generated_at": "2026-03-04T12:00:00",
  "total_signals": 142,
  "signals": [
    {
      "signal_time": "2025-06-10 00:00:00",
      "direction": "LONG",
      "signal_price": 1850.50,
      "tp": 1892.30,
      "sl": 1808.70,
      "lookahead": 4,
      "confidence": 0.62,
      "confidence_tier": "HIGH",
      "position_size_pct": 0.04
    }
  ]
}
```
