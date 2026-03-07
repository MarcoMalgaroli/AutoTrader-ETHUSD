//+------------------------------------------------------------------+
//|                              AutoTrader_SignalFollower.mq5       |
//|                              AI Signal Replay Expert Advisor     |
//|                                                                  |
//|  Reads a JSON file of ML predictions exported from the dashboard |
//|  (Download Signals button) and replays them as real MT5 trades.  |
//|  Works in both Strategy Tester and Live mode.                    |
//|                                                                  |
//|  JSON file placement:                                            |
//|    <Common Data>\Files\   (the MT5 common data folder)           |
//|  Find it via MT5: File → Open Data Folder, then go up one        |
//|  level and open the "Common" folder, then "Files".               |
//|                                                                  |
//|  Workflow per new bar:                                           |
//|    1. Look up signal_time == previous candle time in the JSON    |
//|    2. If LONG/SHORT → open market order with TP + SL             |
//|    3. Track open positions; force-close on time barrier          |
//|       (lookahead candles elapsed without TP/SL hit)              |
//|                                                                  |
//|  This mirrors the exact behaviour of live_trading/trader.py      |
//+------------------------------------------------------------------+
#property copyright   "AutoTrader-ETHUSD"
#property link        ""
#property version     "1.00"
#property strict
#property description "Replay AI predictions from JSON signals file"

#include <Trade\Trade.mqh>

//--- Input parameters --------------------------------------------------------
input string   InpSignalFile     = "signals_lstm_D1.json"; // JSON signals file (in MQL5/Files)
input double   InpInitialCapital = 100000;                 // Initial capital for position sizing
input int      InpMagicNumber    = 112233;                 // Magic number (matches Python trader)
input int      InpSlippage       = 10;                     // Max slippage in points
input bool     InpUseFileSizing  = true;                   // Use position_size_pct from file
input double   InpFixedLots      = 0.01;                   // Fixed lot size (if UseFileSizing=false)
input int      InpExecHour       = 1;                      // Execution hour (server time, 0-23)
input int      InpExecMinute     = 15;                     // Execution minute (0-59)
input int      InpMaxRetries     = 50;                     // Max retry ticks if market closed

//--- Signal structure --------------------------------------------------------
struct Signal
{
   datetime signal_time;       // candle time that generated the signal
   int      direction;         // +1 = LONG, -1 = SHORT
   double   signal_price;      // close price used for barrier calc
   double   tp;                // absolute take-profit price
   double   sl;                // absolute stop-loss price
   int      lookahead;         // bars before time barrier
   double   confidence;        // model confidence
   string   confidence_tier;   // LOW / AVG / HIGH
   double   position_size_pct; // fraction of capital to risk
};

//--- Globals -----------------------------------------------------------------
Signal   g_signals[];          // loaded signals array
int      g_signalCount = 0;
CTrade   g_trade;
datetime g_lastBarTime = 0;
bool     g_signalCheckedThisBar = true;  // have we done the signal lookup for the current bar?

//--- Track open trades for time-barrier management ---------------------------
struct OpenTrade
{
   ulong    ticket;
   datetime open_bar_time;     // bar time when the trade was opened
   int      lookahead;         // bars to wait before force-close
   int      bars_elapsed;      // bars elapsed since open
};

OpenTrade g_openTrades[];
int       g_openTradeCount = 0;

//--- Pending actions (retry when market is closed) --------------------------
struct PendingAction
{
   bool     active;            // is there a pending action?
   int      type;              // 0 = new trade, 1 = time-barrier close
   int      signal_idx;        // index in g_signals (for type 0)
   ulong    ticket;            // ticket to close (for type 1)
   int      retries;           // remaining retry attempts
};

PendingAction g_pendingNew;              // pending new-trade action
PendingAction g_pendingCloses[];         // pending time-barrier closes (can be multiple)
int           g_pendingCloseCount = 0;

//--- Track which signal indices have already been traded (no duplicates) -----
int  g_tradedSignals[];
int  g_tradedSignalCount = 0;

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(InpSlippage);
   g_trade.SetTypeFilling(ORDER_FILLING_IOC);

   if(!LoadSignals(InpSignalFile))
   {
      Print("ERROR: Failed to load signals from ", InpSignalFile);
      return INIT_FAILED;
   }

   //--- Initialize pending actions
   g_pendingNew.active   = false;
   g_pendingCloseCount   = 0;
   g_tradedSignalCount   = 0;

   Print("AutoTrader SignalFollower initialized - ", g_signalCount, " signals loaded");
   Print("Execution window: ", InpExecHour, ":", StringFormat("%02d", InpExecMinute), " server time");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   PrintBacktestResults();
   Print("AutoTrader SignalFollower stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Print a summary of backtest results at the end of the run        |
//+------------------------------------------------------------------+
void PrintBacktestResults()
{
   //--- Select full deal history
   if(!HistorySelect(0, TimeCurrent()))
   {
      Print("WARNING: Could not load deal history for summary");
      return;
   }

   int    totalDeals   = HistoryDealsTotal();
   int    totalTrades  = 0;
   int    wins         = 0;
   int    losses       = 0;
   double grossProfit  = 0.0;
   double grossLoss    = 0.0;
   double netProfit    = 0.0;

   //--- Walk through all deals, count completed trades (DEAL_ENTRY_OUT)
   for(int i = 0; i < totalDeals; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;

      long magic = HistoryDealGetInteger(ticket, DEAL_MAGIC);
      if(magic != InpMagicNumber) continue;

      long entry = HistoryDealGetInteger(ticket, DEAL_ENTRY);
      if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_INOUT) continue;

      double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT)
                    + HistoryDealGetDouble(ticket, DEAL_SWAP)
                    + HistoryDealGetDouble(ticket, DEAL_COMMISSION);

      totalTrades++;
      netProfit += profit;

      if(profit >= 0.0)
      {
         wins++;
         grossProfit += profit;
      }
      else
      {
         losses++;
         grossLoss += MathAbs(profit);
      }
   }

   //--- Equity curve & max drawdown (rebuild from deal profits)
   double equity       = InpInitialCapital;
   double peakEquity   = equity;
   double maxDrawdown  = 0.0;
   double maxDDpct     = 0.0;

   for(int i = 0; i < totalDeals; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;

      long magic = HistoryDealGetInteger(ticket, DEAL_MAGIC);
      if(magic != InpMagicNumber) continue;

      long entry = HistoryDealGetInteger(ticket, DEAL_ENTRY);
      if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_INOUT) continue;

      double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT)
                    + HistoryDealGetDouble(ticket, DEAL_SWAP)
                    + HistoryDealGetDouble(ticket, DEAL_COMMISSION);

      equity += profit;
      if(equity > peakEquity)
         peakEquity = equity;

      double dd    = peakEquity - equity;
      double ddPct = (peakEquity > 0) ? (dd / peakEquity * 100.0) : 0.0;

      if(dd > maxDrawdown)
         maxDrawdown = dd;
      if(ddPct > maxDDpct)
         maxDDpct = ddPct;
   }

   double finalEquity = equity;
   double totalReturn = (InpInitialCapital > 0) ? ((finalEquity - InpInitialCapital) / InpInitialCapital * 100.0) : 0.0;
   double winRate     = (totalTrades > 0) ? ((double)wins / totalTrades * 100.0) : 0.0;
   double profitFactor = (grossLoss > 0) ? (grossProfit / grossLoss) : 0.0;
   double avgTrade    = (totalTrades > 0) ? (netProfit / totalTrades) : 0.0;

   //--- Print summary
   Print("================================================================");
   Print("          BACKTEST RESULTS SUMMARY");
   Print("================================================================");
   Print(StringFormat("  Initial Capital:   $%.2f",  InpInitialCapital));
   Print(StringFormat("  Final Equity:      $%.2f",  finalEquity));
   Print(StringFormat("  Net Profit:        $%.2f (%+.2f%%)", netProfit, totalReturn));
   Print(StringFormat("  Gross Profit:      $%.2f",  grossProfit));
   Print(StringFormat("  Gross Loss:       -$%.2f",  grossLoss));
   Print("----------------------------------------------------------------");
   Print(StringFormat("  Total Trades:      %d",     totalTrades));
   Print(StringFormat("  Wins / Losses:     %d / %d", wins, losses));
   Print(StringFormat("  Win Rate:          %.1f%%", winRate));
   Print(StringFormat("  Profit Factor:     %.2f",   profitFactor));
   Print(StringFormat("  Avg Trade P/L:     $%.2f",  avgTrade));
   Print("----------------------------------------------------------------");
   Print(StringFormat("  Max Drawdown:      $%.2f (%.2f%%)", maxDrawdown, maxDDpct));
   Print(StringFormat("  Peak Equity:       $%.2f",  peakEquity));
   Print(StringFormat("  Signals Loaded:    %d",     g_signalCount));
   Print(StringFormat("  Signals Traded:    %d",     g_tradedSignalCount));
   Print("================================================================");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Always try to execute pending retries (market may have just opened)
   RetryPendingCloses();
   RetryPendingNew();

   //--- Detect new bar
   datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   bool isNewBar = (currentBarTime != g_lastBarTime);

   if(isNewBar)
   {
      g_lastBarTime = currentBarTime;
      g_signalCheckedThisBar = false;

      //--- Check time barriers once per new bar
      CheckTimeBarriers();
   }

   //--- Signal lookup: deferred until execution window, then done once per bar
   if(!g_signalCheckedThisBar)
   {
      MqlDateTime now;
      TimeCurrent(now);
      if(now.hour < InpExecHour || (now.hour == InpExecHour && now.min < InpExecMinute))
         return;  // too early - wait for execution window

      g_signalCheckedThisBar = true;

      //--- Look for signal on the PREVIOUS bar
      datetime prevBarTime = iTime(_Symbol, PERIOD_CURRENT, 1);
      int idx = FindSignal(prevBarTime);

      if(idx < 0)
         return;  // no signal for this bar

      //--- Already queued or already traded this signal?
      if(g_pendingNew.active && g_pendingNew.signal_idx == idx)
         return;
      if(IsSignalAlreadyTraded(idx))
         return;

      //--- Attempt to open the trade (or queue for retry)
      if(!ExecuteNewTrade(idx))
      {
         g_pendingNew.active     = true;
         g_pendingNew.type       = 0;
         g_pendingNew.signal_idx = idx;
         g_pendingNew.ticket     = 0;
         g_pendingNew.retries    = InpMaxRetries;
         Print("Trade queued for retry (market closed)");
      }
   }
}

//+------------------------------------------------------------------+
//| Execute a new trade from signal index. Returns true on success.  |
//+------------------------------------------------------------------+
bool ExecuteNewTrade(int idx)
{
   Signal sig = g_signals[idx];

   //--- Calculate volume
   double volume = InpFixedLots;
   if(InpUseFileSizing)
   {
      double equity = AccountInfoDouble(ACCOUNT_EQUITY);
      if(equity <= 0) equity = InpInitialCapital;
      double dollar_risk = equity * sig.position_size_pct;

      double price = (sig.direction > 0) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                                          : SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double contract_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
      if(contract_size <= 0) contract_size = 1.0;
      if(price <= 0) price = sig.signal_price;

      volume = dollar_risk / (price * contract_size);

      //--- Round to volume step
      double vol_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
      if(vol_step > 0)
         volume = MathFloor(volume / vol_step + 0.5) * vol_step;

      double vol_min = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
      double vol_max = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
      if(volume < vol_min) volume = vol_min;
      if(volume > vol_max) volume = vol_max;

      int decimals = (int)MathMax(0, -MathLog10(vol_step));
      volume = NormalizeDouble(volume, decimals);
   }

   //--- Place order
   string comment = StringFormat("AI-%s-%.0f%%",
                                 sig.confidence_tier,
                                 sig.confidence * 100);

   bool ok = false;
   if(sig.direction > 0)
      ok = g_trade.Buy(volume, _Symbol, 0, sig.sl, sig.tp, comment);
   else
      ok = g_trade.Sell(volume, _Symbol, 0, sig.sl, sig.tp, comment);

   if(ok)
   {
      ulong ticket = g_trade.ResultOrder();
      datetime barTime = iTime(_Symbol, PERIOD_CURRENT, 0);
      Print(StringFormat("TRADE OPENED: %s  vol=%.2f  TP=%.5f  SL=%.5f  conf=%.2f%%  ticket=%d",
                         (sig.direction > 0 ? "LONG" : "SHORT"),
                         volume, sig.tp, sig.sl, sig.confidence * 100, ticket));
      AddOpenTrade(ticket, barTime, sig.lookahead);
      MarkSignalTraded(idx);
      g_pendingNew.active = false;
      return true;
   }

   //--- Check if failure is "market closed" (retryable) vs other error
   uint retcode = g_trade.ResultRetcode();
   if(retcode == TRADE_RETCODE_MARKET_CLOSED  || retcode == 10031 /*NO_CONNECTION*/ ||
      retcode == TRADE_RETCODE_PRICE_OFF      || retcode == TRADE_RETCODE_TIMEOUT)
   {
      return false;  // retryable
   }

   //--- Non-retryable error
   Print("TRADE FAILED (permanent): ", g_trade.ResultRetcodeDescription());
   return true;  // return true to prevent retry
}

//+------------------------------------------------------------------+
//| Retry pending new-trade if market was closed                     |
//+------------------------------------------------------------------+
void RetryPendingNew()
{
   if(!g_pendingNew.active) return;
   if(g_pendingNew.retries <= 0)
   {
      Print("Pending new trade exhausted retries - giving up");
      g_pendingNew.active = false;
      return;
   }

   g_pendingNew.retries--;
   if(ExecuteNewTrade(g_pendingNew.signal_idx))
      g_pendingNew.active = false;
}

//+------------------------------------------------------------------+
//| Retry all pending time-barrier closes if market was closed       |
//+------------------------------------------------------------------+
void RetryPendingCloses()
{
   for(int i = g_pendingCloseCount - 1; i >= 0; i--)
   {
      if(g_pendingCloses[i].retries <= 0)
      {
         Print("Pending close exhausted retries - giving up for ticket ", g_pendingCloses[i].ticket);
         RemovePendingClose(i);
         continue;
      }

      //--- Position may have been closed by TP/SL in the meantime
      if(!PositionSelectByTicket(g_pendingCloses[i].ticket))
      {
         Print("Pending close: position already closed (TP/SL hit)");
         RemovePendingClose(i);
         continue;
      }

      g_pendingCloses[i].retries--;
      if(g_trade.PositionClose(g_pendingCloses[i].ticket, InpSlippage))
      {
         Print("TIME BARRIER close succeeded on retry for ticket ", g_pendingCloses[i].ticket);
         RemovePendingClose(i);
      }
   }
}

void RemovePendingClose(int index)
{
   for(int j = index; j < g_pendingCloseCount - 1; j++)
      g_pendingCloses[j] = g_pendingCloses[j + 1];
   g_pendingCloseCount--;
   ArrayResize(g_pendingCloses, g_pendingCloseCount);
}

//+------------------------------------------------------------------+
//| Track which signals have been traded (prevent duplicates)        |
//+------------------------------------------------------------------+
void MarkSignalTraded(int idx)
{
   g_tradedSignalCount++;
   ArrayResize(g_tradedSignals, g_tradedSignalCount);
   g_tradedSignals[g_tradedSignalCount - 1] = idx;
}

bool IsSignalAlreadyTraded(int idx)
{
   for(int i = 0; i < g_tradedSignalCount; i++)
      if(g_tradedSignals[i] == idx) return true;
   return false;
}

//+------------------------------------------------------------------+
//| Binary search for signal by datetime                              |
//+------------------------------------------------------------------+
int FindSignal(datetime dt)
{
   int lo = 0, hi = g_signalCount - 1;
   while(lo <= hi)
   {
      int mid = (lo + hi) / 2;
      if(g_signals[mid].signal_time == dt)
         return mid;
      else if(g_signals[mid].signal_time < dt)
         lo = mid + 1;
      else
         hi = mid - 1;
   }
   return -1;  // not found
}

//+------------------------------------------------------------------+
//| Add a trade to the time-barrier tracking array                   |
//+------------------------------------------------------------------+
void AddOpenTrade(ulong ticket, datetime openBarTime, int lookahead)
{
   g_openTradeCount++;
   ArrayResize(g_openTrades, g_openTradeCount);
   g_openTrades[g_openTradeCount - 1].ticket        = ticket;
   g_openTrades[g_openTradeCount - 1].open_bar_time  = openBarTime;
   g_openTrades[g_openTradeCount - 1].lookahead      = lookahead;
   g_openTrades[g_openTradeCount - 1].bars_elapsed   = 0;
}

//+------------------------------------------------------------------+
//| Check time barriers and force-close expired trades               |
//+------------------------------------------------------------------+
void CheckTimeBarriers()
{
   for(int i = g_openTradeCount - 1; i >= 0; i--)
   {
      //--- Check if position is still open
      if(!PositionSelectByTicket(g_openTrades[i].ticket))
      {
         //--- Position was closed (TP/SL hit) - remove from tracking
         RemoveOpenTrade(i);
         continue;
      }

      //--- Increment bar counter
      g_openTrades[i].bars_elapsed++;

      //--- Time barrier hit?
      if(g_openTrades[i].bars_elapsed >= g_openTrades[i].lookahead)
      {
         Print(StringFormat("TIME BARRIER: Closing ticket %d after %d bars",
                            g_openTrades[i].ticket, g_openTrades[i].bars_elapsed));

         if(g_trade.PositionClose(g_openTrades[i].ticket, InpSlippage))
         {
            RemoveOpenTrade(i);
         }
         else
         {
            //--- Market closed - queue for retry
            g_pendingCloseCount++;
            ArrayResize(g_pendingCloses, g_pendingCloseCount);
            g_pendingCloses[g_pendingCloseCount - 1].active  = true;
            g_pendingCloses[g_pendingCloseCount - 1].type    = 1;
            g_pendingCloses[g_pendingCloseCount - 1].ticket  = g_openTrades[i].ticket;
            g_pendingCloses[g_pendingCloseCount - 1].retries = InpMaxRetries;
            RemoveOpenTrade(i);
            Print("Time-barrier close queued for retry (market closed)");
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Remove trade from tracking array by index                        |
//+------------------------------------------------------------------+
void RemoveOpenTrade(int index)
{
   for(int j = index; j < g_openTradeCount - 1; j++)
      g_openTrades[j] = g_openTrades[j + 1];
   g_openTradeCount--;
   ArrayResize(g_openTrades, g_openTradeCount);
}

//+------------------------------------------------------------------+
//| Load signals from JSON file                                      |
//|                                                                  |
//| The JSON is exported from the dashboard (Download Signals).      |
//| Place it in the MT5 common data folder:                          |
//|   <Common Data>\Files\                                           |
//| This is the only folder accessible by both the terminal and      |
//| Strategy Tester agents.                                          |
//|                                                                  |
//| To find it: MT5 → File → Open Data Folder, go up one level,      |
//| open "Common" → "Files".                                         |
//|                                                                  |
//| MQL5 has no native JSON parser, so we do minimal line-based      |
//| parsing.  The file is structured with one signal per block.      |
//+------------------------------------------------------------------+
bool LoadSignals(string filename)
{
   //--- Open from the common Files folder (shared between terminal and all tester agents).
   int handle = FileOpen(filename, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
   {
      Print("Cannot open file: ", filename, " - Error: ", GetLastError());
      Print("Place the JSON in the MT5 common data folder:");
      Print("  <Common Data>\\Files\\", filename);
      Print("  To find it: File → Open Data Folder → go up one level → Common → Files");
      return false;
   }

   //--- Read entire file into a single string
   string content = "";
   while(!FileIsEnding(handle))
   {
      content += FileReadString(handle) + "\n";
   }
   FileClose(handle);

   //--- Parse signals array from JSON content
   return ParseSignals(content);
}

//+------------------------------------------------------------------+
//| Minimal JSON parser - extracts the "signals" array               |
//+------------------------------------------------------------------+
bool ParseSignals(string &content)
{
   //--- Find the "signals" array
   int sigStart = StringFind(content, "\"signals\"");
   if(sigStart < 0)
   {
      Print("ERROR: 'signals' key not found in JSON");
      return false;
   }

   //--- Find the opening '[' of the signals array
   int arrStart = StringFind(content, "[", sigStart);
   if(arrStart < 0) return false;

   //--- Now parse each signal object { ... }
   int pos = arrStart + 1;
   int len = StringLen(content);

   while(pos < len)
   {
      //--- Find next '{'
      int objStart = StringFind(content, "{", pos);
      if(objStart < 0) break;

      //--- Check we haven't gone past the closing ']'
      int arrEnd = StringFind(content, "]", pos);
      if(arrEnd >= 0 && objStart > arrEnd) break;

      //--- Find matching '}'
      int objEnd = StringFind(content, "}", objStart);
      if(objEnd < 0) break;

      string block = StringSubstr(content, objStart, objEnd - objStart + 1);

      Signal sig;
      if(ParseOneSignal(block, sig))
      {
         g_signalCount++;
         ArrayResize(g_signals, g_signalCount);
         g_signals[g_signalCount - 1] = sig;
      }

      pos = objEnd + 1;
   }

   Print("Parsed ", g_signalCount, " signals from JSON");
   return (g_signalCount > 0);
}

//+------------------------------------------------------------------+
//| Parse one signal JSON object                                     |
//+------------------------------------------------------------------+
bool ParseOneSignal(string &block, Signal &sig)
{
   //--- signal_time
   string timeStr = ExtractStringValue(block, "signal_time");
   if(timeStr == "") return false;
   sig.signal_time = ParseDateTime(timeStr);
   if(sig.signal_time == 0) return false;

   //--- direction
   string dirStr = ExtractStringValue(block, "direction");
   if(dirStr == "LONG")       sig.direction = 1;
   else if(dirStr == "SHORT") sig.direction = -1;
   else return false;

   //--- numeric fields
   sig.signal_price      = ExtractDoubleValue(block, "signal_price");
   sig.tp                = ExtractDoubleValue(block, "tp");
   sig.sl                = ExtractDoubleValue(block, "sl");
   sig.lookahead         = (int)ExtractDoubleValue(block, "lookahead");
   sig.confidence        = ExtractDoubleValue(block, "confidence");
   sig.position_size_pct = ExtractDoubleValue(block, "position_size_pct");

   //--- confidence_tier
   sig.confidence_tier = ExtractStringValue(block, "confidence_tier");

   return true;
}

//+------------------------------------------------------------------+
//| Extract a string value for a given key from a JSON block         |
//+------------------------------------------------------------------+
string ExtractStringValue(string &block, string key)
{
   string search = "\"" + key + "\"";
   int idx = StringFind(block, search);
   if(idx < 0) return "";

   //--- Find the colon after the key
   int colon = StringFind(block, ":", idx + StringLen(search));
   if(colon < 0) return "";

   //--- Find opening quote of value
   int qStart = StringFind(block, "\"", colon + 1);
   if(qStart < 0) return "";

   //--- Find closing quote
   int qEnd = StringFind(block, "\"", qStart + 1);
   if(qEnd < 0) return "";

   return StringSubstr(block, qStart + 1, qEnd - qStart - 1);
}

//+------------------------------------------------------------------+
//| Extract a numeric value for a given key from a JSON block        |
//+------------------------------------------------------------------+
double ExtractDoubleValue(string &block, string key)
{
   string search = "\"" + key + "\"";
   int idx = StringFind(block, search);
   if(idx < 0) return 0.0;

   //--- Find the colon
   int colon = StringFind(block, ":", idx + StringLen(search));
   if(colon < 0) return 0.0;

   //--- Extract the numeric substring (skip whitespace)
   int start = colon + 1;
   int blen = StringLen(block);

   //--- Skip spaces
   while(start < blen)
   {
      ushort ch = StringGetCharacter(block, start);
      if(ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r')
         break;
      start++;
   }

   //--- Collect digits, '.', '-', 'e', 'E', '+'
   string numStr = "";
   while(start < blen)
   {
      ushort ch = StringGetCharacter(block, start);
      if((ch >= '0' && ch <= '9') || ch == '.' || ch == '-' || ch == '+' || ch == 'e' || ch == 'E')
      {
         numStr += ShortToString(ch);
         start++;
      }
      else
         break;
   }

   if(numStr == "") return 0.0;
   return StringToDouble(numStr);
}

//+------------------------------------------------------------------+
//| Parse datetime from "YYYY-MM-DD HH:MM:SS" string                 |
//+------------------------------------------------------------------+
datetime ParseDateTime(string dtStr)
{
   // Expected format: "2025-06-10 00:00:00"
   // Also handle:     "2025-06-10T00:00:00"  (ISO format)
   StringReplace(dtStr, "T", " ");

   if(StringLen(dtStr) < 10) return 0;

   string datePart = StringSubstr(dtStr, 0, 10);  // "2025-06-10"
   string timePart = "00:00:00";
   if(StringLen(dtStr) >= 19)
      timePart = StringSubstr(dtStr, 11, 8);       // "00:00:00"

   // Parse date
   string parts[];
   int n = StringSplit(datePart, '-', parts);
   if(n < 3) return 0;

   int year  = (int)StringToInteger(parts[0]);
   int month = (int)StringToInteger(parts[1]);
   int day   = (int)StringToInteger(parts[2]);

   // Parse time
   string tparts[];
   int tn = StringSplit(timePart, ':', tparts);
   int hour = 0, minute = 0, second = 0;
   if(tn >= 1) hour   = (int)StringToInteger(tparts[0]);
   if(tn >= 2) minute = (int)StringToInteger(tparts[1]);
   if(tn >= 3) second = (int)StringToInteger(tparts[2]);

   MqlDateTime mdt;
   mdt.year  = year;
   mdt.mon   = month;
   mdt.day   = day;
   mdt.hour  = hour;
   mdt.min   = minute;
   mdt.sec   = second;
   mdt.day_of_week = 0;
   mdt.day_of_year = 0;

   return StructToTime(mdt);
}
//+------------------------------------------------------------------+
