from datetime import datetime, timedelta, timezone
import pandas as pd
import MetaTrader5 as mt5
import time
from typing import Optional

pd.set_option('display.max_columns', 500) # how many columns to show
pd.set_option('display.width', 1500)      # max. table width to display

class MT5Services:
    def __init__(self, symbol: str = "ETHUSD"):
        if not self.connect_mt5():
            raise ConnectionError("Unable to connect to MT5 terminal")
        self.select_mt5_symbol(symbol)
        self.timeframes = {
            "M1": mt5.TIMEFRAME_M1,
            "M2": mt5.TIMEFRAME_M2,
            "M3": mt5.TIMEFRAME_M3,
            "M4": mt5.TIMEFRAME_M4,
            "M5": mt5.TIMEFRAME_M5,
            "M6": mt5.TIMEFRAME_M6,
            "M10": mt5.TIMEFRAME_M10,
            "M12": mt5.TIMEFRAME_M12,
            "M15": mt5.TIMEFRAME_M15,
            "M20": mt5.TIMEFRAME_M20,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H2": mt5.TIMEFRAME_H2,
            "H3": mt5.TIMEFRAME_H3,
            "H4": mt5.TIMEFRAME_H4,
            "H6": mt5.TIMEFRAME_H6,
            "H8": mt5.TIMEFRAME_H8,
            "H12": mt5.TIMEFRAME_H12,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1
        }

    def connect_mt5(self) -> bool:
        """
        MetaTrader5 connection
        """
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            self.shutdown()
            return False
        return True

    def shutdown(self) -> None:
        mt5.shutdown()

    def get_account_info(self) -> Optional[dict]:
        """
        Account informations
        """
        account_info = mt5.account_info()
        if account_info is not None:
            return account_info._asdict()
    
    def get_terminal_info(self) -> Optional[dict]:
        """
        Terminal informations
        """
        terminal_info = mt5.terminal_info()
        if terminal_info is not None:
            return terminal_info._asdict()

    def select_mt5_symbol(self, symbol: str = "ETHUSD") -> bool:
        """
        make sure symbol is present in the Market Watch, or abort the algorithm
        """
        if not mt5.symbol_select(symbol, True):
            raise ValueError(f"Symbol {symbol} not present in Market Watch")
            return False
        self.symbol = symbol
        return True
    
    def get_selected_symbol(self) -> str:
        """
        Return currently selected symbol
        """
        return self.symbol
    
    def get_symbol_info(self, symbol: str = None) -> Optional[dict]:
        """
        Return selected or specified symbol info
        """
        symbol_info = mt5.symbol_info(symbol or self.symbol)
        if symbol_info is None:
            raise Exception(f"Symbol \"{symbol}\" not found")
        return symbol_info._asdict()
    
    def get_last_tick(self, symbol: str = None):
        """
        Returns last tick of the selected or specified symbol
        """
        last_tick = mt5.symbol_info_tick(symbol or self.symbol)
        if last_tick is not None:
            return last_tick._asdict()

    def __safe_convert_to_datetime(self, value, unit='s'):
        return pd.to_datetime(value, unit=unit, errors='coerce') if value != 0 else 0
    
    def get_current_terminal_time(self):
        return self.__safe_convert_to_datetime(self.get_last_tick()['time'])
    
    def get_historical_data_range(self, symbol: str = None, timeframe: str = "M5", from_date: datetime = None, to_date: datetime = None) -> Optional[pd.DataFrame]:
        """
        Return historical data of selected or specified symbol between dates (or in a day interval from now)
        """
        to_date = to_date or self.get_current_terminal_time()
        from_date = from_date or (to_date - timedelta(days = 1))
        symbol = symbol or self.symbol
        if timeframe not in self.timeframes:
            raise ValueError(f"Invalid timeframe ({timeframe})")

        print(f"\n---> Start downloading {symbol} [{timeframe}] {from_date} - {to_date}")
        rates = mt5.copy_rates_range(symbol, self.timeframes[timeframe], from_date, to_date)
        if rates is None or len(rates) == 0:
            raise Exception(f"No data found for {symbol}_{timeframe}: {from_date} - {to_date}")
        rates_frame = pd.DataFrame(rates)       
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit = 's')
        rates_frame = rates_frame[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread']]
        return rates_frame
    
    def get_historical_data_pos(self, symbol: str = None, timeframe: str = "M5", pos: int = 0, count: int = 1000) -> pd.DataFrame | None:
        """
        Returns historical data of selected or specified symbol starting from pos, selecting count bars
        """
        symbol = symbol or self.symbol
        if timeframe not in self.timeframes:
            raise ValueError(f"Invalid timeframe ({timeframe})")

        print(f"\n---> Start downloading {symbol} [{timeframe}] - {count} bars starting from {pos}")
        rates = mt5.copy_rates_from_pos(symbol, self.timeframes[timeframe], pos, count)
        if rates is None or len(rates) == 0:
            raise Exception(f"No data found for {symbol}_{timeframe} - pos: {pos} - count: {count}")
        rates_frame = pd.DataFrame(rates)       
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit = 's')
        rates_frame = rates_frame[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread']]
        print(f"  -> Done: downloaded {len(rates)} bars")
        return rates_frame

    def get_historical_data_all(self, symbol: str = None, timeframe: str = "M5", chunk: int = 10000):
        """
        Returns all available history for the selected or specified symbol
        """
        symbol = symbol or self.symbol
        if timeframe not in self.timeframes:
            raise ValueError(f"Invalid timeframe ({timeframe})")

        print(f"\n---> Start downloading full history of {symbol} [{timeframe}]")
        pos = 0
        dfs = []
        while True:
            rates = mt5.copy_rates_from_pos(symbol, self.timeframes[timeframe], pos, chunk)
            if rates is None or len(rates) == 0:
                break

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            dfs.append(df)

            pos += len(df)
            time.sleep(0.1)

        all_df = pd.concat(dfs, ignore_index=True)
        all_df = all_df.sort_values("time").reset_index(drop=True)
        all_df = all_df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread']]
        print(f"  -> Done: downloaded {len(all_df)} bars")
        return all_df

    # Active orders
    def get_active_orders_count(self):
        """
        Get active pending orders count
        """
        return mt5.orders_total()
    
    def get_active_orders(self):
        """
        Get active pending orders
        """
        orders = mt5.orders_get()
        if orders == None:
            raise Exception(f"No pensing orders found")
        if len(orders) == 0:
            return None
        df = pd.DataFrame(list(orders), columns = orders[0]._asdict().keys())

        df['time_setup'] = df['time_setup'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='s'))
        df['time_setup_msc'] = df['time_setup_msc'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='ms'))
        df['time_done'] = df['time_done'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='s'))
        df['time_done_msc'] = df['time_done_msc'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='ms'))
        df['time_expiration'] = df['time_expiration'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='ms'))
        return df
    
    # open positions
    def get_active_positions_count(self):
        """
        Get active positions count
        """
        return mt5.positions_total()
    
    def get_active_positions(self):
        """
        Get active positions
        """
        positions = mt5.positions_get()
        if positions == None:
            raise Exception(f"No open positions found")
        if len(positions) == 0:
            return None

        df = pd.DataFrame(list(positions), columns = positions[0]._asdict().keys())
        df['time'] = df['time'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='s'))
        df['time_msc'] = df['time_msc'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='ms'))
        df['time_update'] = df['time_update'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='s'))
        df['time_update_msc'] = df['time_update_msc'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='ms'))
        return df

    # Past orders
    def get_history_orders_count(self, from_date: datetime = None, to_date: datetime = None):
        """
        Get past orders count in a date range (default -> 1 day)
        """
        to_date = to_date or self.get_current_terminal_time()
        from_date = from_date or (to_date - timedelta(days = 1))
        return mt5.history_orders_total(from_date, to_date)
    
    def get_history_orders(self, from_date: datetime = None, to_date: datetime = None):
        """
        Get past orders in a date range (default -> 1 day)
        """
        to_date = to_date or self.get_current_terminal_time()
        from_date = from_date or (to_date - timedelta(days = 1))
        orders = mt5.history_orders_get(from_date, to_date)
        if orders == None:
            raise Exception(f"No pensing orders found")
        if len(orders) == 0:
            return None
        
        df = pd.DataFrame(list(orders), columns = orders[0]._asdict().keys())
        df['time_setup'] = df['time_setup'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='s'))
        df['time_setup_msc'] = df['time_setup_msc'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='ms'))
        df['time_done'] = df['time_done'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='s'))
        df['time_done_msc'] = df['time_done_msc'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='ms'))
        df['time_expiration'] = df['time_expiration'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='ms'))
        return df
    
    # Past deals
    def get_history_deals_count(self, from_date: datetime = None, to_date: datetime = None):
        """
        Get past orders count in a date range (default -> 1 day)
        """
        to_date = to_date or self.get_current_terminal_time()
        from_date = from_date or (to_date - timedelta(days = 1))
        return mt5.history_deals_total(from_date, to_date)
    
    def get_history_deals(self, from_date: datetime = None, to_date: datetime = None):
        """
        Get past orders in a date range (default -> 1 day)
        """
        to_date = to_date or self.get_current_terminal_time()
        from_date = from_date or (to_date - timedelta(days = 1))
        orders = mt5.history_deals_get(from_date, to_date)
        if orders == None:
            raise Exception(f"No pensing orders found")
        if len(orders) == 0:
            return None
        
        df = pd.DataFrame(list(orders), columns = orders[0]._asdict().keys())
        df['time'] = df['time'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='s'))
        df['time_msc'] = df['time_msc'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='ms'))
        return df