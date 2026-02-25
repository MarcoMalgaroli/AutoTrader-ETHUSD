from datetime import datetime, timedelta, timezone
import pandas as pd
import MetaTrader5 as mt5
import time
from typing import Optional

pd.set_option('display.max_columns', 500) # how many columns to show
pd.set_option('display.width', 1500)      # max. table width to display

class MT5Services:
    def __init__(self, symbol: str = "ETHUSD"):
        try:
            self.connect()
            self.select_symbol(symbol)
        except Exception as e:
            self.shutdown()
            raise e
        
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

    def connect(self) -> bool:
        """
        MetaTrader5 connection
        """
        if not mt5.initialize():
            raise ConnectionError(f"initialize() failed, error code = {mt5.last_error()}")
        print("\x1b[32;1mConnected to MT5 terminal successfully.\x1b[0m")
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

    def select_symbol(self, symbol: str = "ETHUSD") -> bool:
        """
        make sure symbol is present in the Market Watch, or abort the algorithm
        """
        if not mt5.symbol_select(symbol, True):
            raise ValueError(f"Symbol {symbol} not present in Market Watch")
        self.symbol = symbol
        print(f"\x1b[32;1mSelected symbol: {symbol}\x1b[0m")
        return True
    
    def get_selected_symbol(self) -> str:
        """
        Return currently selected symbol
        """
        return self.symbol
    
    def get_symbol_info(self, symbol: Optional[str] = None) -> Optional[dict]:
        """
        Return selected or specified symbol info
        """
        s = symbol or self.symbol
        symbol_info = mt5.symbol_info(s)
        if symbol_info is None:
            raise Exception(f"Symbol \"{s}\" not found")
        return symbol_info._asdict()
    
    def get_last_tick(self, symbol: Optional[str] = None) -> Optional[dict]:
        """
        Returns last tick of the selected or specified symbol
        """
        s = symbol or self.symbol
        last_tick = mt5.symbol_info_tick(s)
        if last_tick is None:
            raise Exception(f"Failed to get last tick for symbol \"{s}\"")
        return last_tick._asdict()

    def __safe_convert_to_datetime(self, value, unit='s') -> Optional[datetime]:
        return pd.to_datetime(value, unit=unit, errors='coerce') if value != 0 else None
    
    def get_current_terminal_time(self) -> Optional[datetime]:
        return self.__safe_convert_to_datetime(self.get_last_tick()['time'])
    
    def get_historical_data_pos(self, symbol: Optional[str] = None, timeframe: str = "M5", pos: int = 0, count: int = 1000) -> Optional[pd.DataFrame]:
        """
        Returns historical data of selected or specified symbol starting from pos, selecting count bars
        """
        symbol = symbol or self.symbol
        if timeframe not in self.timeframes:
            raise ValueError(f"Invalid timeframe ({timeframe})")

        print(f"    ---> Start downloading {symbol} [{timeframe}] - {count} bars starting from {pos}")
        rates = mt5.copy_rates_from_pos(symbol, self.timeframes[timeframe], pos, count)
        if rates is None or len(rates) == 0:
            raise Exception(f"No data found for {symbol}_{timeframe} - pos: {pos} - count: {count}")
        rates_frame = pd.DataFrame(rates)       
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit = 's')
        rates_frame = rates_frame[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread']]
        print(f"\x1b[92m      -> Done: downloaded {len(rates_frame)} bars\x1b[0m")
        return rates_frame

    def get_historical_data_all(self, symbol: Optional[str] = None, timeframe: str = "M5", chunk: int = 10000) -> Optional[pd.DataFrame]:
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
            try:
                rates = self.get_historical_data_pos(symbol, timeframe, pos, chunk)
            except Exception as e:
                break

            if rates is None or len(rates) == 0:
                break

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            dfs.append(df)

            pos += len(df)
            time.sleep(0.1)

        if not dfs:
            print(f"\x1b[91;1m X-> No data found for {symbol}_{timeframe} - full history\x1b[0m")
            raise Exception(f"No data found for {symbol}_{timeframe} - full history")
        
        all_df = pd.concat(dfs, ignore_index=True)
        all_df = all_df.sort_values("time").reset_index(drop=True)
        all_df = all_df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread']]
        print(f"\x1b[92;1m  -> Done: downloaded all ({len(all_df)} bars)\x1b[0m")
        return all_df

    # Active orders
    def get_active_orders_count(self) -> int:
        """
        Get active pending orders count
        """
        return mt5.orders_total()
    
    def get_active_orders(self) -> Optional[pd.DataFrame]:
        """
        Get active pending orders
        """
        orders = mt5.orders_get()
        if orders == None:
            raise Exception(f"Error retrieving active orders: {mt5.last_error()}")
        if len(orders) == 0:
            return None
        df = pd.DataFrame(list(orders), columns = orders[0]._asdict().keys())
        print(f"\x1b[92m  -> Retrieved {len(df)} active orders\x1b[0m")

        df['time_setup'] = df['time_setup'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='s'))
        df['time_setup_msc'] = df['time_setup_msc'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='ms'))
        df['time_done'] = df['time_done'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='s'))
        df['time_done_msc'] = df['time_done_msc'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='ms'))
        df['time_expiration'] = df['time_expiration'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='ms'))
        return df
    
    # open positions
    def get_active_positions_count(self) -> int:
        """
        Get active positions count
        """
        return mt5.positions_total()
    
    def get_active_positions(self) -> Optional[pd.DataFrame]:
        """
        Get active positions
        """
        positions = mt5.positions_get()
        if positions == None:
            raise Exception(f"Error retrieving active positions: {mt5.last_error()}")
        if len(positions) == 0:
            return None

        df = pd.DataFrame(list(positions), columns = positions[0]._asdict().keys())
        print(f"\x1b[92m  -> Retrieved {len(df)} active positions\x1b[0m")
        df['time'] = df['time'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='s'))
        df['time_msc'] = df['time_msc'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='ms'))
        df['time_update'] = df['time_update'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='s'))
        df['time_update_msc'] = df['time_update_msc'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='ms'))
        return df

    # Past orders
    def get_history_orders_count(self, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None) -> int:
        """
        Get past orders count in a date range (default -> 1 day)
        """
        to_date = to_date or self.get_current_terminal_time()
        from_date = from_date or (to_date - timedelta(days = 1))
        return mt5.history_orders_total(from_date, to_date)
    
    def get_history_orders(self, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Get past orders in a date range (default -> 1 day)
        """
        to_date = to_date or self.get_current_terminal_time()
        from_date = from_date or (to_date - timedelta(days = 1))
        orders = mt5.history_orders_get(from_date, to_date)
        if orders == None:
            raise Exception(f"Error retrieving history orders: {mt5.last_error()}")
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
    def get_history_deals_count(self, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None) -> int:
        """
        Get past deals count in a date range (default -> 1 day)
        """
        to_date = to_date or self.get_current_terminal_time()
        from_date = from_date or (to_date - timedelta(days = 1))
        return mt5.history_deals_total(from_date, to_date)
    
    def get_history_deals(self, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Get past deals in a date range (default -> 1 day)
        """
        to_date = to_date or self.get_current_terminal_time()
        from_date = from_date or (to_date - timedelta(days = 1))
        deals = mt5.history_deals_get(from_date, to_date)
        if deals == None:
            raise Exception(f"Error retrieving history deals: {mt5.last_error()}")
        if len(deals) == 0:
            return None
        
        df = pd.DataFrame(list(deals), columns = deals[0]._asdict().keys())
        df['time'] = df['time'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='s'))
        df['time_msc'] = df['time_msc'].apply(lambda x: self.__safe_convert_to_datetime(x, unit='ms'))
        return df
    # Note: When Pending orders are processed, they become Deals. A list of deals make a Position


    def __build_order_request(self, order_type: str, position: Optional[str] = None, symbol: Optional[str] = None, volume: Optional[float] = 0.01, price: Optional[float] = None, sl_mult: Optional[float] = 0.0, tp_mult: Optional[float] = 0.0, deviation: int = 10, comment: str = "python script") -> dict:
        symbol = symbol or self.symbol

        if position is not None:
            pos = mt5.positions_get(ticket=position)
            if pos is None or len(pos) == 0:
                raise Exception(f"Position {position} not found")
            pos = pos[0]
            symbol = pos['symbol'] or symbol

            if pos['type'] == mt5.ORDER_TYPE_BUY:
                order_type = "SELL"
            elif pos['type'] == mt5.ORDER_TYPE_SELL:
                order_type = "BUY"
            else:
                raise ValueError(f"Invalid position type: {pos['type']}")
                
        price_info = self.get_last_tick(symbol)
        symbol_info = self.get_symbol_info(symbol)

        if order_type == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = price or price_info['ask']
            sl = (price_info['bid'] - sl_mult * symbol_info['point']) if sl_mult > 0 else 0.0
            tp = (price_info['bid'] + tp_mult * symbol_info['point']) if tp_mult > 0 else 0.0
        elif order_type == "SELL":
            order_type = mt5.ORDER_TYPE_SELL
            price = price or price_info['bid']
            sl = (price_info['ask'] + sl_mult * symbol_info['point']) if sl_mult > 0 else 0.0
            tp = (price_info['ask'] - tp_mult * symbol_info['point']) if tp_mult > 0 else 0.0
        else:
            raise ValueError(f"Invalid order_type: {order_type}")
        
        match symbol_info['filling_mode']:
            case 1:
                filling_mode = mt5.ORDER_FILLING_FOK
            case 2:
                filling_mode = mt5.ORDER_FILLING_IOC
            case 4:
                filling_mode = mt5.ORDER_FILLING_BOC
            case _:
                filling_mode = mt5.ORDER_FILLING_RETURN

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "magic": 112233,
            "symbol": symbol,
            "volume": volume,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "type": order_type,
            "type_filling": filling_mode,
            "type_time": mt5.ORDER_TIME_GTC,
            "comment": comment
        }
        if position is not None:
            request["position"] = position
            request["volume"] = pos.volume
            del request["sl"]
            del request["tp"]
        return request
    
    def check_order(self, order_type: str, position: Optional[str] = None, symbol: Optional[str] = None, volume: Optional[float] = 0.01, price: Optional[float] = None, sl_mult: Optional[float] = 0.0, tp_mult: Optional[float] = 0.0, deviation: int = 10, comment: str = "python script") -> None:
        """
        Check order request result and raise exception on failure
        """
        
        request = self.__build_order_request(order_type, position, symbol, volume, price, sl_mult, tp_mult, deviation, comment)
        result = mt5.order_check(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE and result.retcode != 0:
            raise Exception(f"Order check failed: {result.comment} (retcode: {result.retcode})")
        return result
    
    def place_order(self, order_type: str, position: Optional[str] = None, symbol: Optional[str] = None, volume: Optional[float] = 0.01, price: Optional[float] = None, sl_mult: Optional[float] = 0.0, tp_mult: Optional[float] = 0.0, deviation: int = 10, comment: str = "python script") -> None:
        """
        Place an order and check the result
        """
        request = self.__build_order_request(order_type, position, symbol, volume, price, sl_mult, tp_mult, deviation, comment)

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise Exception(f"Order placement failed: {result.comment} (retcode: {result.retcode})")
        return result

    def close_position(self, ticket: int, symbol: Optional[str] = None, deviation: int = 10, comment: str = "close by AI") -> None:
        """
        Close an opened position given the ticket.
        """
        request = self.__build_order_request("CLOSE", position=ticket, symbol=symbol, deviation=deviation, comment=comment)

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise Exception(f"Close position failed: {result.comment} (retcode: {result.retcode})")
        return result
    