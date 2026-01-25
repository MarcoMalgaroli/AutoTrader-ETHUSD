from models.MT5Services import MT5Services
import json

PRINT_WIDTH = 100
SYMBOL = "ETHUSD"
mt5 = MT5Services(SYMBOL)

account_info = mt5.get_account_info()
terminal_info = mt5.get_terminal_info()
symbol_info = mt5.get_symbol_info()

# print(json.dumps(account_info, indent = 2))
# print(json.dumps(terminal_info, indent = 2))
# print(json.dumps(symbol_info, indent = 2))

active_orders = mt5.get_active_orders_count()
print("\n" + f" ACTIVE Orders: {active_orders} ".center(PRINT_WIDTH, "="))
print(mt5.get_active_orders())


positions = mt5.get_active_positions_count()
print("\n" + f" ACTIVE Positions: {positions} ".center(PRINT_WIDTH, "="))
print(mt5.get_active_positions())


history_orders = mt5.get_history_orders_count()
print("\n" + f" HISTORY Orders: {history_orders} ".center(PRINT_WIDTH, "="))
print(mt5.get_history_orders())


deals = mt5.get_history_deals_count()
print("\n" + f" HISTORY Deals: {deals} ".center(PRINT_WIDTH, "="))
print(mt5.get_history_deals())

mt5.shutdown()