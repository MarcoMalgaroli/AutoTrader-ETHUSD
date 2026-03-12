from models.MT5Services import MT5Services

mt5 = MT5Services()

print("Testing MT5 connection...")
print(mt5.get_symbol_info("ETHUSD"))