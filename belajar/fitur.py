import pandas_ta as ta
import pandas as pd
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import mplfinance as mpf

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
TOTAL_CANDLES = 500


if not mt5.initialize():
    print("gagal Terhubung")
    quit()

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, TOTAL_CANDLES)
mt5.shutdown()

df = pd.DataFrame(rates)
df.drop(columns=['spread', 'real_volume',], inplace=True)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index(df['time'], inplace=True)

df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)

mpf.plot(df, type='candle', style='charles', volume=True, mav=(3,6,9), hlines=[1.1855])