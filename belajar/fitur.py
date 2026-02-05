import pandas_ta as ta
import pandas as pd
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import mplfinance as mpf

#BOT KONFIGURASI

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
TOTAL_CANDLES = 2000

def get_fitur():
    if not mt5.initialize():
        print(f"gagal Terhubung : {mt5.last_error()}")
        quit()

    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, TOTAL_CANDLES)
    mt5.shutdown()

    df = pd.DataFrame(rates)
    df.drop(columns=['spread', 'real_volume',], inplace=True)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index(df['time'], inplace=True)

    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    ap =[mpf.make_addplot(df.ta.sma(length=20), color='blue'),
      mpf.make_addplot(df.ta.sma(length=50), color='red'),
      mpf.make_addplot(df.ta.pivot())
    ] 

    
    mpf.plot(df, type='candle',style='charles', addplot=ap)

get_fitur()