import pandas_ta as ta
import pandas as pd
import MetaTrader5 as mt5
import mplfinance as mpf

# BOT CONFIGURATION

SYMBOL = "AUDCAD"
TIMEFRAME = mt5.TIMEFRAME_M15
TOTAL_CANDLES = 2000

SMA_SHORT = 20
SMA_LONG = 50
EMA = 200

ATR_PERIODE = 14

RISK_MULTI = 1.5
REWARD_MULTI = 3


def get_data_history():
    # inisialisasi ke MetaTrader 5
    if not mt5.initialize():
        print(f"Connection failed : {mt5.last_error()}")
        quit()

    # Pengambilan data dan pembersihan kolom yang tidak diperlukan 
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, TOTAL_CANDLES)
    mt5.shutdown()
    df = pd.DataFrame(rates)
    df.drop(columns=['spread', 'real_volume'], inplace=True) # <<<<<< Kolom yang tidak diperlukan
    df['time'] = pd.to_datetime(df['time'], unit='s') # merubah waktu delam bentuk standart agar bisa digunakan

    df.rename(columns={'time': 'Time','open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True) # Stabdarisasi nama kolom
    return df


def add_feature(df):

    """
    indikator yang ingin ditambahkan sebagai parameter

    """
    # =====================================
    # TREND INDICATORS
    # =====================================

    #   SMA SHORT
    df.ta.sma(length=SMA_SHORT, append=True)

    #  SMA LONG
    df.ta.sma(length=SMA_LONG, append=True)

    #  EMA 
    df.ta.ema(length=EMA, append=True)

    # =====================================
    # VOLATILITY INDICATORS
    # =====================================

    # ATR ( Average True Range)
    df.ta.atr(length=ATR_PERIODE, append=True) # Untuk risk management 

    # =====================================
    # CROSSOVER SMA
    # =====================================

    cross_up = ta.cross(df[f'SMA_{SMA_SHORT}'], df[f'SMA_{SMA_LONG}'], above=True)
    cross_down = ta.cross(df[f'SMA_{SMA_SHORT}'], df[f'SMA_{SMA_LONG}'], above=False)
    df['Cross_up'] = cross_up
    df['Cross_down'] = cross_down



    """
    Untuk backup data yang telah diberi fitur
    """
    df.to_csv('belajar\\data_feature.csv', index=False)

    return df


def trade_algoritma(df):
    pass


def risk_management(df):
    atr = df[f'ATRr_{ATR_PERIODE}']
    """
    Perhitungan RR berdasarkan nilai ATR dan RR yang diinginkan

    Perhitungan :

    Stop lost : candle penutup - ( risk ratio * atr)
    Take Profit : candle penutup + ( reward ratio * atr)

    """
    sl_buy = df['Close'] - (atr * RISK_MULTI )
    tp_buy = df['Close'] + (atr * REWARD_MULTI)

    sl_sell = df['Close'] + (atr * RISK_MULTI)
    tp_sell = df['Close'] - (atr * REWARD_MULTI)


def ploting(df):
    df['Time'] = pd.to_datetime(df['Time'], unit='s') # merubah waktu delam bentuk standart agar bisa digunakan
    df.set_index('Time', inplace=True)
    
    sma_short = df[f'SMA_{SMA_SHORT}']
    sma_long = df[f'SMA_{SMA_LONG}']
    cross_up = df.apply(lambda x: x[f'SMA_{SMA_LONG}'] if x['Cross_up'] == 1 else None, axis=1)
    cross_down = df.apply(lambda x: x[f'SMA_{SMA_LONG}'] if x['Cross_down'] == 1 else None, axis=1)
    """
    Ploting data untuk melihat gambaran sinyal yang kita inginkan
    """
    #  Ploting data
    ap =[mpf.make_addplot(sma_short),
      mpf.make_addplot(sma_long),
      mpf.make_addplot(cross_up, type='scatter', markersize=100, marker="^" ,color='blue'),
      mpf.make_addplot(cross_down, type='scatter', markersize=100,marker='v', color='red')
    ] 
    mpf.plot(df, type='candle',style='charles', addplot=ap, title=f'{SYMBOL}')


data = get_data_history()
data_csv = pd.read_csv('belajar\\data_feature.csv')

ploting(data_csv)
# print(data.head(5))
# add_feature(data)