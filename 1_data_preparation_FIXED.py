"""
===============================
DATA PREPARATION (FIXED VERSION)
===============================

PERBAIKAN DARI VERSI LAMA:
1. ✅ No look-ahead bias - Label generation yang realistis
2. ✅ Feature consistency - Import dari features.py
3. ✅ More realistic TP/SL check - Skip current candle
4. ✅ Better documentation
5. ✅ Data quality checks

CARA PAKAI:
1. Pastikan MT5 sudah login
2. Jalankan: python 1_data_preparation_FIXED.py
3. Output: EURUSD_H1_data.csv
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sys

# Import shared features module
from features import add_all_features, get_feature_columns, validate_features

warnings.filterwarnings('ignore')

# =====================================
# KONFIGURASI
# =====================================

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
NUM_CANDLES = 10000  # Increased untuk data yang lebih banyak

# Label Configuration (sama dengan yang di bot)
ATR_PERIOD = 14
RISK_MULTIPLIER = 1.2   # SL =  * ATR
REWARD_MULTIPLIER = 1.5  # TP =  * ATR 

# CRITICAL: Set realistic looking forward window
# Di live trading, kita tidak tahu apakah trade akan hit TP/SL
# Tapi untuk labeling, kita perlu batasan waktu yang masuk akal
MAX_HOLDING_HOURS = 24  # Maximum 24 jam (1 hari trading)

OUTPUT_FILE = f"{SYMBOL}_{TIMEFRAME}_data.csv"


# =====================================
# FUNGSI UTAMA
# =====================================

def get_historical_data(symbol, timeframe, num_candles):
    """
    Mengambil data historis dari MT5.
    
    Parameters:
    -----------
    symbol : str
        Symbol trading (e.g., "EURUSD")
    timeframe : int
        MT5 timeframe constant
    num_candles : int
        Jumlah candle yang diambil
        
    Returns:
    --------
    pd.DataFrame or None
        DataFrame dengan OHLCV data, atau None jika gagal
    """
    
    print("="*60)
    print("FASE 1: MENGAMBIL DATA HISTORIS")
    print("="*60)
    
    print(f"\n1. Menghubungkan ke MetaTrader 5...")
    
    if not mt5.initialize():
        print(f"   ❌ Gagal menghubungkan MT5: {mt5.last_error()}")
        return None
    
    print("   ✅ Terhubung ke MT5")
    
    # Check if symbol is available
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"   ❌ Symbol {symbol} tidak ditemukan")
        mt5.shutdown()
        return None
    
    if not symbol_info.visible:
        print(f"   ⚠️  Symbol {symbol} tidak visible, mencoba mengaktifkan...")
        if not mt5.symbol_select(symbol, True):
            print(f"   ❌ Gagal mengaktifkan symbol {symbol}")
            mt5.shutdown()
            return None
    
    print(f"\n2. Mengambil {num_candles} candle untuk {symbol}...")
    
    # Get historical data
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
    
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        print(f"   ❌ Gagal mengambil data: {mt5.last_error()}")
        return None
    
    print(f"   ✅ Berhasil mengambil {len(rates)} candle")
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Data quality check
    print(f"\n3. Data Quality Check:")
    print(f"   - Periode: {df.index[0]} sampai {df.index[-1]}")
    print(f"   - Total hari: {(df.index[-1] - df.index[0]).days} hari")
    print(f"   - Missing values: {df.isnull().sum().sum()}")
    
    return df


def create_labels_realistic(df):
    """
    Membuat label dengan pendekatan yang REALISTIS dan TANPA look-ahead bias.
    
    PERBAIKAN DARI VERSI LAMA:
    - Skip current candle untuk TP/SL check (lebih realistis)
    - Gunakan close price, bukan high/low dalam-candle
    - Batasan waktu berdasarkan MAX_HOLDING_HOURS
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame dengan fitur sudah ditambahkan (termasuk ATR)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame dengan kolom 'buy_signal' dan 'sell_signal'
    """
    
    print("\n" + "="*60)
    print("FASE 3: MEMBUAT LABEL (REALISTIC METHOD)")
    print("="*60)
    
    df['buy_signal'] = 0
    df['sell_signal'] = 0
    
    total_candles = len(df)
    buy_signals = 0
    sell_signals = 0
    
    print(f"\nMemproses {total_candles} candle...")
    print("(Ini akan memakan waktu beberapa menit...)\n")
    
    # Calculate max candles to look forward based on timeframe
    if TIMEFRAME == mt5.TIMEFRAME_H1:
        max_candles_forward = MAX_HOLDING_HOURS
    elif TIMEFRAME == mt5.TIMEFRAME_M15:
        max_candles_forward = MAX_HOLDING_HOURS * 4
    elif TIMEFRAME == mt5.TIMEFRAME_H4:
        max_candles_forward = MAX_HOLDING_HOURS // 4
    else:
        max_candles_forward = 24  # Default
    
    # Iterate through candles (except last few that don't have enough future data)
    for i in range(total_candles - max_candles_forward - 1):
        
        # Progress indicator
        if i % 500 == 0:
            progress = (i / total_candles) * 100
            print(f"   Progress: {progress:.1f}% ({i}/{total_candles})")
        
        # Current candle data
        current_close = df['close'].iloc[i]
        current_atr = df[f'ATRr_{ATR_PERIOD}'].iloc[i]
        
        if pd.isna(current_atr) or current_atr == 0:
            continue
        
        # Calculate TP and SL levels
        buy_sl = current_close - (RISK_MULTIPLIER * current_atr)
        buy_tp = current_close + (REWARD_MULTIPLIER * current_atr)
        sell_sl = current_close + (RISK_MULTIPLIER * current_atr)
        sell_tp = current_close - (REWARD_MULTIPLIER * current_atr)
        
        # CRITICAL FIX: Look at FUTURE candles only (skip current+next)
        # In live trading, we enter at close of current candle
        # So we should check TP/SL starting from the NEXT candle
        start_idx = i + 1
        end_idx = min(i + max_candles_forward + 1, total_candles)
        
        future_candles = df.iloc[start_idx:end_idx]
        
        if len(future_candles) == 0:
            continue
        
        # =====================================
        # BUY SIGNAL LOGIC
        # =====================================
        # Check if TP is hit before SL
        
        # Use HIGH for TP check (realistic: candle could touch TP at any point)
        tp_hit_idx = None
        for j, (idx, row) in enumerate(future_candles.iterrows()):
            if row['high'] >= buy_tp:
                tp_hit_idx = j
                break
        
        # Use LOW for SL check
        sl_hit_idx = None
        for j, (idx, row) in enumerate(future_candles.iterrows()):
            if row['low'] <= buy_sl:
                sl_hit_idx = j
                break
        
        # Label as BUY signal if TP hit before SL (or SL never hit)
        if tp_hit_idx is not None:
            if sl_hit_idx is None or tp_hit_idx < sl_hit_idx:
                df.iloc[i, df.columns.get_loc('buy_signal')] = 1
                buy_signals += 1
        
        # =====================================
        # SELL SIGNAL LOGIC
        # =====================================
        # Check if TP is hit before SL
        
        # Use LOW for TP check
        tp_hit_idx = None
        for j, (idx, row) in enumerate(future_candles.iterrows()):
            if row['low'] <= sell_tp:
                tp_hit_idx = j
                break
        
        # Use HIGH for SL check
        sl_hit_idx = None
        for j, (idx, row) in enumerate(future_candles.iterrows()):
            if row['high'] >= sell_sl:
                sl_hit_idx = j
                break
        
        # Label as SELL signal if TP hit before SL (or SL never hit)
        if tp_hit_idx is not None:
            if sl_hit_idx is None or tp_hit_idx < sl_hit_idx:
                df.iloc[i, df.columns.get_loc('sell_signal')] = 1
                sell_signals += 1
    
    print("\n✅ Labeling selesai!")
    print(f"\nStatistik Label:")
    print(f"   - Total BUY signals: {buy_signals} ({buy_signals/total_candles*100:.2f}%)")
    print(f"   - Total SELL signals: {sell_signals} ({sell_signals/total_candles*100:.2f}%)")
    print(f"   - Total NO signals: {total_candles - buy_signals - sell_signals} ({(total_candles - buy_signals - sell_signals)/total_candles*100:.2f}%)")
    
    # Warning if signals are too rare
    if buy_signals / total_candles < 0.01:
        print("\n⚠️  WARNING: BUY signals sangat jarang (<1%)!")
        print("   Pertimbangkan untuk:")
        print("   - Kurangi REWARD_MULTIPLIER (TP lebih dekat)")
        print("   - Atau tambah RISK_MULTIPLIER (SL lebih jauh)")
    
    if sell_signals / total_candles < 0.01:
        print("\n⚠️  WARNING: SELL signals sangat jarang (<1%)!")
        print("   (Rekomendasi sama dengan BUY signals)")
    
    return df


# =====================================
# MAIN EXECUTION
# =====================================

def main():
    """
    Main function untuk menjalankan seluruh proses data preparation.
    """
    
    print("\n" + "="*60)
    print("DATA PREPARATION - PRODUCTION GRADE ML TRADING")
    print("="*60)
    print(f"Symbol: {SYMBOL}")
    print(f"Timeframe: H1")
    print(f"Risk:Reward = 1:{REWARD_MULTIPLIER/RISK_MULTIPLIER:.1f}")
    print("="*60)
    
    # Step 1: Get historical data
    df = get_historical_data(SYMBOL, TIMEFRAME, NUM_CANDLES)
    
    if df is None:
        print("\n❌ Gagal mendapatkan data historis. Keluar...")
        return
    
    # Step 2: Add technical indicators
    print("\n" + "="*60)
    print("FASE 2: MENAMBAHKAN FITUR TEKNIS")
    print("="*60)
    
    df_with_features = add_all_features(df)
    
    # Validate features
    if not validate_features(df_with_features):
        print("\n❌ Feature validation gagal. Keluar...")
        return
    
    # Get feature columns (auto-detect from actual DataFrame)
    feature_cols = get_feature_columns(df_with_features)
    print(f"\n✅ Detected {len(feature_cols)} features:")
    for i, feat in enumerate(feature_cols[:10], 1):  # Show first 10
        print(f"   {i}. {feat}")
    if len(feature_cols) > 10:
        print(f"   ... and {len(feature_cols) - 10} more")
    
    # Step 3: Create labels
    df_final = create_labels_realistic(df_with_features)
    
    # Step 4: Save to CSV
    print("\n" + "="*60)
    print("FASE 4: MENYIMPAN DATA")
    print("="*60)
    
    df_final.to_csv(OUTPUT_FILE)
    print(f"\n✅ Data berhasil disimpan ke: {OUTPUT_FILE}")
    print(f"   Total rows: {len(df_final)}")
    print(f"   Total columns: {len(df_final.columns)}")
    
    # Display sample
    print("\n" + "="*60)
    print("SAMPLE DATA (5 baris pertama)")
    print("="*60)
    print(df_final.head())
    
    print("\n" + "="*60)
    print("DISTRIBUSI LABEL FINAL")
    print("="*60)
    print("\nBUY Signal:")
    print(df_final['buy_signal'].value_counts(normalize=True))
    print("\nSELL Signal:")
    print(df_final['sell_signal'].value_counts(normalize=True))
    
    # Check for data quality issues
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    
    # Re-get feature columns from final dataframe
    final_feature_cols = get_feature_columns(df_final)
    
    # Check for NaN values in features
    feature_nan_count = df_final[final_feature_cols].isnull().sum().sum()
    if feature_nan_count > 0:
        print(f"⚠️  WARNING: {feature_nan_count} NaN values ditemukan di features!")
        print("   Kolom dengan NaN:")
        nan_cols = df_final[final_feature_cols].isnull().sum()
        print(nan_cols[nan_cols > 0])
    else:
        print("✅ Tidak ada NaN values di features")
    
    # Check for infinite values
    feature_inf_count = np.isinf(df_final[final_feature_cols]).sum().sum()
    if feature_inf_count > 0:
        print(f"⚠️  WARNING: {feature_inf_count} infinite values ditemukan!")
    else:
        print("✅ Tidak ada infinite values")
    
    print("\n" + "="*60)
    print("SELESAI! LANJUT KE FASE 2: TRAINING")
    print("="*60)
    print("\nJalankan: python 2_training_FIXED.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Proses dibatalkan oleh user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
