"""
===============================
SHARED FEATURES MODULE
===============================
File ini WAJIB digunakan oleh:
1. Data preparation script
2. Training script
3. Live trading bot

Tujuan: Memastikan fitur yang digunakan untuk training
        SAMA PERSIS dengan fitur yang digunakan saat live trading
"""

import pandas as pd
import pandas_ta as ta
import numpy as np

# --- KONFIGURASI FITUR ---
ATR_PERIOD = 14
RSI_PERIOD = 14
STOCH_K = 14
STOCH_D = 3
STOCH_SMOOTH = 3
SMA_SHORT = 20
SMA_LONG = 50
EMA_PERIOD = 100
BB_PERIOD = 20
BB_STD = 2.0
ADX_PERIOD = 14


def add_all_features(df):
    """
    Menambahkan SEMUA indikator teknis sebagai fitur.
    
    CRITICAL: Function ini harus digunakan di:
    - Data preparation (Fase 1)
    - Training (Fase 2)
    - Live bot (Fase 3)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame dengan kolom OHLCV (open, high, low, close, volume)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame dengan fitur tambahan, baris NaN sudah dihapus
    """
    
    # Buat copy untuk menghindari SettingWithCopyWarning
    df = df.copy()
    
    print("Menambahkan fitur teknis...")
    
    # =====================================
    # 1. MOMENTUM INDICATORS
    # =====================================
    
    # RSI (Relative Strength Index)
    df.ta.rsi(length=RSI_PERIOD, append=True)
    
    # Stochastic Oscillator
    df.ta.stoch(k=STOCH_K, d=STOCH_D, smooth_k=STOCH_SMOOTH, append=True)
    
    # =====================================
    # 2. TREND INDICATORS
    # =====================================
    
    # Simple Moving Averages
    df.ta.sma(length=SMA_SHORT, append=True)
    df.ta.sma(length=SMA_LONG, append=True)
    
    # Exponential Moving Average
    df.ta.ema(length=EMA_PERIOD, append=True)
    
    # =====================================
    # 3. VOLATILITY INDICATORS
    # =====================================
    
    # Average True Range (untuk risk management)
    df.ta.atr(length=ATR_PERIOD, append=True)
    
    # Bollinger Bands
    df.ta.bbands(length=BB_PERIOD, std=BB_STD, append=True)
    
    # =====================================
    # 4. VOLUME/STRENGTH INDICATORS
    # =====================================
    
    # Average Directional Index
    df.ta.adx(length=ADX_PERIOD, append=True)
    
    # =====================================
    # 5. CUSTOM FEATURES (opsional tapi recommended)
    # =====================================
    
    # AUTO-DETECT Bollinger Bands column names (different pandas_ta versions use different formats)
    bb_cols = [col for col in df.columns if col.startswith('BBL_') or col.startswith('BBU_')]
    if len(bb_cols) >= 2:
        # Find lower and upper band columns
        bb_lower_col = [col for col in df.columns if col.startswith('BBL_')][0]
        bb_upper_col = [col for col in df.columns if col.startswith('BBU_')][0]
        
        bb_lower = df[bb_lower_col]
        bb_upper = df[bb_upper_col]
        bb_range = bb_upper - bb_lower
        
        # Normalize: 0 = di lower band, 1 = di upper band
        df['bb_position'] = (df['close'] - bb_lower) / (bb_range + 1e-10)  # Add small epsilon to avoid division by zero
        df['bb_position'] = df['bb_position'].clip(0, 1)  # Bound between 0-1
    
    # SMA crossover signal
    if f'SMA_{SMA_SHORT}' in df.columns and f'SMA_{SMA_LONG}' in df.columns:
        df['sma_diff'] = df[f'SMA_{SMA_SHORT}'] - df[f'SMA_{SMA_LONG}']
        df['sma_diff_pct'] = (df['sma_diff'] / (df['close'] + 1e-10)) * 100
    
    # Price distance from EMA
    if f'EMA_{EMA_PERIOD}' in df.columns:
        df['price_to_ema'] = ((df['close'] - df[f'EMA_{EMA_PERIOD}']) / (df['close'] + 1e-10)) * 100
    
    # =====================================
    # 6. CLEANUP
    # =====================================
    
    # Hapus baris dengan NaN (biasanya di awal karena perhitungan MA)
    initial_rows = len(df)
    df.dropna(inplace=True)
    rows_dropped = initial_rows - len(df)
    
    if rows_dropped > 0:
        print(f"   → {rows_dropped} baris dengan NaN dihapus")
    
    print(f"   → Total {len(df.columns)} kolom (termasuk OHLCV)")
    
    return df


def get_feature_columns(df=None):
    """
    Mengembalikan list nama kolom fitur yang akan digunakan untuk ML.
    
    CRITICAL: Jika df provided, akan auto-detect column names.
    Ini penting karena pandas_ta naming bisa berbeda antar versi.
    
    Parameters:
    -----------
    df : pd.DataFrame, optional
        DataFrame yang sudah memiliki features. Jika provided, akan extract
        feature names dari DataFrame ini.
    
    Returns:
    --------
    list
        List nama kolom fitur
    """
    
    if df is not None:
        # AUTO-DETECT dari DataFrame yang actual
        feature_cols = []
        
        # Exclude OHLCV columns dan target columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'tick_volume', 
                       'spread', 'real_volume', 'time', 'buy_signal', 'sell_signal']
        
        for col in df.columns:
            if col not in exclude_cols:
                feature_cols.append(col)
        
        return feature_cols
    
    else:
        # FALLBACK: Expected column names (might not match actual!)
        # This is a GUESS - better to provide df parameter!
        feature_cols = [
            # Momentum
            f'RSI_{RSI_PERIOD}',
            f'STOCHk_{STOCH_K}_{STOCH_D}_{STOCH_SMOOTH}',
            f'STOCHd_{STOCH_K}_{STOCH_D}_{STOCH_SMOOTH}',
            
            # Trend
            f'SMA_{SMA_SHORT}',
            f'SMA_{SMA_LONG}',
            f'EMA_{EMA_PERIOD}',
            
            # Volatility  
            f'ATRr_{ATR_PERIOD}',  # Note: pandas_ta uses 'ATRr' not 'ATR'
            
            # Bollinger Bands - WILL AUTO-DETECT when df provided
            # Placeholder patterns (actual names vary by version)
            
            # Strength
            f'ADX_{ADX_PERIOD}',
            
            # Custom features
            'bb_position',
            'sma_diff',
            'sma_diff_pct',
            'price_to_ema',
        ]
        
        print("⚠️  WARNING: Using fallback feature names. Better to provide df parameter!")
        
        return feature_cols


def validate_features(df):
    """
    Validasi bahwa DataFrame memiliki fitur yang cukup.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame yang akan divalidasi
        
    Returns:
    --------
    bool
        True jika valid
    """
    
    # Check for required base columns
    required_cols = ['close']
    missing_base = [col for col in required_cols if col not in df.columns]
    
    if missing_base:
        print(f"\n❌ ERROR: Missing base columns: {missing_base}")
        return False
    
    # Check that we have some features (excluding OHLCV)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'tick_volume', 
                   'spread', 'real_volume', 'time', 'buy_signal', 'sell_signal']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if len(feature_cols) < 5:
        print(f"\n❌ ERROR: Too few features detected ({len(feature_cols)})")
        print(f"   Available features: {feature_cols}")
        return False
    
    print(f"✅ Valid DataFrame with {len(feature_cols)} features")
    
    return True


# =====================================
# USAGE EXAMPLE
# =====================================

if __name__ == "__main__":
    """
    Contoh penggunaan untuk testing
    """
    import MetaTrader5 as mt5
    
    # Initialize MT5
    if not mt5.initialize():
        print("MT5 initialization failed")
        exit()
    
    # Get sample data
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 500)
    mt5.shutdown()
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    print("Original DataFrame:")
    print(df.head())
    print(f"Shape: {df.shape}")
    
    # Add features
    df_with_features = add_all_features(df)
    
    print("\nDataFrame dengan fitur:")
    print(df_with_features.head())
    print(f"Shape: {df_with_features.shape}")
    
    # Get feature column names
    features = get_feature_columns()
    print(f"\nTotal fitur untuk ML: {len(features)}")
    print("Nama fitur:")
    for i, feat in enumerate(features, 1):
        print(f"{i:2d}. {feat}")
    
    # Validate
    validate_features(df_with_features)
