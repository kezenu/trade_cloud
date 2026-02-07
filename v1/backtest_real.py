import pandas as pd
import numpy as np
import joblib
from datetime import time, timedelta
import pytz
import warnings
import MetaTrader5 as mt5 # Pastikan MT5 diimpor di awal

warnings.filterwarnings('ignore')

# --- KONFIGURASI (HARUS SAMA DENGAN BOT UTAMA) ---
DATA_FILE = "v1\\EURUSD_15_data.csv"
MODEL_BUY_FILE = "v1\\model_buy.pkl"
MODEL_SELL_FILE = "v1\\model_sell.pkl"
SYMBOL = "EURUSD" # Tambahkan konstanta simbol

# Manajemen Risiko
RISK_PERCENT = 2.0
RR_RATIO = 2.0
ATR_PERIOD = 14
RISK_MULTIPLIER = 1.5
REWARD_MULTIPLIER = 1

# Filter Waktu
TRADING_SESSION_START_TIME = time(13, 0, 0)
TRADING_SESSION_END_TIME = time(22, 0, 0)

# Filter Model
MODEL_CONFIDENCE_THRESHOLD = 0.70

# Batasan Harian
MAX_TRADES_PER_DAY = 3

# --- FUNGSI-FUNGSI PENDUKUNG ---

def load_data_and_models():
    try:
        df = pd.read_csv(DATA_FILE, index_col='time', parse_dates=True)
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Jakarta')
        print(f"Data '{DATA_FILE}' berhasil dimuat. Zona waktu diubah ke WIB.")
        
        buy_model = joblib.load(MODEL_BUY_FILE)
        sell_model = joblib.load(MODEL_SELL_FILE)
        print("Model BUY dan SELL berhasil dimuat.")
        
        return df, buy_model, sell_model
    except FileNotFoundError as e:
        print(f"ERROR: File tidak ditemukan - {e}")
        return None, None, None

def is_trading_session(current_time):
    return TRADING_SESSION_START_TIME <= current_time.time() <= TRADING_SESSION_END_TIME

def predict_signal(models, feature_columns, current_features):
    buy_model, sell_model = models
    features_df = pd.DataFrame([current_features], columns=feature_columns)
    
    buy_proba = buy_model.predict_proba(features_df)[0]
    sell_proba = sell_model.predict_proba(features_df)[0]
    
    buy_confidence = buy_proba[1]
    sell_confidence = sell_proba[1]
    
    if buy_confidence >= MODEL_CONFIDENCE_THRESHOLD:
        return 'BUY', buy_confidence
    if sell_confidence >= MODEL_CONFIDENCE_THRESHOLD:
        return 'SELL', sell_confidence
        
    return None, 0.0

# --- PERBAIKAN: Fungsi P&L yang benar ---
def calculate_pnl(entry_price, exit_price, action, symbol_point):
    """Menghitung P&L dalam poin."""
    if action == 'BUY':
        return (exit_price - entry_price) / symbol_point
    else: # SELL
        return (entry_price - exit_price) / symbol_point

# --- EKSEKUSI BACKTEST REALISTIS ---
def main():
    # Inisialisasi MT5 di awal untuk mendapatkan info simbol
    if not mt5.initialize():
        print("Gagal menghubungkan MT5, tidak bisa mendapatkan info simbol.")
        return
        
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        print(f"Tidak bisa mendapatkan info untuk simbol {SYMBOL}.")
        mt5.shutdown()
        return
    symbol_point = symbol_info.point
    
    df, buy_model, sell_model = load_data_and_models()
    if df is None:
        mt5.shutdown()
        return

    feature_columns = buy_model.feature_names_in_
    
    # State untuk simulasi
    trades = []
    is_position_open = False
    position = {}
    daily_state = {'date': None, 'trades_today': 0}
    
    print("\n--- Memulai Backtest Realistis ---")
    print(f"Periode Data: {df.index.min().date()} hingga {df.index.max().date()}")
    
    try:
        for timestamp, row in df.iterrows():
            
            # --- State Management ---
            if daily_state['date'] != timestamp.date():
                daily_state['date'] = timestamp.date()
                daily_state['trades_today'] = 0

            # --- JIKA ADA POSISI TERBUKA ---
            if is_position_open:
                # --- TAMBAHKAN DEBUG PRINT INI ---
                print(f"[DEBUG] Memonitor posisi terbuka pada {timestamp.strftime('%Y-%m-%d %H:%M')} | Harga: {row['close']:.5f}")
                
                hit_tp = (position['action'] == 'BUY' and row['high'] >= position['tp']) or \
                        (position['action'] == 'SELL' and row['low'] <= position['tp'])
                hit_sl = (position['action'] == 'BUY' and row['low'] <= position['sl']) or \
                        (position['action'] == 'SELL' and row['high'] >= position['sl'])
                
                exit_reason = ""
                exit_price = 0

                if hit_sl:
                    exit_reason = "SL"
                    exit_price = position['sl']
                elif hit_tp:
                    exit_reason = "TP"
                    exit_price = position['tp']
                elif timestamp.time() >= time(2, 0, 0) and timestamp > position['entry_time']: # Tutup pukul 02:00 pagi
                    # --- TAMBAHKAN DEBUG PRINT INI JUGA ---
                    print(f"[DEBUG] WAKTU TUTUP TERPICU! Menutup posisi pada {timestamp.strftime('%Y-%m-%d %H:%M')}")
                    exit_reason = "MIDNIGHT CLOSE"
                    exit_price = row['open']
                
                if exit_reason:
                    # --- PERBAIKAN: Panggil fungsi dengan symbol_point yang sudah ada ---
                    pnl = calculate_pnl(position['entry_price'], exit_price, position['action'], symbol_point)
                    trades.append({
                        'entry_time': position['entry_time'], 'action': position['action'],
                        'entry_price': position['entry_price'], 'lot': position['lot'],
                        'exit_time': timestamp, 'exit_price': exit_price,
                        'exit_reason': exit_reason, 'pnl_points': pnl
                    })
                    print(f"TRADE TUTUP: {position['action']} @ {position['entry_price']:.5f} -> {exit_price:.5f} ({exit_reason}) | PnL: {pnl:.1f} poin")
                    is_position_open = False
                    position = {}
                    continue

            # --- JIKA TIDAK ADA POSISI TERBUKA ---
            else:
                if not is_trading_session(timestamp):
                    continue
                if daily_state['trades_today'] >= MAX_TRADES_PER_DAY:
                    continue

                current_features = row[feature_columns].values
                signal, confidence = predict_signal((buy_model, sell_model), feature_columns, current_features)
                
                if signal:
                    # Simulasi eksekusi
                    entry_price = row['close'] # Eksekusi di harga close candle
                    atr = row[f'ATRr_{ATR_PERIOD}']
                    
                    if signal == 'BUY':
                        sl_price = entry_price - (RISK_MULTIPLIER * atr)
                        tp_price = entry_price + (REWARD_MULTIPLIER * atr)
                    else: # SELL
                        sl_price = entry_price + (RISK_MULTIPLIER * atr)
                        tp_price = entry_price - (REWARD_MULTIPLIER * atr)
                    
                    # Simulasi perhitungan lot (disederhanakan, asumsi balance tetap)
                    lot_size = 0.01 # Placeholder
                    
                    position = {
                        'entry_time': timestamp, 'action': signal, 'entry_price': entry_price,
                        'sl': sl_price, 'tp': tp_price, 'lot': lot_size
                    }
                    is_position_open = True
                    daily_state['trades_today'] += 1
                    print(f"TRADE BUKA: {signal} @ {entry_price:.5f} | SL: {sl_price:.5f}, TP: {tp_price:.5f}")
        if is_position_open:
            print(f"Trade terakhir tidak ditutup. Menutup secara paksa di akhir data.")
            last_row = df.iloc[-1]
            exit_price = last_row['close']
            pnl = calculate_pnl(position['entry_price'], exit_price, position['action'], symbol_point)
            trades.append({
                'entry_time': position['entry_time'], 'action': position['action'],
                'entry_price': position['entry_price'], 'lot': position['lot'],
                'exit_time': df.index[-1], 'exit_price': exit_price,
                'exit_reason': 'END_OF_DATA', 'pnl_points': pnl
            })

    finally:
        # Pastikan koneksi MT5 ditutup
        mt5.shutdown()

    # --- Tampilkan Hasil Akhir ---
    if not trades:
        print("\nTidak ada trade yang dieksekusi selama backtest.")
        return

    trades_df = pd.DataFrame(trades)
    print("\n--- Hasil Backtest Realistis ---")
    print(f"Total Trade Dieksekusi: {len(trades_df)}")

    # --- TAMBAHAN: Analisis Alasan Keluar ---
    print("\n--- Analisis Alasan Keluar (Exit Reason) ---")
    exit_reason_counts = trades_df['exit_reason'].value_counts()
    print(exit_reason_counts)
    print(f"\nPersentase Midnight Close: {exit_reason_counts.get('MIDNIGHT CLOSE', 0) / len(trades_df) * 100:.2f}%")

    # ... (sisa kode untuk win rate, PnL, dll)
    wins = trades_df[trades_df['pnl_points'] > 0]
    losses = trades_df[trades_df['pnl_points'] < 0]

    win_rate = len(wins) / len(trades_df) * 100
    total_pnl = trades_df['pnl_points'].sum()
    avg_win = wins['pnl_points'].mean() if not wins.empty else 0
    avg_loss = losses['pnl_points'].mean() if not losses.empty else 0
    profit_factor = (wins['pnl_points'].sum() / abs(losses['pnl_points'].sum())) if not losses.empty else np.inf

    print(f"\nWin Rate: {win_rate:.2f}%")
    print(f"Total PnL (Poin): {total_pnl:.2f}")
    print(f"Rata-rata Kemenangan: {avg_win:.2f} poin")
    print(f"Rata-rata Kerugian: {avg_loss:.2f} poin")
    print(f"Profit Factor: {profit_factor:.2f}")

    print("\nDetail 10 Trade Terakhir:")
    print(trades_df.tail(10).to_string())

if __name__ == "__main__":
    main()