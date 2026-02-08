import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# 1. Download data
df = yf.download('AAPL', start='2023-01-01', end='2024-01-01')

# --- PERBAIKAN: Flatten MultiIndex kolom jika ada ---
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
# ----------------------------------------------------

# 2. Cari local highs/lows (Pivot Points)
n = 5 
# Perbaikan indexing untuk argrelextrema
min_idx = argrelextrema(df['Low'].values, np.less, order=n)[0]
max_idx = argrelextrema(df['High'].values, np.greater, order=n)[0]

df['Min'] = np.nan
df['Max'] = np.nan
df.iloc[min_idx, df.columns.get_loc('Min')] = df['Low'].iloc[min_idx]
df.iloc[max_idx, df.columns.get_loc('Max')] = df['High'].iloc[max_idx]

def get_regression_line(df_subset, col_name):
    # Membersihkan baris yang NaN untuk kolom target
    df_clean = df_subset.dropna(subset=[col_name]).copy()
    
    if len(df_clean) < 2:
        return [], []
        
    # Menggunakan index numerik (0, 1, 2...) untuk sumbu X regresi
    X = np.arange(len(df_clean)).reshape(-1, 1)
    y = df_clean[col_name].values
    
    model = LinearRegression().fit(X, y)
    predictions = model.predict(X)
    
    return df_clean.index, predictions

# 3. Hitung Garis Regresi
idx_res, pred_res = get_regression_line(df, 'Max')
idx_sup, pred_sup = get_regression_line(df, 'Min')

# 4. Plot
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label='Close Price', alpha=0.3, color='gray')

if len(idx_res) > 0:
    plt.scatter(idx_res, df.loc[idx_res, 'Max'], color='red', s=20, label='Resistance Points')
    plt.plot(idx_res, pred_res, color='red', linewidth=2, label='Resistance Line')

if len(idx_sup) > 0:
    plt.scatter(idx_sup, df.loc[idx_sup, 'Min'], color='green', s=20, label='Support Points')
    plt.plot(idx_sup, pred_sup, color='green', linewidth=2, label='Support Line')

plt.title('AAPL Support & Resistance Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
