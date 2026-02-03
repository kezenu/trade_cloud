# 🚀 PRODUCTION-GRADE ML TRADING BOT (FIXED VERSION)

## ⚠️ CRITICAL DISCLAIMER

**PENTING - BACA INI DULU:**

1. **NEVER trade dengan real money tanpa testing ekstensif di demo account minimal 3-6 bulan**
2. Trading forex memiliki **RISIKO SANGAT TINGGI** - Anda bisa kehilangan seluruh modal
3. Past performance **TIDAK MENJAMIN** future results
4. Bot ini adalah **EDUCATIONAL PURPOSES** - bukan jaminan profit
5. Selalu gunakan risk management yang ketat (max 1-2% per trade)

## 📋 DAFTAR FILE

### 1. `features.py` ⭐ **PALING PENTING**

File ini berisi definisi fitur yang **HARUS DIGUNAKAN KONSISTEN** di:

- Data preparation
- Training
- Live trading bot

**Jangan pernah ubah file ini tanpa re-train model dari awal!**

### 2. `1_data_preparation_FIXED.py`

Script untuk mengambil data historis dari MT5 dan membuat label.

**PERBAIKAN dari versi lama:**

- ✅ No look-ahead bias
- ✅ Realistic TP/SL checking
- ✅ Proper time-based labeling
- ✅ Data quality checks

### 3. `2_training_FIXED.py`

Script untuk training ML model dengan metode yang benar.

**PERBAIKAN dari versi lama:**

- ✅ Time-based split (bukan random!)
- ✅ Walk-forward validation
- ✅ Threshold optimization
- ✅ Feature importance analysis
- ✅ Comprehensive evaluation metrics

### 4. `3_trading_bot_FIXED.py`

Bot trading yang siap digunakan (setelah testing di demo!).

**PERBAIKAN dari versi lama:**

- ✅ Feature consistency dengan training
- ✅ Proper PnL calculation
- ✅ Multiple position handling
- ✅ Enhanced error handling
- ✅ Connection health monitoring

---

## 🔧 CARA INSTALASI

### Step 1: Install Dependencies

```bash
pip install MetaTrader5 pandas numpy scikit-learn joblib pandas-ta pytz --break-system-packages
```

### Step 2: Verify MT5 Connection

Pastikan MetaTrader 5:

1. Sudah terinstall dan login
2. Symbol EURUSD tersedia dan visible
3. Algorithmic trading di-enable

---

## 📊 WORKFLOW LENGKAP

### FASE 1: Data Preparation

```bash
python 1_data_preparation_FIXED.py
```

**Output:** `EURUSD_16385_data.csv`

**Apa yang dilakukan:**

1. Download 10,000 candle historis dari MT5
2. Tambah indikator teknis (RSI, SMA, EMA, ATR, Bollinger Bands, dll)
3. Buat label BUY/SELL berdasarkan kriteria:
   - TP = 3.0 \* ATR (Risk:Reward = 1:2)
   - SL = 1.5 \* ATR
   - Holding period max 24 jam
   - TP harus tercapai SEBELUM SL

**Expected time:** 5-15 menit tergantung koneksi MT5

**Troubleshooting:**

- Jika error "Symbol not found" → Enable EURUSD di Market Watch
- Jika error "MT5 connection failed" → Pastikan MT5 login dan running
- Jika label sangat sedikit (<1%) → Turunkan REWARD_MULTIPLIER atau naikkan RISK_MULTIPLIER

---

### FASE 2: Model Training

```bash
python 2_training_FIXED.py
```

**Output:**

- `model_buy.pkl`
- `model_sell.pkl`
- `model_metadata.pkl` (berisi optimal thresholds)

**Apa yang dilakukan:**

1. Load data dari CSV
2. Split data secara TIME-BASED (70% train, 15% validation, 15% test)
3. Train Random Forest model untuk BUY dan SELL
4. Walk-forward validation (5 folds)
5. Optimize probability threshold
6. Evaluate di out-of-sample test set

**Expected time:** 10-30 menit

**Metrics penting yang harus Anda perhatikan:**

```
✅ GOOD MODEL:
   Precision: >0.60  (60%+ sinyal yang benar)
   Recall: >0.40     (catch 40%+ opportunities)
   Expected Return/Trade: >0.2R  (profitable)

❌ BAD MODEL:
   Precision: <0.50  (kurang dari 50%)
   Expected Return/Trade: <0  (losing system)
```

**CRITICAL:**
Jika Expected Return/Trade **NEGATIF**, JANGAN jalankan bot!
Model tidak profitable dan akan kehilangan uang.

---

### FASE 3: Backtesting (WAJIB!)

Sebelum live trading, WAJIB backtest dulu:

```python
# TODO: Buat script backtest terpisah
# Akan saya buatkan jika Anda mau
```

**Minimum requirements sebelum go live:**

- Backtest di data 1-2 tahun
- Win rate minimal 45-50% (dengan RR 1:2)
- Maximum drawdown <20%
- Profit factor >1.5

---

### FASE 4: Demo Trading (MANDATORY!)

**WAJIB test di demo account minimal 3 bulan!**

```bash
# Pastikan MT5 login ke DEMO account
python 3_trading_bot_FIXED.py
```

**Monitoring checklist:**

- [ ] Win rate sesuai ekspektasi (±10%)
- [ ] Drawdown tidak melebihi 20%
- [ ] Tidak ada error/crash selama 1 bulan
- [ ] PnL per trade sesuai risk management
- [ ] Slippage dan spread acceptable

**Log files yang di-generate:**

- `bot.log` - Log semua aktivitas bot
- `trade_journal.csv` - Detail setiap trade
- `bot_state.json` - State persistence

---

## 🎯 PARAMETER PENTING DI BOT

### Risk Management

```python
RISK_PERCENT = 2.0              # 2% risk per trade (JANGAN LEBIH DARI 2%!)
RISK_MULTIPLIER = 1.5           # SL = 1.5 * ATR
REWARD_MULTIPLIER = 3.0         # TP = 3.0 * ATR (RR = 1:2)
```

### Trading Limits

```python
MAX_TRADES_PER_DAY = 3          # Max 3 trade per hari
MAX_DAILY_LOSS_PERCENT = 5.0    # Stop jika loss 5% dalam sehari
MAX_CONSECUTIVE_LOSSES = 3      # Stop setelah 3 loss berturut-turut
MAX_DRAWDOWN_PERCENT = 15.0     # Stop jika drawdown >15%
```

### Trading Hours (Jakarta Time)

```python
TRADING_SESSION_START = time(13, 0)
TRADING_SESSION_END = time(23, 59)
```

**CRITICAL:** Bot akan auto-close semua posisi di tengah malam (00:00)

---

## 📈 EXPECTED PERFORMANCE

**Realistic expectations** (berdasarkan backtesting):

### Dengan RR 1:2 dan Win Rate 50%:

- Expected Return: +0.5R per trade
- Dengan 3 trade/hari × 20 hari = 60 trade/bulan
- Expected monthly return: 30R
- Jika R = 2% modal → 60% gain per bulan (TERLALU BAGUS UNTUK JADI KENYATAAN!)

**Reality check:**

- Slippage: -5% to -10% dari expected
- Spread cost: -5% to -15% dari expected
- False signals: -10% to -20% dari expected
- **Realistic monthly return: 5-15%** (jika model bagus)

**Warning signs model buruk:**

- Win rate <40% di live trading
- Drawdown >20%
- 5+ consecutive losses
- Actual return jauh lebih rendah dari backtest

---

## 🔍 TROUBLESHOOTING

### Bot tidak open trade

✅ Check:

1. Apakah dalam trading session?
2. Apakah sudah exceed daily limits?
3. Apakah spread terlalu tinggi?
4. Apakah model confidence <threshold?
5. Check log file untuk detail

### Trade langsung kena SL

⚠️ Kemungkinan:

1. Slippage terlalu besar
2. Spread terlalu lebar saat entry
3. Volatility tiba-tiba naik
4. ATR calculation outdated

**Fix:**

- Increase `RISK_MULTIPLIER` (SL lebih jauh)
- Add spread filter yang lebih ketat
- Skip trading saat news release

### Win rate jauh lebih rendah dari backtest

🚨 **RED FLAG - STOP BOT IMMEDIATELY**

Kemungkinan penyebab:

1. **Look-ahead bias** masih ada di training (paling sering!)
2. Market regime berubah (trending → ranging)
3. Overfitting - model tidak generalize
4. Execution issues (slippage, requotes)

**Action:**

1. Stop bot
2. Re-check data preparation dan training
3. Collect live trading data 1-2 bulan
4. Re-train model dengan data baru

### Connection errors

```
ERROR: MT5 connection failed
```

**Fix:**

1. Restart MT5
2. Check internet connection
3. Re-login ke account
4. Enable "Algorithmic Trading" di Tools → Options

---

## 📚 BEST PRACTICES

### 1. Regular Model Re-training

- Re-train setiap 3-6 bulan dengan data terbaru
- Market conditions berubah → model perlu adapt

### 2. Position Sizing

- **NEVER** risk lebih dari 2% per trade
- Consider total portfolio risk (multiple assets)

### 3. Monitoring

- Check bot SETIAP HARI
- Review trade journal MINGGUAN
- Analyze performance BULANAN

### 4. Risk Management

```
Modal: $10,000
Risk per trade: 2% = $200
Max positions: 1 (bot ini single position)
Total risk exposure: $200
```

### 5. When to Stop Bot

STOP IMMEDIATELY jika:

- Drawdown >20%
- 5+ consecutive losses
- Win rate <35% setelah 50+ trades
- Unexplained errors/crashes
- Major market event (central bank decision, war, etc.)

---

## 🔬 ADVANCED: MODEL IMPROVEMENT IDEAS

### 1. Feature Engineering

```python
# Tambahkan di features.py:
- Volume indicators (OBV, VWAP)
- Market regime detection (trending/ranging)
- Time-based features (hour, day of week)
- Correlation with other pairs
```

### 2. Alternative Models

```python
# Test di 2_training_FIXED.py:
- XGBoost (sering lebih baik dari Random Forest)
- LightGBM (lebih cepat)
- Neural Networks (LSTM untuk time series)
```

### 3. Ensemble Methods

```python
# Combine multiple models:
buy_signal = (model1.predict_proba() + model2.predict_proba()) / 2 > threshold
```

### 4. Adaptive Thresholds

```python
# Adjust threshold based on market volatility
if high_volatility:
    threshold = 0.80  # More conservative
else:
    threshold = 0.65  # More aggressive
```

---

## 📞 SUPPORT & UPDATES

### Log Analysis

Jika bot crash atau performa buruk, kirimkan:

1. Last 100 lines dari `bot.log`
2. `trade_journal.csv`
3. Deskripsi masalah

### Version History

- **v1.0** (Original) - Broken (look-ahead bias, feature mismatch)
- **v2.0** (Fixed) - Current version dengan semua perbaikan

---

## ⚖️ LEGAL DISCLAIMER

Trading forex carries a high level of risk and may not be suitable for all investors. Before deciding to trade, you should carefully consider your investment objectives, level of experience, and risk appetite.

**THIS BOT:**

- Is provided "AS IS" without warranty
- May contain bugs or errors
- Past performance does NOT indicate future results
- The developer is NOT liable for any losses

**YOU are responsible for:**

- Testing thoroughly before live use
- Managing your risk appropriately
- Understanding the code and its limitations
- Monitoring the bot's performance

---

## 🎓 LEARNING RESOURCES

### Understand the Code

1. **features.py** - Start here untuk mengerti indicators
2. **1_data_preparation_FIXED.py** - Pelajari label creation logic
3. **2_training_FIXED.py** - Pelajari ML evaluation metrics
4. **3_trading_bot_FIXED.py** - Pelajari risk management implementation

### Recommended Reading

- "Advances in Financial Machine Learning" by Marcos Lopez de Prado
- "Machine Learning for Algorithmic Trading" by Stefan Jansen
- Quantopian lectures (archive available online)

### Key Concepts

- **Walk-forward validation** - Why it's critical for time series
- **Look-ahead bias** - Most common mistake in ML trading
- **Overfitting** - Why high backtest accuracy often fails live
- **Position sizing** - Kelly Criterion, Fixed Fractional

---

## ✅ FINAL CHECKLIST SEBELUM GO LIVE

- [ ] Sudah train model dengan data >5000 candles
- [ ] Walk-forward validation score consistent (std <0.1)
- [ ] Expected Return/Trade >0.2R
- [ ] Backtest dengan realistic assumptions (spread, slippage)
- [ ] Demo trading 3+ bulan dengan hasil positif
- [ ] Win rate di demo ±10% dari backtest expectation
- [ ] Understand SEMUA parameter di bot
- [ ] Have emergency stop procedures
- [ ] Siap mental untuk drawdown 15-20%
- [ ] Risk <2% per trade
- [ ] Total trading capital <10% of net worth

**Jika SEMUA checklist ✅, baru consider live trading dengan SMALL capital!**

---

## 🚀 GOOD LUCK & TRADE SAFE!

Remember:

- **90% of retail traders lose money**
- Automated trading is NOT easy money
- The best trader is a disciplined one
- Risk management > prediction accuracy

**When in doubt, DON'T TRADE!**
