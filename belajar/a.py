from sklearn.ensemble import RandomForestClassifier

# 1. Data Latihan (X = Indikator, y = Target/Hasil)
# Format: [RSI, MA_Cross(1=Yes, 0=No)]
X = [[75, 1], [30, 0], [40, 1], [80, 1], [20, 0]]
y = [0, 1, 1, 0, 1] 

# 2. Buat Model Random Forest (Gunakan 100 pohon keputusan)
model = RandomForestClassifier(n_estimators=2000,)

# 3. Training model
model.fit(X, y)

# 4. Prediksi data baru (Misal RSI 25 dan MA tidak cross)
prediksi = model.predict([[10, 0]])
print(f"Hasil Prediksi: {'Naik' if prediksi[0] == 1 else 'Turun'}")
