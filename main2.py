import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. LOAD & CLEANING DATA
df = pd.read_csv('dataset_multivariate_siap_lstm.csv')

def clean_currency(x):
    if isinstance(x, str):
        # Hilangkan titik ribuan dan ubah koma jadi titik desimal
        return x.replace('.', '').replace(',', '.')
    return x

# Pastikan harga emas dan DXY bersih dari format string
df['Price'] = df['Price'].apply(clean_currency).astype(float)
if 'Price_dxy' in df.columns:
    df['Price_dxy'] = df['Price_dxy'].apply(clean_currency).astype(float)

# WAJIB: Urutkan tanggal dari masa lalu ke masa depan
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 2. SELECT FEATURES 

features = ['Price', 'Price_dxy', 'FEDFUNDS', 'CPIAUCSL', 'PAYEMS']
data_values = df[features].values

# 3. NORMALISASI
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_values)

# 4. SLIDING WINDOW (Window size 45 hari biasanya sweet spot)
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window, :])
        y.append(data[i+window, 0]) # Target tetap Price (Index 0)
    return np.array(X), np.array(y)

WINDOW_SIZE = 45
X, y = create_sequences(scaled_data, WINDOW_SIZE)

# Split data (90% training, 10% testing karena data 2026 itu krusial)
split = int(len(X) * 0.9)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. MODEL LSTM OPTIMIZED
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

# Gunakan learning rate yang lebih halus
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')

# Early stopping biar gak overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

print("\n🚀 Memulai Training Model Optimized...")
history = model.fit(
    X_train, y_train, 
    batch_size=16, 
    epochs=100, 
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# 6. PREDIKSI & EVALUASI
predictions = model.predict(X_test)

# Balikin skala ke harga asli
dummy_pred = np.zeros((len(predictions), len(features)))
dummy_pred[:, 0] = predictions.flatten()
inv_predictions = scaler.inverse_transform(dummy_pred)[:, 0]

dummy_actual = np.zeros((len(y_test), len(features)))
dummy_actual[:, 0] = y_test
inv_y_test = scaler.inverse_transform(dummy_actual)[:, 0]

# Hitung MAE Baru
final_mae = mean_absolute_error(inv_y_test, inv_predictions)
print(f"\n✅ MAE FINAL: {final_mae:.2f}")

# 7. VISUALISASI
plt.figure(figsize=(15, 7))
plt.plot(inv_y_test, label='Harga Emas Asli', color='blue', linewidth=2)
plt.plot(inv_predictions, label='Prediksi LSTM', color='red', linestyle='--', linewidth=2)
plt.title(f'Prediksi Harga Emas Multivariate LSTM - Kelompok 5 (MAE: {final_mae:.2f})')
plt.xlabel('Waktu (Data Testing)')
plt.ylabel('Harga')
plt.legend()
plt.grid(True)
plt.show()

# 1. Ambil 45 hari terakhir dari dataset asli
last_45_days = scaled_data[-WINDOW_SIZE:]

# 2. Reshape biar sesuai input LSTM (1, 45, 5)
X_besok = np.array([last_45_days])

# 3. Prediksi!S
prediksi_scaled = model.predict(X_besok)

# 4. Balikin ke harga asli
dummy_besok = np.zeros((1, len(features)))
dummy_besok[0, 0] = prediksi_scaled
harga_besok = scaler.inverse_transform(dummy_besok)[0, 0]

print(f"\n🔮 PREDIKSI HARGA EMAS UNTUK HARI KERJA BERIKUTNYA:")
print(f"Estimasi: ${harga_besok:.2f}")