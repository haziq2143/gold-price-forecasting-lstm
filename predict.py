import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- 1. MEMBERSIHKAN DATA (CRITICAL) ---
# Pastikan data urut dari TAHUN LAMA ke TAHUN BARU
df_final = df_final.sort_values('Date', ascending=True).reset_index(drop=True)

def clean_number(x):
    if isinstance(x, str):
        # Hapus titik (ribuan) dan ganti koma jadi titik (desimal)
        x = x.replace('.', '').replace(',', '.')
    return float(x)

# List semua kolom yang butuh dibersihin (Emas dan DXY biasanya formatnya sama)
df_final['Price'] = df_final['Price'].apply(clean_number)
df_final['Price_dxy'] = df_final['Price_dxy'].apply(clean_number)

# --- 2. PREPARASI DATA ---
features = ['Price', 'Price_dxy', 'FEDFUNDS', 'CPIAUCSL', 'PAYEMS']
data_values = df_final[features].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_values)

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window, :])
        y.append(data[i+window, 0]) 
    return np.array(X), np.array(y)

WINDOW_SIZE = 30 # Coba persempit ke 30 hari biar lebih responsif
X, y = create_sequences(scaled_data, WINDOW_SIZE)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- 3. MODEL LSTM YANG LEBIH STABIL ---
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

# Pake learning rate yang lebih kecil biar gak 'over shoot'
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

print("\n🔥 Training ulang dengan data yang sudah di-sorting...")
model.fit(X_train, y_train, batch_size=16, epochs=50, validation_data=(X_test, y_test), verbose=1)

# --- 4. PREDIKSI & EVALUASI ---
predictions = model.predict(X_test)

# Inverse Scale
dummy_pred = np.zeros((len(predictions), len(features)))
dummy_pred[:, 0] = predictions.flatten()
inv_predictions = scaler.inverse_transform(dummy_pred)[:, 0]

dummy_actual = np.zeros((len(y_test), len(features)))
dummy_actual[:, 0] = y_test
inv_y_test = scaler.inverse_transform(dummy_actual)[:, 0]

# --- 5. TAMPILKAN HASIL ---
from sklearn.metrics import mean_absolute_error
new_mae = mean_absolute_error(inv_y_test, inv_predictions)

print(f"\n✅ MAE BARU: {new_mae:.2f}")

plt.figure(figsize=(15, 6))
plt.plot(inv_y_test, label='Harga Asli', color='blue')
plt.plot(inv_predictions, label='Prediksi LSTM', color='red', linestyle='--')
plt.title('Perbaikan Prediksi Harga Emas Kelompok 5')
plt.legend()
plt.show()