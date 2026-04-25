import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error

# 1. LOAD & CLEANING
df = pd.read_csv('dataset_multivariate_siap_lstm.csv')

def clean_currency(x):
    if isinstance(x, str):
        return x.replace('.', '').replace(',', '.')
    return x

df['Price'] = df['Price'].apply(clean_currency).astype(float)
if 'Price_dxy' in df.columns:
    df['Price_dxy'] = df['Price_dxy'].apply(clean_currency).astype(float)

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 2. FEATURES
features = ['Price', 'Price_dxy', 'FEDFUNDS', 'CPIAUCSL', 'PAYEMS']
data_values = df[features].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_values)

# 3. WINDOW SIZE (Kita pake 50 hari, angka tengah yang pas)
WINDOW_SIZE = 50 
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window, :])
        y.append(data[i+window, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, WINDOW_SIZE)

split = int(len(X) * 0.9)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. MODEL "THE SWEET SPOT" (2 Layer tapi lebih padat)
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.15), # Dropout dikecilin biar info gak banyak ilang
    LSTM(50, return_sequences=False),
    Dropout(0.15),
    Dense(25),
    Dense(1)
])

# 5. OPTIMIZER & CALLBACKS (Senjata Rahasia)
optimizer = Adam(learning_rate=0.001) 
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# --- INI BARIS YANG TADI KETINGGALAN ---
model.compile(optimizer=optimizer, loss='mean_squared_error')
# ---------------------------------------

print("\n🔥 Menjalankan Model Sweet Spot (Optimasi MAE)...")
model.fit(
    X_train, y_train, 
    batch_size=16, 
    epochs=100, 
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# 6. PREDIKSI & EVALUASI
predictions = model.predict(X_test)
dummy_pred = np.zeros((len(predictions), len(features)))
dummy_pred[:, 0] = predictions.flatten()
inv_predictions = scaler.inverse_transform(dummy_pred)[:, 0]

dummy_actual = np.zeros((len(y_test), len(features)))
dummy_actual[:, 0] = y_test
inv_y_test = scaler.inverse_transform(dummy_actual)[:, 0]

final_mae = mean_absolute_error(inv_y_test, inv_predictions)
print(f"\n✅ MAE FINAL: {final_mae:.2f}")

# 7. FORECASTING & PLOT
future_days = 30
current_batch = scaled_data[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, len(features))
future_outputs = []

for _ in range(future_days):
    next_pred = model.predict(current_batch, verbose=0)
    future_outputs.append(next_pred[0, 0])
    new_row = current_batch[0, -1, :].copy()
    new_row[0] = next_pred[0, 0]
    current_batch = np.append(current_batch[:, 1:, :], new_row.reshape(1, 1, len(features)), axis=1)

inv_future = scaler.inverse_transform(np.column_stack([future_outputs, np.zeros((future_days, 4))]))[:, 0]

plt.figure(figsize=(15, 7))
plt.plot(inv_y_test, label='Harga Asli', color='blue', alpha=0.6)
plt.plot(inv_predictions, label='Prediksi Model', color='red', linestyle='--')
x_fut = np.arange(len(inv_y_test), len(inv_y_test) + future_days)
plt.plot(x_fut, inv_future, label='Ramalan Masa Depan', color='green', linewidth=3)
plt.title(f'Final Analisis Prediksi Harga Emas (MAE: {final_mae:.2f})')
plt.legend()
plt.grid(True)
plt.show()

print(f"🔮 Estimasi Besok: ${inv_future[0]:.2f}")