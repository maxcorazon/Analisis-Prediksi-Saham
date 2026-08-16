import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go

def pred_lstm(data_saham, hari_kedepan=90):
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.experimental.enable_op_determinism()

    # Persiapan & Normalisasi Data
    df = data_saham[['Date', 'Close']].copy()
    df['Close'] = df['Close'].round(2)
    data_close = df['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_close)

    # Membuat Sequence (Time step = 60 hari)
    time_step = 60
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Train/Test Split 80:20 (Mencegah Kebocoran Data)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Print Pembuktian 
    print("\nBukti Tabel MinMaxScaler")
    df_bukti_scaler = pd.DataFrame({
        'Tanggal': df['Date'].iloc[:5],
        'Harga Penutupan Asli (Rp)': data_close[:5].flatten(),
        'Harga Hasil Skalasi (Scaled)': scaled_data[:5].flatten()
    })
    print(df_bukti_scaler.to_string(index=False))
    
    print("\nBukti Tensor Shape LSTM")
    print(f"Dimensi X_train (Data Latih) : {X_train.shape}")
    print(f"Dimensi y_train (Target Latih) : {y_train.shape}")
    print(f"Dimensi X_test (Data Uji)  : {X_test.shape}")
    print(f"Dimensi y_test (Target Uji)  : {y_test.shape}")
    print("")

    # Membangun Model LSTM dengan Bidirectional & Dropout 
    model = Sequential()
    
    # Layer 1: Bidirectional LSTM + Dropout (Membaca pola dari dua arah)
    model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    
    # Layer 2: LSTM + Dropout
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Layer Output: Menebak 1 harga
    model.add(Dense(units=1))
    model.summary()

    # Kompilasi Model dengan Optimizer Adam dan Loss MSE
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Callbacks untuk optimasi training
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Latih model dengan validation pada data test
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                        validation_data=(X_test, y_test),
                        callbacks=[early_stop, reduce_lr], verbose=0)

    print(f"\nJumlah epoch yang benar-benar dijalankan: {len(history.history['loss'])} dari maksimal 50")
    if len(history.history['loss']) < 50:
        print("Early Stopping aktif — pelatihan berhenti lebih awal.")
    else:
        print("Early Stopping tidak terpicu — pelatihan selesai hingga epoch maksimal.")

    # Prediksi Data Test (Untuk Evaluasi)
    prediksi_test_scaled = model.predict(X_test, verbose=0)
    prediksi_test = scaler.inverse_transform(prediksi_test_scaled)
    y_test_asli = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Prediksi Data Training (Untuk Visualisasi)
    prediksi_train_scaled = model.predict(X_train, verbose=0)
    prediksi_train = scaler.inverse_transform(prediksi_train_scaled)

    # Menghitung MAE, RMSE, MAPE pada data test
    mae = mean_absolute_error(y_test_asli, prediksi_test)
    rmse = np.sqrt(mean_squared_error(y_test_asli, prediksi_test))
    mape = np.mean(np.abs((y_test_asli - prediksi_test) / y_test_asli)) * 100
    metrik = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    # Prediksi Masa Depan dari 60 hari terakhir
    last_60_days = scaled_data[-time_step:]
    current_batch = last_60_days.reshape(1, time_step, 1)
    prediksi_masa_depan = []

    for i in range(hari_kedepan):
        pred = model.predict(current_batch, verbose=0)
        prediksi_masa_depan.append(pred[0, 0])
        current_batch = np.append(current_batch[:, 1:, :], [[pred[0]]], axis=1)

    prediksi_masa_depan = scaler.inverse_transform(np.array(prediksi_masa_depan).reshape(-1, 1))

    # Menyiapkan DataFrame untuk Prediksi Masa Depan
    tanggal_terakhir = df['Date'].iloc[-1]
    # Menggunakan bdate_range agar hanya bussines day saja
    tanggal_masa_depan = pd.bdate_range(start=tanggal_terakhir + pd.Timedelta(days=1), periods=hari_kedepan)
    
    df_future = pd.DataFrame({
        'Tanggal': tanggal_masa_depan,
        'Prediksi Harga': prediksi_masa_depan.flatten().astype(float).round(2)
    })

    # Visualisasi Grafik 
    grafik = go.Figure()
    
    # Data asli (seluruh data)
    grafik.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'], 
        name='Data Asli', line_color='black'
    ))
    
    # Fitted Training (prediksi pada data training)
    grafik.add_trace(go.Scatter(
        x=df['Date'].iloc[time_step:time_step + split_index], 
        y=prediksi_train.flatten(), 
        name='Fitted Training (80%)', line_color='blue'
    ))
    
    # Prediksi Test (digunakan untuk evaluasi)
    grafik.add_trace(go.Scatter(
        x=df['Date'].iloc[time_step + split_index:], 
        y=prediksi_test.flatten(), 
        name='Prediksi Test (20%)', line_color='orange'
    ))
    
    # Forecast Masa Depan
    grafik.add_trace(go.Scatter(
        x=df_future['Tanggal'], y=df_future['Prediksi Harga'], 
        name='Forecast Masa Depan', line_color='red'
    ))
    
    # Garis vertikal pembatas Train/Test Split
    tanggal_split = str(df['Date'].iloc[time_step + split_index])
    grafik.add_shape(
        type="line", x0=tanggal_split, x1=tanggal_split, y0=0, y1=1,
        yref="paper", line=dict(color="gray", dash="dash", width=1.5)
    )
    grafik.add_annotation(
        x=tanggal_split, y=1.05, yref="paper",
        text="Train/Test Split (80:20)", showarrow=False,
        font=dict(size=11, color="gray")
    )

    grafik.update_layout(
        title="Prediksi LSTM (Bidirectional, Dropout, Early Stopping & Business Days)", 
        xaxis_title="Tanggal", 
        yaxis_title="Harga (Rp)", 
        template="plotly_white"
    )
    
    # Tabel Pembuktian Historis
    df_historis = pd.DataFrame({
        'Tanggal': df['Date'].iloc[time_step + split_index:].values,
        'Harga Asli': y_test_asli.flatten().astype(float).round(2),
        'Harga Prediksi LSTM': prediksi_test.flatten().astype(float).round(2)
    })
    
    # Grafik Learning Curve (Loss vs Validation Loss)
    grafik_loss = go.Figure()
    grafik_loss.add_trace(go.Scatter(
        y=history.history['loss'], 
        name='Training Loss', line_color='blue'
    ))
    grafik_loss.add_trace(go.Scatter(
        y=history.history['val_loss'], 
        name='Validation Loss', line_color='orange'
    ))
    grafik_loss.update_layout(
        title='Kurva Pelatihan Model (Training vs Validation Loss)',
        xaxis_title='Epoch',
        yaxis_title='Mean Squared Error (Loss)',
        template='plotly_white'
    )
    
    return df_future, metrik, grafik, df_historis, grafik_loss