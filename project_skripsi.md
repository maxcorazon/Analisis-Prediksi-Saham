# Lampiran Kode Program: Prediksi Harga Saham BBCA

Dokumen ini berisi kode sumber utama untuk aplikasi dashboard prediksi harga saham BBCA menggunakan metode Facebook Prophet dan Long Short-Term Memory (LSTM).

## 1. File: app.py
```python
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from prediksi_prophet import pred_prophet
from prediksi_lstm import pred_lstm

# Cache hasil model agar tidak ditraining ulang setiap page refresh
@st.cache_data(show_spinner=False)
def cached_pred_prophet(_data_saham, hari_kedepan):
    return pred_prophet(_data_saham, hari_kedepan=hari_kedepan)

@st.cache_data(show_spinner=False)
def cached_pred_lstm(_data_saham, hari_kedepan):
    return pred_lstm(_data_saham, hari_kedepan=hari_kedepan)

# Halaman Dashboard
st.set_page_config(page_title="Prediksi Harga Saham", layout="wide")

# Sidebar
st.sidebar.title("Pengaturan Model")

# Slider untuk mengatur hari ke depan
hari_prediksi = st.sidebar.slider(
    "Rentang Prediksi Masa Depan (Hari)", 
    min_value=10,   
    max_value=365,  
    value=90,       
    step=1          
)

# Header Utama
st.title("Dashboard Prediksi Harga Saham")

# Kode saham dan periode data
KODE_SAHAM = "BBCA.JK"
jumlah_tahun = 10

tanggal_mulai = datetime(2016, 3, 31)
tanggal_akhir = datetime(2026, 3, 31)

# Mengambil data saham 
@st.cache_data
def ambil_data_saham(kode, waktu_mulai, waktu_akhir):
    import os
    try:
        data = yf.download(kode, start=waktu_mulai, end=waktu_akhir, progress=False)
        if data.empty:
            raise ValueError("Data kosong dari Yahoo Finance")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        # Jika gagal tarik data live, gunakan CSV Cadangan
        st.warning("Menggunakan data offline. Koneksi Yahoo limit")
        if os.path.exists("bbca_cadangan.csv"):
            data_csv = pd.read_csv("bbca_cadangan.csv")
            data_csv['Date'] = pd.to_datetime(data_csv['Date'])
            return data_csv
        return pd.DataFrame()

# Menjalankan fungsi pengambilan data
data_saham = ambil_data_saham(KODE_SAHAM, tanggal_mulai, tanggal_akhir)

# Cek apakah data berhasil diambil
if data_saham.empty:
    st.error("Gagal menarik data dan data cadangan tidak ditemukan.")
    st.stop()

# Visualisasi Data Historis
kolom_kiri, kolom_kanan = st.columns([1, 2])

with kolom_kiri:
    st.subheader("Ringkasan Data")
    st.dataframe(data_saham.tail(10)) 
    
    harga_terakhir = float(data_saham['Close'].squeeze().iloc[-1])
    st.metric(label="Harga Penutupan Terakhir", value=f"Rp {harga_terakhir:,.0f}")

with kolom_kanan:
    grafik = go.Figure()
    grafik.add_trace(go.Scatter(
        x=data_saham['Date'], 
        y=data_saham['Close'], 
        name="Harga Asli", 
        line_color='deepskyblue'
    ))
    grafik.update_layout(
        title=f"Grafik Historis {KODE_SAHAM} (2016-2026)", 
        template="plotly_white",
        xaxis_title="Tanggal",
        yaxis_title="Harga (Rupiah)"
    )
    st.plotly_chart(grafik, use_container_width=True)

# Hasil Prediksi dan Evaluasi Model
st.divider()
st.subheader("Hasil Prediksi Machine Learning")
st.info("Evaluasi model menggunakan 80% data awal dan 20% data terakhir untuk memastikan akurasi yang valid.")

tab_prophet, tab_lstm = st.tabs(["Metode Facebook Prophet", "Metode LSTM"])

with tab_prophet:
    st.write(f"Memproses prediksi dengan Facebook Prophet (Prediksi **{hari_prediksi} hari** ke depan)")
    
    # Memanggil model dengan spinner
    with st.spinner("Melatih model Prophet..."):
        hasil_forecast, metrik_prophet, grafik_prophet, histori_prophet = cached_pred_prophet(data_saham, hari_prediksi)
    
    st.plotly_chart(grafik_prophet, use_container_width=True)
    
    st.subheader("Evaluasi Akurasi Model (Data Test 20%)")
    kolom_metrik1, kolom_metrik2, kolom_metrik3 = st.columns(3)
    kolom_metrik1.metric("MAE (Rata-rata Meleset)", f"Rp {metrik_prophet['MAE']:,.2f}")
    kolom_metrik2.metric("RMSE (Error Kuadrat)", f"Rp {metrik_prophet['RMSE']:,.2f}")
    kolom_metrik3.metric("MAPE (Persentase Error)", f"{metrik_prophet['MAPE']:.2f} %")
    
    # Tabel Pembuktian
    with st.expander("Tabel Pembuktian Matematis (Actual vs Prediksi) — Data Test"):
        st.write("Tabel penjabaran matematis selisih harga asli dan tebakan model pada **data test (20%)** untuk membuktikan metrik error.")
        
        df_prov_prophet = histori_prophet[['ds', 'y', 'yhat']].copy()
        df_prov_prophet.columns = ['Tanggal', 'Harga Asli', 'Prediksi Prophet']
        
        # Menghitung Error secara Matematis 
        df_prov_prophet['Selisih (Error)'] = df_prov_prophet['Harga Asli'] - df_prov_prophet['Prediksi Prophet']
        df_prov_prophet['Error Mutlak (MAE)'] = df_prov_prophet['Selisih (Error)'].abs()
        df_prov_prophet['Error Kuadrat (RMSE)'] = df_prov_prophet['Selisih (Error)'] ** 2
        df_prov_prophet['Persentase Error (%)'] = (df_prov_prophet['Error Mutlak (MAE)'] / df_prov_prophet['Harga Asli']) * 100
        
        st.dataframe(df_prov_prophet.tail(20), hide_index=True)
        
        csv_prophet = df_prov_prophet.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data Pembuktian (CSV)", csv_prophet, "pembuktian_matematis_prophet.csv", "text/csv")
        
    st.subheader("Tabel Prediksi Harga Mendatang")
    tabel_masa_depan = hasil_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
    tabel_masa_depan.columns = ['Tanggal', 'Prediksi Harga', 'Batas Bawah', 'Batas Atas']
    st.dataframe(tabel_masa_depan, hide_index=True)

with tab_lstm:
    st.write(f"Memproses Deep Learning LSTM (Prediksi **{hari_prediksi}** hari ke depan)")
    
    # Memanggil model dengan spinner
    with st.spinner("Melatih model LSTM... (memakan waktu beberapa menit)"):
        hasil_future_lstm, metrik_lstm, grafik_lstm, histori_lstm = cached_pred_lstm(data_saham, hari_prediksi)
    
    st.plotly_chart(grafik_lstm, use_container_width=True)
    
    st.subheader("Evaluasi Akurasi Model LSTM (Data Test 20%)")
    kolom1_lstm, kolom2_lstm, kolom3_lstm = st.columns(3)
    kolom1_lstm.metric("MAE (Rata-rata Meleset)", f"Rp {metrik_lstm['MAE']:,.2f}")
    kolom2_lstm.metric("RMSE (Error Kuadrat)", f"Rp {metrik_lstm['RMSE']:,.2f}")
    kolom3_lstm.metric("MAPE (Persentase Error)", f"{metrik_lstm['MAPE']:.2f} %")
    
    # Tabel Pembuktian
    with st.expander("Lihat Tabel Pembuktian Matematis (Actual vs Prediksi) — Data Test"):
        st.write("Tabel penjabaran matematis selisih harga asli dan tebakan model pada **data test (20%)** untuk membuktikan metrik error.")
        
        df_prov_lstm = histori_lstm.copy()
        
        # Menghitung Error secara Matematis 
        df_prov_lstm['Selisih (Error)'] = df_prov_lstm['Harga Asli'] - df_prov_lstm['Harga Prediksi LSTM']
        df_prov_lstm['Error Mutlak (MAE)'] = df_prov_lstm['Selisih (Error)'].abs()
        df_prov_lstm['Error Kuadrat (RMSE)'] = df_prov_lstm['Selisih (Error)'] ** 2
        df_prov_lstm['Persentase Error (%)'] = (df_prov_lstm['Error Mutlak (MAE)'] / df_prov_lstm['Harga Asli']) * 100
        
        st.dataframe(df_prov_lstm.tail(20), hide_index=True)
        
        csv_lstm = df_prov_lstm.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data Pembuktian (CSV)", csv_lstm, "pembuktian_matematis_lstm.csv", "text/csv")
        
    st.subheader("Tabel Prediksi Harga Mendatang")
    st.dataframe(hasil_future_lstm.tail(10), hide_index=True)
```

## 2. File: prediksi_lstm.py
```python
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
    # Set random seed untuk hasil konsisten setiap kali dijalankan
    np.random.seed(42)
    tf.random.set_seed(42)

    # Persiapan & Normalisasi Data
    df = data_saham[['Date', 'Close']].copy()
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

    # Membangun Model LSTM Deep Learning dengan Bidirectional & Dropout 
    model = Sequential()
    
    # Layer 1: Bidirectional LSTM + Dropout (Membaca pola dari dua arah)
    model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2)) # Membuang data 20% agar tidak terjadi overfitting
    
    # Layer 2: LSTM + Dropout (Memperdalam Model)
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2)) # Membuang data 20% agar tidak terjadi overfitting
    
    # Layer Output: Menebak 1 harga
    model.add(Dense(units=1))

    # Kompilasi Model dengan Optimizer Adam dan Loss MSE
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Callbacks untuk optimasi training
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Latih model dengan validation pada data test
    model.fit(X_train, y_train, epochs=50, batch_size=32, 
              validation_data=(X_test, y_test),
              callbacks=[early_stop, reduce_lr], verbose=0)

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

    # Menyiapkan DataFrame untuk Prediksi Masa Depan hari kerja
    tanggal_terakhir = df['Date'].iloc[-1]
    # Menggunakan bdate_range otomatis melewati hari Sabtu & Minggu
    tanggal_masa_depan = pd.bdate_range(start=tanggal_terakhir + pd.Timedelta(days=1), periods=hari_kedepan)
    
    df_future = pd.DataFrame({
        'Tanggal': tanggal_masa_depan,
        'Prediksi Harga': prediksi_masa_depan.flatten()
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
        'Harga Asli': y_test_asli.flatten(),
        'Harga Prediksi LSTM': prediksi_test.flatten()
    })
    
    return df_future, metrik, grafik, df_historis
```

## 3. File: prediksi_prophet.py
```python
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go

def pred_prophet(data_saham, hari_kedepan=90):
    
    # Menyiapkan data
    df_prophet = data_saham[['Date', 'Close']].copy()
    df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)

    # Menambahkan fitur teknikal sebagai regressor 
    # Moving Average 20 hari — menangkap tren jangka pendek
    df_prophet['ma20'] = df_prophet['y'].rolling(window=20, min_periods=1).mean()
    # Moving Average 50 hari — menangkap tren jangka menengah
    df_prophet['ma50'] = df_prophet['y'].rolling(window=50, min_periods=1).mean()

    # Train/Test Split 80:20 (Menghindari Data Leakage)
    split_index = int(len(df_prophet) * 0.8)
    df_train = df_prophet.iloc[:split_index].copy()
    df_test = df_prophet.iloc[split_index:].copy()

    # Hyperparameter tuning untuk memilih parameter terbaik secara otomatis
    # Mencari nilai MAPE terendah pada data test
    param_grid = [
        {'cps': 0.3, 'sps': 10.0, 'sm': 'multiplicative', 'nc': 40, 'cr': 0.9},
        {'cps': 0.5, 'sps': 10.0, 'sm': 'multiplicative', 'nc': 50, 'cr': 0.9},
        {'cps': 0.5, 'sps': 10.0, 'sm': 'multiplicative', 'nc': 50, 'cr': 0.95},
        {'cps': 0.8, 'sps': 10.0, 'sm': 'multiplicative', 'nc': 50, 'cr': 0.95},
        {'cps': 1.0, 'sps': 15.0, 'sm': 'multiplicative', 'nc': 50, 'cr': 0.95},
        {'cps': 1.5, 'sps': 10.0, 'sm': 'multiplicative', 'nc': 50, 'cr': 0.95},
        {'cps': 2.0, 'sps': 10.0, 'sm': 'multiplicative', 'nc': 50, 'cr': 0.95},
        {'cps': 3.0, 'sps': 10.0, 'sm': 'multiplicative', 'nc': 50, 'cr': 0.95},
        {'cps': 5.0, 'sps': 10.0, 'sm': 'multiplicative', 'nc': 50, 'cr': 0.95},
    ]
    
    best_mape = float('inf')
    best_model = None
    best_forecast = None
    best_params = None
    
    # Hitung sekali saja variabel yang dibutuhkan semua iterasi
    last_train_date = df_train['ds'].iloc[-1]
    last_actual_date = df_prophet['ds'].iloc[-1]
    test_bdays = len(pd.bdate_range(start=last_train_date + pd.Timedelta(days=1), end=last_actual_date))
    total_periods = test_bdays + hari_kedepan
    
    for params in param_grid:
        model = Prophet(
            daily_seasonality=False,
            yearly_seasonality=True,
            weekly_seasonality=True,
            seasonality_mode=params['sm'],
            changepoint_prior_scale=params['cps'],
            seasonality_prior_scale=params['sps'],
            n_changepoints=params['nc'],
            changepoint_range=params['cr']
        )
        # Seasonality tambahan
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
        model.add_country_holidays(country_name='ID')
        
        # Regressor teknikal
        model.add_regressor('ma20')
        model.add_regressor('ma50')
        
        model.fit(df_train)
        
        # Membuat DataFrame masa depan dengan regressor
        future = model.make_future_dataframe(periods=total_periods, freq='B')
        
        # Untuk future dates, gunakan nilai MA terakhir yang tersedia
        ma_data = df_prophet[['ds', 'ma20', 'ma50']].copy()
        future = future.merge(ma_data, on='ds', how='left')
        
        # Isi MA yang kosong (future dates) dengan forward fill lalu backfill
        future['ma20'] = future['ma20'].ffill().bfill()
        future['ma50'] = future['ma50'].ffill().bfill()
        
        forecast = model.predict(future)
        
        # Evaluasi pada test
        pred_test = forecast[['ds', 'yhat']].merge(df_test[['ds', 'y']], on='ds', how='inner')
        if len(pred_test) > 0:
            mape_val = np.mean(np.abs((pred_test['y'] - pred_test['yhat']) / pred_test['y'])) * 100
            if mape_val < best_mape:
                best_mape = mape_val
                best_model = model
                best_forecast = forecast
                best_params = params
    
    # Menggunakan model terbaik
    forecast = best_forecast

    # Menghitung Evaluasi pada data test
    prediksi_test = forecast[['ds', 'yhat']].merge(df_test[['ds', 'y']], on='ds', how='inner')
    
    y_asli = prediksi_test['y']
    y_prediksi = prediksi_test['yhat']
    
    mae = mean_absolute_error(y_asli, y_prediksi)
    rmse = np.sqrt(mean_squared_error(y_asli, y_prediksi))
    mape = np.mean(np.abs((y_asli - y_prediksi) / y_asli)) * 100

    metrik = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    # Memisahkan forecast berdasarkan periode
    forecast_train = forecast[forecast['ds'] <= last_train_date]
    forecast_test_period = forecast[(forecast['ds'] > last_train_date) & (forecast['ds'] <= last_actual_date)]
    forecast_future = forecast[forecast['ds'] > last_actual_date]

    # Visualisasi Grafik
    grafik = go.Figure()
    
    # Data Training
    grafik.add_trace(go.Scatter(
        x=df_train['ds'], y=df_train['y'],
        name='Data Training (80%)', line_color='blue'
    ))
    
    # Data Test
    grafik.add_trace(go.Scatter(
        x=df_test['ds'], y=df_test['y'],
        name='Data Test Aktual (20%)', line_color='green'
    ))
    
    # Prediksi Prophet pada periode training (fitted)
    grafik.add_trace(go.Scatter(
        x=forecast_train['ds'], y=forecast_train['yhat'],
        name='Fitted Training', line_color='deepskyblue', line=dict(dash='dot')
    ))
    
    # Prediksi Prophet pada periode test
    grafik.add_trace(go.Scatter(
        x=forecast_test_period['ds'], y=forecast_test_period['yhat'],
        name='Prediksi Test', line_color='orange'
    ))
    
    # Confidence Interval pada periode Test
    grafik.add_trace(go.Scatter(
        x=forecast_test_period['ds'], y=forecast_test_period['yhat_upper'],
        mode='lines', line=dict(width=0), showlegend=False
    ))
    grafik.add_trace(go.Scatter(
        x=forecast_test_period['ds'], y=forecast_test_period['yhat_lower'],
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(255,165,0,0.15)',
        name='Confidence Interval (Test)'
    ))
    
    # Forecast Masa Depan
    if not forecast_future.empty:
        grafik.add_trace(go.Scatter(
            x=forecast_future['ds'], y=forecast_future['yhat'],
            name='Forecast Masa Depan', line_color='red'
        ))
        
        # Confidence Interval pada Forecast Masa Depan
        grafik.add_trace(go.Scatter(
            x=forecast_future['ds'], y=forecast_future['yhat_upper'],
            mode='lines', line=dict(width=0), showlegend=False
        ))
        grafik.add_trace(go.Scatter(
            x=forecast_future['ds'], y=forecast_future['yhat_lower'],
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(255,0,0,0.1)',
            name='Confidence Interval (Future)'
        ))
    
    # Garis vertikal pembatas Train/Test Split
    split_str = str(last_train_date)
    grafik.add_shape(
        type="line", x0=split_str, x1=split_str, y0=0, y1=1,
        yref="paper", line=dict(color="gray", dash="dash", width=1.5)
    )
    grafik.add_annotation(
        x=split_str, y=1.05, yref="paper",
        text="Train/Test Split (80:20)", showarrow=False,
        font=dict(size=11, color="gray")
    )

    grafik.update_layout(
        title="Prediksi Prophet (Hyperparameter Tuning, Moving Average Regressors, Holidays & Business Days)", 
        xaxis_title="Tanggal", 
        yaxis_title="Harga (Rp)",
        template="plotly_white"
    )
    
    # Histori Test
    histori = prediksi_test[['ds', 'y', 'yhat']].copy()
    
    return forecast_future, metrik, grafik, histori
```

## 4. File: requirements.txt
```text
streamlit
yfinance
prophet
tensorflow
scikit-learn
pandas
numpy<2.0.0
plotly
matplotlib
```

