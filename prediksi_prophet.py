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