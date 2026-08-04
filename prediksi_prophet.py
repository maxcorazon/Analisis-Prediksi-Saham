import pandas as pd
import numpy as np
import random
from prophet import Prophet
from prophet.plot import plot_components_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from prophet.make_holidays import make_holidays_df

def pred_prophet(data_saham, hari_kedepan=90):
    random.seed(42)
    np.random.seed(42)
    
    # Menyiapkan data
    df_prophet = data_saham[['Date', 'Close']].copy()
    df_prophet['Close'] = df_prophet['Close'].round(2)
    df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None).dt.normalize()

    # Menambahkan moving average sebagai regressor (MA20 & MA50)
    df_prophet['ma20'] = df_prophet['y'].rolling(window=20, min_periods=1).mean()
    df_prophet['ma50'] = df_prophet['y'].rolling(window=50, min_periods=1).mean()

    print("\nBukti Tabel Prophet")
    print(df_prophet.head(5).to_string(index=False))
    print("")

    # Train/Test Split 80:20 (Menghindari Data Leakage)
    split_index = int(len(df_prophet) * 0.8)
    df_train = df_prophet.iloc[:split_index].copy()
    df_test = df_prophet.iloc[split_index:].copy()

    print(f"Total Baris Data Asli: {len(df_prophet)}")
    print(f"Baris Data Latih (80%): {len(df_train)}")
    print(f"Baris Data Uji (20%): {len(df_test)}")
    print(f"Batas Train Akhir: {df_train['ds'].iloc[-1]}")
    print(f"Batas Test Awal: {df_test['ds'].iloc[0]}")
    print("")

    # Grid Search Hyperparameter Tuning (9 Kombinasi Parameter)
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

    # Membangun tabel hari libur Indonesia (Efek Bursa H-1 & H+1)
    holidays_custom = make_holidays_df(year_list=list(range(2016, 2027)), country='ID')
    holidays_custom['lower_window'] = -1
    holidays_custom['upper_window'] = 1

    last_train_date = df_train['ds'].iloc[-1]
    last_actual_date = df_prophet['ds'].iloc[-1]
    test_bdays = len(pd.bdate_range(start=last_train_date + pd.Timedelta(days=1), end=last_actual_date))
    total_periods = test_bdays + hari_kedepan

    grid_results = []
    print("\nMemulai Grid Search Hyperparameter Tuning")
    for idx, params in enumerate(param_grid, 1):
        random.seed(42)
        np.random.seed(42)
        
        model_tune = Prophet(
            daily_seasonality=False,
            yearly_seasonality=True,
            weekly_seasonality=True,
            seasonality_mode=params['sm'],
            changepoint_prior_scale=params['cps'],
            seasonality_prior_scale=params['sps'],
            n_changepoints=params['nc'],
            changepoint_range=params['cr'],
            holidays=holidays_custom
        )
        model_tune.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model_tune.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
        model_tune.add_regressor('ma20')
        model_tune.add_regressor('ma50')
        
        model_tune.fit(df_train, seed=42, inits=0, iter=1000)
        
        future_tune = model_tune.make_future_dataframe(periods=total_periods, freq='B')
        future_tune['ds'] = future_tune['ds'].dt.normalize()
        ma_data = df_prophet[['ds', 'ma20', 'ma50']].copy()
        future_tune = future_tune.merge(ma_data, on='ds', how='left')
        future_tune['ma20'] = future_tune['ma20'].ffill().bfill()
        future_tune['ma50'] = future_tune['ma50'].ffill().bfill()
        
        forecast_tune = model_tune.predict(future_tune)
        pred_test_tune = forecast_tune[['ds', 'yhat']].merge(df_test[['ds', 'y']], on='ds', how='inner')
        mape_tune = np.mean(np.abs((pred_test_tune['y'] - pred_test_tune['yhat']) / pred_test_tune['y'])) * 100
        
        grid_results.append({
            'params': params,
            'mape': mape_tune,
            'model': model_tune,
            'forecast': forecast_tune
        })
        print(f"Uji {idx}/9: Params={params} -> MAPE: {mape_tune:.2f}%")

    # Memilih kombinasi terbaik
    best_result = min(grid_results, key=lambda x: x['mape'])
    best_params = best_result['params']
    best_mape = best_result['mape']
    best_model = best_result['model']
    best_forecast = best_result['forecast']
    
    # Menghitung Evaluasi pada data test dari model terbaik
    prediksi_test = best_forecast[['ds', 'yhat']].merge(df_test[['ds', 'y']], on='ds', how='inner')
    y_asli = prediksi_test['y']
    y_prediksi = prediksi_test['yhat']
    
    mae = mean_absolute_error(y_asli, y_prediksi)
    rmse = np.sqrt(mean_squared_error(y_asli, y_prediksi))
    mape = np.mean(np.abs((y_asli - y_prediksi) / y_asli)) * 100

    metrik = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    print("\nHasil Grid Search Terbaik")
    print(f"Best Params: {best_params}")
    print(f"Best MAPE (saat tuning): {best_mape:.2f}%\n")

    # Menggunakan model terbaik
    forecast = best_forecast

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
    
    # Grafik Komponen Prophet (Trend, Weekly, Yearly Seasonality)
    grafik_komponen = plot_components_plotly(best_model, forecast)
    grafik_komponen.update_layout(
        title="Dekomposisi Komponen Model Prophet",
        template="plotly_white"
    )
    
    return forecast_future, metrik, grafik, histori, grafik_komponen