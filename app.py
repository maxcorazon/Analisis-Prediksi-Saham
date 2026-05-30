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
        hasil_forecast, metrik_prophet, grafik_prophet, histori_prophet, grafik_komponen = cached_pred_prophet(data_saham, hari_prediksi)
    
    st.plotly_chart(grafik_prophet, use_container_width=True)
    
    st.subheader("Evaluasi Akurasi Model (Data Test 20%)")
    kolom_metrik1, kolom_metrik2, kolom_metrik3 = st.columns(3)
    kolom_metrik1.metric("MAE (Rata-rata Meleset)", f"Rp {metrik_prophet['MAE']:,.2f}")
    kolom_metrik2.metric("RMSE (Error Kuadrat)", f"Rp {metrik_prophet['RMSE']:,.2f}")
    kolom_metrik3.metric("MAPE (Persentase Error)", f"{metrik_prophet['MAPE']:.2f} %")
    
    # Komponen Prophet (Tren & Musiman)
    with st.expander("Lihat Dekomposisi Komponen Prophet (Tren & Musiman)"):
        st.write("Grafik ini membedah prediksi Prophet menjadi komponen pembentuknya: Tren jangka panjang, pola mingguan, dan pola tahunan. Ini membuktikan bahwa model mampu menangkap fluktuasi pasar.")
        st.plotly_chart(grafik_komponen, use_container_width=True)
    
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
        hasil_future_lstm, metrik_lstm, grafik_lstm, histori_lstm, grafik_loss_lstm = cached_pred_lstm(data_saham, hari_prediksi)
    
    st.plotly_chart(grafik_lstm, use_container_width=True)
    
    st.subheader("Evaluasi Akurasi Model LSTM (Data Test 20%)")
    kolom1_lstm, kolom2_lstm, kolom3_lstm = st.columns(3)
    kolom1_lstm.metric("MAE (Rata-rata Meleset)", f"Rp {metrik_lstm['MAE']:,.2f}")
    kolom2_lstm.metric("RMSE (Error Kuadrat)", f"Rp {metrik_lstm['RMSE']:,.2f}")
    kolom3_lstm.metric("MAPE (Persentase Error)", f"{metrik_lstm['MAPE']:.2f} %")
    
    # Kurva Pelatihan Model (Loss vs Epoch)
    with st.expander("Lihat Kurva Pelatihan Model (Loss vs Epoch)"):
        st.write("Grafik ini menunjukkan perbandingan nilai error pada data latih (Training Loss) dan data uji (Validation Loss) selama proses pelatihan model (Epoch). Penurunan yang stabil membuktikan bahwa model tidak mengalami *Overfitting* atau *Underfitting*.")
        st.plotly_chart(grafik_loss_lstm, use_container_width=True)
    
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