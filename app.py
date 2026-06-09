
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Spotify Audio Profiling AI", layout="centered")

st.title("🎵 AI Pengelompok Karakter Musik Spotify")
st.write("Aplikasi ini menggunakan model Unsupervised Machine Learning (K-Means) untuk mendeteksi jenis kelompok musik berdasarkan 9 fitur audio.")

@st.cache_resource
def load_artifacts():
    model = joblib.load('model_kmeans_spotify.pkl')
    scaler = joblib.load('scaler_spotify.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
    st.success("✅ Model AI & Scaler Berhasil Dimuat secara Real-Time!")
except:
    st.error("❌ File model tidak ditemukan. Pastikan Tahap 4 sudah di-run!")

st.markdown("---")
st.subheader("🎛️ Masukkan Karakteristik Audio Lagu Baru:")
st.caption("Silakan masukkan angka sesuai karakteristik lagu (Rentang standar umumnya 0.0 sampai 1.0)")

# MENGGUNAKAN NUMBER INPUT (Jauh lebih ringan dan anti-error dibanding slider)
col1, col2, col3 = st.columns(3)

with col1:
    danceability = st.number_input("Danceability (Kelayakan Dansa)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    energy = st.number_input("Energy (Intensitas Lagu)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    loudness = st.number_input("Loudness (Kebisingan dB)", min_value=-60.0, max_value=0.0, value=-10.0,  step=0.1)

with col2:
    speechiness = st.number_input("Speechiness (Rasio Kata)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    acousticness = st.number_input("Acousticness (Akustik)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    instrumentalness = st.number_input("Instrumentalness (Tanpa Vokal)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

with col3:
    liveness = st.number_input("Liveness (Suasana Live)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    valence = st.number_input("Valence (Tingkat Keceriaan)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    tempo = st.number_input("Tempo (Kecepatan BPM)", min_value=50.0, max_value=220.0, value=120.0, step=1.0)

if st.button("🚀 KLIK UNTUK PROSES DETEKSI AI", type="primary"):
    input_data = np.array([[danceability, energy, loudness, speechiness,
                            acousticness, instrumentalness, liveness, valence, tempo]])

    input_scaled = scaler.transform(input_data)
    prediksi_cluster = model.predict(input_scaled)[0]

    st.markdown("---")
    st.subheader(f"🔮 Hasil Analisis AI: **CLUSTER {prediksi_cluster}**")

    if prediksi_cluster == 2:
        st.info("🟢 **JENIS MUSIK: THE CHILL & ACOUSTIC VIBE**\n\nLagu ini dikategorikan ke dalam musik akustik, balada, atau instrumental slow yang sangat menenangkan dan cocok untuk relaksasi.")
    elif prediksi_cluster == 3:
        st.error("🟣 **JENIS MUSIK: THE HIGH-ENERGY & PARTY VIBE**\n\nLagu ini memiliki intensitas, energi, dan beat yang sangat tinggi. Sangat cocok untuk jenis musik Rock, Metal, EDM, atau musik workout.")
    elif prediksi_cluster == 1:
        st.success("🔵 **JENIS MUSIK: THE GROOVY & DANCEABLE BEATS**\n\nLagu ini memiliki ritme yang sangat asyik untuk dipakai bergoyang. Cocok untuk kategori R&B, Hip-Hop Modern, atau Pop komersial.")
    else:
        st.warning("🔴 **JENIS MUSIK: THE HYBRID / POP MAINSTREAM**\n\nLagu ini memiliki struktur fitur yang seimbang di tengah-tengah. Merupakan lagu pop radio konvensional yang ramah didengar di mana saja.")
