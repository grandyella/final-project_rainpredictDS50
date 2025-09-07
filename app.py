import streamlit as st
import pandas as pd
import joblib
import gdown
import os

st.set_page_config(page_title="Prediksi Hujan Besok", layout="centered")
st.title("☔ Prediksi Hujan Besok")

# Lokasi file lokal
MODEL_PATH = "random_forest_model.joblib"
SCALER_PATH = "scaler.joblib"

# Ganti dengan ID file Google Drive model kamu
ID_MODEL = "1vw9qq0NPiVOdZpRfq26nTvggEv9mQWBs"

# Download model dari Google Drive kalau belum ada
if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={ID_MODEL}", MODEL_PATH, quiet=False)

# Load model & scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.write("### Masukkan Data Cuaca")

# Input user
min_temp = st.text_input("MinTemp (°C)", "15.0")
max_temp = st.text_input("MaxTemp (°C)", "30.0")
rainfall = st.text_input("Rainfall (mm)", "0.0")
humidity9am = st.text_input("Humidity 9am (%)", "65.0")
humidity3pm = st.text_input("Humidity 3pm (%)", "50.0")
wind_speed9am = st.text_input("Wind Speed 9am (km/h)", "10.0")
wind_speed3pm = st.text_input("Wind Speed 3pm (km/h)", "15.0")

if st.button("Prediksi"):
    try:
        # Konversi input ke float (ganti koma jadi titik dulu)
        data = [
            float(min_temp.replace(",", ".")),
            float(max_temp.replace(",", ".")),
            float(rainfall.replace(",", ".")),
            float(humidity9am.replace(",", ".")),
            float(humidity3pm.replace(",", ".")),
            float(wind_speed9am.replace(",", ".")),
            float(wind_speed3pm.replace(",", "."))
        ]

        # Buat DataFrame
        X_new = pd.DataFrame([data],
                             columns=["MinTemp","MaxTemp","Rainfall",
                                      "Humidity9am","Humidity3pm",
                                      "WindSpeed9am","WindSpeed3pm"])

        # Coba scaling
        try:
            X_scaled = scaler.transform(X_new)
        except ValueError:
            # fallback: pakai values kalau nama kolom beda
            X_scaled = scaler.transform(X_new.values)

        # Prediksi
        pred = model.predict(X_scaled)
        hasil = "🌧️ Hujan" if pred[0] == 1 else "☀️ Tidak Hujan"

        st.success(f"Hasil Prediksi: {hasil}")

    except Exception as e:
        st.error(f"Input tidak valid: {e}")
