import streamlit as st
import pandas as pd
import joblib
import gdown
import os

st.title("Prediksi Hujan Besok 🌧️")

# Lokasi file lokal
MODEL_PATH = "random_forest_model.joblib"
SCALER_PATH = "scaler.joblib"

# ID Google Drive untuk model besar
ID_MODEL = "1vw9qq0NPiVOdZpRfq26nTvggEv9mQWBs"   # ganti dengan ID model kamu

# Download model dari Google Drive kalau belum ada di lokal
if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={ID_MODEL}", MODEL_PATH, quiet=False)

# Load model dan scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)   # ini kecil, bisa langsung upload ke GitHub

# Input dari user
min_temp = st.number_input("MinTemp")
max_temp = st.number_input("MaxTemp")
rainfall = st.number_input("Rainfall")
humidity9am = st.number_input("Humidity 9am")
humidity3pm = st.number_input("Humidity 3pm")
wind_speed9am = st.number_input("Wind Speed 9am")
wind_speed3pm = st.number_input("Wind Speed 3pm")

if st.button("Prediksi"):
    X_new = pd.DataFrame([[min_temp, max_temp, rainfall,
                           humidity9am, humidity3pm,
                           wind_speed9am, wind_speed3pm]],
                         columns=["MinTemp","MaxTemp","Rainfall",
                                  "Humidity9am","Humidity3pm",
                                  "WindSpeed9am","WindSpeed3pm"])
    X_scaled = scaler.transform(X_new)
    pred = model.predict(X_scaled)
    hasil = "🌧️ Hujan" if pred[0]==1 else "☀️ Tidak Hujan"
    st.subheader(f"Hasil Prediksi: {hasil}")
