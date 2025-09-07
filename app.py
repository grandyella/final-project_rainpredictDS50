import streamlit as st
import pandas as pd
import joblib
import gdown
import os
from datetime import date

st.set_page_config(page_title="Prediksi Hujan Besok", layout="centered")
st.title("☔ Prediksi Hujan Besok")

# ----------------------------
# CONFIG
# ----------------------------
ID_MODEL = "1vw9qq0NPiVOdZpRfq26nTvggEv9mQWBs"   # ID model di Google Drive
MODEL_PATH = "random_forest_model.joblib"
SCALER_PATH = "scaler.joblib"  # harus ada di repo (kecil)

# Download model dari Drive jika belum ada
if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={ID_MODEL}", MODEL_PATH, quiet=False)

# Load model & scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = list(scaler.feature_names_in_)

# Template DataFrame kosong sesuai scaler
X_template = pd.DataFrame([[0]*len(feature_names)], columns=feature_names, dtype=float)

# ----------------------------
# FRONTEND
# ----------------------------
st.write("### Masukkan Data Cuaca")

today = date.today()
year = st.selectbox("Year", list(range(2000, today.year+1)), index=(today.year-2000))
month = st.selectbox("Month", list(range(1, 13)), index=today.month-1)
day = st.selectbox("Day", list(range(1, 32)), index=today.day-1)

location = st.text_input("Location (contoh: Sydney)")

min_temp = st.number_input("MinTemp (°C)", format="%.2f")
max_temp = st.number_input("MaxTemp (°C)", format="%.2f")
rainfall = st.number_input("Rainfall (mm)", format="%.2f")
humidity9am = st.number_input("Humidity 9am (%)", format="%.2f")
humidity3pm = st.number_input("Humidity 3pm (%)", format="%.2f")
wind_speed9am = st.number_input("Wind Speed 9am (km/h)", format="%.2f")
wind_speed3pm = st.number_input("Wind Speed 3pm (km/h)", format="%.2f")

rain_today = st.selectbox("RainToday", ["No", "Yes"])

# ----------------------------
# BUILD INPUT
# ----------------------------
X = X_template.copy()

try:
    # numeric
    X.loc[0, "Year"] = year
    X.loc[0, "Month"] = month
    X.loc[0, "Day"] = day
    X.loc[0, "MinTemp"] = min_temp
    X.loc[0, "MaxTemp"] = max_temp
    X.loc[0, "Rainfall"] = rainfall
    X.loc[0, "Humidity9am"] = humidity9am
    X.loc[0, "Humidity3pm"] = humidity3pm
    X.loc[0, "WindSpeed9am"] = wind_speed9am
    X.loc[0, "WindSpeed3pm"] = wind_speed3pm
    X.loc[0, "RainToday"] = 1 if rain_today == "Yes" else 0

    # one-hot Location
    if location:
        col_location = f"Location_{location}"
        if col_location in X.columns:
            X.loc[0, col_location] = 1.0
        else:
            st.warning(f"Location '{location}' tidak dikenali model.")

except Exception:
    st.info("Isi semua input terlebih dahulu untuk bisa prediksi.")

# ----------------------------
# PREDIKSI + PENJELASAN
# ----------------------------
if st.button("Prediksi"):
    try:
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]

        if int(pred) == 1:
            hasil = "🌧️ Besok kemungkinan **Hujan**"
            alasan = []
            if rainfall > 0:
                alasan.append("Sudah ada curah hujan hari ini.")
            if humidity3pm > 70 or humidity9am > 80:
                alasan.append("Kelembapan udara cukup tinggi.")
            if wind_speed3pm > 20:
                alasan.append("Kecepatan angin sore cukup kencang, berpotensi membawa hujan.")
            if not alasan:
                alasan.append("Model mendeteksi pola cuaca yang mendukung hujan.")
        else:
            hasil = "☀️ Besok kemungkinan **Tidak Hujan**"
            alasan = []
            if rainfall == 0:
                alasan.append("Hari ini tidak ada curah hujan.")
            if humidity3pm < 60 and humidity9am < 70:
                alasan.append("Kelembapan udara relatif rendah.")
            if max_temp > 25:
                alasan.append("Suhu cukup tinggi, cenderung cerah.")
            if not alasan:
                alasan.append("Model mendeteksi kondisi cuaca stabil tanpa potensi hujan.")

        st.success(hasil)
        st.write("**Penjelasan:**")
        for a in alasan:
            st.write(f"- {a}")

    except Exception as e:
        st.error(f"Gagal prediksi: {e}")
