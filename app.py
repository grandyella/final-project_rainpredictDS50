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

today = date.today()

# ----------------------------
# FRONTEND INPUT
# ----------------------------

with st.expander("📍 Informasi Lokasi & Tanggal", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        year = st.selectbox("Year", list(range(2000, today.year+1)), index=(today.year-2000))
    with col2:
        month = st.selectbox("Month", list(range(1, 13)), index=today.month-1)
    with col3:
        day = st.selectbox("Day", list(range(1, 32)), index=today.day-1)

    # Location dropdown
    locations = sorted({f.split("_", 1)[1] for f in feature_names if f.startswith("Location_")})
    location = st.selectbox("🏙️ Location", locations, help="Pilih lokasi stasiun cuaca")

    season = st.selectbox("🍂 Season", [1, 2, 3, 4],
                          format_func=lambda x: ["Summer","Fall","Winter","Spring"][x-1])

    regions = sorted({f.split("_", 1)[1] for f in feature_names if f.startswith("Region_")})
    region = st.selectbox("🌍 Region", regions)

with st.expander("🌡️ Suhu & Curah Hujan", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        min_temp = st.number_input("MinTemp (°C)", format="%.2f",
                                   help="Suhu minimum harian dalam °C")
        max_temp = st.number_input("MaxTemp (°C)", format="%.2f",
                                   help="Suhu maksimum harian dalam °C")
        rainfall = st.number_input("🌧️ Rainfall (mm)", format="%.2f",
                                   help="Jumlah curah hujan hari ini (mm)")
    with col2:
        temp9am = st.number_input("Temp 9am (°C)", format="%.2f")
        temp3pm = st.number_input("Temp 3pm (°C)", format="%.2f")

with st.expander("💧 Kelembapan & Tekanan Udara", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        humidity9am = st.number_input("Humidity 9am (%)", format="%.2f",
                                      help="Kelembapan udara pagi hari (%)")
        humidity3pm = st.number_input("Humidity 3pm (%)", format="%.2f",
                                      help="Kelembapan udara sore hari (%)")
    with col2:
        pressure9am = st.number_input("Pressure 9am (hPa)", format="%.2f",
                                      help="Tekanan udara pagi hari (hPa)")
        pressure3pm = st.number_input("Pressure 3pm (hPa)", format="%.2f",
                                      help="Tekanan udara sore hari (hPa)")

with st.expander("💨 Angin", expanded=False):
    wind_directions = sorted({f.split("_")[1] for f in feature_names if f.startswith("WindDir9am_")})

    col1, col2 = st.columns(2)
    with col1:
        wind_gust_speed = st.number_input("Wind Gust Speed (km/h)", format="%.2f",
                                          help="Kecepatan angin paling kencang")
        wind_speed9am = st.number_input("Wind Speed 9am (km/h)", format="%.2f")
        wind_gust_dir = st.selectbox("Wind Gust Dir", wind_directions)
    with col2:
        wind_speed3pm = st.number_input("Wind Speed 3pm (km/h)", format="%.2f")
        wind_dir9am = st.selectbox("Wind Dir 9am", wind_directions)
        wind_dir3pm = st.selectbox("Wind Dir 3pm", wind_directions)

rain_today = st.selectbox("☔ RainToday", ["No", "Yes"])

# ----------------------------
# BUILD INPUT
# ----------------------------
X = X_template.copy()

X.loc[0, "Year"] = year
X.loc[0, "Month"] = month
X.loc[0, "Day"] = day
X.loc[0, "Season"] = season
X.loc[0, "MinTemp"] = min_temp
X.loc[0, "MaxTemp"] = max_temp
X.loc[0, "Rainfall"] = rainfall
X.loc[0, "Temp9am"] = temp9am
X.loc[0, "Temp3pm"] = temp3pm
X.loc[0, "Humidity9am"] = humidity9am
X.loc[0, "Humidity3pm"] = humidity3pm
X.loc[0, "Pressure9am"] = pressure9am
X.loc[0, "Pressure3pm"] = pressure3pm
X.loc[0, "WindGustSpeed"] = wind_gust_speed
X.loc[0, "WindSpeed9am"] = wind_speed9am
X.loc[0, "WindSpeed3pm"] = wind_speed3pm
X.loc[0, "RainToday"] = 1 if rain_today == "Yes" else 0

# One-hot kategori
for prefix, val in [("Location", location), ("Region", region),
                    ("WindGustDir", wind_gust_dir),
                    ("WindDir9am", wind_dir9am),
                    ("WindDir3pm", wind_dir3pm)]:
    col = f"{prefix}_{val}"
    if col in X.columns:
        X.loc[0, col] = 1.0

# ----------------------------
# PREVIEW DATA
# ----------------------------
st.subheader("🔎 Preview Data Input")
st.dataframe(X.T.rename(columns={0: "Value"}))

# ----------------------------
# PREDIKSI + HASIL
# ----------------------------
if st.button("🚀 Prediksi"):
    try:
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]

        st.subheader("📊 Hasil Prediksi")
        if int(pred) == 1:
            st.success(f"🌧️ Besok kemungkinan **Hujan** (Probabilitas: {prob:.2%})")
        else:
            st.info(f"☀️ Besok kemungkinan **Tidak Hujan** (Probabilitas hujan: {prob:.2%})")

    except Exception as e:
        st.error(f"Gagal prediksi: {e}")
