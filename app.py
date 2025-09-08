import streamlit as st
import pandas as pd
import joblib
import gdown
import os

st.title("Prediksi Hujan Besok")

# ambil model dari google drive
model_id = "1vw9qq0NPiVOdZpRfq26nTvggEv9mQWBs"
model_path = "random_forest_model.joblib"
scaler_path = "scaler.joblib"

# download model kalau belum ada
if not os.path.exists(model_path):
    gdown.download(f"https://drive.google.com/uc?id={model_id}", model_path, quiet=False)

# load model dan scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = list(scaler.feature_names_in_)

# buat template dataframe kosong
X = pd.DataFrame([[0]*len(feature_names)], columns=feature_names, dtype=float)

st.write("Silakan masukkan data cuaca hari ini. Data ini akan dipakai untuk memprediksi apakah besok hujan atau tidak.")

# input tanggal dan lokasi
year = st.selectbox("Tahun", list(range(2000, 2031)), index=20)
month = st.selectbox("Bulan", list(range(1, 13)))
day = st.selectbox("Hari", list(range(1, 32)))

# lokasi (dropdown ambil dari scaler)
locations = sorted([f.split("_", 1)[1] for f in feature_names if f.startswith("Location_")])
location = st.selectbox("Lokasi", locations)

# season manual
season = st.selectbox("Musim (1=Summer,2=Fall,3=Winter,4=Spring)", [1, 2, 3, 4])

# region (dropdown)
regions = sorted([f.split("_", 1)[1] for f in feature_names if f.startswith("Region_")])
region = st.selectbox("Region", regions)

# suhu dan curah hujan
min_temp = st.number_input("MinTemp (°C)", -5.0, 45.0, step=0.1)
max_temp = st.number_input("MaxTemp (°C)", -5.0, 50.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", 0.0, 370.0, step=0.1)
temp9am = st.number_input("Temp jam 9 pagi (°C)", -5.0, 45.0, step=0.1)
temp3pm = st.number_input("Temp jam 3 sore (°C)", -5.0, 45.0, step=0.1)

# kelembapan dan tekanan
humidity9am = st.number_input("Humidity 9am (%)", 0.0, 100.0, step=0.1)
humidity3pm = st.number_input("Humidity 3pm (%)", 0.0, 100.0, step=0.1)
pressure9am = st.number_input("Pressure 9am (hPa)", 980.0, 1045.0, step=0.1)
pressure3pm = st.number_input("Pressure 3pm (hPa)", 980.0, 1045.0, step=0.1)

# angin
wind_directions = sorted([f.split("_")[1] for f in feature_names if f.startswith("WindDir9am_")])
wind_gust_speed = st.number_input("Wind Gust Speed (km/h)", 0.0, 135.0, step=0.1)
wind_speed9am = st.number_input("Wind Speed 9am (km/h)", 0.0, 80.0, step=0.1)
wind_speed3pm = st.number_input("Wind Speed 3pm (km/h)", 0.0, 80.0, step=0.1)
wind_gust_dir = st.selectbox("Wind Gust Direction", wind_directions)
wind_dir9am = st.selectbox("Wind Direction 9am", wind_directions)
wind_dir3pm = st.selectbox("Wind Direction 3pm", wind_directions)

# rain today
rain_today = st.selectbox("Apakah hari ini hujan?", ["No", "Yes"])

# isi data ke X
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

# one-hot kategori
for prefix, val in [("Location", location), ("Region", region),
                    ("WindGustDir", wind_gust_dir),
                    ("WindDir9am", wind_dir9am),
                    ("WindDir3pm", wind_dir3pm)]:
    col = f"{prefix}_{val}"
    if col in X.columns:
        X.loc[0, col] = 1.0

# tombol prediksi
if st.button("Prediksi"):
    try:
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]

        if pred == 1:
            st.write(f"Hasil: Besok kemungkinan HUJAN 🌧️ (Probabilitas: {prob:.2%})")
        else:
            st.write(f"Hasil: Besok kemungkinan TIDAK HUJAN ☀️ (Probabilitas hujan: {prob:.2%})")

    except Exception as e:
        st.write("Terjadi error saat prediksi:", e)
